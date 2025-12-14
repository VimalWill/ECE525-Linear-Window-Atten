"""
RULER Benchmark implementation for Infini-LLM
Tests real context understanding and length capabilities
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer

from .tasks import RULERTaskType, RULERTaskConfig, RULERTaskFactory


@dataclass
class RULERResult:
    """Result for a RULER evaluation"""
    task_type: str
    context_length: int
    num_examples: int
    accuracy: float
    avg_latency_ms: float
    examples: List[Dict[str, Any]]
    metrics: Dict[str, float]


class RULERBenchmark:
    """
    RULER Benchmark for evaluating long-context understanding

    Tests models on:
    1. Retrieval (NIAH variants)
    2. Multi-hop tracing
    3. Aggregation
    4. Question answering
    """

    def __init__(
        self,
        model,
        tokenizer_path: str,
        output_dir: str = "ruler_results",
        device: str = "cuda",
    ):
        """
        Initialize RULER benchmark

        Args:
            model: Infini-LLM model instance
            tokenizer_path: Path to tokenizer
            output_dir: Directory for saving results
            device: Device to run on
        """
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = device
        self.model = self.model.to(device)
        self.model.eval()

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.task_factory = RULERTaskFactory(self.tokenizer)

    def run_task(
        self,
        task_config: RULERTaskConfig,
        num_examples: int = 10,
        max_new_tokens: int = 20,
    ) -> RULERResult:
        """
        Run a RULER task

        Args:
            task_config: Task configuration
            num_examples: Number of examples to generate and test
            max_new_tokens: Maximum tokens to generate

        Returns:
            RULERResult object
        """
        print(f"\n{'='*70}")
        print(f"RULER Task: {task_config.task_type.value}")
        print(f"Context Length: {task_config.context_length} tokens")
        print(f"Num Examples: {num_examples}")
        print(f"{'='*70}")

        results = []
        correct = 0
        total_latency = 0.0

        for ex_idx in range(num_examples):
            print(f"\nExample {ex_idx + 1}/{num_examples}")

            # Generate task
            task_data = self.task_factory.create_task(task_config)

            # Run evaluation
            result = self._evaluate_example(task_data, max_new_tokens)

            results.append(result)

            if result["correct"]:
                correct += 1

            total_latency += result["latency_ms"]

            print(f"Expected: {result['expected']}")
            print(f"Generated: {result['generated']}")
            print(f"Correct: {result['correct']}")
            print(f"Latency: {result['latency_ms']:.2f}ms")

        # Calculate metrics
        accuracy = correct / num_examples
        avg_latency = total_latency / num_examples

        print(f"\n{'='*70}")
        print(f"Task Complete!")
        print(f"Accuracy: {accuracy:.2%} ({correct}/{num_examples})")
        print(f"Avg Latency: {avg_latency:.2f}ms")
        print(f"{'='*70}")

        # Create result object
        ruler_result = RULERResult(
            task_type=task_config.task_type.value,
            context_length=task_config.context_length,
            num_examples=num_examples,
            accuracy=accuracy,
            avg_latency_ms=avg_latency,
            examples=results,
            metrics={
                "accuracy": accuracy,
                "latency_ms": avg_latency,
                "num_correct": correct,
                "num_total": num_examples,
            }
        )

        # Save result
        self._save_result(ruler_result)

        return ruler_result

    def run_full_evaluation(
        self,
        context_lengths: List[int] = [4096, 8192, 16384, 32768],
        num_examples_per_task: int = 10,
    ) -> Dict[str, List[RULERResult]]:
        """
        Run full RULER evaluation across multiple context lengths

        Args:
            context_lengths: List of context lengths to test
            num_examples_per_task: Number of examples per task

        Returns:
            Dictionary mapping task types to lists of results
        """
        print(f"\n{'='*70}")
        print(f"Running Full RULER Evaluation")
        print(f"Context Lengths: {context_lengths}")
        print(f"{'='*70}")

        all_results = {}

        # Define tasks to run
        task_configs = [
            # Single needle - most basic
            (RULERTaskType.NIAH_SINGLE, 1, 1),
            # Multi-needle
            (RULERTaskType.NIAH_MULTI, 5, 1),
            # Multi-key retrieval
            (RULERTaskType.NIAH_MULTIKEY, 10, 1),
            # Multi-hop tracing
            (RULERTaskType.MULTIHOP_TRACING, 1, 3),
            # Aggregation
            (RULERTaskType.AGGREGATION, 20, 1),
        ]

        for task_type, num_needles, num_hops in task_configs:
            task_results = []

            for ctx_len in context_lengths:
                config = RULERTaskConfig(
                    task_type=task_type,
                    context_length=ctx_len,
                    num_needles=num_needles,
                    num_hops=num_hops,
                )

                result = self.run_task(config, num_examples_per_task)
                task_results.append(result)

            all_results[task_type.value] = task_results

        # Save summary
        self._save_summary(all_results, context_lengths)

        return all_results

    def _evaluate_example(
        self,
        task_data: Dict[str, Any],
        max_new_tokens: int,
    ) -> Dict[str, Any]:
        """
        Evaluate a single example

        Args:
            task_data: Task data dictionary
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with evaluation results
        """
        context = task_data["context"]
        query = task_data["query"]
        expected_answer = str(task_data["answer"])

        # Construct prompt
        prompt = f"{context}\n\nQuestion: {query}\nAnswer:"

        # Tokenize
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        prompt_len = input_ids.shape[1]

        # Generate
        start_time = time.time()
        generated_ids = self._generate(input_ids, max_new_tokens)
        latency = (time.time() - start_time) * 1000

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        # Extract answer (remove prompt)
        if prompt in generated_text:
            generated_answer = generated_text[len(prompt):].strip()
        else:
            generated_answer = generated_text[prompt_len:].strip()

        # Check correctness
        correct = self._check_answer(generated_answer, expected_answer, task_data["task_type"])

        return {
            "prompt_length": prompt_len,
            "context_length": len(self.tokenizer.encode(context)),
            "expected": expected_answer,
            "generated": generated_answer,
            "correct": correct,
            "latency_ms": latency,
        }

    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens from model"""
        generated = input_ids.clone()

        # Stop tokens: newline, EOS, and common answer separators
        stop_tokens = {
            self.tokenizer.eos_token_id,
            self.tokenizer.encode('\n', add_special_tokens=False)[0] if self.tokenizer.encode('\n', add_special_tokens=False) else None,
        }
        stop_tokens = {t for t in stop_tokens if t is not None}

        with torch.no_grad():
            outputs = self.model(generated, 0)

            for _ in range(max_new_tokens):
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                # Stop if we hit EOS or newline
                if next_token.item() in stop_tokens:
                    break

                generated = torch.cat([generated, next_token], dim=1)
                current_pos = generated.shape[1] - 1
                outputs = self.model(next_token, current_pos)

        return generated

    def _check_answer(
        self,
        generated: str,
        expected: str,
        task_type: str,
    ) -> bool:
        """
        Check if generated answer is correct

        Args:
            generated: Generated answer
            expected: Expected answer
            task_type: Type of task

        Returns:
            True if correct, False otherwise
        """
        generated_lower = generated.lower().strip()
        expected_lower = expected.lower().strip()

        # For numeric answers
        if expected.isdigit():
            # Extract numbers from generated text
            import re
            numbers = re.findall(r'\d+', generated)
            if numbers:
                return numbers[0] == expected

        # For multi-needle tasks, check if answer contains expected
        if "multi" in task_type:
            return expected_lower in generated_lower

        # For exact match
        return expected_lower == generated_lower or expected_lower in generated_lower

    def _save_result(self, result: RULERResult):
        """Save single task result"""
        filename = f"{result.task_type}_ctx{result.context_length}.json"
        output_file = self.output_dir / filename

        with open(output_file, 'w') as f:
            json.dump(asdict(result), f, indent=2)

        print(f"\nResult saved to: {output_file}")

    def _save_summary(
        self,
        all_results: Dict[str, List[RULERResult]],
        context_lengths: List[int],
    ):
        """Save summary of all results"""
        summary = {
            "context_lengths": context_lengths,
            "tasks": {},
        }

        for task_type, results in all_results.items():
            task_summary = {
                "accuracies": [r.accuracy for r in results],
                "latencies": [r.avg_latency_ms for r in results],
                "context_lengths": [r.context_length for r in results],
            }
            summary["tasks"][task_type] = task_summary

        output_file = self.output_dir / "ruler_summary.json"
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*70}")
        print(f"Summary saved to: {output_file}")
        print(f"{'='*70}")

        # Print summary table
        print("\nRULER Benchmark Summary:")
        print(f"{'Task':<25} | " + " | ".join(f"{cl:>6}K" for cl in [cl//1024 for cl in context_lengths]))
        print("-" * 70)

        for task_type, task_summary in summary["tasks"].items():
            accs = [f"{a:>6.1%}" for a in task_summary["accuracies"]]
            print(f"{task_type:<25} | " + " | ".join(accs))
