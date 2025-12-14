"""
Runner for SCBench (SharedContextBench) evaluation
Focuses on KV cache-centric analysis across multi-turn and multi-request scenarios
"""

import os
import json
import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from transformers import AutoTokenizer

from .tasks import SCBenchTaskRegistry, SCBenchMode, SCBenchTaskType


@dataclass
class KVCacheStats:
    """Statistics about KV cache usage"""
    cache_size_mb: float
    cache_hits: int
    cache_misses: int
    reuse_ratio: float
    compression_ratio: Optional[float] = None


@dataclass
class SCBenchResult:
    """Result for a single SCBench evaluation"""
    task_name: str
    mode: str
    num_turns: int
    accuracy: float
    latency_ms: float
    kv_cache_stats: KVCacheStats
    metrics: Dict[str, float]
    examples: List[Dict[str, Any]]


class SCBenchRunner:
    """
    Runner for SCBench evaluation on Infini-LLM

    SCBench evaluates models across 4 stages of KV cache lifecycle:
    1. KV cache generation (prefill)
    2. KV cache compression
    3. KV cache retrieval (decode)
    4. KV cache loading (reuse)
    """

    def __init__(
        self,
        model,
        tokenizer_path: str,
        output_dir: str = "scbench_results",
        device: str = "cuda",
        track_kv_cache: bool = True,
    ):
        """
        Initialize SCBench runner

        Args:
            model: Infini-LLM model instance
            tokenizer_path: Path to tokenizer
            output_dir: Directory for results
            device: Device to run on
            track_kv_cache: Whether to track KV cache statistics
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

        self.track_kv_cache = track_kv_cache
        self.task_registry = SCBenchTaskRegistry()

    def run_multi_turn_session(
        self,
        shared_context: str,
        queries: List[str],
        expected_answers: Optional[List[str]] = None,
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Run a multi-turn evaluation session with shared context

        Args:
            shared_context: The shared context (e.g., long document)
            queries: List of queries to ask about the context
            expected_answers: Optional list of expected answers
            max_new_tokens: Maximum tokens to generate per query

        Returns:
            Dictionary with results including KV cache statistics
        """
        print(f"\n{'='*70}")
        print(f"Multi-Turn Session: {len(queries)} queries")
        print(f"Shared context length: {len(self.tokenizer.encode(shared_context))} tokens")
        print(f"{'='*70}")

        # Encode shared context once
        context_tokens = self.tokenizer.encode(shared_context, return_tensors="pt").to(self.device)
        context_len = context_tokens.shape[1]

        # Process shared context (prefill stage)
        start_time = time.time()
        with torch.no_grad():
            _ = self.model(context_tokens, 0)
        prefill_time = (time.time() - start_time) * 1000

        print(f"Prefill time: {prefill_time:.2f}ms")

        # Track results for each turn
        turn_results = []
        total_decode_time = 0.0

        for turn_idx, query in enumerate(queries):
            print(f"\nTurn {turn_idx + 1}/{len(queries)}: {query}")

            # Encode query
            full_prompt = f"{shared_context}\n\nQuery: {query}\nAnswer:"
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

            # Generate response
            start_time = time.time()
            generated_ids = self._generate(input_ids, max_new_tokens)
            decode_time = (time.time() - start_time) * 1000
            total_decode_time += decode_time

            # Decode response
            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # Remove prompt from response
            if full_prompt in response:
                response = response[len(full_prompt):].strip()

            print(f"Response: {response}")
            print(f"Decode time: {decode_time:.2f}ms")

            # Calculate accuracy if expected answer provided
            accuracy = None
            if expected_answers and turn_idx < len(expected_answers):
                accuracy = self._calculate_accuracy(response, expected_answers[turn_idx])
                print(f"Accuracy: {accuracy:.2f}")

            turn_results.append({
                "turn": turn_idx + 1,
                "query": query,
                "response": response,
                "expected": expected_answers[turn_idx] if expected_answers and turn_idx < len(expected_answers) else None,
                "accuracy": accuracy,
                "latency_ms": decode_time,
            })

        # Calculate KV cache statistics
        kv_stats = self._get_kv_cache_stats() if self.track_kv_cache else None

        return {
            "mode": "multi_turn",
            "num_turns": len(queries),
            "context_length": context_len,
            "prefill_time_ms": prefill_time,
            "total_decode_time_ms": total_decode_time,
            "avg_decode_time_ms": total_decode_time / len(queries),
            "kv_cache_stats": kv_stats,
            "turns": turn_results,
        }

    def run_multi_request_batch(
        self,
        shared_context: str,
        requests: List[Dict[str, str]],
        max_new_tokens: int = 100,
    ) -> Dict[str, Any]:
        """
        Run multi-request evaluation with shared context (simulating parallel requests)

        Args:
            shared_context: The shared context
            requests: List of request dicts with 'query' and optionally 'expected'
            max_new_tokens: Maximum tokens to generate

        Returns:
            Dictionary with results
        """
        print(f"\n{'='*70}")
        print(f"Multi-Request Batch: {len(requests)} parallel requests")
        print(f"Shared context length: {len(self.tokenizer.encode(shared_context))} tokens")
        print(f"{'='*70}")

        # In multi-request mode, context is shared but queries are processed in parallel
        # For single-batch models, we process sequentially but measure cache reuse
        results = []

        for req_idx, request in enumerate(requests):
            query = request.get("query", "")
            expected = request.get("expected", None)

            print(f"\nRequest {req_idx + 1}/{len(requests)}: {query}")

            full_prompt = f"{shared_context}\n\nQuery: {query}\nAnswer:"
            input_ids = self.tokenizer.encode(full_prompt, return_tensors="pt").to(self.device)

            start_time = time.time()
            generated_ids = self._generate(input_ids, max_new_tokens)
            latency = (time.time() - start_time) * 1000

            response = self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if full_prompt in response:
                response = response[len(full_prompt):].strip()

            accuracy = self._calculate_accuracy(response, expected) if expected else None

            print(f"Response: {response}")
            print(f"Latency: {latency:.2f}ms")
            if accuracy is not None:
                print(f"Accuracy: {accuracy:.2f}")

            results.append({
                "request_id": req_idx + 1,
                "query": query,
                "response": response,
                "expected": expected,
                "accuracy": accuracy,
                "latency_ms": latency,
            })

        # Calculate aggregate statistics
        accuracies = [r["accuracy"] for r in results if r["accuracy"] is not None]
        avg_accuracy = np.mean(accuracies) if accuracies else None

        return {
            "mode": "multi_request",
            "num_requests": len(requests),
            "avg_accuracy": avg_accuracy,
            "avg_latency_ms": np.mean([r["latency_ms"] for r in results]),
            "kv_cache_stats": self._get_kv_cache_stats() if self.track_kv_cache else None,
            "requests": results,
        }

    def run_task(
        self,
        task_name: str,
        dataset: Dict[str, Any],
        limit: Optional[int] = None,
    ) -> SCBenchResult:
        """
        Run a specific SCBench task

        Args:
            task_name: Name of the task
            dataset: Dataset containing examples
            limit: Limit number of examples

        Returns:
            SCBenchResult object
        """
        task = self.task_registry.get_task(task_name)
        print(f"\n{'='*70}")
        print(f"Running SCBench Task: {task.name}")
        print(f"Type: {task.task_type.value}")
        print(f"Mode: {task.mode.value}")
        print(f"Description: {task.description}")
        print(f"{'='*70}")

        examples = dataset.get("examples", [])
        if limit:
            examples = examples[:limit]

        all_results = []
        total_latency = 0.0

        for idx, example in enumerate(examples):
            shared_context = example.get("context", "")
            queries = example.get("queries", [])
            expected = example.get("expected", [])

            if task.mode == SCBenchMode.MULTI_TURN:
                result = self.run_multi_turn_session(
                    shared_context=shared_context,
                    queries=queries,
                    expected_answers=expected,
                )
            else:  # MULTI_REQUEST
                requests = [
                    {"query": q, "expected": e}
                    for q, e in zip(queries, expected)
                ]
                result = self.run_multi_request_batch(
                    shared_context=shared_context,
                    requests=requests,
                )

            all_results.append(result)
            total_latency += result.get("total_decode_time_ms", result.get("avg_latency_ms", 0))

        # Aggregate metrics
        avg_latency = total_latency / len(all_results) if all_results else 0
        kv_stats = self._get_kv_cache_stats() if self.track_kv_cache else None

        result = SCBenchResult(
            task_name=task.name,
            mode=task.mode.value,
            num_turns=len(examples),
            accuracy=0.0,  # Calculate based on task-specific metrics
            latency_ms=avg_latency,
            kv_cache_stats=kv_stats,
            metrics={},
            examples=all_results,
        )

        # Save result
        self._save_result(result)

        return result

    def _generate(self, input_ids: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        """Generate tokens from the model"""
        generated = input_ids.clone()

        with torch.no_grad():
            outputs = self.model(generated, 0)

            for _ in range(max_new_tokens):
                next_token_logits = outputs[:, -1, :]
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

                if next_token.item() == self.tokenizer.eos_token_id:
                    break

                generated = torch.cat([generated, next_token], dim=1)
                current_pos = generated.shape[1] - 1
                outputs = self.model(next_token, current_pos)

        return generated

    def _calculate_accuracy(self, response: str, expected: str) -> float:
        """Calculate accuracy score between response and expected answer"""
        # Simple exact match or substring match
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        if response_lower == expected_lower:
            return 1.0
        elif expected_lower in response_lower:
            return 0.8
        else:
            # Could use more sophisticated metrics (F1, ROUGE, etc.)
            return 0.0

    def _get_kv_cache_stats(self) -> Optional[KVCacheStats]:
        """Extract KV cache statistics from the model"""
        if not self.track_kv_cache:
            return None

        # This is a placeholder - actual implementation depends on your model's cache structure
        # You'll need to adapt this based on how Infini-LLM exposes cache stats
        try:
            # Example: access cache from first layer
            if hasattr(self.model, 'layers') and len(self.model.layers) > 0:
                layer = self.model.layers[0]
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'cache'):
                    cache = layer.attention.cache
                    # Calculate approximate cache size
                    cache_size = 0
                    if cache is not None:
                        cache_size = cache.element_size() * cache.nelement() / (1024 * 1024)

                    return KVCacheStats(
                        cache_size_mb=cache_size,
                        cache_hits=0,  # Implement if available
                        cache_misses=0,  # Implement if available
                        reuse_ratio=0.0,  # Implement if available
                    )
        except Exception as e:
            print(f"Warning: Could not extract KV cache stats: {e}")

        return None

    def _save_result(self, result: SCBenchResult):
        """Save result to file"""
        output_file = self.output_dir / f"{result.task_name}_{result.mode}_result.json"

        # Convert to dict
        result_dict = asdict(result)

        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)

        print(f"\nResults saved to: {output_file}")
