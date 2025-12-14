"""
Runner for LM Evaluation Harness benchmarks
"""

import os
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

try:
    import lm_eval
    from lm_eval import evaluator
    from lm_eval.tasks import TaskManager
except ImportError:
    print("Warning: lm-evaluation-harness not installed. Run: pip install lm-eval")
    lm_eval = None

from .model_wrapper import InfiniLLMWrapper


class LMEvalHarnessRunner:
    """
    Runner for executing LM Evaluation Harness benchmarks on Infini-LLM
    """

    def __init__(
        self,
        model,
        tokenizer_path: str,
        output_dir: str = "eval_results",
        batch_size: int = 1,
        device: str = "cuda",
    ):
        """
        Initialize the LM Eval Harness runner

        Args:
            model: Infini-LLM model instance
            tokenizer_path: Path to the tokenizer
            output_dir: Directory to save evaluation results
            batch_size: Batch size for evaluation
            device: Device to run on
        """
        if lm_eval is None:
            raise ImportError(
                "lm-evaluation-harness is required. "
                "Install it with: pip install lm-eval"
            )

        self.model_wrapper = InfiniLLMWrapper(
            model=model,
            tokenizer_path=tokenizer_path,
            batch_size=batch_size,
            device=device,
        )

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(
        self,
        tasks: Optional[List[str]] = None,
        num_fewshot: int = 0,
        limit: Optional[int] = None,
        bootstrap_iters: int = 100000,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation on specified tasks

        Args:
            tasks: List of task names to evaluate (e.g., ['hellaswag', 'arc_easy'])
                  If None, defaults to common benchmarks
            num_fewshot: Number of few-shot examples
            limit: Limit number of examples per task (useful for testing)
            bootstrap_iters: Number of bootstrap iterations for uncertainty estimation
            **kwargs: Additional arguments to pass to evaluator

        Returns:
            Dictionary containing evaluation results
        """
        if tasks is None:
            # Default to common benchmarks
            tasks = [
                "hellaswag",
                "arc_easy",
                "arc_challenge",
                "winogrande",
                "piqa",
            ]

        print(f"Running evaluation on tasks: {tasks}")
        print(f"Few-shot: {num_fewshot}, Limit: {limit}")

        # Run evaluation
        results = evaluator.simple_evaluate(
            model=self.model_wrapper,
            tasks=tasks,
            num_fewshot=num_fewshot,
            limit=limit,
            bootstrap_iters=bootstrap_iters,
            **kwargs
        )

        # Save results
        self._save_results(results, tasks)

        return results

    def run_custom_tasks(
        self,
        task_config_path: str,
        tasks: List[str],
        **kwargs
    ) -> Dict[str, Any]:
        """
        Run evaluation on custom tasks defined in a config file

        Args:
            task_config_path: Path to custom task configuration directory
            tasks: List of custom task names
            **kwargs: Additional arguments for evaluation

        Returns:
            Dictionary containing evaluation results
        """
        # Initialize task manager with custom tasks
        task_manager = TaskManager(include_path=task_config_path)

        results = evaluator.simple_evaluate(
            model=self.model_wrapper,
            tasks=tasks,
            task_manager=task_manager,
            **kwargs
        )

        self._save_results(results, tasks, prefix="custom")

        return results

    def _save_results(
        self,
        results: Dict[str, Any],
        tasks: List[str],
        prefix: str = ""
    ):
        """
        Save evaluation results to file

        Args:
            results: Evaluation results dictionary
            tasks: List of task names
            prefix: Prefix for output filename
        """
        # Create filename
        tasks_str = "_".join(tasks[:3])  # Use first 3 tasks in filename
        if len(tasks) > 3:
            tasks_str += "_etc"

        if prefix:
            filename = f"{prefix}_{tasks_str}_results.json"
        else:
            filename = f"{tasks_str}_results.json"

        output_path = self.output_dir / filename

        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {output_path}")

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)

        if "results" in results:
            for task_name, task_results in results["results"].items():
                print(f"\n{task_name}:")
                for metric_name, metric_value in task_results.items():
                    if isinstance(metric_value, (int, float)):
                        print(f"  {metric_name}: {metric_value:.4f}")

    def list_available_tasks(self) -> List[str]:
        """
        List all available tasks in lm-evaluation-harness

        Returns:
            List of available task names
        """
        task_manager = TaskManager()
        return task_manager.all_tasks

    def get_task_info(self, task_name: str) -> Dict[str, Any]:
        """
        Get information about a specific task

        Args:
            task_name: Name of the task

        Returns:
            Dictionary containing task information
        """
        task_manager = TaskManager()
        task = task_manager.load_task_or_group(task_name)

        return {
            "name": task_name,
            "description": getattr(task, "DESCRIPTION", "No description available"),
            "metrics": getattr(task, "metrics", []),
        }
