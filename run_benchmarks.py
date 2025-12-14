"""
Unified benchmark runner for Infini-LLM

Runs SCBench, LM Eval Harness, and RULER benchmarks on Infini-LLM models

Usage:
    # Run all benchmarks
    python run_benchmarks.py --model_path /path/to/model --benchmark all

    # Run specific benchmark
    python run_benchmarks.py --model_path /path/to/model --benchmark lm_eval

    # Run RULER with custom context lengths
    python run_benchmarks.py --model_path /path/to/model --benchmark ruler \
        --context_lengths 4096 8192 16384

    # Run LM Eval on specific tasks
    python run_benchmarks.py --model_path /path/to/model --benchmark lm_eval \
        --tasks hellaswag arc_easy winogrande
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from InfiniLLM import Llama
from benchmarks.lm_eval import LMEvalHarnessRunner
from benchmarks.scbench import SCBenchRunner
from benchmarks.ruler import RULERBenchmark, RULERTaskType, RULERTaskConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Run benchmarks on Infini-LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model configuration
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model checkpoint (defaults to LLAMA_DIR env variable)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 8192)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (default: 1)",
    )
    parser.add_argument(
        "--attn_method",
        type=str,
        default="window",
        choices=["softmax", "window", "linear"],
        help="Attention method (default: window)",
    )
    parser.add_argument(
        "--window_size",
        type=int,
        default=512,
        help="Window size for window attention (default: 512)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu)",
    )

    # Benchmark selection
    parser.add_argument(
        "--benchmark",
        type=str,
        required=True,
        choices=["all", "lm_eval", "scbench", "ruler"],
        help="Which benchmark(s) to run",
    )

    # Output configuration
    parser.add_argument(
        "--output_dir",
        type=str,
        default="benchmark_results",
        help="Output directory for results (default: benchmark_results)",
    )

    # LM Eval Harness specific
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="LM Eval tasks to run (default: hellaswag arc_easy arc_challenge winogrande piqa)",
    )
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for LM Eval (default: 0)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of examples per task (useful for testing)",
    )

    # RULER specific
    parser.add_argument(
        "--context_lengths",
        nargs="+",
        type=int,
        default=[4096, 8192, 16384],
        help="Context lengths for RULER (default: 4096 8192 16384)",
    )
    parser.add_argument(
        "--ruler_tasks",
        nargs="+",
        choices=["niah_single", "niah_multi", "niah_multikey", "multihop", "aggregation", "all"],
        default=["all"],
        help="RULER tasks to run (default: all)",
    )
    parser.add_argument(
        "--num_examples",
        type=int,
        default=10,
        help="Number of examples per RULER task (default: 10)",
    )

    # SCBench specific
    parser.add_argument(
        "--scbench_mode",
        type=str,
        choices=["multi_turn", "multi_request", "both"],
        default="both",
        help="SCBench evaluation mode (default: both)",
    )

    return parser.parse_args()


def load_model(args):
    """Load Infini-LLM model"""
    model_path = args.model_path or os.getenv("LLAMA_DIR")
    if not model_path:
        raise ValueError(
            "Model path must be provided via --model_path or LLAMA_DIR environment variable"
        )

    print(f"\n{'='*70}")
    print(f"Loading Infini-LLM Model")
    print(f"{'='*70}")
    print(f"Model Path: {model_path}")
    print(f"Sequence Length: {args.seq_len}")
    print(f"Attention Method: {args.attn_method}")
    print(f"Window Size: {args.window_size}")
    print(f"Device: {args.device}")
    print(f"{'='*70}\n")

    model = Llama.LLama.build(
        chkpt_dir=model_path,
        model_seq_len=args.seq_len,
        model_batch_size=args.batch_size,
        use_cache=True,
        attn_method=args.attn_method,
        window_size=args.window_size,
    )

    model = model.to(args.device)
    model.eval()

    return model, model_path


def run_lm_eval(model, model_path, args):
    """Run LM Evaluation Harness"""
    print(f"\n{'='*70}")
    print(f"Running LM Evaluation Harness")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir) / "lm_eval"
    runner = LMEvalHarnessRunner(
        model=model,
        tokenizer_path=model_path,
        output_dir=str(output_dir),
        batch_size=args.batch_size,
        device=args.device,
    )

    # Use default tasks if none specified
    tasks = args.tasks or ["hellaswag", "arc_easy", "arc_challenge", "winogrande", "piqa"]

    results = runner.run(
        tasks=tasks,
        num_fewshot=args.num_fewshot,
        limit=args.limit,
    )

    print(f"\n{'='*70}")
    print(f"LM Eval Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")

    return results


def run_scbench(model, model_path, args):
    """Run SCBench evaluation"""
    print(f"\n{'='*70}")
    print(f"Running SCBench (KV Cache-Centric Benchmark)")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir) / "scbench"
    runner = SCBenchRunner(
        model=model,
        tokenizer_path=model_path,
        output_dir=str(output_dir),
        device=args.device,
        track_kv_cache=True,
    )

    # Example: Run a simple multi-turn session
    # In practice, you would load SCBench dataset
    shared_context = """
    The Python programming language was created by Guido van Rossum and first released in 1991.
    Python emphasizes code readability with significant whitespace. It supports multiple programming
    paradigms including procedural, object-oriented, and functional programming. The language has
    a comprehensive standard library and a large ecosystem of third-party packages.
    """

    queries = [
        "Who created Python?",
        "When was Python first released?",
        "What does Python emphasize?",
        "What programming paradigms does Python support?",
    ]

    expected = [
        "Guido van Rossum",
        "1991",
        "code readability",
        "procedural, object-oriented, and functional programming",
    ]

    if args.scbench_mode in ["multi_turn", "both"]:
        print("\nRunning Multi-Turn Session...")
        result = runner.run_multi_turn_session(
            shared_context=shared_context,
            queries=queries,
            expected_answers=expected,
        )
        print(f"\nMulti-turn results: {result}")

    if args.scbench_mode in ["multi_request", "both"]:
        print("\nRunning Multi-Request Batch...")
        requests = [{"query": q, "expected": e} for q, e in zip(queries, expected)]
        result = runner.run_multi_request_batch(
            shared_context=shared_context,
            requests=requests,
        )
        print(f"\nMulti-request results: {result}")

    print(f"\n{'='*70}")
    print(f"SCBench Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


def run_ruler(model, model_path, args):
    """Run RULER benchmark"""
    print(f"\n{'='*70}")
    print(f"Running RULER Benchmark")
    print(f"{'='*70}\n")

    output_dir = Path(args.output_dir) / "ruler"
    benchmark = RULERBenchmark(
        model=model,
        tokenizer_path=model_path,
        output_dir=str(output_dir),
        device=args.device,
    )

    # Map task names
    task_map = {
        "niah_single": RULERTaskType.NIAH_SINGLE,
        "niah_multi": RULERTaskType.NIAH_MULTI,
        "niah_multikey": RULERTaskType.NIAH_MULTIKEY,
        "multihop": RULERTaskType.MULTIHOP_TRACING,
        "aggregation": RULERTaskType.AGGREGATION,
    }

    if "all" in args.ruler_tasks:
        # Run full evaluation
        results = benchmark.run_full_evaluation(
            context_lengths=args.context_lengths,
            num_examples_per_task=args.num_examples,
        )
    else:
        # Run specific tasks
        results = {}
        for task_name in args.ruler_tasks:
            task_type = task_map.get(task_name)
            if not task_type:
                print(f"Warning: Unknown task {task_name}, skipping...")
                continue

            task_results = []
            for ctx_len in args.context_lengths:
                config = RULERTaskConfig(
                    task_type=task_type,
                    context_length=ctx_len,
                    num_needles=5 if "multi" in task_name else 1,
                    num_hops=3 if task_name == "multihop" else 1,
                )

                result = benchmark.run_task(config, args.num_examples)
                task_results.append(result)

            results[task_name] = task_results

    print(f"\n{'='*70}")
    print(f"RULER Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")

    return results


def main():
    """Main entry point"""
    args = parse_args()

    # Load model
    model, model_path = load_model(args)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run requested benchmarks
    if args.benchmark == "all":
        run_lm_eval(model, model_path, args)
        run_scbench(model, model_path, args)
        run_ruler(model, model_path, args)
    elif args.benchmark == "lm_eval":
        run_lm_eval(model, model_path, args)
    elif args.benchmark == "scbench":
        run_scbench(model, model_path, args)
    elif args.benchmark == "ruler":
        run_ruler(model, model_path, args)

    print(f"\n{'='*70}")
    print(f"All Benchmarks Complete!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
