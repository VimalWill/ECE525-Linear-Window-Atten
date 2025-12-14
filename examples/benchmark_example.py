"""
Example script demonstrating how to use the benchmark suite programmatically

This shows how to:
1. Load a model
2. Run each benchmark individually
3. Customize benchmark parameters
4. Process and visualize results
"""

import os
import sys
import torch
import json
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from InfiniLLM import Llama
from benchmarks.lm_eval import LMEvalHarnessRunner
from benchmarks.scbench import SCBenchRunner
from benchmarks.ruler import RULERBenchmark, RULERTaskType, RULERTaskConfig


def load_model():
    """Load Infini-LLM model"""
    chkpt_dir = os.getenv("LLAMA_DIR")
    if not chkpt_dir:
        raise ValueError("Please set LLAMA_DIR environment variable")

    print("Loading model...")
    model = Llama.LLama.build(
        chkpt_dir=chkpt_dir,
        max_seq_len=8192,
        max_batch_size=1,
        use_cache=True,
        attn_method="window",
        window_size=512,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    return model, chkpt_dir, device


def example_lm_eval(model, tokenizer_path, device):
    """Example: Running LM Evaluation Harness"""
    print("\n" + "="*70)
    print("Example 1: LM Evaluation Harness")
    print("="*70)

    runner = LMEvalHarnessRunner(
        model=model,
        tokenizer_path=tokenizer_path,
        output_dir="example_results/lm_eval",
        batch_size=1,
        device=device,
    )

    # Run on a subset of tasks with limited examples
    results = runner.run(
        tasks=["hellaswag", "arc_easy"],
        num_fewshot=0,
        limit=10,  # Limit to 10 examples for quick testing
    )

    print("\nResults:")
    for task, metrics in results.get("results", {}).items():
        print(f"\n{task}:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")

    return results


def example_scbench(model, tokenizer_path, device):
    """Example: Running SCBench"""
    print("\n" + "="*70)
    print("Example 2: SCBench (KV Cache-Centric)")
    print("="*70)

    runner = SCBenchRunner(
        model=model,
        tokenizer_path=tokenizer_path,
        output_dir="example_results/scbench",
        device=device,
        track_kv_cache=True,
    )

    # Example shared context
    shared_context = """
    Machine learning is a subset of artificial intelligence that enables systems to learn
    and improve from experience without being explicitly programmed. Deep learning, a subset
    of machine learning, uses neural networks with multiple layers to progressively extract
    higher-level features from raw input. Transformers, introduced in 2017, revolutionized
    natural language processing by using self-attention mechanisms. These models can process
    entire sequences simultaneously, making them highly parallelizable and efficient.
    """

    # Multi-turn session
    queries = [
        "What is machine learning?",
        "What is deep learning?",
        "When were transformers introduced?",
        "What makes transformers efficient?",
    ]

    expected = [
        "subset of artificial intelligence",
        "subset of machine learning using neural networks",
        "2017",
        "self-attention and parallelization",
    ]

    print("\nRunning multi-turn session...")
    result = runner.run_multi_turn_session(
        shared_context=shared_context,
        queries=queries,
        expected_answers=expected,
        max_new_tokens=50,
    )

    print(f"\nPrefill time: {result['prefill_time_ms']:.2f}ms")
    print(f"Avg decode time per query: {result['avg_decode_time_ms']:.2f}ms")
    print(f"\nTurn results:")
    for turn in result['turns']:
        print(f"  Turn {turn['turn']}: {turn['query']}")
        print(f"    Response: {turn['response']}")
        print(f"    Latency: {turn['latency_ms']:.2f}ms")

    return result


def example_ruler(model, tokenizer_path, device):
    """Example: Running RULER"""
    print("\n" + "="*70)
    print("Example 3: RULER Benchmark")
    print("="*70)

    benchmark = RULERBenchmark(
        model=model,
        tokenizer_path=tokenizer_path,
        output_dir="example_results/ruler",
        device=device,
    )

    # Test multiple tasks at different context lengths
    context_lengths = [2048, 4096]
    results = {}

    for ctx_len in context_lengths:
        print(f"\n--- Testing at {ctx_len} tokens ---")

        # Single needle task
        config = RULERTaskConfig(
            task_type=RULERTaskType.NIAH_SINGLE,
            context_length=ctx_len,
            num_needles=1,
            haystack_type="essay",
        )

        result = benchmark.run_task(config, num_examples=5)

        results[f"niah_single_{ctx_len}"] = result

        print(f"\nResults at {ctx_len} tokens:")
        print(f"  Accuracy: {result.accuracy:.2%}")
        print(f"  Avg Latency: {result.avg_latency_ms:.2f}ms")

    return results


def example_ruler_full_suite(model, tokenizer_path, device):
    """Example: Running full RULER evaluation suite"""
    print("\n" + "="*70)
    print("Example 4: Full RULER Suite (Multiple Tasks)")
    print("="*70)

    benchmark = RULERBenchmark(
        model=model,
        tokenizer_path=tokenizer_path,
        output_dir="example_results/ruler_full",
        device=device,
    )

    # Run comprehensive evaluation
    results = benchmark.run_full_evaluation(
        context_lengths=[2048, 4096],  # Using smaller lengths for quick demo
        num_examples_per_task=5,  # Small number for quick testing
    )

    print("\nFull evaluation complete!")
    print("\nSummary:")
    for task_type, task_results in results.items():
        print(f"\n{task_type}:")
        for result in task_results:
            print(f"  {result.context_length} tokens: {result.accuracy:.2%} accuracy")

    return results


def visualize_ruler_results(results):
    """Visualize RULER results"""
    print("\n" + "="*70)
    print("Example 5: Visualizing Results")
    print("="*70)

    # Extract data for plotting
    context_lengths = []
    accuracies = []

    for key, result in results.items():
        if "niah_single" in key:
            context_lengths.append(result.context_length)
            accuracies.append(result.accuracy)

    # Create plot
    plt.figure(figsize=(10, 6))
    plt.plot(context_lengths, accuracies, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Context Length (tokens)', fontsize=12, fontweight='bold')
    plt.ylabel('Accuracy', fontsize=12, fontweight='bold')
    plt.title('RULER NIAH Performance vs Context Length', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.1])

    # Add horizontal line at 100%
    plt.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Perfect Score')
    plt.legend()

    # Save plot
    output_dir = Path("example_results")
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / "ruler_visualization.png", dpi=300, bbox_inches='tight')
    print(f"\nVisualization saved to: {output_dir / 'ruler_visualization.png'}")

    plt.show()


def compare_attention_methods():
    """Example: Compare different attention methods"""
    print("\n" + "="*70)
    print("Example 6: Comparing Attention Methods")
    print("="*70)

    chkpt_dir = os.getenv("LLAMA_DIR")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    methods = ["full", "window"]
    results = {}

    for method in methods:
        print(f"\nTesting {method} attention...")

        # Load model with specific attention method
        model = Llama.LLama.build(
            chkpt_dir=chkpt_dir,
            max_seq_len=4096,
            max_batch_size=1,
            use_cache=True,
            attn_method=method,
            window_size=512 if method == "window" else None,
        )
        model = model.to(device)
        model.eval()

        # Run quick RULER test
        benchmark = RULERBenchmark(
            model=model,
            tokenizer_path=chkpt_dir,
            output_dir=f"example_results/ruler_{method}",
            device=device,
        )

        config = RULERTaskConfig(
            task_type=RULERTaskType.NIAH_SINGLE,
            context_length=2048,
            num_needles=1,
        )

        result = benchmark.run_task(config, num_examples=3)
        results[method] = result

        print(f"{method} attention - Accuracy: {result.accuracy:.2%}, "
              f"Latency: {result.avg_latency_ms:.2f}ms")

    return results


def main():
    """Run all examples"""
    print("="*70)
    print("Infini-LLM Benchmark Examples")
    print("="*70)

    # Load model once
    model, tokenizer_path, device = load_model()

    # Example 1: LM Eval Harness
    try:
        lm_eval_results = example_lm_eval(model, tokenizer_path, device)
    except ImportError as e:
        print(f"\nSkipping LM Eval example (not installed): {e}")
        lm_eval_results = None

    # Example 2: SCBench
    scbench_results = example_scbench(model, tokenizer_path, device)

    # Example 3: RULER (basic)
    ruler_results = example_ruler(model, tokenizer_path, device)

    # Example 4: RULER (full suite) - uncomment to run
    # ruler_full_results = example_ruler_full_suite(model, tokenizer_path, device)

    # Example 5: Visualization
    if ruler_results:
        visualize_ruler_results(ruler_results)

    # Example 6: Compare attention methods - uncomment to run
    # comparison_results = compare_attention_methods()

    print("\n" + "="*70)
    print("All examples complete!")
    print("Results saved to: example_results/")
    print("="*70)


if __name__ == "__main__":
    main()
