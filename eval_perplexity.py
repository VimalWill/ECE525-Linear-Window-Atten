"""
Evaluate perplexity on WikiText-2 or WikiText-103

This will properly test window attention because it processes long sequences
for language modeling, unlike the multiple-choice tasks.

Usage:
    python eval_perplexity.py --model_path <path> --attn_method window --window_size 64
    python eval_perplexity.py --model_path <path> --attn_method softmax
"""

import os
import sys
import argparse
import torch
import math
from pathlib import Path
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from InfiniLLM import Llama
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate perplexity on WikiText")

    # Model configuration
    parser.add_argument("--model_path", type=str,
                       default=os.getenv("LLAMA_DIR"),
                       help="Path to model checkpoint (defaults to LLAMA_DIR env variable)")
    parser.add_argument("--seq_len", type=int, default=1024,
                       help="Maximum sequence length (default: 1024)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size")
    parser.add_argument("--attn_method", type=str, default="window",
                       choices=["softmax", "window", "linear"],
                       help="Attention method")
    parser.add_argument("--window_size", type=int, default=512,
                       help="Window size for window attention")
    parser.add_argument("--device", type=str,
                       default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to run on")

    # Dataset configuration
    parser.add_argument("--dataset", type=str, default="wikitext-2",
                       choices=["wikitext-2", "wikitext-103"],
                       help="WikiText dataset to use")
    parser.add_argument("--stride", type=int, default=512,
                       help="Stride for sliding window over text")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to evaluate (for testing)")

    return parser.parse_args()


def load_wikitext(dataset_name="wikitext-2"):
    """Load WikiText dataset"""
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: datasets library not installed. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading {dataset_name}...")
    if dataset_name == "wikitext-2":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    else:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")

    # Concatenate all text
    text = "\n\n".join(dataset["text"])
    return text


def evaluate_perplexity(model, tokenizer, text, args):
    """
    Evaluate perplexity on text using sliding window approach
    """
    print(f"\n{'='*70}")
    print(f"Evaluating Perplexity")
    print(f"{'='*70}")
    print(f"Dataset: {args.dataset}")
    print(f"Attention method: {args.attn_method}")
    print(f"Window size: {args.window_size}")
    print(f"Sequence length: {args.seq_len}")
    print(f"Stride: {args.stride}")
    print(f"{'='*70}\n")

    # Tokenize entire text
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(args.device)

    seq_len = args.seq_len
    stride = args.stride

    nlls = []  # negative log-likelihoods
    prev_end_loc = 0

    num_samples = 0
    total_tokens = input_ids.size(1)

    print(f"Total tokens in dataset: {total_tokens}")
    print(f"Processing with stride {stride}...\n")

    pbar = tqdm(total=total_tokens if args.max_samples is None else args.max_samples * stride,
                desc="Computing perplexity")

    for begin_loc in range(0, input_ids.size(1), stride):
        if args.max_samples and num_samples >= args.max_samples:
            break

        end_loc = min(begin_loc + seq_len, input_ids.size(1))
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_batch = input_ids[:, begin_loc:end_loc]

        if input_batch.size(1) < 2:
            break

        with torch.no_grad():
            # Reset cache for each sequence
            if hasattr(model, 'layers'):
                for layer in model.layers:
                    if hasattr(layer.attention, 'prompt_len'):
                        layer.attention.prompt_len = 0
                    # Also zero out cache tensors to free memory
                    if hasattr(layer.attention, 'cache_k') and layer.attention.cache_k is not None:
                        layer.attention.cache_k.zero_()
                    if hasattr(layer.attention, 'cache_v') and layer.attention.cache_v is not None:
                        layer.attention.cache_v.zero_()

            # Forward pass
            outputs = model(input_batch, 0)
            logits = outputs[:, :-1, :]  # Shift for next-token prediction
            labels = input_batch[:, 1:]   # Target tokens

            # Compute log probabilities
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

            # Get log prob of actual next tokens
            nll = -log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)

            # Only use the last trg_len tokens for evaluation to avoid double counting
            nll = nll[:, -trg_len:]
            nlls.append(nll.cpu())  # Move to CPU to save GPU memory

            # Clear GPU cache periodically
            if num_samples % 10 == 0:
                torch.cuda.empty_cache()

        prev_end_loc = end_loc
        num_samples += 1
        pbar.update(stride)

        if begin_loc % (stride * 10) == 0 and nlls:
            # Print intermediate perplexity
            current_nll = torch.cat(nlls, dim=1).mean()
            current_ppl = torch.exp(current_nll).item()
            pbar.set_postfix({"ppl": f"{current_ppl:.2f}"})

    pbar.close()

    # Compute final perplexity (tensors are on CPU)
    nll = torch.cat(nlls, dim=1).mean()
    perplexity = torch.exp(nll).item()

    # Final cleanup
    torch.cuda.empty_cache()

    print(f"\n{'='*70}")
    print(f"RESULTS")
    print(f"{'='*70}")
    print(f"Samples processed: {num_samples}")
    print(f"Total tokens evaluated: {sum(n.size(1) for n in nlls)}")
    print(f"Negative Log-Likelihood: {nll.item():.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print(f"{'='*70}\n")

    return perplexity


def main():
    args = parse_args()

    # Load model
    model_path = args.model_path
    if not model_path:
        raise ValueError("Model path must be provided via --model_path or LLAMA_DIR environment variable")

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

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Load WikiText
    text = load_wikitext(args.dataset)

    # Evaluate perplexity
    perplexity = evaluate_perplexity(model, tokenizer, text, args)

    # Save results
    results = {
        "dataset": args.dataset,
        "attn_method": args.attn_method,
        "window_size": args.window_size,
        "seq_len": args.seq_len,
        "stride": args.stride,
        "perplexity": perplexity,
    }

    output_dir = Path("benchmark_results/perplexity")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{args.dataset}_{args.attn_method}_w{args.window_size}.json"

    import json
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
