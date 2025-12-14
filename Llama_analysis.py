import torch
import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from InfiniLLM import Llama
from transformers import AutoTokenizer
import sys 


def apply_repetition_penalty(logits, generated_tokens, penalty=1.2):
    """Apply repetition penalty to discourage repeated tokens"""
    for token in set(generated_tokens.tolist()):
        logits[:, token] /= penalty
    return logits


def top_p_sampling(logits, top_p=0.9):
    """Nucleus sampling (top-p sampling)"""
    sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)

    sorted_indices_to_remove = cumulative_probs > top_p
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
    indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)

    logits[indices_to_remove] = float('-inf')
    probs = torch.softmax(logits, dim=-1)

    if torch.isnan(probs).any() or torch.isinf(probs).any() or probs.sum() == 0:
        next_token = torch.argmax(logits, dim=-1, keepdim=True)
    else:
        next_token = torch.multinomial(probs, num_samples=1)

    return next_token


def collect_atten_viz(model, tokensizer, prompt, temp, rep_penality, top_p, max_token_len):
    """Collect and visualize attention patterns across model layers"""
    inputs = tokensizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"]

    model_device = next(model.parameters()).device
    generated = generated.to(model_device)

    n_layers = len(model.layers)
    layer_attention_matrices = [[] for _ in range(n_layers)]
    prompt_attention_matrices = [None for _ in range(n_layers)]
    generated_tokens = []
    prompt_len = len(inputs["input_ids"][0])

    attn_method = model.layers[0].attn_method
    window_size = model.layers[0].window_size if hasattr(model.layers[0], 'window_size') else 128
    print(f"Using attention method: {attn_method}")
    print(f"Window size: {window_size}")

    atten_maps_path = os.path.join(os.getcwd(), "attention_maps")
    os.makedirs(atten_maps_path, exist_ok=True)

    with torch.no_grad():
        outputs, _ = model(generated, 0)
        for layer_idx, layer in enumerate(model.layers):
            attn_weights = layer.attention.last_attention_weights

            if attn_weights is not None:
                avg_attention = attn_weights[0].mean(dim=0)
                prompt_attn = avg_attention[:prompt_len, :prompt_len].cpu().numpy()
                prompt_attention_matrices[layer_idx] = prompt_attn

        print(f"Prompt processed. Starting token generation with {attn_method} attention (window_size={window_size})...")

        for gen_step in range(max_token_len):
            # Get logits for next token
            next_token_logits = outputs[:, -1, :] / temp
            next_token_logits = apply_repetition_penalty(next_token_logits, generated[0], rep_penality)
            next_token = top_p_sampling(next_token_logits, top_p=top_p)

            generated_tokens.append(next_token.item())

            current_pos = generated.shape[1]
            outputs, _ = model(next_token, current_pos)
            generated = torch.cat([generated, next_token], dim=1)

            for layer_idx, layer in enumerate(model.layers):
                attn_weights = layer.attention.last_attention_weights

                if attn_weights is not None:
                    avg_attention = attn_weights[0].mean(dim=0)  
                    token_attention = avg_attention[-1, :].cpu().numpy()  

                    full_attention = np.zeros(current_pos)
                    if hasattr(layer.attention, '_cache_positions'):
                        cache_positions = layer.attention._cache_positions.cpu().numpy()
                        for cache_idx, abs_pos in enumerate(cache_positions):
                            if abs_pos < current_pos:
                                full_attention[abs_pos] = token_attention[cache_idx]
                    else:
                        copy_len = min(len(token_attention), current_pos)
                        full_attention[:copy_len] = token_attention[:copy_len]

                    layer_attention_matrices[layer_idx].append(full_attention)

            if next_token.item() == tokensizer.eos_token_id:
                break

    output_text = tokensizer.decode(generated[0], skip_special_tokens=True)
    print(output_text)

    prompt_only_path = os.path.join(atten_maps_path, "prompt_only")
    gen_only_path = os.path.join(atten_maps_path, "generated_only")
    full_context_path = os.path.join(atten_maps_path, "full_context")
    os.makedirs(prompt_only_path, exist_ok=True)
    os.makedirs(gen_only_path, exist_ok=True)
    os.makedirs(full_context_path, exist_ok=True)

    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 10,
        'axes.linewidth': 1.0,
        'axes.labelweight': 'normal',
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.labelsize': 12,
        'axes.titlesize': 14,
        'legend.fontsize': 10,
    })

    for layer_idx in range(n_layers):
        if len(layer_attention_matrices[layer_idx]) > 0:
            num_generated = len(generated_tokens)

            attention_matrix = np.zeros((num_generated, num_generated))
            for gen_idx, full_attention in enumerate(layer_attention_matrices[layer_idx]):
                current_seq_len = prompt_len + gen_idx + 1
                attention_to_generated = full_attention[prompt_len:current_seq_len].copy()

                if attention_to_generated.sum() > 0:
                    attention_to_generated = attention_to_generated / attention_to_generated.sum()

                attention_matrix[gen_idx, :len(attention_to_generated)] = attention_to_generated
            gen_token_labels = [tokensizer.decode([token]).strip() for token in generated_tokens]

            fig_size = max(16, num_generated * 0.8)
            fig, ax = plt.subplots(figsize=(fig_size, fig_size))
            sns.heatmap(attention_matrix,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       xticklabels=gen_token_labels,
                       yticklabels=gen_token_labels,
                       cbar_kws={'label': 'Attention Weight'},
                       square=True,
                       vmin=0,
                       vmax=1.0,
                       ax=ax)

            ax.set_xlabel('Generated Tokens (Attended To)')
            ax.set_ylabel('Generated Tokens (Attending From)')
            ax.set_title(f'Self-Attention Pattern (Generated Only, Renormalized): Layer {layer_idx + 1}')

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            fig.savefig(f'{gen_only_path}/attention_layer_{layer_idx}.png', dpi=400, bbox_inches='tight', facecolor='white')
            fig.savefig(f'{gen_only_path}/attention_layer_{layer_idx}.pdf', bbox_inches='tight', facecolor='white')
            print(f"Saved generated-only attention heatmap for layer {layer_idx}")
            plt.close(fig)

    prompt_tokens = tokensizer.encode(prompt)
    all_tokens = prompt_tokens + generated_tokens
    all_token_labels = [tokensizer.decode([token]).strip() for token in all_tokens]

    for layer_idx in range(n_layers):
        if len(layer_attention_matrices[layer_idx]) > 0:
            num_generated = len(generated_tokens)
            max_len = prompt_len + num_generated
            attention_matrix_full = np.zeros((num_generated, max_len))

            for gen_idx, full_attention in enumerate(layer_attention_matrices[layer_idx]):
                actual_len = len(full_attention)
                copy_len = min(actual_len, max_len)
                attention_matrix_full[gen_idx, :copy_len] = full_attention[:copy_len]

            total_tokens = len(all_tokens)
            fig_size_full = max(18, total_tokens * 0.6)
            fig, ax = plt.subplots(figsize=(fig_size_full, max(14, num_generated * 0.6)))

            sns.heatmap(attention_matrix_full,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       xticklabels=all_token_labels,
                       yticklabels=[tokensizer.decode([token]).strip() for token in generated_tokens],
                       cbar_kws={'label': 'Attention Weight'},
                       vmin=0,
                       vmax=attention_matrix_full.max(),
                       ax=ax)

            ax.set_xlabel('All Tokens (Prompt + Generated)')
            ax.set_ylabel('Generated Tokens')

            # Add window size to title if using window attention
            title_suffix = f" [Window={window_size}]" if attn_method == "window" else ""
            ax.set_title(f'Self-Attention Pattern (Full Context): Layer {layer_idx + 1}{title_suffix}')

            if prompt_len > 0:
                ax.axvline(x=prompt_len - 0.5, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Prompt End')

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            fig.savefig(f'{full_context_path}/attention_layer_{layer_idx}.png', dpi=400, bbox_inches='tight', facecolor='white')
            fig.savefig(f'{full_context_path}/attention_layer_{layer_idx}.pdf', bbox_inches='tight', facecolor='white')
            print(f"Saved full-context attention heatmap for layer {layer_idx}")
            plt.close(fig)

    prompt_tokens = tokensizer.encode(prompt)
    prompt_token_labels = [tokensizer.decode([token]).strip() for token in prompt_tokens]

    for layer_idx in range(n_layers):
        if prompt_attention_matrices[layer_idx] is not None:
            prompt_attn_matrix = prompt_attention_matrices[layer_idx]
            num_prompt_tokens = len(prompt_tokens)

            fig_size_prompt = max(12, num_prompt_tokens * 0.8)
            fig, ax = plt.subplots(figsize=(fig_size_prompt, fig_size_prompt))

            sns.heatmap(prompt_attn_matrix,
                       annot=True,
                       fmt='.2f',
                       cmap='RdYlBu_r',
                       xticklabels=prompt_token_labels,
                       yticklabels=prompt_token_labels,
                       cbar_kws={'label': 'Attention Weight'},
                       square=True,
                       vmin=0,
                       vmax=prompt_attn_matrix.max(),
                       ax=ax)

            ax.set_xlabel('Prompt Tokens (Attended To)')
            ax.set_ylabel('Prompt Tokens (Attending From)')
            ax.set_title(f'Self-Attention Pattern (Prompt Only): Layer {layer_idx + 1}')

            plt.xticks(rotation=45, ha='right')
            plt.yticks(rotation=0)
            plt.tight_layout()

            fig.savefig(f'{prompt_only_path}/attention_layer_{layer_idx}.png', dpi=400, bbox_inches='tight', facecolor='white')
            fig.savefig(f'{prompt_only_path}/attention_layer_{layer_idx}.pdf', bbox_inches='tight', facecolor='white')
            print(f"Saved prompt-only attention heatmap for layer {layer_idx}")
            plt.close(fig)


def main():
    """Analyze Llama-3.2 performance and efficiency for SoftMax Attention"""
    chkpt_dir = os.getenv("LLAMA_DIR")

    profiler_flag = False
    if len(sys.argv) > 1 and sys.argv[1] == "--enable-profiler":
        profiler_flag = True

    # [64, 64, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 64, 64]
    seq_len = 1024
    batch_size = 1
    window_size = 64
    attn_method = "swa_linear"
    delta_values = None
    if attn_method == "swa_linear":
        # Test with delta=0 (pure softmax) first to verify base mechanism
        delta_values = [64, 64, 64, 64, 64, 0, 0, 0, 0, 0, 0, 64, 64, 64, 64, 64]
    model = Llama.LLama.build(chkpt_dir, seq_len, batch_size, True, attn_method, window_size, delta_values)

    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     print(f"Model moved to GPU: {torch.cuda.get_device_name(0)}")
    # else:
    #     print("CUDA not available, using CPU")

    device = next(model.parameters()).device
    print(f"Model is running on: {device}")

    print(f"\n{'='*50}")
    print(f"Model Configuration:")
    print(f"Attention Method: {model.layers[0].attn_method}")
    print(f"Window Size: {model.layers[0].window_size}")
    print(f"Using Cache: {model.layers[0].attention.use_cache}")
    print(f"{'='*50}\n")


    print(model)
    prompt = '''When I stepped into the abandoned research cabin, the logs scattered across the desk no longer described the experiments I remembered. Entire paragraphs had reshaped themselves into observations of places I’d never visited, written in a tone that didn’t feel like mine. A sealed tin box now held a folded strip of fabric that pulsed'''

    tokenizer = AutoTokenizer.from_pretrained(chkpt_dir)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    inputs = tokenizer(prompt, return_tensors="pt")
    generated = inputs["input_ids"]

    actual_prompt_len = inputs["input_ids"].shape[1]
    print(f"\n{'='*50}")
    print(f"Prompt token count: {actual_prompt_len}")
    print(f"Window size: {window_size}")
    print(f"{'='*50}\n")

    model_device = next(model.parameters()).device
    generated = generated.to(model_device)

    max_new_tokens = 200
    temperature = 0.8
    top_p = 0.9
    repetition_penalty = 1.2

    num_rounds = 1 if profiler_flag else 5
    all_rounds_times = []
    all_outputs = []

    print(f"\n{'='*50}")
    if profiler_flag:
        print("PROFILER MODE: Running 1 round for nsys profiling...")
    else:
        print(f"Running {num_rounds} rounds for latency analysis...")
    print(f"{'='*50}\n")

    for round_num in range(num_rounds):
        print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

        prompt_len = inputs["input_ids"].shape[1]
        max_total_len = prompt_len + max_new_tokens
        generated_round = torch.zeros((1, max_total_len), dtype=torch.long, device=model_device)
        generated_round[:, :prompt_len] = inputs["input_ids"].to(model_device)
        current_len = prompt_len
        token_times = []

        first_5_hidden_states = []
        prefill_end = window_size

        with torch.no_grad():
            # Process prompt
            outputs, _ = model(generated_round[:, :current_len], 0)

            for i in range(max_new_tokens):
                start_time = time.time()

                next_token_logits = outputs[:, -1, :] / temperature


                next_token_logits = apply_repetition_penalty(
                    next_token_logits, generated_round[0, :current_len], repetition_penalty)
                next_token = top_p_sampling(next_token_logits, top_p=top_p)

                # Store the generated token
                generated_round[:, current_len] = next_token
                current_len += 1
                outputs, hidden_states = model(next_token, current_len - 1)

                if current_len > prefill_end and len(first_5_hidden_states) < 5:
                    if len(first_5_hidden_states) == 0:
                        input_after_prefill = generated_round[:, :current_len].cpu().clone()
                        np.save('input_after_prefill.npy', input_after_prefill.numpy())
                    first_5_hidden_states.append([(h[0, -1, :].cpu().clone()) for h in hidden_states])

                end_time = time.time()

                token_time = (end_time - start_time) * 1000
                token_times.append(token_time)

                print(f"Token {i+1}: {tokenizer.decode(next_token[0])} - {token_time:.2f}ms")

                if next_token.item() == tokenizer.eos_token_id:
                    print(f"  [EOS encountered, continuing to benchmark full sequence...]")

        all_rounds_times.append(token_times)
        output_text = tokenizer.decode(generated_round[0, :current_len], skip_special_tokens=True)
        all_outputs.append(output_text)

        print(f"\nRound {round_num + 1} complete: {len(token_times)} tokens, "
              f"avg {sum(token_times)/len(token_times):.2f}ms per token")
        print(f"Output: {output_text}\n")

    if profiler_flag:
        print("\n" + "="*70)
        print("PROFILER MODE: Profiling complete.")
        print("="*70)

        print("\nCollecting time records from model layers...")
        n_layers = len(model.layers)

        has_time_records = False
        has_io_records = False
        for layer_idx in range(n_layers):
            if len(model.layers[layer_idx].attention.time_record) > 0:
                has_time_records = True
            if len(model.layers[layer_idx].attention.time_record_io) > 0:
                has_io_records = True
            if has_time_records and has_io_records:
                break

        if has_time_records or has_io_records:

            print("\n=== Computing IO vs Compute Latency Comparison with Error Bounds (Across Layers) ===")

            # Find minimum token length across all layers
            min_tokens = min(len(model.layers[layer_idx].attention.time_record)
                           for layer_idx in range(n_layers)
                           if len(model.layers[layer_idx].attention.time_record) > 0)

            # Collect data from all layers: shape (n_layers, n_tokens)
            compute_matrix = []
            io_matrix = []

            for layer_idx in range(n_layers):
                compute_records = model.layers[layer_idx].attention.time_record
                io_records = model.layers[layer_idx].attention.time_record_io

                if len(compute_records) > 0:
                    compute_matrix.append(compute_records[:min_tokens])
                if len(io_records) > 0:
                    io_matrix.append(io_records[:min_tokens])

            if len(compute_matrix) > 0 and len(io_matrix) > 0:
                compute_matrix = np.array(compute_matrix)  # shape: (n_layers, n_tokens)
                io_matrix = np.array(io_matrix)

                # Calculate sum across all layers (total latency per token)
                compute_sum_per_token = np.sum(compute_matrix, axis=0)
                io_sum_per_token = np.sum(io_matrix, axis=0)

                # Calculate std across layers to show variance
                compute_std_per_token = np.std(compute_matrix, axis=0)
                io_std_per_token = np.std(io_matrix, axis=0)

                # Overall statistics
                compute_overall_mean = np.mean(compute_sum_per_token)
                io_overall_mean = np.mean(io_sum_per_token)

                # Get attention method info
                attn_method = model.layers[0].attn_method
                window_size_val = model.layers[0].window_size if hasattr(model.layers[0], 'window_size') else None

                fig, ax = plt.subplots(figsize=(14, 8))
                token_numbers = np.arange(1, min_tokens + 1)

                # Plot Compute total with error bounds showing layer variance
                ax.plot(token_numbers, compute_sum_per_token, '-',
                       linewidth=2.0, color='#2E86AB', alpha=0.9)
                ax.fill_between(token_numbers,
                               compute_sum_per_token - compute_std_per_token * np.sqrt(n_layers),
                               compute_sum_per_token + compute_std_per_token * np.sqrt(n_layers),
                               alpha=0.25, color='#2E86AB',
                               label=f'Compute (σ={np.mean(compute_std_per_token):.2f}ms)')

                # Plot IO total with error bounds showing layer variance
                ax.plot(token_numbers, io_sum_per_token, '-',
                       linewidth=2.0, color='#A23B72', alpha=0.9)
                ax.fill_between(token_numbers,
                               io_sum_per_token - io_std_per_token * np.sqrt(n_layers),
                               io_sum_per_token + io_std_per_token * np.sqrt(n_layers),
                               alpha=0.25, color='#A23B72',
                               label=f'IO (σ={np.mean(io_std_per_token):.2f}ms)')

                # Add window size line for window attention
                if attn_method == "window" and window_size_val is not None and window_size_val < min_tokens:
                    ax.axvline(x=window_size_val, color='red', linestyle='--',
                              linewidth=2, alpha=0.7, label=f'Window Size ({window_size_val})')

                ax.set_xlabel('Number of Processed Tokens (Prefill + Decode)')
                ax.set_ylabel('Aggregated Latency Across Layers (ms)')
                ax.set_title('Error-Bound Analysis of IO and Compute Latency Across Layers')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='best')

                plt.tight_layout()
                plt.savefig('latency_comparison_across_layers.png',
                           dpi=300, bbox_inches='tight')
                plt.savefig('latency_comparison_across_layers.pdf',
                           bbox_inches='tight')
                plt.close(fig)

                print(f"Compute Total (sum across layers): {compute_overall_mean:.2f}ms")
                print(f"IO Total (sum across layers): {io_overall_mean:.2f}ms")
                print(f"✓ Comparison plot with error bounds saved as 'latency_comparison_across_layers.png/pdf'")

                # Create second plot excluding TFFT (first token)
                if min_tokens > 1:
                    compute_sum_no_tfft = compute_sum_per_token[1:]
                    io_sum_no_tfft = io_sum_per_token[1:]
                    compute_std_no_tfft = compute_std_per_token[1:]
                    io_std_no_tfft = io_std_per_token[1:]

                    compute_mean_no_tfft = np.mean(compute_sum_no_tfft)
                    io_mean_no_tfft = np.mean(io_sum_no_tfft)

                    fig2, ax2 = plt.subplots(figsize=(14, 8))
                    token_numbers_no_tfft = np.arange(2, min_tokens + 1)

                    # Plot Compute without TFFT
                    ax2.plot(token_numbers_no_tfft, compute_sum_no_tfft, '-',
                           linewidth=2.0, color='#2E86AB', alpha=0.9)
                    ax2.fill_between(token_numbers_no_tfft,
                                   compute_sum_no_tfft - compute_std_no_tfft * np.sqrt(n_layers),
                                   compute_sum_no_tfft + compute_std_no_tfft * np.sqrt(n_layers),
                                   alpha=0.25, color='#2E86AB',
                                   label=f'Compute (σ={np.mean(compute_std_no_tfft):.2f}ms)')

                    # Plot IO without TFFT
                    ax2.plot(token_numbers_no_tfft, io_sum_no_tfft, '-',
                           linewidth=2.0, color='#A23B72', alpha=0.9)
                    ax2.fill_between(token_numbers_no_tfft,
                                   io_sum_no_tfft - io_std_no_tfft * np.sqrt(n_layers),
                                   io_sum_no_tfft + io_std_no_tfft * np.sqrt(n_layers),
                                   alpha=0.25, color='#A23B72',
                                   label=f'IO (σ={np.mean(io_std_no_tfft):.2f}ms)')

                    # Add window size line for window attention
                    if attn_method == "window" and window_size_val is not None and window_size_val < min_tokens:
                        ax2.axvline(x=window_size_val, color='red', linestyle='--',
                                  linewidth=2, alpha=0.7, label=f'Window Size ({window_size_val})')

                    ax2.set_xlabel('Number of Processed Tokens (Prefill + Decode, TTFT Excluded)')
                    ax2.set_ylabel('Aggregated Latency Across Layers (ms)')
                    ax2.set_title('Error-Bound Analysis of IO and Compute Latency Across Layers (TTFT Excluded)')
                    ax2.grid(True, alpha=0.3)
                    ax2.legend(loc='best')

                    plt.tight_layout()
                    plt.savefig('latency_comparison_no_ttft.png',
                               dpi=300, bbox_inches='tight')
                    plt.savefig('latency_comparison_no_ttft.pdf',
                               bbox_inches='tight')
                    plt.close(fig2)

                    print(f"Compute Total (excluding TTFT): {compute_mean_no_tfft:.2f}ms")
                    print(f"IO Total (excluding TTFT): {io_mean_no_tfft:.2f}ms")
                    print(f"✓ Comparison plot without TTFT saved as 'latency_comparison_no_ttft.png/pdf'")

                # Export IO vs Compute data to CSV
                csv_io_compute = 'profiler_io_compute_latency.csv'
                print(f"\nExporting IO vs Compute latency data to {csv_io_compute}...")
                with open(csv_io_compute, 'w', newline='') as csvfile:
                    writer = csv.writer(csvfile)

                    # Write header with per-layer breakdown
                    header = ['Token_Number']
                    for layer_idx in range(n_layers):
                        header.append(f'Layer_{layer_idx}_Compute_ms')
                        header.append(f'Layer_{layer_idx}_IO_ms')
                    header.extend(['Total_Compute_ms', 'Total_IO_ms', 'Compute_StdDev', 'IO_StdDev'])
                    writer.writerow(header)

                    # Write data for each token
                    for token_idx in range(min_tokens):
                        row = [token_idx + 1]  # Token number (1-indexed)

                        # Add per-layer compute and IO values
                        for layer_idx in range(n_layers):
                            row.append(f'{compute_matrix[layer_idx, token_idx]:.4f}')
                            row.append(f'{io_matrix[layer_idx, token_idx]:.4f}')

                        # Add aggregated totals and std dev
                        row.append(f'{compute_sum_per_token[token_idx]:.4f}')
                        row.append(f'{io_sum_per_token[token_idx]:.4f}')
                        row.append(f'{compute_std_per_token[token_idx]:.4f}')
                        row.append(f'{io_std_per_token[token_idx]:.4f}')

                        writer.writerow(row)

                print(f"✓ IO vs Compute latency data exported to '{csv_io_compute}'")

            min_tokens = min(len(model.layers[layer_idx].attention.time_record)
                           for layer_idx in range(n_layers)
                           if len(model.layers[layer_idx].attention.time_record) > 0)

            overall_latency = np.zeros(min_tokens)
            for layer_idx in range(n_layers):
                time_records = model.layers[layer_idx].attention.time_record
                if len(time_records) > 0:
                    overall_latency += np.array(time_records[:min_tokens])

            if min_tokens > 1:
                overall_latency_no_tfft = overall_latency[1:]
                token_numbers_no_tfft = np.arange(2, min_tokens + 1)

                fig_overall, ax_overall = plt.subplots(figsize=(14, 8))
                ax_overall.plot(token_numbers_no_tfft, overall_latency_no_tfft, '-o',
                               linewidth=2.0, markersize=5, color='#A23B72',
                               label='Overall Latency (All Layers)')

                mean_overall = np.mean(overall_latency_no_tfft)
                ax_overall.axhline(y=mean_overall, color='#F18F01', linestyle='--',
                                  linewidth=2, alpha=0.7,
                                  label=f'Mean: {mean_overall:.2f}ms')

                ax_overall.set_xlabel('Token Number (excluding TFFT)')
                ax_overall.set_ylabel('Total Latency (ms)')
                ax_overall.set_title('Overall Per-Token Latency (Sum of All Layers, TFFT Excluded)')
                ax_overall.grid(True, alpha=0.3)
                ax_overall.legend(loc='best')

                stats_text = (f"Total tokens: {len(overall_latency_no_tfft)}\n"
                             f"Mean: {mean_overall:.2f}ms\n"
                             f"Min: {np.min(overall_latency_no_tfft):.2f}ms\n"
                             f"Max: {np.max(overall_latency_no_tfft):.2f}ms\n"
                             f"TFFT: {overall_latency[0]:.2f}ms (excluded)")
                ax_overall.text(0.02, 0.98, stats_text,
                               transform=ax_overall.transAxes,
                               fontsize=10, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

                plt.tight_layout()
                plt.savefig('profiler_overall_latency_no_tfft.png', dpi=300, bbox_inches='tight')
                plt.savefig('profiler_overall_latency_no_tfft.pdf', bbox_inches='tight')
                print("\n Overall latency plot (excluding TFFT) saved as 'profiler_overall_latency_no_tfft.png/pdf'")

                print(f"\nOverall Latency Statistics (TFFT excluded):")
                print(f"  TFFT (Token 1): {overall_latency[0]:.2f}ms")
                print(f"  Tokens 2-{min_tokens}: mean={mean_overall:.2f}ms, "
                      f"min={np.min(overall_latency_no_tfft):.2f}ms, "
                      f"max={np.max(overall_latency_no_tfft):.2f}ms")

                plt.close(fig_overall)

            csv_filename = 'profiler_time_records.csv'
            print(f"\nExporting time records to {csv_filename}...")
            with open(csv_filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)

                header = ['Token_Number']
                for layer_idx in range(n_layers):
                    header.append(f'Layer_{layer_idx}_ms')
                header.append('Overall_Latency_ms')
                writer.writerow(header)

                max_len = max(len(model.layers[layer_idx].attention.time_record)
                             for layer_idx in range(n_layers))


                for token_idx in range(max_len):
                    row = [token_idx + 1]
                    token_sum = 0
                    valid_layers = 0
                    for layer_idx in range(n_layers):
                        time_records = model.layers[layer_idx].attention.time_record
                        if token_idx < len(time_records):
                            row.append(f'{time_records[token_idx]:.4f}')
                            token_sum += time_records[token_idx]
                            valid_layers += 1
                        else:
                            row.append('')
                   
                    if valid_layers == n_layers:
                        row.append(f'{token_sum:.4f}')
                    else:
                        row.append('')
                    writer.writerow(row)

            print(f"✓ Time records exported to '{csv_filename}'")
        else:
            print("No time records found in model layers.")

        print("\n=== Saving Hidden States for First 5 Generated Tokens ===")
        if len(first_5_hidden_states) > 0:
            hidden_states_dir = 'first_5_token_hidden_states'
            os.makedirs(hidden_states_dir, exist_ok=True)

            num_tokens = len(first_5_hidden_states)
            n_layers = len(first_5_hidden_states[0])

            for token_idx in range(num_tokens):
                token_dir = os.path.join(hidden_states_dir, f'token_{token_idx+1}')
                os.makedirs(token_dir, exist_ok=True)

                for layer_idx in range(n_layers):
                    hidden_array = first_5_hidden_states[token_idx][layer_idx].numpy()
                    npy_filename = os.path.join(token_dir, f'layer_{layer_idx}.npy')
                    np.save(npy_filename, hidden_array)

                print(f"✓ Token {token_idx+1}: Saved {n_layers} layer hidden states as .npy files")

            print(f"✓ All hidden states saved to '{hidden_states_dir}/'")
        else:
            print("No hidden states collected for first 5 tokens.")

        print("="*70)
        return

    min_length = min(len(times) for times in all_rounds_times)
    truncated_times = [times[:min_length] for times in all_rounds_times]

    times_array = np.array(truncated_times)
    mean_times = np.mean(times_array, axis=0)
    std_times = np.std(times_array, axis=0)

    print("\n" + "="*70)
    print("LATENCY ANALYSIS - Summary Statistics (across all rounds):")
    print("="*70)
    print(f"Number of rounds: {num_rounds}")
    print(f"Tokens generated per round: {min_length}")
    print(f"Prompt length: {inputs['input_ids'].shape[1]} tokens")
    print(f"\nLatency Statistics:")
    print(f"  Overall mean: {np.mean(mean_times):.2f} ms/token")
    print(f"  Overall std:  {np.mean(std_times):.2f} ms")
    print(f"  Min latency:  {np.min(mean_times):.2f} ms (token {np.argmin(mean_times)+1})")
    print(f"  Max latency:  {np.max(mean_times):.2f} ms (token {np.argmax(mean_times)+1})")

    print(f"\nFirst 10 tokens (ms):")
    for i in range(min(10, min_length)):
        print(f"  Token {i+1:3d}: {mean_times[i]:6.2f} ± {std_times[i]:5.2f}")

    if min_length > 20:
        print(f"\nLast 10 tokens (ms):")
        for i in range(max(10, min_length-10), min_length):
            print(f"  Token {i+1:3d}: {mean_times[i]:6.2f} ± {std_times[i]:5.2f}")

    print(f"\nSample output (Round 1):")
    print("Prompt:", prompt)
    print("Generated:", all_outputs[0])
    print("="*70)

    attn_method = model.layers[0].attn_method
    window_size_val = model.layers[0].window_size if hasattr(model.layers[0], 'window_size') else 'N/A'
    csv_filename = f'latency_data_{attn_method}_window{window_size_val}.csv'

    print(f"\nExporting latency data to {csv_filename}...")
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        writer.writerow(['# Latency Analysis Results'])
        writer.writerow(['# Attention Method:', attn_method])
        writer.writerow(['# Window Size:', window_size_val])
        writer.writerow(['# Number of Rounds:', num_rounds])
        writer.writerow(['# Prompt Length:', inputs['input_ids'].shape[1]])
        writer.writerow(['# Tokens Generated:', min_length])
        writer.writerow([])

        # Write column headers
        header = ['Token_Number', 'Mean_Latency_ms', 'Std_Latency_ms']
        for i in range(num_rounds):
            header.append(f'Round_{i+1}_ms')
        writer.writerow(header)

        # Write data rows
        for i in range(min_length):
            row = [i+1, f'{mean_times[i]:.4f}', f'{std_times[i]:.4f}']
            for round_idx in range(num_rounds):
                row.append(f'{truncated_times[round_idx][i]:.4f}')
            writer.writerow(row)

    print(f"✓ Latency data exported to '{csv_filename}'")

    # Create publication-quality plot for poster
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 2.5

    fig, ax = plt.subplots(1, 1, figsize=(16, 9))  # Larger figure for poster
    token_generation_num = np.arange(1, min_length + 1)
    ax.plot(token_generation_num, mean_times, '-', linewidth=4.0,  # Thicker line
            color='#2E86AB', label='Mean Generation Latency', zorder=3, alpha=0.9)

    marker_interval = max(1, min_length // 20)
    ax.plot(token_generation_num[::marker_interval], mean_times[::marker_interval],
            'o', markersize=10, color='#2E86AB', zorder=4,  # Larger markers
            markeredgecolor='black', markeredgewidth=1.5)  # Black marker edges

    ax.fill_between(token_generation_num,
                     mean_times - std_times,
                     mean_times + std_times,
                     alpha=0.25,
                     color='#2E86AB',
                     label=f'±1 Std Dev (n={num_rounds} rounds)',
                     zorder=2)

    overall_mean = np.mean(mean_times)
    ax.axhline(y=overall_mean, color='#A23B72', linestyle='--', linewidth=3.5,  # Thicker line
               alpha=0.8, label=f'Overall Mean: {overall_mean:.2f}ms', zorder=1)

    prompt_len = inputs['input_ids'].shape[1]
    attn_method = model.layers[0].attn_method
    if attn_method == "window":
        window_boundary = window_size
        if window_boundary < min_length:
            ax.axvline(x=window_boundary, color='#F18F01', linestyle=':', linewidth=3.5,  # Thicker line
                       alpha=0.8, label=f'Window Size ({window_size} tokens)', zorder=1)

    ax.set_xlabel('Token Generation Number (nth generated token)', fontweight='bold', fontsize=20)
    ax.set_ylabel('Latency per Token (ms)', fontweight='bold', fontsize=20)

    # Set appropriate title based on attention method
    if attn_method == "window":
        title = f'Per-Token Generation Latency - Sliding Window Attention (window={window_size})\n'
    else:
        title = f'Per-Token Generation Latency - {attn_method.capitalize()} Attention\n'
    ax.set_title(f'{title}Mean across {num_rounds} rounds',
                 fontweight='bold', fontsize=22, pad=20)
    ax.grid(True, alpha=0.4, linestyle='--', linewidth=1.5)  # More visible grid
    ax.tick_params(axis='both', which='major', labelsize=16, width=2.5, length=10)

    # Create legend and set frame properties separately
    legend_token = ax.legend(frameon=True, fancybox=False, edgecolor='black',
                            framealpha=1.0, fontsize=16, loc='best')
    legend_token.get_frame().set_linewidth(2.5)  # Set legend border width

    # Keep all spines visible and make them thick with black edges
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(2.5)
        spine.set_edgecolor('black')

    ax.set_ylim(bottom=0)

    plt.tight_layout()
    plt.savefig('token_generation_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('token_generation_analysis.pdf', bbox_inches='tight')
    print("\n✓ Plot saved as 'token_generation_analysis.png' and 'token_generation_analysis.pdf'")

    print("\nGenerating attention maps...")
    max_itm_tokens = 20
    collect_atten_viz(model, tokenizer, prompt, temperature, repetition_penalty, top_p, max_itm_tokens)


if __name__ == "__main__":
    main()