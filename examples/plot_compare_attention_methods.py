#!/usr/bin/env python3
"""
Compare IO vs Compute Latency across different attention methods

This script reads multiple profiler_io_compute_latency.csv files from different
attention methods (linear, softmax, sliding window) and plots them together
for comparison.

Usage:
    python plot_compare_attention_methods.py \
        --linear path/to/linear_profiler_io_compute_latency.csv \
        --softmax path/to/softmax_profiler_io_compute_latency.csv \
        --window path/to/window_profiler_io_compute_latency.csv \
        --output comparison_plot.png
"""

import argparse
import csv
import numpy as np
import matplotlib.pyplot as plt
import sys
import os


def load_csv_data(csv_path):
    """Load IO and Compute data from CSV file."""
    token_numbers = []
    compute_totals = []
    io_totals = []
    compute_stds = []
    io_stds = []

    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            token_numbers.append(int(row['Token_Number']))
            compute_totals.append(float(row['Total_Compute_ms']))
            io_totals.append(float(row['Total_IO_ms']))
            compute_stds.append(float(row['Compute_StdDev']))
            io_stds.append(float(row['IO_StdDev']))

    return (np.array(token_numbers), np.array(compute_totals),
            np.array(io_totals), np.array(compute_stds), np.array(io_stds))


def plot_comparison(methods_data, output_path, exclude_tfft=True, window_size=128):
    """
    Plot comparison of different attention methods with both Compute and IO lines.

    Args:
        methods_data: Dict mapping method names to (token_numbers, compute, io, compute_std, io_std)
        output_path: Path to save the output plot
        exclude_tfft: If True, exclude the first token (TFFT)
        window_size: Window size to mark with vertical line (default: 128)
    """
    # Set IEEE paper style parameters
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.linewidth': 1.5,
        'axes.labelsize': 13,
        'axes.titlesize': 14,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'xtick.major.width': 1.2,
        'ytick.major.width': 1.2,
        'xtick.major.size': 5,
        'ytick.major.size': 5,
        'legend.fontsize': 10,
        'legend.frameon': True,
        'legend.fancybox': False,
        'legend.edgecolor': 'black',
        'legend.framealpha': 1.0,
        'grid.linewidth': 0.8,
        'grid.alpha': 0.4,
    })

    # Color schemes for different methods - IEEE compliant colors
    colors = {
        'linear': '#0072BD',      # IEEE Blue
        'softmax': '#D95319',     # IEEE Red-Orange
        'window': '#EDB120',      # IEEE Yellow
        'infini': '#77AC30',      # IEEE Green
    }

    # Create single plot with all methods showing both Compute and IO
    # IEEE single-column width is typically 3.5 inches, double-column is 7.16 inches
    fig, ax = plt.subplots(figsize=(7.16, 5))

    for method_name, data in methods_data.items():
        token_numbers, compute_totals, io_totals, compute_stds, io_stds = data

        # Exclude TFFT if requested
        if exclude_tfft:
            token_numbers = token_numbers[1:]
            compute_totals = compute_totals[1:]
            io_totals = io_totals[1:]
            compute_stds = compute_stds[1:]
            io_stds = io_stds[1:]

        color = colors.get(method_name.lower(), '#333333')

        # Plot Compute line (solid, thicker for IEEE papers)
        ax.plot(token_numbers, compute_totals, '-',
               linewidth=2.5, color=color, alpha=1.0,
               label=f'{method_name.capitalize()} Compute',
               marker='o', markersize=0, markevery=50)  # No markers but allows adding later

        # Plot IO line (dashed, thicker for IEEE papers)
        ax.plot(token_numbers, io_totals, '--',
               linewidth=2.5, color=color, alpha=0.85,
               label=f'{method_name.capitalize()} IO',
               marker='s', markersize=0, markevery=50)

    # Add window size vertical line (typically 128 for sliding window attention)
    # This shows where the window attention transitions to constant cost
    if 'window' in methods_data and window_size is not None:
        ax.axvline(x=window_size, color='red', linestyle='-.',
                  linewidth=2.0, alpha=0.7, label=f'Window Size ({window_size})',
                  zorder=0)

    # Configure axes - IEEE style
    ax.set_xlabel('Token Index (Prefill + Decode) (TTFT Excluded)' if exclude_tfft
                 else 'Token Index (Prefill + Decode)',
                 fontsize=13, fontweight='bold')
    ax.set_ylabel('Aggregated Latency Across Layers (ms)', fontsize=13, fontweight='bold')

    # IEEE papers typically don't use titles in figures (titles go in captions)
    # But if needed, uncomment below:
    # ax.set_title('Comparison of IO and Compute Latency Across Attention Mechanisms',
    #             fontsize=14, fontweight='bold', pad=15)

    # Proper IEEE grid - both major grids visible
    ax.grid(True, which='major', linestyle='-', linewidth=0.8, alpha=0.4, color='gray')
    ax.minorticks_on()
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.2, color='gray')

    # IEEE legend style - upper right or best position
    ax.legend(loc='best', fontsize=10, frameon=True, fancybox=False,
             edgecolor='black', framealpha=1.0, ncol=2)

    # Make spine (border) thicker for IEEE style
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_edgecolor('black')

    plt.tight_layout()

    # Save as PNG and PDF
    png_path = output_path
    pdf_path = output_path.replace('.png', '.pdf')

    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    plt.savefig(pdf_path, bbox_inches='tight')

    print(f"\n✓ Comparison plot saved to: {png_path}")
    print(f"✓ Comparison plot saved to: {pdf_path}")

    # Print statistics
    print(f"\nComparison Statistics ({'TTFT excluded' if exclude_tfft else 'TTFT included'}):")
    for method_name, data in methods_data.items():
        token_numbers, compute_totals, io_totals, _, _ = data

        if exclude_tfft:
            compute_totals = compute_totals[1:]
            io_totals = io_totals[1:]

        total = compute_totals + io_totals
        print(f"\n  {method_name.capitalize()}:")
        print(f"    Compute - Mean: {np.mean(compute_totals):.2f}ms, "
              f"Min: {np.min(compute_totals):.2f}ms, Max: {np.max(compute_totals):.2f}ms")
        print(f"    IO      - Mean: {np.mean(io_totals):.2f}ms, "
              f"Min: {np.min(io_totals):.2f}ms, Max: {np.max(io_totals):.2f}ms")
        print(f"    Total   - Mean: {np.mean(total):.2f}ms, "
              f"Min: {np.min(total):.2f}ms, Max: {np.max(total):.2f}ms")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Compare IO vs Compute latency across different attention methods'
    )
    parser.add_argument('--linear', type=str,
                       help='Path to Linear attention CSV file')
    parser.add_argument('--softmax', type=str,
                       help='Path to Softmax attention CSV file')
    parser.add_argument('--window', type=str,
                       help='Path to Sliding Window attention CSV file')
    parser.add_argument('--infini', type=str,
                       help='Path to Infini-attention CSV file (optional)')
    parser.add_argument('--output', type=str,
                       default='attention_methods_comparison.png',
                       help='Output plot filename (default: attention_methods_comparison.png)')
    parser.add_argument('--include-tfft', action='store_true',
                       help='Include TFFT (first token) in the plot (default: excluded)')
    parser.add_argument('--window-size', type=int, default=128,
                       help='Window size to mark with vertical line (default: 128)')

    args = parser.parse_args()

    # Collect methods data
    methods_data = {}

    if args.linear:
        if not os.path.exists(args.linear):
            print(f"Error: Linear CSV file not found: {args.linear}")
            sys.exit(1)
        print(f"Loading Linear attention data from: {args.linear}")
        methods_data['linear'] = load_csv_data(args.linear)

    if args.softmax:
        if not os.path.exists(args.softmax):
            print(f"Error: Softmax CSV file not found: {args.softmax}")
            sys.exit(1)
        print(f"Loading Softmax attention data from: {args.softmax}")
        methods_data['softmax'] = load_csv_data(args.softmax)

    if args.window:
        if not os.path.exists(args.window):
            print(f"Error: Window CSV file not found: {args.window}")
            sys.exit(1)
        print(f"Loading Sliding Window attention data from: {args.window}")
        methods_data['window'] = load_csv_data(args.window)

    if args.infini:
        if not os.path.exists(args.infini):
            print(f"Error: Infini-attention CSV file not found: {args.infini}")
            sys.exit(1)
        print(f"Loading Infini-attention data from: {args.infini}")
        methods_data['infini'] = load_csv_data(args.infini)

    if not methods_data:
        print("\nError: No CSV files provided!")
        print("\nUsage example:")
        print("  python plot_compare_attention_methods.py \\")
        print("    --linear linear_profiler_io_compute_latency.csv \\")
        print("    --softmax softmax_profiler_io_compute_latency.csv \\")
        print("    --window window_profiler_io_compute_latency.csv \\")
        print("    --output comparison.png")
        sys.exit(1)

    print(f"\nGenerating comparison plot with {len(methods_data)} attention methods...")

    # Generate comparison plot
    plot_comparison(methods_data, args.output,
                   exclude_tfft=not args.include_tfft,
                   window_size=args.window_size)


if __name__ == '__main__':
    main()
