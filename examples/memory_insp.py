import numpy as np
import matplotlib.pyplot as plt


def get_softmax_attn_cache_size(d_model: int, 
                                n_heads: int, 
                                seq_len: int = 512,
                                batch_size: int = 1,
                                n_layers: int = 1):
    """Calculate KV cache size for standard softmax attention"""
    if seq_len == 0 and d_model == 0 and n_heads == 0:
        return None
    
    head_d = d_model // n_heads
    if head_d <= 0:
        raise ValueError("invalid head dims")
    
    kv_cache_size = 2 * n_layers * (batch_size * n_heads * seq_len * head_d)
    return kv_cache_size


def get_swa_cache_size(d_model: int, 
                       n_heads: int, 
                       window_size: int = 10, 
                       seq_len: int = 512, 
                       batch_size: int = 1,
                       n_layers: int = 1):
    """Calculate KV cache size for Sliding Window Attention"""
    if seq_len == 0 and d_model == 0 and n_heads == 0:
        return None
    
    head_d = d_model // n_heads
    if head_d <= 0:
        raise ValueError("invalid head dims")
    
    effective_window = min(window_size, seq_len)
    kv_cache_size = 2 * n_layers * (batch_size * n_heads * effective_window * head_d)
    return kv_cache_size


def get_linear_cache_size(d_model: int,
                          n_heads: int, 
                          seq_len: int = 512, 
                          batch_size: int = 1,
                          n_layers: int = 1):
    """Calculate cache size for Linear Attention (constant memory)
    
    Linear attention maintains a d_k × d_v state matrix per head per layer,
    which remains constant regardless of sequence length.
    """
    if seq_len == 0 and d_model == 0 and n_heads == 0:
        return None
    
    head_d = d_model // n_heads
    if head_d <= 0:
        raise ValueError("invalid head dims")
    
    # Linear attention stores a state matrix of size (d_k × d_v) per head
    # d_k = d_v = head_d
    # State matrix size: head_d × head_d per head per layer
    # Note: This is constant regardless of sequence length!
    kv_cache_size = n_layers * (batch_size * n_heads * head_d * head_d)
    return kv_cache_size


def create_elements_growth_plot(tokens_range, softmax_sizes, swa_sizes, linear_sizes, 
                               window_size, d_model, n_heads, n_layers):
    """Create line plot with number of elements (log scale) on y-axis"""
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Plot lines
    line1 = ax.plot(tokens_range, softmax_sizes, marker='o', linewidth=3, markersize=9, 
                    color='#C55A11', markeredgecolor='black', markeredgewidth=1.5, 
                    label='Standard Attention', zorder=3)
    
    line2 = ax.plot(tokens_range, swa_sizes, marker='s', linewidth=3, markersize=9, 
                    color='#70AD47', markeredgecolor='black', markeredgewidth=1.5, 
                    label=f'Sliding Window Attention (w={window_size})', zorder=3)
    
    line3 = ax.plot(tokens_range, linear_sizes, marker='^', linewidth=3, markersize=9, 
                    color='#4472C4', markeredgecolor='black', markeredgewidth=1.5, 
                    label='Linear Attention (Constant Memory)', zorder=3)
    
    # Add window limit line
    if window_size <= max(tokens_range):
        ax.axvline(x=window_size, color='red', linestyle='--', linewidth=2, 
                  alpha=0.7, zorder=2)
        # Minimal annotation
        ax.text(window_size, ax.get_ylim()[1] * 0.95, f'  window = {window_size}', 
               rotation=0, verticalalignment='top',
               fontsize=11, fontweight='bold', color='red')
    
    # Calculate and mark crossover point where linear becomes more efficient
    head_d = d_model // n_heads
    crossover_point = head_d / 2  # seq_len where 2*seq_len*d_k = d_k^2
    if crossover_point <= max(tokens_range):
        ax.axvline(x=crossover_point, color='purple', linestyle=':', linewidth=2.5, 
                  alpha=0.8, zorder=2)
        ax.text(crossover_point, ax.get_ylim()[1] * 0.85, 
               f'  Linear becomes\n  efficient at {int(crossover_point)} tokens', 
               rotation=0, verticalalignment='top',
               fontsize=10, fontweight='bold', color='purple',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                        edgecolor='purple', alpha=0.9))
    
    # Set logarithmic scale on y-axis with base 2
    ax.set_yscale('log', base=2)
    
    # Labels and title
    ax.set_xlabel('Number of Tokens Generated', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Elements (log₂ scale)', fontsize=14, fontweight='bold')
    ax.set_title(f'KV Cache Growth: Standard vs Sliding Window vs Linear Attention\n' + 
                f'd_model={d_model}, n_heads={n_heads}, n_layers={n_layers}, window_size={window_size}',
                fontsize=15, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, alpha=0.3, linestyle='--', which='both')
    ax.grid(True, alpha=0.15, linestyle=':', which='minor')
    
    # Legend
    ax.legend(fontsize=12, loc='upper left', framealpha=0.95, 
             edgecolor='black', fancybox=True, shadow=True)
    
    # Format y-axis to show powers of 2
    def format_power_of_2(y, pos):
        if y <= 0:
            return ''
        power = int(np.round(np.log2(y)))
        return f'$2^{{{power}}}$'
    
    ax.yaxis.set_major_formatter(plt.FuncFormatter(format_power_of_2))
    
    # Set x-axis limits with some padding
    ax.set_xlim(0, max(tokens_range) + 1)
    
    plt.tight_layout()
    plt.savefig('kv_cache_elements_growth.png', 
               dpi=300, bbox_inches='tight', facecolor='white')
    print("\n✓ Visualization saved: kv_cache_elements_growth.png")
    
    return fig


def print_summary_table(tokens_range, softmax_sizes, swa_sizes, linear_sizes, window_size, d_model, n_heads):
    """Print a summary table"""
    
    print("\n" + "=" * 100)
    print("KV CACHE ELEMENTS SUMMARY")
    print("=" * 100)
    print(f"{'Token':<10} {'Standard Attn':<25} {'SWA':<25} {'Linear Attn':<25} {'Best Savings':<15}")
    print("-" * 100)
    
    for i in range(0, len(tokens_range), max(1, len(tokens_range) // 10)):
        token = tokens_range[i]
        swa_savings = (softmax_sizes[i] - swa_sizes[i]) / softmax_sizes[i] * 100 if softmax_sizes[i] > 0 else 0
        linear_savings = (softmax_sizes[i] - linear_sizes[i]) / softmax_sizes[i] * 100 if softmax_sizes[i] > 0 else 0
        best_savings = max(swa_savings, linear_savings)
        
        print(f"{token:<10} {softmax_sizes[i]:>20,} elem   {swa_sizes[i]:>20,} elem   "
              f"{linear_sizes[i]:>20,} elem   {best_savings:>6.1f}%")
    
    # Last token if not already printed
    if (len(tokens_range) - 1) % max(1, len(tokens_range) // 10) != 0:
        token = tokens_range[-1]
        swa_savings = (softmax_sizes[-1] - swa_sizes[-1]) / softmax_sizes[-1] * 100 if softmax_sizes[-1] > 0 else 0
        linear_savings = (softmax_sizes[-1] - linear_sizes[-1]) / softmax_sizes[-1] * 100 if softmax_sizes[-1] > 0 else 0
        best_savings = max(swa_savings, linear_savings)
        
        print(f"{token:<10} {softmax_sizes[-1]:>20,} elem   {swa_sizes[-1]:>20,} elem   "
              f"{linear_sizes[-1]:>20,} elem   {best_savings:>6.1f}%")
    
    print("=" * 100)
    
    # Print key insights
    print("\n" + "=" * 100)
    print("KEY INSIGHTS")
    print("=" * 100)
    head_d = d_model // n_heads
    crossover = head_d / 2
    print(f"Linear Attention Cache Size: {linear_sizes[0]:,} elements (CONSTANT for all sequence lengths)")
    print(f"  - Formula: n_layers × batch_size × n_heads × (d_k × d_v)")
    print(f"  - Crossover point: {int(crossover)} tokens (where linear becomes more efficient than standard)")
    print(f"  - Linear attention is MORE EFFICIENT for sequences longer than {int(crossover)} tokens")
    print(f"  - For short sequences (<{int(crossover)} tokens), standard attention uses less memory")
    print(f"\nAt {tokens_range[-1]} tokens:")
    print(f"  - Standard Attention: {softmax_sizes[-1]:,} elements")
    print(f"  - Sliding Window: {swa_sizes[-1]:,} elements ({(softmax_sizes[-1] - swa_sizes[-1]) / softmax_sizes[-1] * 100:.1f}% saved vs standard)")
    linear_vs_standard = (softmax_sizes[-1] - linear_sizes[-1]) / softmax_sizes[-1] * 100
    if linear_vs_standard > 0:
        print(f"  - Linear Attention: {linear_sizes[-1]:,} elements ({linear_vs_standard:.1f}% saved vs standard)")
    else:
        print(f"  - Linear Attention: {linear_sizes[-1]:,} elements ({abs(linear_vs_standard):.1f}% MORE than standard)")
    print("=" * 100)


def main():
    """Main execution function"""
    
    # Configuration
    D_MODEL = 2048
    N_HEADS = 32
    N_LAYERS = 40
    BATCH_SIZE = 1
    WINDOW_SIZE = 128
    MAX_TOKENS = 100  # Increased to show crossover point
    
    print("=" * 100)
    print("KV CACHE ELEMENTS GROWTH ANALYSIS")
    print("=" * 100)
    print(f"\nConfiguration:")
    print(f"  d_model = {D_MODEL}")
    print(f"  n_heads = {N_HEADS}")
    print(f"  d_k (per head) = {D_MODEL // N_HEADS}")
    print(f"  n_layers = {N_LAYERS}")
    print(f"  batch_size = {BATCH_SIZE}")
    print(f"  window_size = {WINDOW_SIZE}")
    print(f"  tokens = 1 to {MAX_TOKENS}")
    
    # Calculate cache sizes
    tokens_range = list(range(1, MAX_TOKENS + 1))
    softmax_sizes = []
    swa_sizes = []
    linear_sizes = []
    
    for token in tokens_range:
        softmax_size = get_softmax_attn_cache_size(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            seq_len=token,
            batch_size=BATCH_SIZE,
            n_layers=N_LAYERS
        )
        
        swa_size = get_swa_cache_size(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            window_size=WINDOW_SIZE,
            seq_len=token,
            batch_size=BATCH_SIZE,
            n_layers=N_LAYERS
        )
        
        linear_size = get_linear_cache_size(
            d_model=D_MODEL,
            n_heads=N_HEADS,
            seq_len=token,
            batch_size=BATCH_SIZE,
            n_layers=N_LAYERS
        )
        
        if softmax_size is None or swa_size is None or linear_size is None:
            raise ValueError(f"Failed to get cache size for token {token}")
        
        softmax_sizes.append(softmax_size)
        swa_sizes.append(swa_size)
        linear_sizes.append(linear_size)
    
    # Print summary
    print_summary_table(tokens_range, softmax_sizes, swa_sizes, linear_sizes, WINDOW_SIZE, D_MODEL, N_HEADS)
    
    # Create visualization
    fig = create_elements_growth_plot(tokens_range, softmax_sizes, swa_sizes, linear_sizes,
                                     WINDOW_SIZE, D_MODEL, N_HEADS, N_LAYERS)
    
    print("\n✓ Analysis complete!")
    print(f"✓ Generated plot with logarithmic y-axis (powers of 2)")
    print(f"✓ Token range: 1 to {MAX_TOKENS}")
    print(f"✓ Standard @ {MAX_TOKENS} tokens: {softmax_sizes[-1]:,} elements")
    print(f"✓ SWA @ {MAX_TOKENS} tokens: {swa_sizes[-1]:,} elements")
    print(f"✓ Linear @ {MAX_TOKENS} tokens: {linear_sizes[-1]:,} elements (constant)")
    
    swa_savings_pct = ((softmax_sizes[-1] - swa_sizes[-1]) / softmax_sizes[-1] * 100)
    linear_savings_pct = ((softmax_sizes[-1] - linear_sizes[-1]) / softmax_sizes[-1] * 100)
    print(f"✓ SWA Memory saved: {swa_savings_pct:.1f}%")
    print(f"✓ Linear Attention Memory saved: {linear_savings_pct:.1f}%\n")
    
    plt.show()


if __name__ == "__main__":
    main()