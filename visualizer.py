import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set academic publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 14,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True
})

# Color palette for academic plots
COLORS = {
    'attention': '#E74C3C',      # Red
    'softmax': '#3498DB',        # Blue
    'matmul': '#2ECC71',         # Green
    'sgemm': '#F39C12',          # Orange
    'elementwise': '#9B59B6',    # Purple
    'other': '#95A5A6'           # Gray
}

def categorize_kernel(name):
    """Categorize CUDA kernels by operation type"""
    name_lower = name.lower()
    
    if 'softmax' in name_lower:
        return 'Softmax'
    elif 'sgemm' in name_lower or 'gemm' in name_lower:
        return 'Matrix Multiply (GEMM)'
    elif 'maxwell' in name_lower or 'matmul' in name_lower:
        return 'Matrix Operations'
    elif 'elementwise' in name_lower or 'vectorized' in name_lower:
        return 'Elementwise Operations'
    elif 'attention' in name_lower:
        return 'Attention'
    else:
        return 'Other'

def load_and_process_data():
    """Load and categorize profiling data"""
    
    # Load NVTX data
    try:
        nvtx_df = pd.read_csv('unified_report_nvtx_sum.csv')
        print(f"NVTX columns: {nvtx_df.columns.tolist()}")
    except:
        nvtx_df = pd.DataFrame()
    
    # Load CUDA kernel data
    try:
        cuda_df = pd.read_csv('unified_report_cuda_gpu_kern_sum.csv')
        print(f"CUDA columns: {cuda_df.columns.tolist()}")
        
        # Find the correct column names
        time_col = [c for c in cuda_df.columns if 'time' in c.lower()][0]
        name_col = [c for c in cuda_df.columns if 'name' in c.lower()][0]
        
        # Clean and convert time column
        if cuda_df[time_col].dtype == 'object':
            cuda_df[time_col] = cuda_df[time_col].str.replace('%', '').str.replace(',', '')
        cuda_df[time_col] = pd.to_numeric(cuda_df[time_col], errors='coerce')
        
        # Add category
        cuda_df['Category'] = cuda_df[name_col].apply(categorize_kernel)
        
        return nvtx_df, cuda_df, time_col, name_col
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None, None, None

def plot_kernel_breakdown(cuda_df, time_col, name_col):
    """Create academic-style kernel breakdown plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Top 15 kernels by execution time
    ax1 = axes[0, 0]
    top_kernels = cuda_df.nlargest(15, time_col)
    
    # Color by category
    colors = [COLORS.get(cat.split()[0].lower(), COLORS['other']) 
              for cat in top_kernels['Category']]
    
    y_pos = np.arange(len(top_kernels))
    bars = ax1.barh(y_pos, top_kernels[time_col], color=colors, edgecolor='black', linewidth=0.5)
    ax1.set_yticks(y_pos)
    
    # Truncate long kernel names for readability
    labels = []
    for name in top_kernels[name_col]:
        if len(name) > 50:
            labels.append(name[:47] + '...')
        else:
            labels.append(name)
    
    ax1.set_yticklabels(labels, fontsize=9)
    ax1.set_xlabel('Execution Time (ns)', fontweight='bold')
    ax1.set_title('(a) Top 15 CUDA Kernels by Execution Time', fontweight='bold', loc='left')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Kernel category breakdown (pie chart)
    ax2 = axes[0, 1]
    category_time = cuda_df.groupby('Category')[time_col].sum().sort_values(ascending=False)
    
    category_colors = [COLORS.get(cat.split()[0].lower(), COLORS['other']) 
                      for cat in category_time.index]
    
    wedges, texts, autotexts = ax2.pie(category_time.values, 
                                        labels=category_time.index,
                                        autopct='%1.1f%%',
                                        startangle=90,
                                        colors=category_colors,
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 0.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(9)
    
    ax2.set_title('(b) Kernel Execution Time by Category', fontweight='bold', loc='left')
    
    # Plot 3: Attention-related kernels
    ax3 = axes[1, 0]
    attention_keywords = ['attention', 'softmax', 'sgemm', 'gemm', 'matmul']
    attention_kernels = cuda_df[cuda_df[name_col].str.lower().str.contains('|'.join(attention_keywords), na=False)]
    attention_kernels = attention_kernels.nlargest(10, time_col)
    
    if not attention_kernels.empty:
        colors_att = [COLORS.get(cat.split()[0].lower(), COLORS['other']) 
                     for cat in attention_kernels['Category']]
        
        y_pos = np.arange(len(attention_kernels))
        ax3.barh(y_pos, attention_kernels[time_col], color=colors_att, 
                edgecolor='black', linewidth=0.5)
        ax3.set_yticks(y_pos)
        
        labels = []
        for name in attention_kernels[name_col]:
            if len(name) > 45:
                labels.append(name[:42] + '...')
            else:
                labels.append(name)
        
        ax3.set_yticklabels(labels, fontsize=9)
        ax3.set_xlabel('Execution Time (ns)', fontweight='bold')
        ax3.set_title('(c) Attention-Related Kernels', fontweight='bold', loc='left')
        ax3.invert_yaxis()
        ax3.grid(axis='x', alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No attention-related kernels found', 
                ha='center', va='center', transform=ax3.transAxes)
        ax3.set_title('(c) Attention-Related Kernels', fontweight='bold', loc='left')
    
    # Plot 4: Cumulative time distribution
    ax4 = axes[1, 1]
    sorted_df = cuda_df.sort_values(time_col, ascending=False).reset_index(drop=True)
    cumulative_time = sorted_df[time_col].cumsum()
    total_time = sorted_df[time_col].sum()
    cumulative_percent = (cumulative_time / total_time) * 100
    
    ax4.plot(range(1, len(cumulative_percent)+1), cumulative_percent, 
            linewidth=2, color='#2C3E50')
    ax4.axhline(y=80, color='red', linestyle='--', linewidth=1.5, label='80% threshold')
    ax4.axhline(y=90, color='orange', linestyle='--', linewidth=1.5, label='90% threshold')
    
    ax4.set_xlabel('Number of Kernels', fontweight='bold')
    ax4.set_ylabel('Cumulative Time (%)', fontweight='bold')
    ax4.set_title('(d) Cumulative Kernel Execution Time', fontweight='bold', loc='left')
    ax4.legend(loc='lower right', frameon=True, fancybox=False, edgecolor='black')
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim(0, min(100, len(cumulative_percent)))
    
    plt.tight_layout()
    plt.savefig('kernel_analysis_academic.png', dpi=300, bbox_inches='tight')
    print("Saved: kernel_analysis_academic.png")
    plt.close()


def plot_top_kernels_only():
    """Plot ONLY the top 15 kernels - single clean figure"""
    
    # Load data
    cuda_df = pd.read_csv('unified_report_cuda_gpu_kern_sum.csv')
    print(f"\nCUDA Kernel Report columns: {cuda_df.columns.tolist()}")
    
    # Find the correct column names
    time_col = [c for c in cuda_df.columns if 'time' in c.lower()][0]
    name_col = [c for c in cuda_df.columns if 'name' in c.lower()][0]
    
    # Clean and convert time column
    if cuda_df[time_col].dtype == 'object':
        cuda_df[time_col] = cuda_df[time_col].str.replace('%', '').str.replace(',', '')
    cuda_df[time_col] = pd.to_numeric(cuda_df[time_col], errors='coerce')
    
    # Get top 15 kernels
    top_kernels = cuda_df.nlargest(15, time_col)
    
    # Categorize and color
    categories = [categorize_kernel(name) for name in top_kernels[name_col]]
    colors = [COLORS.get(cat.split()[0].lower(), COLORS['other']) for cat in categories]
    
    # Create single figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    y_pos = np.arange(len(top_kernels))
    bars = ax.barh(y_pos, top_kernels[time_col], color=colors, edgecolor='black', linewidth=0.5)
    ax.set_yticks(y_pos)
    
    # Truncate long kernel names for readability
    labels = []
    for name in top_kernels[name_col]:
        if len(name) > 50:
            labels.append(name[:47] + '...')
        else:
            labels.append(name)
    
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Execution Time (ns)', fontweight='bold')
    ax.set_title('Top 15 CUDA Kernels by Execution Time', fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # Add legend for kernel types
    unique_cats = sorted(list(set(categories)))
    legend_patches = [mpatches.Patch(color=COLORS.get(cat.split()[0].lower(), COLORS['other']), 
                                     label=cat, edgecolor='black', linewidth=0.5)
                     for cat in unique_cats]
    ax.legend(handles=legend_patches, loc='lower right', frameon=True,
              fancybox=False, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig('top_15_kernels_only.png', dpi=300, bbox_inches='tight')
    print("\n✓ Saved: top_15_kernels_only.png")
    plt.close()


def plot_attention_focus(cuda_df, time_col, name_col):
    """Create focused plot on attention/softmax/matmul operations"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Define operation types
    operations = {
        'Softmax': ['softmax'],
        'GEMM/MatMul': ['sgemm', 'gemm', 'matmul', 'maxwell'],
        'Elementwise': ['elementwise', 'vectorized']
    }
    
    # Extract relevant kernels
    operation_data = {}
    for op_name, keywords in operations.items():
        mask = cuda_df[name_col].str.lower().str.contains('|'.join(keywords), na=False)
        op_kernels = cuda_df[mask]
        if not op_kernels.empty:
            operation_data[op_name] = op_kernels[time_col].sum()
    
    # Plot 1: Operation time comparison
    ax1 = axes[0]
    if operation_data:
        ops = list(operation_data.keys())
        times = list(operation_data.values())
        colors_ops = ['#3498DB', '#2ECC71', '#9B59B6'][:len(ops)]
        
        bars = ax1.bar(ops, times, color=colors_ops, edgecolor='black', linewidth=1.2, alpha=0.8)
        ax1.set_ylabel('Total Execution Time (ns)', fontweight='bold')
        ax1.set_title('(a) Attention Operation Breakdown', fontweight='bold', loc='left')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2e}',
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Plot 2: Detailed GEMM breakdown
    ax2 = axes[1]
    gemm_kernels = cuda_df[cuda_df[name_col].str.lower().str.contains('gemm|sgemm', na=False)]
    gemm_top = gemm_kernels.nlargest(8, time_col)
    
    if not gemm_top.empty:
        y_pos = np.arange(len(gemm_top))
        bars = ax2.barh(y_pos, gemm_top[time_col], color='#2ECC71', 
                       edgecolor='black', linewidth=0.5, alpha=0.8)
        ax2.set_yticks(y_pos)
        
        labels = []
        for name in gemm_top[name_col]:
            # Extract key info from kernel name
            if 'sgemm' in name.lower():
                parts = name.split('sgemm')
                if len(parts) > 1:
                    labels.append('sgemm' + parts[1][:30])
                else:
                    labels.append(name[:35])
            else:
                labels.append(name[:35])
        
        ax2.set_yticklabels(labels, fontsize=9)
        ax2.set_xlabel('Execution Time (ns)', fontweight='bold')
        ax2.set_title('(b) Top Matrix Multiplication Kernels', fontweight='bold', loc='left')
        ax2.invert_yaxis()
        ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('attention_operations_academic.png', dpi=300, bbox_inches='tight')
    print("Saved: attention_operations_academic.png")
    plt.close()

def plot_performance_table(cuda_df, time_col, name_col):
    """Create a performance summary table"""
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
    
    # Get top 20 kernels
    top_kernels = cuda_df.nlargest(20, time_col)
    
    # Prepare table data
    table_data = []
    for idx, row in top_kernels.iterrows():
        kernel_name = row[name_col]
        if len(kernel_name) > 60:
            kernel_name = kernel_name[:57] + '...'
        
        table_data.append([
            row['Category'],
            kernel_name,
            f"{row[time_col]:.2e}"
        ])
    
    # Create table
    table = ax.table(cellText=table_data,
                    colLabels=['Category', 'Kernel Name', 'Time (ns)'],
                    cellLoc='left',
                    loc='center',
                    colWidths=[0.2, 0.6, 0.2])
    
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)
    
    # Style header
    for i in range(3):
        table[(0, i)].set_facecolor('#34495E')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#ECF0F1')
            else:
                table[(i, j)].set_facecolor('white')
            table[(i, j)].set_edgecolor('black')
    
    plt.title('Top 20 CUDA Kernels - Performance Summary', 
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig('kernel_summary_table.png', dpi=300, bbox_inches='tight')
    print("Saved: kernel_summary_table.png")
    plt.close()

def main():
    print("="*70)
    print("ACADEMIC-STYLE NSYS PROFILING ANALYSIS")
    print("="*70)
    
    nvtx_df, cuda_df, time_col, name_col = load_and_process_data()
    
    if cuda_df is None or cuda_df.empty:
        print("Error: No CUDA kernel data found!")
        return
    
    print(f"\nProcessing {len(cuda_df)} CUDA kernels...")
    print(f"Total execution time: {cuda_df[time_col].sum():.2e} ns")
    
    # Generate plots
    print("\nGenerating plots...")
    plot_kernel_breakdown(cuda_df, time_col, name_col)
    plot_attention_focus(cuda_df, time_col, name_col)
    plot_performance_table(cuda_df, time_col, name_col)
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated files:")
    print("  1. kernel_analysis_academic.png - Comprehensive kernel analysis")
    print("  2. attention_operations_academic.png - Attention operations focus")
    print("  3. kernel_summary_table.png - Performance summary table")
    print("\nAll plots are publication-ready (300 DPI, serif fonts)")

if __name__ == "__main__":
    main()