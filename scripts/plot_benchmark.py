import os
import matplotlib.pyplot as plt
import numpy as np

def apply_transparent_style(ax):
    # Set text colors to slate gray which is visible on both black and white backgrounds
    text_color = '#334155'
    tick_color = '#64748b'
    grid_color = '#cbd5e1'
    
    # Configure axes colors
    ax.set_facecolor('none')
    ax.spines['bottom'].set_color(tick_color)
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color(tick_color)
    
    ax.xaxis.label.set_color(text_color)
    ax.yaxis.label.set_color(text_color)
    ax.title.set_color(text_color)
    
    ax.tick_params(colors=tick_color, which='both')
    ax.grid(True, linestyle=':', color=grid_color, alpha=0.3)

def plot_bar(title, val1, val2, ylabel, filename, val_format='{:.2f}'):
    fig, ax = plt.subplots(figsize=(4.5, 3.5), facecolor='none')
    
    color_intextus = '#4f46e5'
    color_fastembed = '#94a3b8'
    
    rects1 = ax.bar([-0.2], [val1], 0.35, label='intextus (Q8_0)', color=color_intextus)
    rects2 = ax.bar([0.2], [val2], 0.35, label='fastembed (ONNX)', color=color_fastembed)
    
    ax.set_ylabel(ylabel, fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    ax.set_xlim(-0.6, 0.6)
    ax.set_xticks([])
    ax.set_xticklabels([])
    
    # Style legend
    legend = ax.legend(frameon=True, facecolor='none', edgecolor='none', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#334155')
        
    apply_transparent_style(ax)
    
    # Add values on top of the bars
    for rect in rects1:
        height = rect.get_height()
        ax.annotate(val_format.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='#4f46e5')
    for rect in rects2:
        height = rect.get_height()
        ax.annotate(val_format.format(height), xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points", ha='center', va='bottom', 
                    fontsize=9, fontweight='bold', color='#64748b')
                    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()

def plot_line(title, batch_sizes, y_intextus, y_fastembed, filename):
    fig, ax = plt.subplots(figsize=(5, 3.5), facecolor='none')
    
    ax.plot(batch_sizes, y_intextus, marker='o', linewidth=2.5, label='intextus (Q8_0)', color='#4f46e5')
    ax.plot(batch_sizes, y_fastembed, marker='x', linestyle='--', linewidth=2, label='fastembed (ONNX)', color='#94a3b8')
    
    ax.set_xscale('log')
    ax.set_xticks(batch_sizes)
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    
    ax.set_xlabel('Batch Size (Log Scale)', fontsize=9, fontweight='bold')
    ax.set_ylabel('Throughput (sent/sec)', fontsize=10, fontweight='bold')
    ax.set_title(title, fontsize=11, fontweight='bold', pad=8)
    
    # Style legend
    legend = ax.legend(frameon=True, facecolor='none', edgecolor='none', fontsize=8)
    for text in legend.get_texts():
        text.set_color('#334155')
        
    apply_transparent_style(ax)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150, transparent=True)
    plt.close()

def main():
    os.makedirs('assets', exist_ok=True)
    batch_sizes = [1, 4, 8, 32, 128]
    
    # ------------------ MiniLM Charts ------------------
    plot_bar(
        title='Single Latency\n(Lower is Better)',
        val1=2.15, val2=8.50,
        ylabel='Mean Latency (ms)',
        filename='assets/minilm_latency.png',
        val_format='{:.2f}ms'
    )
    
    plot_line(
        title='Throughput Scaling\n(Higher is Better)',
        batch_sizes=batch_sizes,
        y_intextus=[821.2, 753.6, 699.5, 693.4, 682.3],
        y_fastembed=[109.8, 141.5, 137.0, 76.2, 53.6],
        filename='assets/minilm_throughput.png'
    )
    
    plot_bar(
        title='Peak Memory Footprint\n(Lower is Better)',
        val1=106.2, val2=945.3,
        ylabel='Peak RSS (MB)',
        filename='assets/minilm_memory.png',
        val_format='{:.1f}MB'
    )
    
    # ------------------ BGE Charts ------------------
    plot_bar(
        title='Single Latency\n(Lower is Better)',
        val1=5.17, val2=5.30,
        ylabel='Mean Latency (ms)',
        filename='assets/bge_latency.png',
        val_format='{:.2f}ms'
    )
    
    plot_line(
        title='Throughput Scaling\n(Higher is Better)',
        batch_sizes=batch_sizes,
        y_intextus=[373.3, 290.9, 269.9, 268.0, 273.0],
        y_fastembed=[200.7, 442.0, 356.9, 405.0, 231.8],
        filename='assets/bge_throughput.png'
    )
    
    plot_bar(
        title='Peak Memory Footprint\n(Lower is Better)',
        val1=120.2, val2=403.5,
        ylabel='Peak RSS (MB)',
        filename='assets/bge_memory.png',
        val_format='{:.1f}MB'
    )
    
    print("All 6 transparent benchmark charts generated successfully in assets/")

if __name__ == "__main__":
    main()
