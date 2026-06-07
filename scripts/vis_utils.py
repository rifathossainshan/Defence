import matplotlib
matplotlib.use('Agg') # Use non-interactive backend to avoid Tkinter GUI memory overhead
import matplotlib.pyplot as plt
import os

def set_academic_style():
    """Apply consistent academic styling for all plots without seaborn."""
    # Use matplotlib default styles to set a clean academic look
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.titleweight': 'normal', 
        'axes.labelweight': 'bold',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.autolayout': True,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.grid': False,
        'font.size': 14
    })

def add_panel_label(ax, label, x=-0.05, y=1.1):
    """Add labels like (A), (B) to the plot."""
    ax.text(x, y, label, transform=ax.transAxes,
            fontsize=24, fontweight='bold', va='top', ha='right')

def get_palette(n_colors=5):
    """Return a visually distinct and professional color palette using matplotlib colormap."""
    cmap = matplotlib.colormaps['viridis']
    return [cmap(val) for val in [i/(n_colors-1) if n_colors > 1 else 0 for i in range(n_colors)]]

def save_fig(fig, filename, output_dir="outputs/visualizations/figures"):
    """Save figure with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Figure saved: {path}")
    return path
