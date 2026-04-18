import matplotlib.pyplot as plt
import seaborn as sns
import os

def set_academic_style():
    """Apply consistent academic styling for all plots."""
    sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.titleweight': 'bold',
        'axes.labelweight': 'bold',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'figure.autolayout': True
    })

def get_palette(n_colors=5):
    """Return a visually distinct and professional color palette."""
    return sns.color_palette("viridis", n_colors)

def save_fig(fig, filename, output_dir="outputs/visualizations/figures"):
    """Save figure with consistent settings."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, bbox_inches='tight', dpi=300)
    print(f"Figure saved: {path}")
    return path
