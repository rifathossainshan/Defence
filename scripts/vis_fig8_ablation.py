import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path for imports
sys.path.append('.')
from scripts.vis_utils import set_academic_style, save_fig

def visualize_ablation():
    set_academic_style()
    
    # Data derived from outputs/logs/ablation_summary.txt
    # and previous session results
    data = {
        'Strategy': ['Whole-Brain', 'Lesion-Centered', 'Single-Modality', 'Multi-Branch (Ours)'],
        'Consistency Score': [0.9894, 0.9890, 0.9450, 0.9921],
        'Category': ['Input Space', 'Input Space', 'Architecture', 'Architecture']
    }
    
    df = pd.DataFrame(data)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    barplot = sns.barplot(
        data=df, 
        x='Strategy', y='Consistency Score', 
        hue='Category',
        palette='rocket',
        ax=ax
    )
    
    ax.set_title("Figure 8. Ablation Analysis of Model Design Choices", fontsize=22, pad=20)
    ax.set_ylabel("Retrieval Consistency (mAP / Sim)", fontweight='bold')
    ax.set_xlabel("Experimental Configuration", fontweight='bold')
    ax.set_ylim(0.9, 1.0)
    
    # Add values
    for p in barplot.patches:
        if p.get_height() > 0:
            barplot.annotate(format(p.get_height(), '.4f'), 
                           (p.get_x() + p.get_width() / 2., p.get_height()), 
                           ha = 'center', va = 'center', 
                           xytext = (0, 10), 
                           textcoords = 'offset points',
                           fontsize=13, fontweight='bold')

    save_fig(fig, "fig8_ablation.png")
    plt.close()

if __name__ == "__main__":
    visualize_ablation()
