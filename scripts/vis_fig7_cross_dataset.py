import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

# Add current directory to path for imports
sys.path.append('.')
from scripts.vis_utils import set_academic_style, save_fig

def get_dataset_name(pid):
    if str(pid).startswith("TCGA-"):
        # Very simplistic mapping for demo: TCGA usually GBM/LGG
        # In a real scenario we'd lookup the metadata
        if "DU-" in pid or "CS-" in pid or "FG-" in pid: return "TCGA-LGG"
        return "TCGA-GBM"
    elif str(pid).startswith("BraTS"):
        return "BraTS 2021"
    return "External"

def visualize_cross_dataset():
    set_academic_style()
    
    summary_path = "outputs/evaluation/retrieval_summary.csv"
    if not os.path.exists(summary_path):
        print("Summary data not found.")
        return

    df = pd.read_csv(summary_path)
    df['Cohort'] = df['query_patient_id'].apply(get_dataset_name)
    
    # Calculate group means and std
    stats = df.groupby('Cohort')['best_match_sim'].agg(['mean', 'std']).reset_index()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    barplot = sns.barplot(
        data=df, 
        x='Cohort', y='best_match_sim', 
        palette='magma', 
        capsize=.1, 
        errorbar='sd',
        ax=ax
    )
    
    ax.set_title("Figure 7. Cross-Dataset Retrieval Generalization", fontsize=22, pad=20)
    ax.set_ylabel("Similarity Score (Cosine)", fontweight='bold')
    ax.set_xlabel("Query Source Cohort", fontweight='bold')
    ax.set_ylim(0.9, 1.01) # Zoom in on the high similarity region
    
    # Add values on top of bars
    for p in barplot.patches:
        barplot.annotate(format(p.get_height(), '.3f'), 
                       (p.get_x() + p.get_width() / 2., p.get_height()), 
                       ha = 'center', va = 'center', 
                       xytext = (0, 15), 
                       textcoords = 'offset points',
                       fontsize=14, fontweight='bold')

    save_fig(fig, "fig7_cross_dataset.png")
    plt.close()

if __name__ == "__main__":
    visualize_cross_dataset()
