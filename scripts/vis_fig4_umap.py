import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import sys

# Add current directory to path for imports
sys.path.append('.')
from scripts.vis_utils import set_academic_style, save_fig, get_palette

def visualize_embedding_space():
    set_academic_style()
    
    # 1. Load Data
    embeddings_path = "outputs/embeddings/hybrid_embeddings.npy"
    metadata_path = "outputs/embeddings/hybrid_metadata.csv"
    
    if not os.path.exists(embeddings_path) or not os.path.exists(metadata_path):
        print("Embedding data not found. Extraction might be needed.")
        return

    features = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    
    # Ensure indices match
    n_samples = min(len(features), len(metadata))
    features = features[:n_samples]
    metadata = metadata.iloc[:n_samples]

    print(f"Running t-SNE on {n_samples} samples...")
    # 2. t-SNE Projection
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    projections = tsne.fit_transform(features)
    
    metadata['tsne_1'] = projections[:, 0]
    metadata['tsne_2'] = projections[:, 1]
    
    # 3. Plotting
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Map dataset names for cleaner legend
    dataset_map = {
        'BraTS2021': 'BraTS 2021 (Training)',
        'GBM': 'TCGA-GBM (External)',
        'LGG': 'TCGA-LGG (External)'
    }
    metadata['Display_Dataset'] = metadata['dataset'].map(dataset_map).fillna(metadata['dataset'])

    scatter = sns.scatterplot(
        data=metadata,
        x='tsne_1', y='tsne_2',
        hue='Display_Dataset',
        style='Display_Dataset',
        palette='viridis',
        s=80, alpha=0.7,
        ax=ax
    )
    
    ax.set_title("Figure 4. Embedding Space Visualization (t-SNE Projection)", fontsize=22, pad=20)
    ax.set_xlabel("t-SNE Component 1")
    ax.set_ylabel("t-SNE Component 2")
    
    legend = ax.legend(title="Dataset Cohort", title_fontsize='16', fontsize='14', loc='best', frameon=True)
    plt.setp(legend.get_title(), multialignment='center')

    save_fig(fig, "fig4_tsne.png")
    plt.close()

if __name__ == "__main__":
    visualize_embedding_space()
