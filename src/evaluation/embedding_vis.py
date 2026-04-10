import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os
import sys

def visualize_embeddings(embeddings_path, metadata_path, output_path):
    """
    Project 512-dim embeddings to 2D using UMAP and plot by dataset.
    """
    # 1. Load Data
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings not found at {embeddings_path}")
        return

    embeddings = np.load(embeddings_path)
    metadata = pd.read_csv(metadata_path)
    
    print(f"Running UMAP projection for {embeddings.shape[0]} samples...")
    
    # 2. UMAP Projection
    # Note: UMAP can be compute-intensive on CPU for large datasets, 
    # but 1381 samples should be fine.
    reducer = umap.UMAP(
        n_neighbors=15, 
        min_dist=0.1, 
        n_components=2, 
        metric='cosine', # Since we used cosine similarity for retrieval
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)
    
    # 3. Plotting preparation
    plt.figure(figsize=(12, 10))
    sns.set_style("darkgrid")
    
    # Scatter plot colored by dataset
    scatter = sns.scatterplot(
        x=embedding_2d[:, 0], 
        y=embedding_2d[:, 1], 
        hue=metadata['dataset'], 
        palette='magma', 
        alpha=0.7, 
        s=50,
        edgecolor='w',
        linewidth=0.5
    )
    
    plt.title("UMAP Visualization of Self-Supervised MRI Embeddings (MVP Phase)", fontsize=15)
    plt.xlabel("UMAP Component 1", fontsize=12)
    plt.ylabel("UMAP Component 2", fontsize=12)
    plt.legend(title="Dataset Source", loc='best', fontsize=10)
    
    # 4. Save Figure
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    
    print(f"\n[SUCCESS] UMAP visualization generated!")
    print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # Default paths for our project
    visualize_embeddings(
        embeddings_path="outputs/embeddings/embeddings.npy",
        metadata_path="outputs/embeddings/embedding_metadata.csv",
        output_path="outputs/figures/umap_embeddings.png"
    )
