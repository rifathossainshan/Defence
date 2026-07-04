import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

def plot_similarity_matrix():
    print("Loading metadata and embeddings...")
    metadata_path = "outputs/faiss/index_metadata.csv"
    embeddings_path = "outputs/embeddings/embeddings.npy"
    
    if not os.path.exists(metadata_path) or not os.path.exists(embeddings_path):
        print("Error: Metadata or embeddings not found. Run extraction first.")
        return

    df = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path).astype('float32')
    
    # Normalize embeddings to calculate cosine similarity via dot product (or just use sklearn)
    # L2 Normalization
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings_normalized = embeddings / (norms + 1e-8)
    
    # Get unique datasets
    datasets = df['dataset'].unique()
    print(f"Found datasets: {datasets}")
    
    # Calculate average similarity matrix between datasets
    matrix_size = len(datasets)
    similarity_matrix = np.zeros((matrix_size, matrix_size))
    
    for i, ds1 in enumerate(datasets):
        for j, ds2 in enumerate(datasets):
            # Get indices for each dataset
            idx1 = df[df['dataset'] == ds1].index.values
            idx2 = df[df['dataset'] == ds2].index.values
            
            # Extract embeddings for the datasets
            emb1 = embeddings_normalized[idx1]
            emb2 = embeddings_normalized[idx2]
            
            # Compute cross-similarity
            # If it's the same dataset, we exclude self-similarity (diagonal of the cross-sim matrix)
            cross_sim = cosine_similarity(emb1, emb2)
            
            if i == j:
                # Remove self-similarity (diagonal values = 1.0) to get true intra-dataset similarity
                np.fill_diagonal(cross_sim, np.nan)
                avg_sim = np.nanmean(cross_sim)
            else:
                avg_sim = np.mean(cross_sim)
                
            similarity_matrix[i, j] = avg_sim
            
    # Plotting
    plt.figure(figsize=(10, 8))
    sns.set_theme(style="white")
    
    # Create a heatmap
    ax = sns.heatmap(
        similarity_matrix, 
        annot=True, 
        fmt=".4f", 
        cmap="YlGnBu", 
        xticklabels=datasets, 
        yticklabels=datasets,
        vmin=np.min(similarity_matrix) - 0.005,
        vmax=np.max(similarity_matrix) + 0.005,
        linewidths=.5,
        cbar_kws={"shrink": .8, "label": "Mean Cosine Similarity"}
    )
    
    # Customize the plot
    plt.title("Cross-Cohort Similarity Coefficient Matrix\n(Average Cosine Similarity in Embedding Space)", pad=20, fontsize=14, fontweight='bold')
    plt.xlabel("Target Cohort", fontsize=12, labelpad=10)
    plt.ylabel("Source Cohort", fontsize=12, labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save the figure
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "similarity_coefficient_matrix.png")
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved successfully to: {output_path}")
    plt.show()

if __name__ == "__main__":
    plot_similarity_matrix()
