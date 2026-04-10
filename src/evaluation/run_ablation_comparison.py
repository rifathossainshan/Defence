import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import os
import faiss

def run_ablation_comparison(config):
    print("--- STARTING ABLATION COMPARISON (Whole-Brain vs Lesion-Centered) ---")
    
    # 1. Load Data
    print("Loading embeddings...")
    emb_w = np.load(config["emb_whole"])
    emb_l = np.load(config["emb_lesion"])
    meta_w = pd.read_csv(config["meta_whole"])
    meta_l = pd.read_csv(config["meta_lesion"])
    
    # 2. UMAP Projections
    print("Computing UMAP for both modes...")
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_w = reducer.fit_transform(emb_w)
    
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_l = reducer.fit_transform(emb_l)
    
    # 3. Side-by-Side Visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    sns.set_style("darkgrid")
    
    # Plot Whole-Brain
    sns.scatterplot(x=umap_w[:, 0], y=umap_w[:, 1], hue=meta_w['dataset'], palette='viridis', alpha=0.6, ax=ax1, s=40)
    ax1.set_title("A: Whole-Brain Embedding Space", fontsize=14)
    ax1.legend(title="Dataset")
    
    # Plot Lesion-Centered
    sns.scatterplot(x=umap_l[:, 0], y=umap_l[:, 1], hue=meta_l['dataset'], palette='viridis', alpha=0.6, ax=ax2, s=40)
    ax2.set_title("B: Lesion-Centered Embedding Space", fontsize=14)
    ax2.legend(title="Dataset")
    
    os.makedirs(os.path.dirname(config["output_fig"]), exist_ok=True)
    plt.tight_layout()
    plt.savefig(config["output_fig"], dpi=300)
    print(f"Comparison plot saved to: {config['output_fig']}")
    
    # 4. Consistency Metric (Average Similarity of Top-5 Neighbors)
    def get_avg_sim(embeddings):
        idx = faiss.IndexFlatIP(embeddings.shape[1])
        faiss.normalize_L2(embeddings)
        idx.add(embeddings)
        # Search top-6 (include self)
        sims, _ = idx.search(embeddings, 6)
        return np.mean(sims[:, 1:]) # Average excluding self
        
    avg_sim_w = get_avg_sim(emb_w.copy())
    avg_sim_l = get_avg_sim(emb_l.copy())
    
    summary = f"""--- ABLATION STUDY RESULTS ---
1. Whole-Brain Consistency (Avg Top-5 Sim): {avg_top5_w:.4f}
2. Lesion-Centered Consistency (Avg Top-5 Sim): {avg_top5_l:.4f}

OBSERVATION:
Higher consistency in 'lesion' mode suggests that the model is refocusing on tumor traits
rather than global brain anatomy. 
------------------------------
""".replace("avg_top5_w", f"{avg_sim_w}").replace("avg_top5_l", f"{avg_sim_l}")
    
    with open(config["output_txt"], "w") as f:
        f.write(summary)
    print("\nSummary Results:")
    print(summary)

if __name__ == "__main__":
    CONFIG = {
        "emb_whole": "outputs/embeddings/embeddings.npy",
        "meta_whole": "outputs/embeddings/embedding_metadata.csv",
        "emb_lesion": "outputs/embeddings/embeddings_lesion.npy",
        "meta_lesion": "outputs/embeddings/embedding_metadata_lesion.csv",
        "output_fig": "outputs/figures/ablation/whole_vs_lesion_umap.png",
        "output_txt": "outputs/logs/ablation_summary.txt"
    }
    
    if os.path.exists(CONFIG["emb_lesion"]):
        run_ablation_comparison(CONFIG)
    else:
        print("Warning: Lesion embeddings not found yet. Please wait for extraction.")
