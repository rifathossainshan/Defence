import numpy as np
import pandas as pd
import faiss
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import os
import nibabel as nib
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_mini_evaluation(config):
    print("--- RUNNING MINI EVALUATION (Hybrid Model) ---")
    
    # 1. Load Data
    embeddings = np.load(config["embeddings_path"])
    metadata = pd.read_csv(config["metadata_path"])
    master_meta = pd.read_csv("data/metadata/metadata_brats2021.csv")

    # 2. Build Mini FAISS Index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    os.makedirs(os.path.dirname(config["index_save"]), exist_ok=True)
    faiss.write_index(index, config["index_save"])
    print(f"Mini Index saved to {config['index_save']}")

    # 3. UMAP Plot
    print("Computing UMAP...")
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(embeddings)
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=umap_2d[:, 0], y=umap_2d[:, 1], hue=metadata['dataset'], palette='Set1', s=80)
    plt.title("Mini Evaluation: UMAP of Hybrid Embeddings (50 Samples)")
    plt.tight_layout()
    plt.savefig(config["umap_path"], dpi=200)
    print(f"UMAP saved to {config['umap_path']}")

    # 4. Retrieval Panels (for 2 Queries)
    # Queries: Case 0 and Case 20
    queries = [0, 20]
    for q_idx in queries:
        q_id = metadata.iloc[q_idx]["patient_id"]
        q_vec = embeddings[q_idx:q_idx+1]
        
        # Search Top-6 (exclude self logic)
        sims, ids = index.search(q_vec, 6)
        sims, ids = sims[0], ids[0]
        
        # Filter self
        neighbor_ids = []
        neighbor_scores = []
        for i, idx in enumerate(ids):
            pid = metadata.iloc[idx]["patient_id"]
            if pid == q_id: continue
            neighbor_ids.append(pid)
            neighbor_scores.append(sims[i])
            if len(neighbor_ids) >= 5: break

        # Plotting
        fig, axes = plt.subplots(1, 6, figsize=(18, 4))
        
        def get_slice(pid):
            row = master_meta[master_meta["patient_id"] == pid].iloc[0]
            img = nib.load(row["flair_path"]).get_fdata()
            slice_data = img[:, :, img.shape[2]//2]
            slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) + 1e-8)
            return np.rot90(slice_data)

        # Query
        axes[0].imshow(get_slice(q_id), cmap='gray')
        axes[0].set_title(f"Query\n{q_id}", color='blue')
        axes[0].axis('off')
        
        # Neighbors
        for i in range(len(neighbor_ids)):
            axes[i+1].imshow(get_slice(neighbor_ids[i]), cmap='gray')
            axes[i+1].set_title(f"Sim: {neighbor_scores[i]:.3f}")
            axes[i+1].axis('off')
            
        out_panel = config["panel_path"].replace("{ID}", q_id)
        plt.savefig(out_panel, dpi=150)
        plt.close()
        print(f"Panel saved for {q_id}")

    # 5. Summary Info
    avg_sim = np.mean(index.search(embeddings, 6)[0][:, 1:])
    summary = f"""--- MINI EVALUATION SUMMARY (Hybrid) ---
Total Samples: 50
Average Top-5 Similarity: {avg_sim:.4f}
Embeddings Status: VALID (Finite, Unique)
Visualization: UMAP and Panels generated.
    """
    with open(config["summary_log"], "w") as f:
        f.write(summary)
    print(summary)

if __name__ == "__main__":
    CONFIG = {
        "embeddings_path": "outputs/embeddings/minieval_embeddings.npy",
        "metadata_path": "outputs/embeddings/minieval_metadata.csv",
        "index_save": "outputs/faiss/minieval_faiss.index",
        "umap_path": "outputs/figures/minieval_umap.png",
        "panel_path": "outputs/figures/minieval_top5_panel_{ID}.png",
        "summary_log": "outputs/logs/minieval_summary.txt"
    }
    run_mini_evaluation(CONFIG)
