import nibabel as nib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import sys
import faiss
from pathlib import Path

def create_retrieval_panel(query_id, config):
    """
    Creates a visual panel showing the query MRI slice and its Top-K retrieved neighbors.
    """
    # 1. Load Index and Metadata
    index = faiss.read_index(config["index_path"])
    index_meta = pd.read_csv(config["index_metadata"])
    master_meta = pd.read_csv(config["master_metadata"])
    embeddings = np.load(config["embeddings_npy"]).astype("float32")
    faiss.normalize_L2(embeddings)

    # 2. Get Query Data
    query_row = index_meta[index_meta["patient_id"] == query_id]
    if query_row.empty:
        print(f"Error: {query_id} not in index.")
        return
    
    q_idx = query_row.index[0]
    q_vec = embeddings[q_idx:q_idx+1]

    # 3. Search Top-K
    # We ask for top_k + 1 to exclude self
    top_k = config.get("top_k", 5)
    sims, ids = index.search(q_vec, top_k + 1)
    sims, ids = sims[0], ids[0]

    # Filter out self
    neighbor_ids = []
    neighbor_scores = []
    for i, idx in enumerate(ids):
        pid = index_meta.iloc[idx]["patient_id"]
        if pid == query_id: continue
        neighbor_ids.append(pid)
        neighbor_scores.append(sims[i])
        if len(neighbor_ids) >= top_k: break

    # 4. Visualization Setup
    fig, axes = plt.subplots(1, top_k + 1, figsize=(18, 4))
    plt.subplots_adjust(wspace=0.1)

    # Function to get central slice
    def get_slice(pid):
        m_row = master_meta[master_meta["patient_id"] == pid].iloc[0]
        # FLAIR is usually the best for tumor visualization
        path = m_row["flair_path"]
        if not os.path.exists(path):
            return np.zeros((240, 240))
        
        img = nib.load(path).get_fdata()
        # Take central axial slice (BraTS shape is 240x240x155)
        # We take z=77
        z_idx = img.shape[2] // 2
        slice_data = img[:, :, z_idx]
        
        # Simple normalization for display
        slice_data = (slice_data - np.min(slice_data)) / (np.max(slice_data) + 1e-8)
        return np.rot90(slice_data) # Rotate for correct orientation

    # Plot Query
    q_slice = get_slice(query_id)
    axes[0].imshow(q_slice, cmap="gray")
    axes[0].set_title(f"Query\n{query_id}", color="blue", fontsize=10)
    axes[0].axis("off")

    # Plot Neighbors
    for i in range(len(neighbor_ids)):
        n_id = neighbor_ids[i]
        n_score = neighbor_scores[i]
        n_slice = get_slice(n_id)
        
        axes[i+1].imshow(n_slice, cmap="gray")
        axes[i+1].set_title(f"Rank {i+1}\nSim: {n_score:.3f}", fontsize=9)
        axes[i+1].axis("off")

    # 5. Save Panel
    os.makedirs(os.path.dirname(config["output_fig"]), exist_ok=True)
    out_path = config["output_fig"].replace("{ID}", query_id)
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Panel saved for {query_id} to {out_path}")

if __name__ == "__main__":
    CONFIG = {
        "index_path": "outputs/faiss/faiss.index",
        "index_metadata": "outputs/faiss/index_metadata.csv",
        "master_metadata": "data/metadata/metadata_brats2021.csv",
        "embeddings_npy": "outputs/embeddings/embeddings.npy",
        "output_fig": "outputs/figures/retrieval_panel_{ID}.png",
        "top_k": 5
    }

    # Select 3 query cases from different datasets for demonstration
    meta = pd.read_csv(CONFIG["index_metadata"])
    # 1. First case (BraTS)
    # 2. A middle case
    # 3. A TCGA-GBM or LGG case if possible
    queries = [meta.iloc[0]["patient_id"], meta.iloc[len(meta)//2]["patient_id"], meta.iloc[len(meta)-1]["patient_id"]]

    print(f"Generating retrieval panels for: {queries}")
    for q in queries:
        create_retrieval_panel(q, CONFIG)
