import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import faiss

# Add current directory to path for imports
sys.path.append('.')
from scripts.vis_utils import set_academic_style, save_fig

DATASET_ROOT = "e:/Cse Engineering/11Defense"

def resolve_path(path, dataset):
    if pd.isna(path): return None
    path = str(path).replace('\\', '/')
    if dataset == 'BraTS2021':
        return os.path.join(DATASET_ROOT, path)
    return path

def load_slice(path):
    if path is None or not os.path.exists(path):
        return np.zeros((64, 64))
    img = nib.load(path).get_fdata()
    # Get center axial slice
    z = img.shape[2] // 2
    return np.rot90(img[:, :, z])

def visualize_retrieval():
    set_academic_style()
    
    # 1. Load Data
    index_path = "outputs/faiss_hybrid/faiss.index"
    index_metadata_path = "outputs/faiss_hybrid/index_metadata.csv"
    embeddings_path = "outputs/embeddings/tcga_embeddings.npy"
    tcga_metadata_path = "data/metadata/metadata_testing_tcga.csv"
    brats_metadata_path = "data/metadata/metadata_brats2021.csv"

    index = faiss.read_index(index_path)
    db_meta = pd.read_csv(index_metadata_path)
    tcga_meta = pd.read_csv(tcga_metadata_path)
    brats_meta = pd.read_csv(brats_metadata_path)
    tcga_embeddings = np.load(embeddings_path)

    # 2. Select Query (TCGA-02-0006)
    q_id = "TCGA-02-0006"
    q_idx = tcga_meta[tcga_meta['patient_id'] == q_id].index[0]
    q_emb = tcga_embeddings[q_idx : q_idx+1].astype('float32')
    faiss.normalize_L2(q_emb)
    
    # 3. Search
    D, I = index.search(q_emb, k=4) # Query + Top 3
    
    # 4. Plotting
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    # Query Plot
    q_row = tcga_meta.iloc[q_idx]
    q_img = load_slice(q_row['flair_path'])
    axes[0].imshow(q_img, cmap='bone')
    axes[0].set_title(f"QUERY\n({q_id})", color='cyan', fontweight='bold')
    axes[0].axis('off')
    
    # Match Plots
    for i in range(1, 4):
        m_idx = I[0][i]
        m_dist = D[0][i]
        m_id = db_meta.iloc[m_idx]['patient_id']
        m_dataset = db_meta.iloc[m_idx]['dataset']
        
        # Find path in respective metadata
        if m_dataset == 'BraTS2021':
            m_row = brats_meta[brats_meta['patient_id'] == m_id].iloc[0]
        else:
            m_row = tcga_meta[tcga_meta['patient_id'] == m_id].iloc[0]
            
        m_path = resolve_path(m_row['flair_path'], m_dataset)
        m_img = load_slice(m_path)
        
        axes[i].imshow(m_img, cmap='bone')
        axes[i].set_title(f"MATCH {i}\nSim: {1-m_dist:.3f}\n({m_id})")
        axes[i].axis('off')

    plt.suptitle("Figure 3. Zero-Shot Similar-Case Retrieval Panel", fontsize=20, y=1.05)
    plt.tight_layout()
    save_fig(fig, "fig3_retrieval.png")
    plt.close()

if __name__ == "__main__":
    visualize_retrieval()
