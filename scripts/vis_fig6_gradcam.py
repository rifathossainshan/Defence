import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
import faiss
from src.models.multibranch_model import MultiBranchHybridSSLModel
from src.evaluation.explainability_gradcam import GradCAM3D
from scipy.ndimage import zoom

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

def load_mri_volume(modality_path, target_shape=(64, 64, 64)):
    if modality_path is None or not os.path.exists(modality_path):
        return np.zeros(target_shape)
    img = nib.load(modality_path).get_fdata()
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    factors = [t/s for t, s in zip(target_shape, img.shape)]
    return zoom(img, factors, order=1)

def visualize_gradcam():
    set_academic_style()
    device = torch.device('cpu') # Use CPU for visualization stability
    
    # 1. Load Data/Model
    model_path = "outputs/checkpoints/multibranch_hybrid_best.pth"
    model = MultiBranchHybridSSLModel(output_size=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    target_layers = {
        'flair': model.branch_flair.encoder[6],
        't1ce': model.branch_t1ce.encoder[6]
    }
    gcam = GradCAM3D(model, target_layers)

    tcga_meta = pd.read_csv("data/metadata/metadata_testing_tcga.csv")
    brats_meta = pd.read_csv("data/metadata/metadata_brats2021.csv")
    db_meta = pd.read_csv("outputs/faiss_hybrid/index_metadata.csv")
    index = faiss.read_index("outputs/faiss_hybrid/faiss.index")

    # 2. Select Query (TCGA-02-0006)
    q_id = "TCGA-02-0006"
    q_row = tcga_meta[tcga_meta['patient_id'] == q_id].iloc[0]
    q_vols = {m: load_mri_volume(q_row[f"{m}_path"]) for m in ['t1', 't1ce', 't2', 'flair']}
    
    input_tensor = np.stack([q_vols['t1'], q_vols['t1ce'], q_vols['t2'], q_vols['flair']], axis=0)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)

    # 3. Find Match
    with torch.no_grad():
        emb, _ = model(input_tensor)
        emb_np = emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(emb_np)
        D, I = index.search(emb_np, k=5)
    
    m_idx = I[0][1] # Top-1 match (skip self)
    m_id = db_meta.iloc[m_idx]['patient_id']
    m_dataset = db_meta.iloc[m_idx]['dataset']
    
    if m_dataset == 'BraTS2021':
        m_row = brats_meta[brats_meta['patient_id'] == m_id].iloc[0]
    else:
        m_row = tcga_meta[tcga_meta['patient_id'] == m_id].iloc[0]
        
    m_vols = {mod: load_mri_volume(resolve_path(m_row[f"{mod}_path"], m_dataset)) for mod in ['t1', 't1ce', 't2', 'flair']}
    m_input_tensor = np.stack([m_vols['t1'], m_vols['t1ce'], m_vols['t2'], m_vols['flair']], axis=0)
    m_input_tensor = torch.from_numpy(m_input_tensor).unsqueeze(0).float().to(device)

    # 4. Generate Heatmaps
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    modalities = ['flair', 't1ce']
    for row_idx, mod in enumerate(modalities):
        # Query CAM
        q_cam = gcam.generate_heatmap(input_tensor, mod)
        # Match CAM
        m_cam = gcam.generate_heatmap(m_input_tensor, mod)
        
        z = 32 # Center slice for 64x64x64
        
        # Plot Row: Query MRI | Query CAM | Match MRI | Match CAM
        q_img = np.rot90(q_vols[mod][:, :, z])
        q_hmp = np.rot90(q_cam[:, :, z])
        m_img = np.rot90(m_vols[mod][:, :, z])
        m_hmp = np.rot90(m_cam[:, :, z])
        
        axes[row_idx, 0].imshow(q_img, cmap='bone')
        axes[row_idx, 0].set_ylabel(mod.upper(), fontsize=18, fontweight='bold')
        axes[row_idx, 0].set_title(f"Query ({q_id})")
        
        axes[row_idx, 1].imshow(q_img, cmap='bone')
        axes[row_idx, 1].imshow(q_hmp, cmap='jet', alpha=0.5)
        axes[row_idx, 1].set_title(f"Query Grad-CAM")
        
        axes[row_idx, 2].imshow(m_img, cmap='bone')
        axes[row_idx, 2].set_title(f"Match ({m_id})")
        
        axes[row_idx, 3].imshow(m_img, cmap='bone')
        axes[row_idx, 3].imshow(m_hmp, cmap='jet', alpha=0.5)
        axes[row_idx, 3].set_title(f"Match Grad-CAM")

        for c in range(4): axes[row_idx, c].axis('off')

    plt.suptitle("Figure 6. Grad-CAM Explainability (FLAIR vs T1ce Branches)", fontsize=22, y=1.02)
    plt.tight_layout()
    save_fig(fig, "fig6_gradcam.png")
    plt.close()
    gcam.remove_hooks()

if __name__ == "__main__":
    visualize_gradcam()
