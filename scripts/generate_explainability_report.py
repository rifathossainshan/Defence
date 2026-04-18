import torch
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from src.models.multibranch_model import MultiBranchHybridSSLModel
from src.datasets.ssl_dataset import BraTSSSLDataset
from src.evaluation.explainability_gradcam import GradCAM3D, save_explainability_panel
import faiss
from scipy.ndimage import zoom

# Configuration
DATASET_ROOT = "e:/Cse Engineering/11Defense"
MODALITY_COLS = ['t1_path', 't1ce_path', 't2_path', 'flair_path']

def resolve_path(path, dataset):
    """Ensure path is absolute regardless of dataset source."""
    if pd.isna(path): return None
    path = str(path).replace('\\', '/')
    if dataset == 'BraTS2021':
        return os.path.join(DATASET_ROOT, path)
    return path # TCGA / BraTS2024 are absolute or already fixed

def load_mri_volume(modality_path, target_shape=(64, 64, 64)):
    import nibabel as nib
    if modality_path is None or not os.path.exists(modality_path):
        print(f"Warning: Path not found: {modality_path}")
        return np.zeros(target_shape)
        
    img = nib.load(modality_path).get_fdata()
    # Basic normalization
    img = (img - np.mean(img)) / (np.std(img) + 1e-8)
    
    # Resize to target shape
    factors = [t/s for t, s in zip(target_shape, img.shape)]
    img_resized = zoom(img, factors, order=1)
    return img_resized

def run_explainability():
    # 1. Config
    device = torch.device('cpu') # Force CPU for stability in this run
    model_path = "outputs/checkpoints/multibranch_hybrid_best.pth"
    tcga_metadata_path = "data/metadata/metadata_testing_tcga.csv"
    brats_metadata_path = "data/metadata/metadata_brats2021.csv"
    index_path = "outputs/faiss_hybrid/faiss.index"
    index_metadata_path = "outputs/faiss_hybrid/index_metadata.csv"
    output_dir = "outputs/explainability"
    os.makedirs(output_dir, exist_ok=True)

    # 2. Load Model
    print("Loading model...")
    model = MultiBranchHybridSSLModel(output_size=64).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Define target layers for Grad-CAM
    target_layers = {
        'flair': model.branch_flair.encoder[6],
        't1ce': model.branch_t1ce.encoder[6]
    }
    gcam = GradCAM3D(model, target_layers)

    # 3. Load Metadata for lookup
    print("Loading metadata...")
    tcga_df = pd.read_csv(tcga_metadata_path)
    brats_df = pd.read_csv(brats_metadata_path)
    
    # Create a lookup dict: patient_id -> modality_paths
    lookup = {}
    for _, row in tcga_df.iterrows():
        lookup[row['patient_id']] = {m: resolve_path(row[m], row['dataset']) for m in MODALITY_COLS}
    for _, row in brats_df.iterrows():
        lookup[row['patient_id']] = {m: resolve_path(row[m], row['dataset']) for m in MODALITY_COLS}

    # Load FAISS index and its metadata
    index = faiss.read_index(index_path)
    db_metadata = pd.read_csv(index_metadata_path)

    # 4. Select Query Patient (High-grade Query from TCGA)
    q_id = "TCGA-02-0006" # Typical GBM case
    if q_id not in lookup:
        q_id = tcga_df.iloc[0]['patient_id']
        
    print(f"\n--- Running Explainability for Query: {q_id} ---")
    q_paths = lookup[q_id]
    
    # Load Query Modalities
    q_vols = {m: load_mri_volume(q_paths[m]) for m in MODALITY_COLS}
    
    # Prepare Tensor [1, 4, 64, 64, 64] (T1, T1ce, T2, FLAIR)
    input_tensor = np.stack([q_vols['t1_path'], q_vols['t1ce_path'], q_vols['t2_path'], q_vols['flair_path']], axis=0)
    input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float().to(device)

    # Get Index Match
    with torch.no_grad():
        emb, _ = model(input_tensor)
        emb_np = emb.cpu().numpy().astype('float32')
        faiss.normalize_L2(emb_np)
        D, I = index.search(emb_np, k=5) # Look for top matches
        
    # Pick a match that is NOT the query itself (if query is in index)
    match_found = False
    for i in range(len(I[0])):
        m_idx = I[0][i]
        m_id = db_metadata.iloc[m_idx]['patient_id']
        if m_id != q_id:
            match_idx = m_idx
            match_found = True
            break
            
    if not match_found: match_idx = I[0][0]
    
    match_row = db_metadata.iloc[match_idx]
    m_id = match_row['patient_id']
    print(f"Top Match Found: {m_id} (Distance: {D[0][0]:.4f})")

    # Load Match Modalities
    m_paths = lookup[m_id]
    m_vols = {m: load_mri_volume(m_paths[m]) for m in MODALITY_COLS}
    
    m_input_tensor = np.stack([m_vols['t1_path'], m_vols['t1ce_path'], m_vols['t2_path'], m_vols['flair_path']], axis=0)
    m_input_tensor = torch.from_numpy(m_input_tensor).unsqueeze(0).float().to(device)

    # 5. Generate Grad-CAM Heatmaps
    for mod in ['flair', 't1ce']:
        print(f"Generating Grad-CAM for {mod.upper()} branch...")
        q_cam = gcam.generate_heatmap(input_tensor, mod)
        m_cam = gcam.generate_heatmap(m_input_tensor, mod)
        
        # Save Panel
        save_name = f"explainability_report_{q_id}_vs_{m_id}_{mod}.png"
        save_path = os.path.join(output_dir, save_name)
        
        q_vol = q_vols['flair_path'] if mod == 'flair' else q_vols['t1ce_path']
        m_vol = m_vols['flair_path'] if mod == 'flair' else m_vols['t1ce_path']
        
        save_explainability_panel(q_vol, q_cam, m_vol, m_cam, mod, save_path)
        print(f"Panel saved: {save_path}")

    gcam.remove_hooks()
    print("\n[FINISH] Explainability analysis complete.")

if __name__ == "__main__":
    run_explainability()

if __name__ == "__main__":
    run_explainability()
