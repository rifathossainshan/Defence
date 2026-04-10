import os
import torch
import nibabel as nib
import numpy as np
import faiss
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.encoder import MRIEncoder

def preprocess_nifti(paths, target_shape=(128, 128, 128)):
    """
    Standard preprocessing for raw NIfTI files:
    Modality stacking -> Z-score norm -> Crop/Pad to 128^3
    """
    imgs = []
    for p in paths:
        img = nib.load(p).get_fdata().astype(np.float32)
        # Foreground Z-score
        mask = img > 0
        if mask.any():
            mean = img[mask].mean()
            std = img[mask].std()
            img = (img - mean) / (std + 1e-8)
        imgs.append(img)
    
    # Stack [4, H, W, D]
    volume = np.stack(imgs, axis=0)
    
    # Center Crop / Pad to 128x128x128
    # BraTS input usually 240x240x155
    c, h, w, d = volume.shape
    th, tw, td = target_shape
    
    # Simple central crop/pad logic
    def get_slice(size, target):
        if size > target:
            start = (size - target) // 2
            return start, start + target, 0, 0
        else:
            pad_total = target - size
            pad_before = pad_total // 2
            return 0, size, pad_before, target - size - pad_before

    h_s, h_e, h_pb, h_pa = get_slice(h, th)
    w_s, w_e, w_pb, w_pa = get_slice(w, tw)
    d_s, d_e, d_pb, d_pa = get_slice(d, td)
    
    cropped = volume[:, h_s:h_e, w_s:w_e, d_s:d_e]
    # Pad if necessary
    padded = np.pad(cropped, ((0,0), (h_pb, h_pa), (w_pb, w_pa), (d_pb, d_pa)), mode='constant')
    
    return torch.from_numpy(padded).unsqueeze(0) # [1, 4, 128, 128, 128]

def query_live(case_dir, config):
    print(f"Starting Live Query for directory: {case_dir}")
    
    # 1. Map modalities
    files = os.listdir(case_dir)
    mapping = {}
    for f in files:
        if '_t1.nii' in f: mapping['t1'] = os.path.join(case_dir, f)
        if '_t1Gd.nii' in f or '_t1ce.nii' in f: mapping['t1ce'] = os.path.join(case_dir, f)
        if '_t2.nii' in f: mapping['t2'] = os.path.join(case_dir, f)
        if '_flair.nii' in f: mapping['flair'] = os.path.join(case_dir, f)
    
    required = ['t1', 't1ce', 't2', 'flair']
    if not all(k in mapping for k in required):
        print(f"Error: Missing modalities in {case_dir}. Found: {list(mapping.keys())}")
        return

    # 2. Preprocess
    print("Preprocessing MRI volumes...")
    input_tensor = preprocess_nifti([mapping['t1'], mapping['t1ce'], mapping['t2'], mapping['flair']])
    
    # 3. Model Inference
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MRIEncoder(in_channels=4, embedding_dim=128).to(device)
    model.load_state_dict(torch.load(config["model_path"], map_location=device))
    model.eval()
    
    with torch.no_grad():
        # Get 512-dim features (backbone)
        query_emb = model.get_features(input_tensor.to(device))
        query_emb = torch.nn.functional.normalize(query_emb, p=2, dim=1).cpu().numpy()

    # 4. FAISS Search
    print("Searching database...")
    index = faiss.read_index(config["index_path"])
    index_meta = pd.read_csv(config["index_metadata"])
    top_k = config.get("top_k", 5)
    
    sims, ids = index.search(query_emb, top_k)
    sims, ids = sims[0], ids[0]

    # 5. Results
    results = []
    for i, idx in enumerate(ids):
        res = index_meta.iloc[idx]
        results.append({
            "retrieved_id": res["patient_id"],
            "dataset": res["dataset"],
            "score": f"{sims[i]:.4f}"
        })
    
    print("\n[MATCHES FOUND]")
    print(pd.DataFrame(results).to_string(index=False))

    # 6. Optional: Qualitative Panel for this query
    # (Reusing visualization logic from previous turn)
    print("\nGenerating visualization panel...")
    # NOTE: Since the query case IS a raw folder, we can't easily use create_retrieval_panel
    # without updating it to handle raw paths. We will just print for now.
    
if __name__ == "__main__":
    CONFIG = {
        "model_path": "outputs/checkpoints/best_model.pth",
        "index_path": "outputs/faiss/faiss.index",
        "index_metadata": "outputs/faiss/index_metadata.csv"
    }
    
    # Path provided by user for TCGA-02-0009
    query_case = r"E:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-GBM\BraTS-TCGA-GBM\Pre-operative_TCGA_GBM_NIfTI_and_Segmentations\TCGA-02-0009"
    query_live(query_case, CONFIG)
