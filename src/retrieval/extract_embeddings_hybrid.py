import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
from pathlib import Path

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.ssl_dataset import BraTSSSLDataset
from models.multibranch_model import MultiBranchHybridSSLModel

def extract_hybrid_mini(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- MINI EVALUATION EXTRACTION (50 Samples) ---")
    print(f"Using device: {device}")

    # 1. Load Model
    model = MultiBranchHybridSSLModel(embedding_dim=128).to(device)
    if os.path.exists(config["model_path"]):
        print(f"Loading weights from {config['model_path']}")
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
    else:
        print(f"Error: Model weights not found at {config['model_path']}")
        return
    
    model.eval()

    # 2. Dataset & Loader (50 samples only)
    dataset = BraTSSSLDataset(
        csv_file=config["csv_path"],
        base_dir=config["base_dir"],
        split="all",
        crop_mode="whole", 
        use_seg=False
    )
    
    indices = list(range(config["sample_size"]))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=2, shuffle=False)

    # 3. Extraction Loop
    embeddings = []
    metadata = []
    recons = [] # Sample reconstructions for sanity check

    print(f"Extracting for {len(subset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["view1"].to(device) # Using only one view for retrieval
            p_ids = batch["id"]
            datasets = batch["dataset"]
            
            # Forward pass: z is 128-dim, recon is 4-modality volume
            z, recon = model(x)
            
            # Use z for retrieval
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            embeddings.append(z.cpu().numpy())
            
            # Store first recon of first batch for sanity check
            if len(recons) == 0:
                recons.append({
                    "id": p_ids[0],
                    "original": x[0].cpu().numpy(),
                    "reconstructed": recon[0].cpu().numpy()
                })

            for i in range(len(p_ids)):
                metadata.append({
                    "patient_id": p_ids[i],
                    "dataset": datasets[i]
                })

    # 4. Save Results
    os.makedirs(os.path.dirname(config["output_npy"]), exist_ok=True)
    
    embeddings_matrix = np.concatenate(embeddings, axis=0) # [50, 128]
    np.save(config["output_npy"], embeddings_matrix)
    
    meta_df = pd.DataFrame(metadata)
    meta_df["embedding_idx"] = meta_df.index
    meta_df.to_csv(config["output_csv"], index=False)
    
    # Save a small recon artifact for visual inspection
    np.save(config["output_recon"], recons[0])

    print(f"\n[SUCCESS] Mini-Extraction Complete!")
    print(f"Saved: {config['output_npy']}")
    print(f"Saved Recon Sample: {config['output_recon']}")

if __name__ == "__main__":
    CONFIG = {
        "model_path": "outputs/checkpoints/multibranch_hybrid_best.pth",
        "csv_path": "data/metadata/metadata_brats2021.csv",
        "base_dir": ".",
        "sample_size": 50,
        "output_npy": "outputs/embeddings/minieval_embeddings.npy",
        "output_csv": "outputs/embeddings/minieval_metadata.csv",
        "output_recon": "outputs/embeddings/minieval_recon_sample.npy"
    }
    extract_hybrid_mini(CONFIG)
