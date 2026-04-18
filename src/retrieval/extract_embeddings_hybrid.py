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

def extract_hybrid_full(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = config.get("output_size", 64)
    print(f"--- HYBRID SSL EMBEDDING EXTRACTION ---")
    print(f"Using device: {device} | Resolution: {output_size}^3")

    # 1. Load Model
    model = MultiBranchHybridSSLModel(embedding_dim=128, output_size=output_size).to(device)
    if os.path.exists(config["model_path"]):
        print(f"Loading weights from {config['model_path']}")
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
    else:
        print(f"Warning: Model weights not found at {config['model_path']}. Using random weights for pipeline test.")
    
    model.eval()

    # 2. Dataset & Loader
    dataset = BraTSSSLDataset(
        csv_file=config["csv_path"],
        base_dir=config["base_dir"],
        split="all",
        crop_mode="whole", 
        use_seg=False,
        output_size=output_size
    )
    
    if config.get("sample_size"):
        indices = list(range(min(config["sample_size"], len(dataset))))
        dataset = torch.utils.data.Subset(dataset, indices)
    
    loader = DataLoader(dataset, batch_size=config.get("batch_size", 4), shuffle=False)

    # 3. Extraction Loop
    embeddings = []
    metadata = []
    recons = [] 

    print(f"Extracting for {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch["view1"].to(device) 
            p_ids = batch["id"]
            datasets = batch["dataset"]
            
            z, recon = model(x)
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            embeddings.append(z.cpu().numpy())
            
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
            
            sys.stdout.flush()

    # 4. Save Results
    os.makedirs(os.path.dirname(config["output_npy"]), exist_ok=True)
    
    embeddings_matrix = np.concatenate(embeddings, axis=0)
    np.save(config["output_npy"], embeddings_matrix)
    
    meta_df = pd.DataFrame(metadata)
    meta_df["embedding_idx"] = meta_df.index
    meta_df.to_csv(config["output_csv"], index=False)
    
    if recons:
        np.save(config["output_recon"], recons[0])

    print(f"\n[SUCCESS] Extraction Complete!")
    print(f"Total Samples: {len(meta_df)}")
    print(f"Saved: {config['output_npy']}")
    sys.stdout.flush()

if __name__ == "__main__":
    CONFIG = {
        "model_path": "outputs/checkpoints/multibranch_hybrid_best.pth",
        "csv_path": "data/metadata/metadata_brats2021.csv",
        "base_dir": ".",
        "batch_size": 4,
        "output_size": 64,
        "sample_size": None, # Set to None for full dataset
        "output_npy": "outputs/embeddings/hybrid_embeddings.npy",
        "output_csv": "outputs/embeddings/hybrid_metadata.csv",
        "output_recon": "outputs/embeddings/hybrid_recon_sample.npy"
    }
    extract_hybrid_full(CONFIG)

