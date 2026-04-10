import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.ssl_dataset import BraTSSSLDataset
from models.encoder import MRIEncoder

def extract_embeddings(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 1. Load Model
    model = MRIEncoder(in_channels=4, embedding_dim=128).to(device)
    if os.path.exists(config["model_path"]):
        print(f"Loading weights from {config['model_path']}")
        model.load_state_dict(torch.load(config["model_path"], map_location=device))
    else:
        print(f"Warning: Model weights not found at {config['model_path']}. Using random initialization.")
    
    model.eval()

    # 2. Dataset & Loader (No augmentation for extraction)
    dataset = BraTSSSLDataset(
        csv_file=config["csv_path"],
        base_dir=config["base_dir"],
        split="all",
        use_seg=False, # Disable seg for extraction to avoid collation errors
        transform=None # Crucial: no augmentations for database indexing
    )
    
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False, num_workers=0)

    # 3. Extraction Loop
    embeddings = []
    metadata = []

    print(f"Extracting embeddings for {len(dataset)} samples...")
    with torch.no_grad():
        for batch in tqdm(loader):
            # We use view1 as the representative image (since transform=None, view1=view2=original)
            x = batch["view1"].to(device)
            p_ids = batch["id"]
            datasets = batch["dataset"]
            
            # Extract backbone features (512-dim)
            features = model.get_features(x)
            
            # L2 Normalize
            features = torch.nn.functional.normalize(features, p=2, dim=1)
            
            embeddings.append(features.cpu().numpy())
            
            for i in range(len(p_ids)):
                metadata.append({
                    "patient_id": p_ids[i],
                    "dataset": datasets[i]
                })

    # 4. Save Results
    embeddings_matrix = np.concatenate(embeddings, axis=0) # [N, 512]
    
    os.makedirs(os.path.dirname(config["output_npy"]), exist_ok=True)
    np.save(config["output_npy"], embeddings_matrix)
    
    meta_df = pd.DataFrame(metadata)
    meta_df["embedding_idx"] = meta_df.index
    meta_df.to_csv(config["output_csv"], index=False)

    print(f"\nExtraction Complete!")
    print(f"Embeddings Matrix Shape: {embeddings_matrix.shape}")
    print(f"Saved Matrix to: {config['output_npy']}")
    print(f"Saved Metadata to: {config['output_csv']}")

if __name__ == "__main__":
    CONFIG = {
        "model_path": "outputs/checkpoints/best_model.pth",
        "csv_path": "data/metadata/metadata_brats2021.csv",
        "base_dir": ".",
        "batch_size": 2,
        "output_npy": "outputs/embeddings/embeddings.npy",
        "output_csv": "outputs/embeddings/embedding_metadata.csv"
    }
    extract_embeddings(CONFIG)
