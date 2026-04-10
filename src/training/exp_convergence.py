import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import faiss
import umap

# Add src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(root_dir)
sys.path.append(src_dir)

from datasets.ssl_dataset import BraTSSSLDataset
from models.encoder import MRIEncoder # Early Fusion
from models.multibranch_model import MultiBranchHybridSSLModel # Multi-Branch
from losses.hybrid_loss import HybridSSLLoss
from preprocessing.simple_transforms import SimpleSSLTransform

class FastHybridDataset(torch.utils.data.Dataset):
    """
    Caches only raw images to save RAM, applies transforms on the fly.
    """
    def __init__(self, raw_images, patient_ids, datasets, transform):
        self.raw_images = raw_images
        self.patient_ids = patient_ids
        self.datasets = datasets
        self.transform = transform
            
    def __len__(self):
        return len(self.raw_images)
        
    def __getitem__(self, idx):
        img = self.raw_images[idx] # [4, 128, 128, 128]
        
        # SSL Needs two views
        v1 = self.transform(img.copy())
        v2 = self.transform(img.copy())
        
        # Convert to tensor [C, D, H, W]
        # img is [D, H, W, C] where 0=D, 1=H, 2=W, 3=C
        # We need [C, D, H, W] which is (3, 0, 1, 2)
        v1 = torch.tensor(v1, dtype=torch.float32).permute(3, 0, 1, 2)
        v2 = torch.tensor(v2, dtype=torch.float32).permute(3, 0, 1, 2)
        
        return {
            "view1": v1,
            "view2": v2,
            "id": self.patient_ids[idx],
            "dataset": self.datasets[idx]
        }

def run_experiment(exp_id, model_type, lambda_recon, config, fast_dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> RUNNING EXPERIMENT {exp_id}: {model_type} (lambda_recon={lambda_recon})")
    
    loader = DataLoader(fast_dataset, batch_size=config["batch_size"], shuffle=True)

    # 2. Model Selection
    if model_type == "EarlyFusion":
        model = MRIEncoder(in_channels=4, embedding_dim=128).to(device)
    else:
        model = MultiBranchHybridSSLModel(embedding_dim=128).to(device)
    
    criterion = HybridSSLLoss(lambda_recon=lambda_recon)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # 3. Training Loop
    model.train()
    for epoch in range(config["epochs"]):
        total_l = 0
        pbar = tqdm(loader, desc=f"Exp {exp_id} Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            x1, x2 = batch["view1"].to(device), batch["view2"].to(device)
            optimizer.zero_grad()
            
            if model_type == "EarlyFusion":
                z1, z2 = model(x1), model(x2)
                recon1, recon2 = x1, x2 # dummy
            else:
                z1, recon1 = model(x1)
                z2, recon2 = model(x2)
            
            loss, _, _ = criterion(z1, z2, recon1, recon2, x1, x2)
            loss.backward()
            optimizer.step()
            total_l += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # 4. Evaluation
    model.eval()
    embeddings = []
    with torch.no_grad():
        eval_loader = DataLoader(fast_dataset, batch_size=config["batch_size"], shuffle=False)
        for batch in eval_loader:
            x = batch["view1"].to(device)
            if model_type == "EarlyFusion":
                z = model(x)
            else:
                z, _ = model(x)
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            embeddings.append(z.cpu().numpy())
    
    emb = np.concatenate(embeddings, axis=0)
    
    # Metrics
    variances = np.var(emb, axis=0)
    avg_var = np.mean(variances)
    index = faiss.IndexFlatIP(128)
    faiss.normalize_L2(emb)
    index.add(emb)
    sims, _ = index.search(emb, 6)
    avg_sim = np.mean(sims[:, 1:])
    
    report = f"EXPERIMENT {exp_id} SUMMARY\nModel: {model_type}\nLambda Recon: {lambda_recon}\nAvg Top-5 Sim: {avg_sim:.6f}\nAvg Var: {avg_var:.8e}\n"
    os.makedirs("outputs/convergence", exist_ok=True)
    with open(f"outputs/convergence/exp{exp_id}_summary.txt", "w") as f:
        f.write(report)
    print(report)

    # UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(emb)
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_2d[:, 0], umap_2d[:, 1], alpha=0.7)
    plt.title(f"Exp {exp_id} UMAP")
    plt.savefig(f"outputs/convergence/exp{exp_id}_umap.png")
    plt.close()

if __name__ == "__main__":
    COMMON_CONFIG = {"csv_path": "data/metadata/metadata_brats2021.csv", "base_dir": ".", "subset_path": "data/metadata/fixed_50_subset.npy", "epochs": 10, "batch_size": 4, "lr": 1e-4}
    
    # Pre-load only RAW images into memory
    print("Pre-loading raw images into RAM (50 samples)...")
    base_dataset = BraTSSSLDataset(csv_file=COMMON_CONFIG["csv_path"], base_dir=COMMON_CONFIG["base_dir"], apply_crop=True, apply_normalize=True, transform=None)
    indices = np.load(COMMON_CONFIG["subset_path"])
    
    raw_images = []
    p_ids = []
    datasets = []
    
    for idx in tqdm(indices, desc="Pre-loading"):
        # We need a way to get the processed image without transform
        # Our dataset by default does normalize/crop
        sample = base_dataset[idx] # This returns dict with 'view1' (normalized image)
        raw_images.append(sample["view1"].permute(1, 2, 3, 0).numpy()) # [H, W, D, C] to [C, H, W, D] in dataset logic back and forth
        p_ids.append(sample["id"])
        datasets.append(sample["dataset"])
        
    fast_ds = FastHybridDataset(raw_images, p_ids, datasets, SimpleSSLTransform())
    # Exp 3: Multi-Branch Hybrid
    try:
        run_experiment(3, "MultiBranch", 0.01, COMMON_CONFIG, fast_ds)
    except Exception as e:
        print(f"\n[FATAL ERROR] Experiment 3 failed: {e}")
        with open("outputs/convergence/exp3_error.txt", "w") as f:
            f.write(str(e))
    
    run_experiment(1, "EarlyFusion", 0.0, COMMON_CONFIG, fast_ds)
