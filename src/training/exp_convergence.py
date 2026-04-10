import os
import sys

# Immediate Environment Fix
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Step 0: Environment patched. Loading core libraries..."); sys.stdout.flush()

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import pandas as pd
from tqdm import tqdm

# Add src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(root_dir)
sys.path.append(src_dir)

print("Step 1: Paths added. Importing project modules..."); sys.stdout.flush()

from datasets.ssl_dataset import BraTSSSLDataset
from models.encoder import MRIEncoder 
from models.multibranch_model import MultiBranchHybridSSLModel, ReconstructionHead
from losses.hybrid_loss import HybridSSLLoss
from preprocessing.simple_transforms import SimpleSSLTransform

class EarlyFusionHybridModel(nn.Module):
    def __init__(self, embedding_dim=128, output_size=64):
        super(EarlyFusionHybridModel, self).__init__()
        self.encoder = MRIEncoder(in_channels=4, embedding_dim=embedding_dim)
        self.reconstruction_head = ReconstructionHead(latent_dim=embedding_dim, out_channels=4, output_size=output_size)

    def forward(self, x):
        z = self.encoder(x) 
        recon = self.reconstruction_head(z)
        return z, recon

class Fast64Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_images_64, patient_ids, datasets, transform):
        self.raw_images = raw_images_64
        self.patient_ids = patient_ids
        self.datasets = datasets
        self.transform = transform
    def __len__(self): return len(self.raw_images)
    def __getitem__(self, idx):
        img = self.raw_images[idx]
        v1 = self.transform(img.copy())
        v2 = self.transform(img.copy())
        return {"view1": torch.tensor(v1, dtype=torch.float32), "view2": torch.tensor(v2, dtype=torch.float32), "id": self.patient_ids[idx], "dataset": self.datasets[idx]}

def run_experiment(exp_id, model_name, lambda_recon, config, dataset):
    import faiss
    import umap
    import matplotlib.pyplot as plt
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> RUNNING EXPERIMENT {exp_id}: {model_name} (Res: 64^3, Lambda={lambda_recon})"); sys.stdout.flush()
    
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    if model_name == "EarlyFusion-SimCLR":
        model = MRIEncoder(in_channels=4, embedding_dim=128).to(device)
    elif model_name == "EarlyFusion-Hybrid":
        model = EarlyFusionHybridModel(embedding_dim=128, output_size=64).to(device)
    else:
        model = MultiBranchHybridSSLModel(embedding_dim=128, output_size=64).to(device)
    
    criterion = HybridSSLLoss(lambda_recon=lambda_recon)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    model.train()
    for epoch in range(config["epochs"]):
        total_l = 0
        pbar = tqdm(loader, desc=f"Exp {exp_id} Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            x1, x2 = batch["view1"].to(device), batch["view2"].to(device)
            optimizer.zero_grad()
            if model_name == "EarlyFusion-SimCLR":
                z1, z2 = model(x1), model(x2)
                recon1, recon2 = x1, x2
            else:
                z1, recon1 = model(x1)
                z2, recon2 = model(x2)
            loss, _, _ = criterion(z1, z2, recon1, recon2, x1, x2)
            loss.backward()
            optimizer.step()
            total_l += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    model.eval()
    embeddings = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        for batch in eval_loader:
            x = batch["view1"].to(device)
            z = model(x) if model_name == "EarlyFusion-SimCLR" else model(x)[0]
            z = torch.nn.functional.normalize(z, p=2, dim=1)
            embeddings.append(z.cpu().numpy())
    
    emb = np.concatenate(embeddings, axis=0)
    variances = np.var(emb, axis=0)
    avg_var = np.mean(variances)
    index = faiss.IndexFlatIP(128)
    faiss.normalize_L2(emb)
    index.add(emb)
    sims, _ = index.search(emb, 6)
    avg_sim = np.mean(sims[:, 1:])
    
    report = f"EXPERIMENT {exp_id} ({model_name}) SUMMARY\nRes: 64^3 (Debugging)\nAvg Top-5 Sim: {avg_sim:.6f}\nAvg Var: {avg_var:.8e}\n"
    os.makedirs("outputs/convergence", exist_ok=True)
    with open(f"outputs/convergence/exp{exp_id}_summary.txt", "w") as f: f.write(report)
    print(report); sys.stdout.flush()

    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(emb)
    plt.figure(figsize=(8, 6)); plt.scatter(umap_2d[:, 0], umap_2d[:, 1], alpha=0.7)
    plt.title(f"Exp {exp_id} 64^3 UMAP ({model_name})")
    plt.savefig(f"outputs/convergence/exp{exp_id}_umap.png"); plt.close()

if __name__ == "__main__":
    print("Step 2: Starting Caching Process..."); sys.stdout.flush()
    CONFIG = {"csv_path": "data/metadata/metadata_brats2021.csv", "base_dir": ".", "subset_path": "data/metadata/fixed_50_subset.npy", "epochs": 10, "batch_size": 4, "lr": 1e-4}
    
    base_dataset = BraTSSSLDataset(csv_file=CONFIG["csv_path"], base_dir=CONFIG["base_dir"], transform=None)
    indices = np.load(CONFIG["subset_path"])
    raw_64 = []
    p_ids, datasets = [], []
    
    for idx in tqdm(indices, desc="Caching 64^3"):
        sample = base_dataset[idx]
        img_64 = sample["view1"].numpy()[:, ::2, ::2, ::2]
        raw_64.append(img_64)
        p_ids.append(sample["id"]); datasets.append(sample["dataset"])
        
    fast_ds = Fast64Dataset(raw_64, p_ids, datasets, SimpleSSLTransform())
    print("Step 3: Caching Complete. Sequence Start."); sys.stdout.flush()
    
    run_experiment(1, "EarlyFusion-SimCLR", 0.0, CONFIG, fast_ds)
    run_experiment(2, "EarlyFusion-Hybrid", 0.01, CONFIG, fast_ds)
    CONFIG["batch_size"] = 2
    run_experiment(3, "MultiBranch-Hybrid", 0.01, CONFIG, fast_ds)
