import os
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from models.encoder import MRIEncoder 
from models.multibranch_model import MultiBranchHybridSSLModel, ReconstructionHead
from losses.hybrid_loss import HybridSSLLoss
from preprocessing.simple_transforms import SimpleSSLTransform

class EarlyFusionHybridModel(nn.Module):
    """
    Exp 2 Model: Standard Early Fusion Encoder + Reconstruction Decoder.
    Flexible for variable resolutions.
    """
    def __init__(self, embedding_dim=128, output_size=64):
        super(EarlyFusionHybridModel, self).__init__()
        self.encoder = MRIEncoder(in_channels=4, embedding_dim=embedding_dim)
        # Latent dim of MRIEncoder is 512 before the final FC usually, 
        # but our MRIEncoder.forward returns embedding_dim (128).
        # We'll use 128 as the latent for the decoder for this experiment.
        self.reconstruction_head = ReconstructionHead(latent_dim=embedding_dim, out_channels=4, output_size=output_size)

    def forward(self, x):
        z = self.encoder(x) # [B, 128]
        recon = self.reconstruction_head(z)
        return z, recon

class Fast64Dataset(torch.utils.data.Dataset):
    """
    Caches 64x64x64 raw images in memory for fast CPU training.
    """
    def __init__(self, raw_images_64, patient_ids, datasets, transform):
        self.raw_images = raw_images_64
        self.patient_ids = patient_ids
        self.datasets = datasets
        self.transform = transform
            
    def __len__(self):
        return len(self.raw_images)
        
    def __getitem__(self, idx):
        img = self.raw_images[idx] # [4, 64, 64, 64]
        
        # Apply SSL Transfroms
        # Transform expects [H, W, D, C] usually but we can handle [C, D, H, W]
        # Our SimpleSSLTransform handles [C, H, W, D] which is equivalent here
        v1 = self.transform(img.copy())
        v2 = self.transform(img.copy())
        
        v1 = torch.tensor(v1, dtype=torch.float32)
        v2 = torch.tensor(v2, dtype=torch.float32)
        
        return {
            "view1": v1,
            "view2": v2,
            "id": self.patient_ids[idx],
            "dataset": self.datasets[idx]
        }

def run_experiment(exp_id, model_name, lambda_recon, config, dataset):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n>>> RUNNING EXPERIMENT {exp_id}: {model_name} (Res: 64^3, Lambda={lambda_recon})")
    
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True)

    # Model Selection
    if model_name == "EarlyFusion-SimCLR":
        model = MRIEncoder(in_channels=4, embedding_dim=128).to(device)
    elif model_name == "EarlyFusion-Hybrid":
        model = EarlyFusionHybridModel(embedding_dim=128, output_size=64).to(device)
    else:
        model = MultiBranchHybridSSLModel(embedding_dim=128, output_size=64).to(device)
    
    criterion = HybridSSLLoss(lambda_recon=lambda_recon)
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # Training
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

    # Eval
    model.eval()
    embeddings = []
    with torch.no_grad():
        eval_loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False)
        for batch in eval_loader:
            x = batch["view1"].to(device)
            if model_name == "EarlyFusion-SimCLR":
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
    
    report = f"EXPERIMENT {exp_id} ({model_name}) SUMMARY\n"
    report += f"Resolution: 64x64x64 (Debugging/Collapse Recovery)\n"
    report += f"Avg Top-5 Sim: {avg_sim:.6f}\nAvg Var: {avg_var:.8e}\nFinal Loss: {total_l/len(loader):.4f}\n"
    
    os.makedirs("outputs/convergence", exist_ok=True)
    with open(f"outputs/convergence/exp{exp_id}_summary.txt", "w") as f:
        f.write(report)
    print(report)

    # UMAP
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.1, n_components=2, metric='cosine', random_state=42)
    umap_2d = reducer.fit_transform(emb)
    plt.figure(figsize=(8, 6))
    plt.scatter(umap_2d[:, 0], umap_2d[:, 1], alpha=0.7)
    plt.title(f"Exp {exp_id} 64^3 UMAP ({model_name})")
    plt.savefig(f"outputs/convergence/exp{exp_id}_umap.png")
    plt.close()

if __name__ == "__main__":
    CONFIG = {"csv_path": "data/metadata/metadata_brats2021.csv", "base_dir": ".", "subset_path": "data/metadata/fixed_50_subset.npy", "epochs": 10, "batch_size": 4, "lr": 1e-4}
    
    print("Pre-loading and Downsampling to 64x64x64 (50 samples)...")
    base_dataset = BraTSSSLDataset(csv_file=CONFIG["csv_path"], base_dir=CONFIG["base_dir"], transform=None)
    indices = np.load(CONFIG["subset_path"])
    
    raw_64 = []
    p_ids, datasets = [], []
    
    for idx in tqdm(indices, desc="Caching 64^3"):
        sample = base_dataset[idx] # [4, 128, 128, 128]
        img_128 = sample["view1"].unsqueeze(0) # [1, 4, 128, 128, 128]
        img_64 = F.interpolate(img_128, size=(64, 64, 64), mode='trilinear', align_corners=False)
        raw_64.append(img_64.squeeze(0).numpy())
        p_ids.append(sample["id"])
        datasets.append(sample["dataset"])
        
    fast_ds = Fast64Dataset(raw_64, p_ids, datasets, SimpleSSLTransform())
    
    # Run the 3 Experiment Sequence
    run_experiment(1, "EarlyFusion-SimCLR", 0.0, CONFIG, fast_ds)
    run_experiment(2, "EarlyFusion-Hybrid", 0.01, CONFIG, fast_ds)
    
    CONFIG["batch_size"] = 2 # Lower batch for heavy multi-branch
    run_experiment(3, "MultiBranch-Hybrid", 0.01, CONFIG, fast_ds)
