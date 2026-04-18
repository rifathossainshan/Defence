import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.ssl_dataset import BraTSSSLDataset
from models.multibranch_model import MultiBranchHybridSSLModel
from losses.hybrid_loss import HybridSSLLoss

def train_hybrid(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_size = config.get("output_size", 64)
    print(f"--- STARTING HYBRID SSL TRAINING (Phase 14) ---")
    print(f"Device: {device} | Batch Size: {config['batch_size']} | Epochs: {config['epochs']} | Resolution: {output_size}^3")

    # 1. Dataset & Loader
    dataset = BraTSSSLDataset(
        csv_file=config["csv_path"],
        base_dir=config["base_dir"],
        split="all",
        crop_mode="whole", 
        use_seg=False,
        output_size=output_size
    )
    
    if config.get("subset_size"):
        indices = list(range(config["subset_size"]))
        dataset = torch.utils.data.Subset(dataset, indices)
        print(f"Using subset of size {len(dataset)}")

    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, num_workers=0)

    # 2. Model & Loss
    model = MultiBranchHybridSSLModel(embedding_dim=128, output_size=output_size).to(device)
    criterion = HybridSSLLoss(temperature=config["temperature"], lambda_recon=config["lambda_recon"])
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])

    # 3. Training Loop
    os.makedirs(os.path.dirname(config["checkpoint_path"]), exist_ok=True)
    os.makedirs("outputs/logs", exist_ok=True)
    
    log_path = "outputs/logs/hybrid_train_log.txt"
    print(f"Logging to {log_path}")

    for epoch in range(config["epochs"]):
        model.train()
        total_epoch_loss = 0
        total_simclr_l = 0
        total_recon_l = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{config['epochs']}")
        for batch in pbar:
            x1 = batch["view1"].to(device)
            x2 = batch["view2"].to(device)
            
            optimizer.zero_grad()
            z1, recon1 = model(x1)
            z2, recon2 = model(x2)
            
            loss, l_sim, l_rec = criterion(z1, z2, recon1, recon2, x1, x2)
            loss.backward()
            optimizer.step()
            
            total_epoch_loss += loss.item()
            total_simclr_l += l_sim.item()
            total_recon_l += l_rec.item()
            
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "sim": f"{l_sim.item():.4f}", "rec": f"{l_rec.item():.4f}"})

        # Logging and Saving
        avg_loss = total_epoch_loss / len(loader)
        msg = f"Epoch [{epoch+1}/{config['epochs']}] | Avg Loss: {avg_loss:.4f} | Sim: {total_simclr_l/len(loader):.4f} | Rec: {total_recon_l/len(loader):.4f}\n"
        print(msg)
        
        with open(log_path, "a") as f:
            f.write(msg)
            f.flush()
        
        torch.save(model.state_dict(), config["checkpoint_path"])
        # Flush stdout for better visibility in some environments
        sys.stdout.flush()

    print("\n[SUCCESS] Hybrid SSL Training Finalized!")

if __name__ == "__main__":
    CONFIG = {
        "csv_path": "data/metadata/metadata_brats2021.csv",
        "base_dir": ".",
        "batch_size": 2,
        "epochs": 10, # Full training run for final validation
        "output_size": 64,
        "lr": 1e-4,
        "temperature": 0.07,
        "lambda_recon": 0.1,
        "subset_size": None, # Using all 1381 volumes
        "checkpoint_path": "outputs/checkpoints/multibranch_hybrid_best.pth"
    }
    
    train_hybrid(CONFIG)
