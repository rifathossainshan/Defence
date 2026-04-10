import os
import torch
import pandas as pd
from torch.utils.data import DataLoader, Subset
from pathlib import Path
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from datasets.ssl_dataset import BraTSSSLDataset
from preprocessing.simple_transforms import SimpleSSLTransform
from models.encoder import MRIEncoder
from losses.simclr_loss import SimCLRLoss
from engine import train_one_epoch

def main():
    # 1. Configuration
    CONFIG = {
        "csv_path": "data/metadata/metadata_brats2021.csv",
        "base_dir": ".",
        "subset_size": 30,
        "epochs": 1,
        "batch_size": 2,
        "lr": 1e-4,
        "temperature": 0.07,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "log_file": "outputs/logs/train_log.txt",
        "save_dir": "outputs/checkpoints"
    }

    print(f"Using device: {CONFIG['device']}")
    
    # 2. Setup Logging
    os.makedirs(os.path.dirname(CONFIG["log_file"]), exist_ok=True)
    os.makedirs(CONFIG["save_dir"], exist_ok=True)
    
    with open(CONFIG["log_file"], "w") as f:
        f.write("epoch,step,loss,avg_loss,lr,time\n")

    # 3. Dataset & DataLoader
    transform = SimpleSSLTransform()
    full_dataset = BraTSSSLDataset(
        csv_file=CONFIG["csv_path"],
        base_dir=CONFIG["base_dir"],
        split="all",
        transform=transform
    )

    # Trial Run: Subset of 30 samples
    indices = list(range(min(CONFIG["subset_size"], len(full_dataset))))
    trial_dataset = Subset(full_dataset, indices)
    
    loader = DataLoader(
        trial_dataset, 
        batch_size=CONFIG["batch_size"], 
        shuffle=True, 
        num_workers=0
    )

    # 4. Model, Loss, Optimizer
    model = MRIEncoder(in_channels=4, embedding_dim=128).to(CONFIG["device"])
    criterion = SimCLRLoss(temperature=CONFIG["temperature"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # 5. Training Loop (Trial Run)
    print(f"Starting Trial Run (Epochs: {CONFIG['epochs']}, Samples: {len(trial_dataset)})")
    
    for epoch in range(1, CONFIG["epochs"] + 1):
        avg_loss = train_one_epoch(
            model=model,
            loader=loader,
            optimizer=optimizer,
            criterion=criterion,
            device=CONFIG["device"],
            epoch=epoch,
            log_file=CONFIG["log_file"]
        )
        
        # Save checkpoints
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "last_model.pth"))
        torch.save(model.state_dict(), os.path.join(CONFIG["save_dir"], "best_model.pth"))
        
        print(f"Epoch {epoch} finished. Average Loss: {avg_loss:.4f}")

    print("\nTraining Trial Run Complete!")
    print(f"Log saved to: {CONFIG['log_file']}")
    print(f"Checkpoints saved to: {CONFIG['save_dir']}")

if __name__ == "__main__":
    main()
