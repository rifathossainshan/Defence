import torch
from tqdm import tqdm
import time

def train_one_epoch(model, loader, optimizer, criterion, device, epoch, log_file=None):
    model.train()
    total_loss = 0
    start_time = time.time()
    
    # Progress bar
    pbar = tqdm(enumerate(loader), total=len(loader), desc=f"Epoch {epoch}")
    
    for step, batch in pbar:
        # Load views to device
        view1 = batch["view1"].to(device)
        view2 = batch["view2"].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        z1 = model(view1)
        z2 = model(view2)
        
        # Compute SimCLR Loss
        loss = criterion(z1, z2)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'avg_loss': f"{avg_loss:.4f}"
        })
        
        # Log to file if provided
        if log_file:
            with open(log_file, "a") as f:
                elapsed = time.time() - start_time
                f.write(f"{epoch},{step},{loss.item():.6f},{avg_loss:.6f},{optimizer.param_groups[0]['lr']:.8f},{elapsed:.2f}\n")
                
    return avg_loss
