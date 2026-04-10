import os
import torch
import torch.optim as optim
import sys
import numpy as np

# Add src to path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
src_dir = os.path.join(root_dir, 'src')
sys.path.append(root_dir)
sys.path.append(src_dir)

from models.multibranch_model import MultiBranchHybridSSLModel
from losses.hybrid_loss import HybridSSLLoss

def sanity_test():
    print("--- MULTI-BRANCH HYBRID SANITY TEST (1 SAMPLE) ---")
    device = torch.device("cpu")
    
    # 1. Setup Model
    model = MultiBranchHybridSSLModel(embedding_dim=128).to(device)
    criterion = HybridSSLLoss(lambda_recon=0.01)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 2. Dummy Batch (4 modalities, 128^3)
    # B=1, C=4, D=128, H=128, W=128
    x1 = torch.randn(1, 4, 128, 128, 128)
    x2 = torch.randn(1, 4, 128, 128, 128)
    
    print("Forward Pass...")
    z1, recon1 = model(x1)
    z2, recon2 = model(x2)
    print(f"z1 shape: {z1.shape}")
    print(f"recon1 shape: {recon1.shape}")
    
    print("Loss Calculation...")
    loss, sim_l, rec_l = criterion(z1, z2, recon1, recon2, x1, x2)
    print(f"Initial Loss: {loss.item()}")
    
    print("Backward Pass...")
    loss.backward()
    
    print("Optimizer Step...")
    optimizer.step()
    
    print("\n[SUCCESS] Multi-Branch Model Sanity Check Passed!")

if __name__ == "__main__":
    sanity_test()
