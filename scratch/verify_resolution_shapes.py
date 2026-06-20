import sys
import os
import torch
from pathlib import Path

# Add src to python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.multibranch_model import MultiBranchHybridSSLModel

def test_resolution(size):
    print(f"\n--- Testing resolution {size}^3 ---")
    try:
        model = MultiBranchHybridSSLModel(embedding_dim=128, output_size=size)
        model.eval()
        
        # input shape: [B, 4, D, H, W]
        x = torch.randn(1, 4, size, size, size)
        print(f"Input shape: {x.shape}")
        
        with torch.no_grad():
            z, recon = model(x)
            
        print(f"Embedding shape: {z.shape} (Expected: [1, 128])")
        print(f"Reconstruction shape: {recon.shape} (Expected: [1, 4, {size}, {size}, {size}])")
        
        assert z.shape == (1, 128), "Embedding shape mismatch!"
        assert recon.shape == (1, 4, size, size, size), "Reconstruction shape mismatch!"
        print(f"[SUCCESS] {size}^3 resolution verification passed!")
    except Exception as e:
        print(f"[ERROR] failed for {size}^3: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    test_resolution(128)
    test_resolution(256)
