import sys
import torch
from pathlib import Path

# Add src to path
sys.path.append(str(Path("e:/Cse Engineering/11Defense/src")))

from models.encoder import MRIEncoder

def verify_model():
    print("Initializing MRIEncoder (ResNet18 3D)...")
    model = MRIEncoder(in_channels=4, embedding_dim=128)
    
    # Check parameters count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total Parameters: {total_params / 1e6:.2f}M")
    
    # Create dummy batch
    batch_size = 2
    x = torch.randn(batch_size, 4, 128, 128, 128)
    print(f"Input shape: {x.shape}")
    
    print("Performing forward pass...")
    try:
        z = model(x)
        print(f"Output embedding shape: {z.shape}")
        
        # Validation
        assert z.shape == (batch_size, 128), f"Error: Expected shape (2, 128), got {z.shape}"
        print("\n[SUCCESS] Phase 6: Model forward pass is functional and shapes are correct!")
        
    except Exception as e:
        print(f"\n[FAILURE] Phase 6: Forward pass failed. Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    verify_model()
