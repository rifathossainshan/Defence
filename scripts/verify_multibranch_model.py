import torch
import os
import sys

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from models.multibranch_model import MultiBranchHybridSSLModel

def verify_model():
    print("--- STARTING MULTI-BRANCH MODEL VERIFICATION ---")
    
    # 1. Initialize Model
    try:
        model = MultiBranchHybridSSLModel(feature_dim=128, fused_dim=512, embedding_dim=128)
        print("[SUCCESS] Model instantiated.")
    except Exception as e:
        print(f"[FAILED] Model instantiation failed: {e}")
        return

    # 2. Preparation
    batch_size = 2
    dummy_input = torch.randn(batch_size, 4, 128, 128, 128)
    print(f"Input Shape: {dummy_input.shape}")

    # 3. Forward Pass
    print("Running forward pass...")
    try:
        z, recon = model(dummy_input)
        
        print("\n--- OUTPUT SHAPES ---")
        print(f"Embedding (z) Shape: {z.shape} | Expected: [2, 128]")
        print(f"Reconstruction Shape: {recon.shape} | Expected: [2, 4, 128, 128, 128]")
        
        # Validation checks
        assert z.shape == (2, 128), "z shape mismatch!"
        assert recon.shape == (2, 4, 128, 128, 128), "recon shape mismatch!"
        
        print("\n[SUCCESS] Model verification passed! All shapes match user requirements.")
        
    except Exception as e:
        print(f"\n[FAILED] Forward pass or shape check failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    verify_model()
