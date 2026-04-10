import numpy as np
import matplotlib.pyplot as plt
import os

def verify_reconstruction(npy_path, output_fig):
    print("--- RUNNING RECONSTRUCTION SANITY CHECK ---")
    
    if not os.path.exists(npy_path):
        print(f"Error: Sample data not found at {npy_path}")
        return
        
    data = np.load(npy_path, allow_pickle=True).item()
    p_id = data["id"]
    orig = data["original"] # [4, 128, 128, 128]
    recon = data["reconstructed"] # [4, 128, 128, 128]
    
    # Pick a modality (FLAIR = index 3) and a central slice
    slice_idx = 64
    mod_idx = 3
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original
    axes[0].imshow(np.rot90(orig[mod_idx, :, :, slice_idx]), cmap='gray')
    axes[0].set_title(f"Original FLAIR (ID: {p_id})")
    axes[0].axis('off')
    
    # Reconstructed
    axes[1].imshow(np.rot90(recon[mod_idx, :, :, slice_idx]), cmap='gray')
    axes[1].set_title(f"Reconstructed FLAIR")
    axes[1].axis('off')
    
    os.makedirs(os.path.dirname(output_fig), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_fig, dpi=150)
    print(f"Reconstruction sanity check saved to {output_fig}")

if __name__ == "__main__":
    verify_reconstruction("outputs/embeddings/minieval_recon_sample.npy", "outputs/figures/minieval_recon_sanity.png")
