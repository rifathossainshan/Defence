import matplotlib.pyplot as plt
import os
import sys
import numpy as np

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from datasets.ssl_dataset import BraTSSSLDataset

def verify_crop(csv_path, base_dir):
    dataset_whole = BraTSSSLDataset(csv_file=csv_path, base_dir=base_dir, crop_mode="whole")
    dataset_lesion = BraTSSSLDataset(csv_file=csv_path, base_dir=base_dir, crop_mode="lesion")
    
    indices = [0, 50, 100] # Sample indices
    os.makedirs("outputs/figures/ablation", exist_ok=True)
    
    fig, axes = plt.subplots(len(indices), 2, figsize=(10, 15))
    plt.suptitle("Crop Mode Comparison: Whole-Brain vs Lesion-Centered", fontsize=16)

    for i, idx in enumerate(indices):
        sample_w = dataset_whole[idx]
        sample_l = dataset_lesion[idx]
        
        # Take FLAIR (index 3) central slice
        img_w = sample_w["view1"][3, :, :, 64].numpy()
        img_l = sample_l["view1"][3, :, :, 64].numpy()
        
        axes[i, 0].imshow(np.rot90(img_w), cmap="gray")
        axes[i, 0].set_title(f"Case {sample_w['id']}\nMode: Whole", fontsize=10)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(np.rot90(img_l), cmap="gray")
        axes[i, 1].set_title(f"Case {sample_l['id']}\nMode: Lesion", fontsize=10)
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("outputs/figures/ablation/crop_comparison.png", dpi=200)
    print("Crop comparison plot saved to outputs/figures/ablation/crop_comparison.png")

if __name__ == "__main__":
    verify_crop("data/metadata/metadata_brats2021.csv", ".")
