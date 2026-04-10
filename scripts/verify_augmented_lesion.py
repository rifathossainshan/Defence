import matplotlib.pyplot as plt
import os
import sys
import numpy as np
import torch

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from datasets.ssl_dataset import BraTSSSLDataset
from preprocessing.simple_transforms import SimpleSSLTransform

def verify_augmented_visuals(csv_path, base_dir):
    print("--- AUGMENTATION VISUAL SANITY CHECK ---")
    
    transform = SimpleSSLTransform()
    dataset = BraTSSSLDataset(
        csv_file=csv_path, 
        base_dir=base_dir, 
        transform=transform, 
        use_seg=True
    )
    
    # Pick a case with visible tumor
    idx = 0 
    sample = dataset[idx]
    
    # view1 and view2 are augmented versions from the same instance
    # To see the "Original", we can load without transform or just assume view1 vs view2 shows the variation
    
    # Let's get original by reloading with no transform
    dataset_no_aug = BraTSSSLDataset(csv_file=csv_path, base_dir=base_dir, transform=None)
    sample_orig = dataset_no_aug[idx]

    # Modality: FLAIR (index 3), Central Slice
    slice_idx = 64
    img_orig = sample_orig["view1"][3, :, :, slice_idx].numpy()
    img_v1 = sample["view1"][3, :, :, slice_idx].numpy()
    img_v2 = sample["view2"][3, :, :, slice_idx].numpy()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(np.rot90(img_orig), cmap='gray')
    axes[0].set_title(f"Original FLAIR\n(ID: {sample['id']})")
    axes[0].axis('off')
    
    axes[1].imshow(np.rot90(img_v1), cmap='gray')
    axes[1].set_title("Augmented View 1\n(Noise + Flip + Shift)")
    axes[1].axis('off')
    
    axes[2].imshow(np.rot90(img_v2), cmap='gray')
    axes[2].set_title("Augmented View 2\n(Noise + Flip + Shift)")
    axes[2].axis('off')
    
    os.makedirs("outputs/figures/sanity", exist_ok=True)
    plt.tight_layout()
    plt.savefig("outputs/figures/sanity/augmentation_check.png", dpi=150)
    print("Visual check saved to outputs/figures/sanity/augmentation_check.png")

if __name__ == "__main__":
    verify_augmented_visuals("data/metadata/metadata_brats2021.csv", ".")
