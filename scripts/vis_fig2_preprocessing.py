import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import os
import sys
import pandas as pd
from scipy.ndimage import zoom

# Add current directory to path for imports
sys.path.append('.')
from scripts.vis_utils import set_academic_style, save_fig

def get_center_slice(vol):
    return vol[:, :, vol.shape[2]//2]

def preprocess_step_demo(path, target_shape=(64, 64, 64)):
    """Simulate preprocessing steps for visualization."""
    # 1. Raw
    img_obj = nib.load(path)
    raw_vol = img_obj.get_fdata()
    
    # 2. Normalized
    norm_vol = (raw_vol - np.mean(raw_vol)) / (np.std(raw_vol) + 1e-8)
    
    # 3. Cropped (Simplified ROI)
    # In actual pipeline we use non-zero bounding box
    coords = np.argwhere(raw_vol > np.max(raw_vol)*0.1)
    if len(coords) > 0:
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)
        crop_vol = norm_vol[z_min:z_max, y_min:y_max, x_min:x_max]
    else:
        crop_vol = norm_vol

    # 4. Resized
    factors = [t/s for t, s in zip(target_shape, crop_vol.shape)]
    resized_vol = zoom(crop_vol, factors, order=1)
    
    return raw_vol, norm_vol, crop_vol, resized_vol

def visualize_workflow():
    set_academic_style()
    
    # Target Patient (TCGA-02-0006 is a good example from GBM)
    tcga_meta = pd.read_csv("data/metadata/metadata_testing_tcga.csv")
    patient_id = "TCGA-02-0006"
    row = tcga_meta[tcga_meta['patient_id'] == patient_id].iloc[0]
    
    flair_path = row['flair_path'].replace('\\', '/')
    
    print(f"Processing workflow for {patient_id}...")
    raw, norm, crop, resized = preprocess_step_demo(flair_path)
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    
    titles = ['(A) Raw Slice', '(B) Normalized', '(C) ROI Cropped', '(D) Final Resized (64³)']
    volumes = [raw, norm, crop, resized]
    
    for ax, vol, title in zip(axes, volumes, titles):
        slice_img = get_center_slice(vol)
        im = ax.imshow(np.rot90(slice_img), cmap='bone')
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    save_fig(fig, "fig2_preprocessing.png")
    plt.close()

if __name__ == "__main__":
    visualize_workflow()
