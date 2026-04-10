import os
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Spacingd,
    NormalizeIntensityd,
    CropForegroundd,
    SpatialPadd,
    ResizeWithPadOrCropd,
    ToTensord,
)
from monai.data import Dataset, DataLoader
from tqdm import tqdm

def get_transforms(spatial_size=(128, 128, 128)):
    return Compose([
        LoadImaged(keys=["t1", "t1ce", "t2", "flair"]),
        EnsureChannelFirstd(keys=["t1", "t1ce", "t2", "flair"]),
        Orientationd(keys=["t1", "t1ce", "t2", "flair"], axcodes="RAS"),
        Spacingd(
            keys=["t1", "t1ce", "t2", "flair"],
            pixdim=(1.0, 1.0, 1.0),
            mode=("bilinear", "bilinear", "bilinear", "bilinear"),
        ),
        # Crop foreground based on FLAIR (modality with usually the widest extent)
        CropForegroundd(keys=["t1", "t1ce", "t2", "flair"], source_key="flair"),
        NormalizeIntensityd(keys=["t1", "t1ce", "t2", "flair"], nonzero=True, channel_wise=True),
        ResizeWithPadOrCropd(keys=["t1", "t1ce", "t2", "flair"], spatial_size=spatial_size),
        ToTensord(keys=["t1", "t1ce", "t2", "flair"]),
    ])

def preprocess_all(data_root, output_root, qc_csv, spatial_size=(128, 128, 128)):
    data_root = Path(data_root)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    
    qc_df = pd.read_csv(qc_csv)
    # Only process cases that passed QC
    passed_cases = qc_df[qc_df['qc_pass'] == 1]
    
    data_dicts = []
    for _, row in passed_cases.iterrows():
        p_id = row['patient_id']
        ds = row['dataset']
        p_path = data_root / ds / p_id
        
        data_dicts.append({
            "patient_id": p_id,
            "dataset": ds,
            "t1": str(p_path / "t1.nii"),
            "t1ce": str(p_path / "t1ce.nii"),
            "t2": str(p_path / "t2.nii"),
            "flair": str(p_path / "flair.nii"),
        })

    transforms = get_transforms(spatial_size)
    
    print(f"Starting preprocessing for {len(data_dicts)} cases...")
    
    for item in tqdm(data_dicts):
        try:
            # Apply transforms
            p_id = item['patient_id']
            ds = item['dataset']
            
            # Create output directory
            save_dir = output_root / ds / p_id
            save_dir.mkdir(parents=True, exist_ok=True)
            save_path = save_dir / "preprocessed_mtal.pt" # Multimodal Tensor
            
            if save_path.exists():
                continue

            # Load and process
            out = transforms(item)
            
            # Stack modalities into a single 4-channel tensor [4, 128, 128, 128]
            stacked = torch.stack([
                out["t1"][0], 
                out["t1ce"][0], 
                out["t2"][0], 
                out["flair"][0]
            ])
            
            # Save as torch tensor (efficient for training)
            torch.save(stacked, save_path)
            
        except Exception as e:
            print(f"Error processing {item['patient_id']}: {str(e)}")

    print(f"Preprocessing complete. Results in {output_root}")

if __name__ == "__main__":
    DATA_ROOT = "e:/Cse Engineering/11Defense/data"
    OUTPUT_ROOT = "e:/Cse Engineering/11Defense/preprocessed_data"
    QC_CSV = "e:/Cse Engineering/11Defense/qc_results.csv"
    
    # Run preprocessing
    preprocess_all(DATA_ROOT, OUTPUT_ROOT, QC_CSV)
