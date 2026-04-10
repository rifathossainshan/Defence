import os
import pandas as pd
import shutil
from pathlib import Path

def standardize_dataset(csv_path, output_root):
    df = pd.read_csv(csv_path)
    output_root = Path(output_root)
    
    # Mapping dataset names to output folders
    dataset_map = {
        'BraTS2021': 'brats2021',
        'TCGA-GBM': 'tcga_gbm',
        'TCGA-LGG': 'tcga_lgg',
        'BraTS2024-Val': 'brats2024'
    }

    count = 0
    total = len(df)

    for _, row in df.iterrows():
        patient_id = row['patient_id']
        ds_name = row['dataset']
        
        if ds_name not in dataset_map:
            print(f"Skipping unknown dataset: {ds_name}")
            continue
            
        target_dir = output_root / dataset_map[ds_name] / patient_id
        target_dir.mkdir(parents=True, exist_ok=True)
        
        modalities = {
            'flair_path': 'flair.nii',
            't1_path': 't1.nii',
            't1ce_path': 't1ce.nii',
            't2_path': 't2.nii',
            'seg_path': 'seg.nii'
        }
        
        for col, target_name in modalities.items():
            src_path_str = row[col]
            if pd.isna(src_path_str):
                continue
                
            # Path correction for BraTS2021
            if "BraTS2021_Training_Data" in src_path_str:
                src_path_str = src_path_str.replace("BraTS2021_Training_Data", "Training_Data")
            
            src_path = Path(src_path_str)
            
            if not src_path.exists():
                # Try relative to the script's run location (project root)
                src_path = Path("e:/Cse Engineering/11Defense") / src_path_str.replace("BraTS2021_Training_Data", "Training_Data")
                if not src_path.exists():
                    print(f"Warning: Source file not found: {src_path}")
                    continue
            
            # Target path (we use .nii for now as source is .nii)
            dest_path = target_dir / target_name
            
            # Create symlink or copy
            if not dest_path.exists():
                try:
                    # On Windows, symlink might need special permissions. 
                    # If it fails, we fall back to hardlink or copy.
                    os.symlink(src_path.absolute(), dest_path)
                except OSError:
                    try:
                        os.link(src_path.absolute(), dest_path) # Hardlink
                    except OSError:
                        shutil.copy2(src_path, dest_path) # Copy if all else fails
        
        count += 1
        if count % 100 == 0:
            print(f"Processed {count}/{total} cases...")

    print(f"Standardization complete. Data organized in {output_root}")

if __name__ == "__main__":
    CSV_PATH = "e:/Cse Engineering/11Defense/master_metadata.csv"
    OUTPUT_DIR = "e:/Cse Engineering/11Defense/data"
    standardize_dataset(CSV_PATH, OUTPUT_DIR)
