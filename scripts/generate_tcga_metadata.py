import os
import pandas as pd
from pathlib import Path

def generate_tcga_metadata():
    gbm_root = Path(r"e:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-GBM\BraTS-TCGA-GBM\Pre-operative_TCGA_GBM_NIfTI_and_Segmentations")
    lgg_root = Path(r"e:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-LGG\BraTS-TCGA-LGG\Pre-operative_TCGA_LGG_NIfTI_and_Segmentations")
    
    output_path = Path(r"e:\Cse Engineering\11Defense\data\metadata\metadata_testing_tcga.csv")
    
    rows = []
    
    for root_dir, cohort_label in [(gbm_root, "GBM"), (lgg_root, "LGG")]:
        if not root_dir.exists():
            print(f"Warning: {root_dir} not found.")
            continue
            
        print(f"Scanning {cohort_label} cohort...")
        
        # Each patient is a subdirectory
        for patient_dir in root_dir.iterdir():
            if not patient_dir.is_dir():
                continue
                
            patient_id = patient_dir.name
            files = list(patient_dir.glob("*.nii"))
            
            # Map modalities
            # Filenames look like TCGA-02-0006_1996.08.23_flair.nii
            mapping = {
                "flair": None,
                "t1": None,
                "t1ce": None, # mapped from t1Gd
                "t2": None
            }
            
            for f in files:
                fname = f.name.lower()
                if "_flair.nii" in fname:
                    mapping["flair"] = str(f)
                elif "_t1.nii" in fname:
                    mapping["t1"] = str(f)
                elif "_t1gd.nii" in fname:
                    mapping["t1ce"] = str(f)
                elif "_t2.nii" in fname:
                    mapping["t2"] = str(f)
            
            # Only add if we have all modalities (or at least some)
            if any(mapping.values()):
                rows.append({
                    "patient_id": patient_id,
                    "t1_path": mapping["t1"],
                    "t1ce_path": mapping["t1ce"],
                    "t2_path": mapping["t2"],
                    "flair_path": mapping["flair"],
                    "dataset": cohort_label
                })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Generated metadata for {len(df)} patients at {output_path}")

if __name__ == "__main__":
    generate_tcga_metadata()
