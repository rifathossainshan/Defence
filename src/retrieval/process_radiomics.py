import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def process_radiomics(gbm_path, lgg_path, master_metadata_path, output_dir):
    print("Loading Radiomics CSVs...")
    gbm_df = pd.read_csv(gbm_path)
    lgg_df = pd.read_csv(lgg_path)
    master_df = pd.read_csv(master_metadata_path)

    # 1. Identity common columns (features)
    # Drop 'Date' and keep 'ID' for mapping
    common_cols = list(set(gbm_df.columns) & set(lgg_df.columns))
    if 'Date' in common_cols:
        common_cols.remove('Date')
    
    # We must have 'ID'
    if 'ID' not in common_cols:
        print("Error: 'ID' column not found in both CSVs.")
        return

    # 2. Merge GSM and LGG
    combined_df = pd.concat([gbm_df[common_cols], lgg_df[common_cols]], axis=0).reset_index(drop=True)
    print(f"Combined Radiomics cases: {len(combined_df)}")

    # 3. Filter to match our master metadata (patient_id)
    # Note: Our metadata has patient_id, radiomics has ID. We match them.
    master_ids = set(master_df['patient_id'].unique())
    combined_df = combined_df[combined_df['ID'].isin(master_ids)].reset_index(drop=True)
    print(f"Matched with Master Metadata: {len(combined_df)} cases")

    # 4. Handle Missing and Infinite Values
    # Replace inf with NaN
    combined_df = combined_df.replace([np.inf, -np.inf], np.nan)
    
    # Drop columns with > 50% NaNs
    threshold = len(combined_df) * 0.5
    combined_df = combined_df.dropna(thresh=threshold, axis=1)
    
    # Fill remaining NaNs with column mean
    feature_cols = [c for c in combined_df.columns if c != 'ID']
    for col in feature_cols:
        combined_df[col] = pd.to_numeric(combined_df[col], errors='coerce')
        # Handle cases where column might still have inf after numeric conversion or very large values
        col_mean = combined_df[col][np.isfinite(combined_df[col])].mean()
        combined_df[col] = combined_df[col].fillna(col_mean if pd.notnull(col_mean) else 0)
        # Final clip for extreme outliers just in case
        combined_df[col] = combined_df[col].clip(lower=-1e10, upper=1e10)

    # 5. Extract Features and Normalize
    X = combined_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 6. Save Results
    os.makedirs(output_dir, exist_ok=True)
    # Ensure C-contiguity for FAISS
    X_contiguous = np.ascontiguousarray(X_scaled).astype('float32')
    np.save(os.path.join(output_dir, "radiomics_features.npy"), X_contiguous)
    
    # Metadata for Radiomics Index
    # We rename 'ID' to 'patient_id' to match our conventions
    radiomics_meta = combined_df[['ID']].copy()
    radiomics_meta.columns = ['patient_id']
    # Add source dataset info via join with master
    radiomics_meta = radiomics_meta.merge(master_df[['patient_id', 'dataset']], on='patient_id', how='left')
    radiomics_meta.to_csv(os.path.join(output_dir, "radiomics_metadata.csv"), index=False)

    print(f"\n[SUCCESS] Radiomics Preprocessing Complete!")
    print(f"Features matrix shape: {X_scaled.shape}")
    print(f"Saved to {output_dir}")

if __name__ == "__main__":
    GBM_PATH = r"E:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-GBM\BraTS-TCGA-GBM\Pre-operative_TCGA_GBM_NIfTI_and_Segmentations\TCGA_GBM_radiomicFeatures.csv"
    LGG_PATH = r"E:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-LGG\BraTS-TCGA-LGG\Pre-operative_TCGA_LGG_NIfTI_and_Segmentations\TCGA_LGG_radiomicFeatures.csv"
    MASTER_CSV = "data/metadata/metadata_brats2021.csv"
    OUTPUT_DIR = "outputs/embeddings"
    
    process_radiomics(GBM_PATH, LGG_PATH, MASTER_CSV, OUTPUT_DIR)
