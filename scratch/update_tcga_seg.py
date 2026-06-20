import os
import glob
import pandas as pd

def main():
    csv_path = "data/metadata/metadata_testing_tcga.csv"
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return
        
    df = pd.read_csv(csv_path)
    print("Original columns:", df.columns.tolist())
    
    seg_paths = []
    for idx, row in df.iterrows():
        # Get patient directory from t1_path
        t1_path = row['t1_path']
        patient_dir = os.path.dirname(t1_path)
        
        # Look for ManuallyCorrected.nii first, then GlistrBoost.nii
        seg_file = None
        if os.path.exists(patient_dir):
            manual_files = glob.glob(os.path.join(patient_dir, "*_ManuallyCorrected.nii"))
            glistr_files = glob.glob(os.path.join(patient_dir, "*_GlistrBoost.nii"))
            
            if manual_files:
                seg_file = manual_files[0]
            elif glistr_files:
                seg_file = glistr_files[0]
                
        seg_paths.append(seg_file)
        
    df['seg_path'] = seg_paths
    df.to_csv(csv_path, index=False)
    print(f"Successfully added seg_path to {csv_path}!")
    print("Sample updated row:")
    print(df.iloc[0].to_dict())

if __name__ == "__main__":
    main()
