import os
import nibabel as nib
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def run_comprehensive_qc(csv_path, data_root):
    df = pd.read_csv(csv_path)
    data_root = Path(data_root)
    
    results = []
    required_modalities = ['t1', 't1ce', 't2', 'flair']
    
    # Dataset name to folder mapping (consistent with standardization script)
    dataset_map = {
        'BraTS2021': 'brats2021',
        'TCGA-GBM': 'tcga_gbm',
        'TCGA-LGG': 'tcga_lgg',
        'BraTS2024-Val': 'brats2024'
    }
    
    print(f"Starting comprehensive QC for {len(df)} cases...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row['patient_id']
        ds_raw = row['dataset']
        ds_folder = dataset_map.get(ds_raw, ds_raw.lower())
        p_path = data_root / ds_folder / patient_id
        
        qc_pass = True
        error_msg = ""
        
        stats = {
            'patient_id': patient_id,
            'dataset': ds_raw,
            'shapes': {},
            'spacings': {},
            'affine_match': True
        }
        
        try:
            first_affine = None
            first_shape = None
            
            for mod in required_modalities:
                mod_file = p_path / f"{mod}.nii"
                if not mod_file.exists():
                    qc_pass = False
                    error_msg = f"Missing {mod}.nii at {mod_file}"
                    break
                
                img = nib.load(mod_file)
                shape = img.shape
                spacing = tuple(np.round(img.header.get_zooms(), 3))
                affine = img.affine
                
                stats['shapes'][mod] = str(shape)
                stats['spacings'][mod] = str(spacing)
                
                if first_affine is None:
                    first_affine = affine
                    first_shape = shape
                else:
                    if shape != first_shape:
                        qc_pass = False
                        error_msg = f"Shape mismatch: {mod} {shape} vs ref {first_shape}"
                    if not np.allclose(affine, first_affine, atol=1e-3):
                        stats['affine_match'] = False
                        # Note: In some multi-modality datasets, affines might differ slightly 
                        # if not perfectly registered. We log it but won't necessarily fail BraTS.
                        # error_msg = f"Affine mismatch in {mod}"
            
        except Exception as e:
            qc_pass = False
            error_msg = f"Error during QC: {str(e)}"
            
        results.append({
            'patient_id': patient_id,
            'dataset': ds_raw,
            'qc_pass': 1 if qc_pass else 0,
            'error_msg': error_msg,
            'shape': stats['shapes'].get('flair', 'N/A'),
            'spacing': stats['spacings'].get('flair', 'N/A'),
            'affine_match': 1 if stats['affine_match'] else 0
        })
        
    qc_df = pd.DataFrame(results)
    report_path = data_root / "metadata" / "qc_report.csv"
    qc_df.to_csv(report_path, index=False)
    
    # Summary
    pass_count = qc_df['qc_pass'].sum()
    fail_count = len(qc_df) - pass_count
    print(f"\nQC Summary:")
    print(f"Total: {len(qc_df)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    
    if fail_count > 0:
        print("\nFirst 5 failures:")
        print(qc_df[qc_df['qc_pass'] == 0][['patient_id', 'dataset', 'error_msg']].head(5))
    
    print("\nSample of 15 cases for manual verification:")
    pd.set_option('display.expand_frame_repr', False)
    print(qc_df.sample(min(15, len(qc_df)))[['patient_id', 'dataset', 'shape', 'spacing', 'qc_pass']])

if __name__ == "__main__":
    CSV_PATH = "e:/Cse Engineering/11Defense/data/metadata/metadata_brats2021.csv"
    DATA_ROOT = "e:/Cse Engineering/11Defense/data"
    run_comprehensive_qc(CSV_PATH, DATA_ROOT)
