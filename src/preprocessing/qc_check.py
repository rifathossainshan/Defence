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
    
    print(f"Starting comprehensive QC for {len(df)} cases...")
    
    for _, row in tqdm(df.iterrows(), total=len(df)):
        patient_id = row['patient_id']
        ds = row['dataset']
        p_path = data_root / ds / patient_id
        
        qc_pass = True
        error_msg = ""
        
        stats = {
            'patient_id': patient_id,
            'dataset': ds,
            'shapes': {},
            'spacings': {},
            'affine_match': True
        }
        
        try:
            first_affine = None
            first_shape = None
            first_spacing = None
            
            for mod in required_modalities:
                mod_file = p_path / f"{mod}.nii"
                if not mod_file.exists():
                    qc_pass = False
                    error_msg = f"Missing {mod}.nii"
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
                    first_spacing = spacing
                else:
                    # Check consistency
                    if shape != first_shape:
                        qc_pass = False
                        error_msg = f"Shape mismatch: {mod} {shape} vs ref {first_shape}"
                    if not np.allclose(affine, first_affine, atol=1e-3):
                        stats['affine_match'] = False
                        # We don't necessarily fail QC for affine mismatch yet if registered later, 
                        # but we should note it. For BraTS they SHOULD match.
                        error_msg = f"Affine mismatch in {mod}"
            
            # Additional BraTS specific check (Expect 1mm and similar shapes)
            if first_spacing and not all(1.0 - 0.1 <= s <= 1.0 + 0.1 for s in first_spacing[:3]):
                error_msg = f"Unusual spacing detected: {first_spacing}"
                # We won't fail it yet, just log it.
                
        except Exception as e:
            qc_pass = False
            error_msg = f"Error during QC: {str(e)}"
            
        results.append({
            'patient_id': patient_id,
            'dataset': ds,
            'qc_pass': 1 if qc_pass else 0,
            'error_msg': error_msg,
            'shape': stats['shapes'].get('flair', 'N/A'),
            'spacing': stats['spacings'].get('flair', 'N/A'),
            'affine_match': 1 if stats['affine_match'] else 0
        })
        
    qc_df = pd.DataFrame(results)
    report_path = Path("e:/Cse Engineering/11Defense/data/metadata/qc_report.csv")
    qc_df.to_csv(report_path, index=False)
    
    # Summary
    pass_count = qc_df['qc_pass'].sum()
    fail_count = len(qc_df) - pass_count
    print(f"\nQC Summary:")
    print(f"Total: {len(qc_df)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    
    # Randomly sample 10 cases for manual verification display
    print("\nSample of 10 cases for manual verification:")
    print(qc_df.sample(min(10, len(qc_df)))[['patient_id', 'shape', 'spacing', 'qc_pass']])

if __name__ == "__main__":
    CSV_PATH = "e:/Cse Engineering/11Defense/data/metadata/metadata_brats2021.csv"
    DATA_ROOT = "e:/Cse Engineering/11Defense/data"
    run_comprehensive_qc(CSV_PATH, DATA_ROOT)
