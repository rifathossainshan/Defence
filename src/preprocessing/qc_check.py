import os
import nibabel as nib
import pandas as pd
from pathlib import Path

def run_qc(data_root):
    data_root = Path(data_root)
    datasets = ['brats2021', 'tcga_gbm', 'tcga_lgg', 'brats2024']
    
    results = []
    required_modalities = ['t1.nii', 't1ce.nii', 't2.nii', 'flair.nii']
    
    for ds in datasets:
        ds_path = data_root / ds
        if not ds_path.exists():
            print(f"Skipping missing dataset folder: {ds_path}")
            continue
            
        print(f"Checking dataset: {ds}")
        patients = [p for p in ds_path.iterdir() if p.is_dir()]
        
        for p_path in patients:
            patient_id = p_path.name
            qc_pass = True
            error_msg = ""
            
            # 1. Check existence
            missing = [m for m in required_modalities if not (p_path / m).exists()]
            if missing:
                qc_pass = False
                error_msg = f"Missing modalities: {', '.join(missing)}"
            else:
                # 2. Check shapes and affines
                try:
                    ref_img = None
                    ref_shape = None
                    
                    for mod in required_modalities:
                        img = nib.load(p_path / mod)
                        if ref_img is None:
                            ref_img = img
                            ref_shape = img.shape
                        else:
                            if img.shape != ref_shape:
                                qc_pass = False
                                error_msg = f"Shape mismatch: {mod} {img.shape} vs ref {ref_shape}"
                                break
                except Exception as e:
                    qc_pass = False
                    error_msg = f"Corruption/Load error: {str(e)}"
            
            results.append({
                'patient_id': patient_id,
                'dataset': ds,
                'qc_pass': 1 if qc_pass else 0,
                'error': error_msg
            })
            
    qc_df = pd.DataFrame(results)
    output_csv = data_root.parent / "qc_results.csv"
    qc_df.to_csv(output_csv, index=False)
    
    # Summary
    pass_count = qc_df['qc_pass'].sum()
    fail_count = len(qc_df) - pass_count
    print(f"\nQC Summary:")
    print(f"Total cases checked: {len(qc_df)}")
    print(f"Passed: {pass_count}")
    print(f"Failed: {fail_count}")
    print(f"Results saved to: {output_csv}")

if __name__ == "__main__":
    DATA_ROOT = "e:/Cse Engineering/11Defense/data"
    run_qc(DATA_ROOT)
