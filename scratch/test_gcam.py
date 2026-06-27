import os
import sys

# Add src/ to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
sys.path.append(os.path.abspath('src'))

import pandas as pd
import numpy as np

# Load metadata like app.py does
metadata_path = "data/retrieval_metadata.csv"
dataset_meta = pd.read_csv(metadata_path)

from web.app import compute_patient_gcam, get_gcam_engine

patient_id = "BraTS-GLI-02095-100"
print(f"Testing Grad-CAM calculation for: {patient_id}")
try:
    cam_vol = compute_patient_gcam(patient_id)
    if cam_vol is None:
        print("RESULT: cam_vol is None!")
    else:
        print(f"RESULT: Success! Shape={cam_vol.shape}, Min={cam_vol.min()}, Max={cam_vol.max()}")
except Exception as e:
    import traceback
    traceback.print_exc()
