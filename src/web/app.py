import os
import sys
import base64
import io
import time
import numpy as np
import pandas as pd
import nibabel as nib
import faiss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from flask import Flask, jsonify, request, render_template

# Add root folder to sys path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.utils.path_resolver import resolve_mri_path

app = Flask(__name__, 
            template_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'templates')),
            static_folder=os.path.abspath(os.path.join(os.path.dirname(__file__), 'static')))

# Load Database index & metadata
EMB_DIR = "outputs/embeddings"
meta_df = pd.read_csv(os.path.join(EMB_DIR, "hybrid_metadata.csv"))
embeddings = np.load(os.path.join(EMB_DIR, "hybrid_embeddings.npy")).astype('float32')
faiss.normalize_L2(embeddings)

index = faiss.read_index(os.path.join(EMB_DIR, "hybrid_faiss.index"))

# Load raw dataset metadata to resolve local NIfTI paths
dataset_meta = pd.read_csv("data/metadata/metadata_brats2021.csv")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/cases', methods=['GET'])
def get_cases():
    """Returns list of patient IDs."""
    patients = dataset_meta['patient_id'].tolist()
    return jsonify({"patients": sorted(patients)})

@app.route('/api/query', methods=['GET'])
def query_case():
    """Queries FAISS index and returns top-5 similar patients (excluding self)."""
    patient_id = request.args.get('patient_id')
    if not patient_id:
        return jsonify({"error": "Missing patient_id parameter"}), 400
        
    # Get index of this patient in our embedding list
    match_rows = meta_df[meta_df['patient_id'] == patient_id]
    if match_rows.empty:
        return jsonify({"error": f"Patient {patient_id} not found in database metadata"}), 404
        
    idx = match_rows.index[0]
    query_vec = embeddings[idx:idx+1]
    
    # Search index (retrieve 10 in case some are filtered)
    scores, indices = index.search(query_vec, k=10)
    scores = scores[0]
    indices = indices[0]
    
    results = []
    rank = 1
    for score, m_idx in zip(scores, indices):
        match_id = meta_df.iloc[m_idx]["patient_id"]
        source_dataset = meta_df.iloc[m_idx]["dataset"]
        
        # Self exclusion
        if match_id == patient_id:
            continue
            
        results.append({
            "rank": rank,
            "patient_id": match_id,
            "score": float(score),
            "dataset": source_dataset
        })
        rank += 1
        if rank > 5:
            break
            
    return jsonify({
        "query_patient_id": patient_id,
        "results": results
    })

@app.route('/api/slice', methods=['GET'])
def get_slice():
    """
    On-the-fly NIfTI 2D slice extractor.
    Parameters: patient_id, modality (t1, t1ce, t2, flair, seg), plane (axial, sagittal, coronal), slice_idx (0 to 100%)
    """
    patient_id = request.args.get('patient_id')
    modality = request.args.get('modality', 'flair').lower()
    plane = request.args.get('plane', 'axial').lower()
    slice_pct = float(request.args.get('slice_pct', 0.5))
    
    if not patient_id:
        return jsonify({"error": "Missing patient_id"}), 400
        
    row = dataset_meta[dataset_meta['patient_id'] == patient_id]
    if row.empty:
        return jsonify({"error": f"Patient {patient_id} not found"}), 404
        
    # Find NIfTI path from columns
    path_col = f"{modality}_path"
    if path_col not in row.columns:
        return jsonify({"error": f"Modality {modality} not supported"}), 400
        
    rel_path = row.iloc[0][path_col]
    if pd.isna(rel_path):
        return jsonify({"error": f"Path for modality {modality} is missing"}), 404
        
    resolved_path = resolve_mri_path(rel_path, base_dir=".")
    if not os.path.exists(resolved_path):
        return jsonify({"error": f"NIfTI file not found at {resolved_path}"}), 404
        
    try:
        # Load NIfTI header-safe using nibabel proxy
        img = nib.load(resolved_path)
        shape = img.shape
        
        # Determine slice plane indices
        # standard BraTS shape is typically (240, 240, 155)
        # axis mapping: 0=sagittal, 1=coronal, 2=axial
        if plane == 'sagittal':
            axis = 0
            max_slices = shape[0]
        elif plane == 'coronal':
            axis = 1
            max_slices = shape[1]
        else: # axial
            axis = 2
            max_slices = shape[2]
            
        slice_idx = int(slice_pct * (max_slices - 1))
        
        # Slice dataobj without loading the whole volume into RAM
        if axis == 0:
            slice_data = img.dataobj[slice_idx, :, :]
        elif axis == 1:
            slice_data = img.dataobj[:, slice_idx, :]
        else:
            slice_data = img.dataobj[:, :, slice_idx]
            
        slice_data = np.asanyarray(slice_data)
        
        # Rotate slice for correct vertical display
        slice_data = np.rot90(slice_data)
        
        # Render slice using Matplotlib to create an elegant PNG
        fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        
        # Apply visual enhancement (grayscale for MRI, colormap for segmentation)
        if modality == 'seg':
            cmap = 'nipy_spectral'
            ax.imshow(slice_data, cmap=cmap, interpolation='nearest')
        else:
            # Normalize MRI intensities to 99th percentile for visual pop (removes outliers)
            p99 = np.percentile(slice_data, 99)
            if p99 > 0:
                slice_data = np.clip(slice_data, 0, p99)
            ax.imshow(slice_data, cmap='gray')
            
        buf = io.BytesIO()
        plt.savefig(buf, format='png', facecolor='black', bbox_inches='tight', pad_inches=0)
        plt.close(fig)
        buf.seek(0)
        
        base64_img = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({
            "slice_idx": slice_idx,
            "max_slices": max_slices,
            "image": f"data:image/png;base64,{base64_img}"
        })
        
    except Exception as e:
        return jsonify({"error": f"Failed to process NIfTI: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
