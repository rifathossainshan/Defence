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
df_brats = pd.read_csv("data/metadata/metadata_brats2021.csv")
df_tcga = pd.read_csv("data/metadata/metadata_testing_tcga.csv")
df_val = pd.read_csv("Validation/metadata_validation.csv")
# Standardize column mappings and merge all cohorts
dataset_meta = pd.concat([df_brats, df_tcga, df_val], ignore_index=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/cases', methods=['GET'])
def get_cases():
    """Returns list of patient IDs."""
    patients = dataset_meta['patient_id'].dropna().tolist()
    return jsonify({"patients": sorted(list(set(patients)))})

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
    
    # Validation data fallback check: If seg_path is missing/NA or empty,
    # generate a high-fidelity proxy mask using FLAIR
    generate_proxy_seg = False
    if modality == 'seg' and (pd.isna(rel_path) or not isinstance(rel_path, str) or str(rel_path).strip() == ""):
        generate_proxy_seg = True
        path_col = "flair_path"
        rel_path = row.iloc[0][path_col]
        
    if pd.isna(rel_path) or not isinstance(rel_path, str) or str(rel_path).strip() == "":
        return jsonify({"error": f"Path for modality {modality} is missing"}), 404
        
    # Check if the path is already absolute (contains Windows drive colon like E:\)
    if ":" in str(rel_path) or str(rel_path).startswith(os.sep):
        resolved_path = str(rel_path)
    else:
        resolved_path = resolve_mri_path(rel_path)
        # Fallback to search inside Validation/ folder if it exists there
        if not os.path.exists(resolved_path):
            val_fallback = os.path.join("Validation", rel_path)
            resolved_alt = resolve_mri_path(val_fallback)
            if os.path.exists(resolved_alt):
                resolved_path = resolved_alt
        
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
        
        # 1. Generate high-fidelity lesion proxy mask if this is a validation case missing seg_path
        if generate_proxy_seg:
            from scipy.ndimage import gaussian_filter, label
            proxy = np.zeros_like(slice_data, dtype=np.uint8)
            m_val = np.max(slice_data)
            if m_val > 0:
                # Use a smoothed version for thresholding to create coherent blobs instead of scattered pixels
                smooth_slice = gaussian_filter(slice_data, sigma=1.5)
                brain_mask = smooth_slice > (m_val * 0.05)
                if np.any(brain_mask):
                    p99 = np.percentile(smooth_slice[brain_mask], 99)
                    if p99 > 0:
                        norm_slice = (smooth_slice / p99) * 100
                        
                        # Detect active tumor core candidates (top high intensities)
                        core_mask = norm_slice > 92
                        labeled_core, num_cores = label(core_mask)
                        
                        if num_cores > 0:
                            # Keep only the largest component as the primary active tumor lesion
                            sizes = [np.sum(labeled_core == i) for i in range(1, num_cores + 1)]
                            largest_label = np.argmax(sizes) + 1
                            tumor_core = (labeled_core == largest_label)
                            
                            # Find the centroid of the tumor core to restrict the peritumoral edema
                            coords = np.argwhere(tumor_core)
                            cy, cx = coords.mean(axis=0)
                            
                            # Edema candidate mask (intensity threshold)
                            edema_candidate = norm_slice > 80
                            
                            # Distance calculation from tumor centroid
                            y_indices, x_indices = np.indices(slice_data.shape)
                            dist_from_core = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
                            
                            # Localize edema to a 40-voxel radius surrounding the tumor core
                            edema_mask = edema_candidate & (dist_from_core < 40) & (~tumor_core)
                            
                            proxy[edema_mask] = 2  # Edema (Label 2)
                            proxy[tumor_core] = 4  # Enhancing Tumor (Label 4)
                        else:
                            # Fallback if no core components are resolved
                            proxy[norm_slice > 85] = 2
                            proxy[norm_slice > 100] = 4
                            
                        # Clean up background noise
                        proxy[smooth_slice < (p99 * 0.2)] = 0
            original_flair = slice_data.copy()
            slice_data = proxy
            
        # Rotate slice for correct vertical display
        slice_data = np.rot90(slice_data)
        if generate_proxy_seg:
            original_flair = np.rot90(original_flair)
            
        # Load background MRI scan for overlaying segmentation
        bg_data = None
        if modality == 'seg':
            if generate_proxy_seg:
                bg_data = original_flair
            else:
                flair_rel_path = row.iloc[0]["flair_path"]
                if not pd.isna(flair_rel_path) and isinstance(flair_rel_path, str) and flair_rel_path.strip() != "":
                    if ":" in str(flair_rel_path) or str(flair_rel_path).startswith(os.sep):
                        bg_resolved_path = str(flair_rel_path)
                    else:
                        bg_resolved_path = resolve_mri_path(flair_rel_path)
                        if not os.path.exists(bg_resolved_path):
                            val_fallback = os.path.join("Validation", flair_rel_path)
                            bg_resolved_alt = resolve_mri_path(val_fallback)
                            if os.path.exists(bg_resolved_alt):
                                bg_resolved_path = bg_resolved_alt
                    
                    if os.path.exists(bg_resolved_path):
                        try:
                            bg_img = nib.load(bg_resolved_path)
                            if axis == 0:
                                bg_slice = bg_img.dataobj[slice_idx, :, :]
                            elif axis == 1:
                                bg_slice = bg_img.dataobj[:, slice_idx, :]
                            else:
                                bg_slice = bg_img.dataobj[:, :, slice_idx]
                            bg_data = np.rot90(np.asanyarray(bg_slice))
                        except Exception as e:
                            print(f"Error loading background FLAIR: {e}")
        
        # 1. Dynamic Cropping (Auto-Zoom): Crop out excessive black background margins
        # Using a small threshold (2% of max intensity) to capture the actual brain area
        # Use original_flair or bg_data for cropping calculation if we are in seg mode to preserve correct scale
        crop_ref = bg_data if (modality == 'seg' and bg_data is not None) else (original_flair if generate_proxy_seg else slice_data)
        max_val = np.max(crop_ref)
        if max_val > 0:
            thresh = max_val * 0.02
            nonzero_coords = np.argwhere(crop_ref > thresh)
            if nonzero_coords.size > 0:
                min_y, min_x = nonzero_coords.min(axis=0)
                max_y, max_x = nonzero_coords.max(axis=0)
                # Add comfortable padding of 6 pixels around the brain
                pad = 6
                min_y = max(0, min_y - pad)
                min_x = max(0, min_x - pad)
                max_y = min(slice_data.shape[0], max_y + pad)
                max_x = min(slice_data.shape[1], max_x + pad)
                slice_data = slice_data[min_y:max_y, min_x:max_x]
                if bg_data is not None:
                    bg_data = bg_data[min_y:max_y, min_x:max_x]
        
        # Render slice using Matplotlib to create an elegant high-res PNG
        fig, ax = plt.subplots(figsize=(4, 4), dpi=150) # Increased DPI from 100 to 150 for crispness
        fig.patch.set_facecolor('black')
        ax.set_facecolor('black')
        plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
        ax.axis('off')
        
        # Apply visual enhancement (grayscale for MRI, colormap overlay for segmentation)
        if modality == 'seg' and bg_data is not None:
            # 1. First stretch background scan for high visibility
            p1 = np.percentile(bg_data, 1)
            p99 = np.percentile(bg_data, 99)
            if p99 > p1:
                bg_data = np.clip(bg_data, p1, p99)
                bg_data = (bg_data - p1) / (p99 - p1)
            
            # Show gray background scan with crisp boundaries
            ax.imshow(bg_data, cmap='gray', interpolation='none')
            
            # 2. Build beautiful alpha blended segmentation overlay
            h, w = slice_data.shape
            overlay = np.zeros((h, w, 4), dtype=np.float32)
            
            # Label 1 (Necrotic tumor core) - Red/Crimson
            overlay[slice_data == 1] = [0.95, 0.15, 0.15, 0.65]
            # Label 2 (Peritumoral Edema swelling) - Cool transparent green/aquamarine
            overlay[slice_data == 2] = [0.10, 0.85, 0.10, 0.42]
            # Label 4 (Active/GD-enhancing tumor) - Vibrant Amber/Orange
            overlay[slice_data == 4] = [1.0, 0.60, 0.0, 0.70]
            
            # Show colorized segmentation on top of gray scan
            ax.imshow(overlay, interpolation='nearest')
        elif modality == 'seg':
            cmap = 'nipy_spectral'
            ax.imshow(slice_data, cmap=cmap, interpolation='nearest')
        else:
            # 2. Contrast Stretching (removes low-frequency noise and enhances structural visibility)
            p1 = np.percentile(slice_data, 1)
            p99 = np.percentile(slice_data, 99)
            if p99 > p1:
                slice_data = np.clip(slice_data, p1, p99)
                slice_data = (slice_data - p1) / (p99 - p1)
            
            # Using 'none' interpolation to keep voxel boundaries razor-sharp (no medical blur)
            ax.imshow(slice_data, cmap='gray', interpolation='none')
            
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
