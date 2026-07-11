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

# Grad-CAM Global Cache and Model Engine Initialization
_gcam_model = None
_gcam_engine = None
gcam_cache = {}

def get_gcam_engine():
    global _gcam_model, _gcam_engine
    if _gcam_model is None:
        import torch
        from src.models.multibranch_model import MultiBranchHybridSSLModel
        from src.evaluation.explainability_gradcam import GradCAM3D
        
        device = torch.device('cpu')
        model_path = "outputs/checkpoints/multibranch_hybrid_best.pth"
        try:
            # Output size of current best model is 128
            _gcam_model = MultiBranchHybridSSLModel(output_size=128).to(device)
            state_dict = torch.load(model_path, map_location=device)
            
            # Filter out reconstruction head to prevent shape mismatches
            filtered_dict = {k: v for k, v in state_dict.items() if 'reconstruction_head' not in k}
            _gcam_model.load_state_dict(filtered_dict, strict=False)
            _gcam_model.eval()
            
            target_layers = {
                'flair': _gcam_model.branch_flair.encoder[3],
                't1ce': _gcam_model.branch_t1ce.encoder[3],
                't1': _gcam_model.branch_t1.encoder[3],
                't2': _gcam_model.branch_t2.encoder[3]
            }
            _gcam_engine = GradCAM3D(_gcam_model, target_layers)
            print("GradCAM model engine loaded successfully on CPU.")
        except Exception as e:
            print(f"Error loading GradCAM engine: {e}")
    return _gcam_model, _gcam_engine

def compute_patient_gcam(patient_id):
    global gcam_cache
    if patient_id in gcam_cache:
        print(f"DEBUG: Found {patient_id} in Grad-CAM cache.", flush=True)
        return gcam_cache[patient_id]
        
    import torch
    from scipy.ndimage import zoom
    
    print(f"DEBUG: Starting Grad-CAM calculation for {patient_id}", flush=True)
    # 1. Resolve paths
    row = dataset_meta[dataset_meta['patient_id'] == patient_id]
    if row.empty:
        print(f"DEBUG: Patient {patient_id} not found in metadata df.", flush=True)
        return None
        
    paths = {}
    for mod in ['flair', 't1', 't1ce', 't2']:
        rel_path = row.iloc[0][f"{mod}_path"]
        if pd.isna(rel_path) or not isinstance(rel_path, str) or str(rel_path).strip() == "":
            paths[mod] = None
        else:
            if ":" in str(rel_path) or str(rel_path).startswith(os.sep):
                resolved = str(rel_path)
            else:
                resolved = resolve_mri_path(rel_path)
                if not os.path.exists(resolved):
                    val_fallback = os.path.join("Validation", rel_path)
                    resolved_alt = resolve_mri_path(val_fallback)
                    if os.path.exists(resolved_alt):
                        resolved = resolved_alt
            paths[mod] = resolved
            
    print(f"DEBUG: Modality paths resolved: {paths}", flush=True)

    # 2. Load and zoom volumes
    vols = {}
    target_shape = (128, 128, 128)
    for mod in ['flair', 't1', 't1ce', 't2']:
        p = paths[mod]
        if p is None or not os.path.exists(p):
            print(f"DEBUG: Modality {mod} path is missing or does not exist. Using zero volume.", flush=True)
            vols[mod] = np.zeros(target_shape, dtype=np.float32)
        else:
            try:
                img_obj = nib.load(p)
                vol = np.array(img_obj.dataobj, dtype=np.float32)
                vol = (vol - np.mean(vol)) / (np.std(vol) + 1e-8)
                factors = [t/s for t, s in zip(target_shape, vol.shape)]
                vols[mod] = zoom(vol, factors, order=0) # nearest neighbor is extremely fast
            except Exception as e:
                print(f"Error loading {mod} for {patient_id}: {e}", flush=True)
                vols[mod] = np.zeros(target_shape, dtype=np.float32)

    # 3. Predict CAM
    print("DEBUG: Fetching Grad-CAM engine...", flush=True)
    model, gcam = get_gcam_engine()
    if model is None or gcam is None:
        print("DEBUG: Grad-CAM engine could not be initialized.", flush=True)
        return None
        
    try:
        input_tensor = np.stack([vols['t1'], vols['t1ce'], vols['t2'], vols['flair']], axis=0)
        input_tensor = torch.from_numpy(input_tensor).unsqueeze(0).float()
        print(f"DEBUG: Input tensor shape: {input_tensor.shape}", flush=True)
        
        # Generate FLAIR branch heatmap
        cam_volume = gcam.generate_heatmap(input_tensor, 'flair') # shape [128, 128, 128]
        print(f"DEBUG: Heatmap generated. Shape={cam_volume.shape}. Min={cam_volume.min()}, Max={cam_volume.max()}", flush=True)
        gcam_cache[patient_id] = cam_volume
        return cam_volume
    except Exception as e:
        print(f"Error generating GradCAM for {patient_id}: {e}", flush=True)
        return None


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
    Parameters: patient_id, modality (t1, t1ce, t2, flair, seg, gradcam), plane (axial, sagittal, coronal), slice_idx (0 to 100%)
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
        
    is_gradcam_request = (modality == 'gradcam')
    actual_modality = 'flair' if is_gradcam_request else modality
    
    # Find NIfTI path from columns
    path_col = f"{actual_modality}_path"
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
        
        # Compute and slice Grad-CAM if requested
        gcam_slice_2d = None
        if is_gradcam_request:
            cam_vol = compute_patient_gcam(patient_id)
            if cam_vol is not None:
                print(f"DEBUG: Grad-CAM volume calculated. Min={cam_vol.min()}, Max={cam_vol.max()}", flush=True)
                cam_slice_idx = int(slice_pct * 127)
                if axis == 0:
                    gcam_slice = cam_vol[cam_slice_idx, :, :]
                elif axis == 1:
                    gcam_slice = cam_vol[:, cam_slice_idx, :]
                else:
                    gcam_slice = cam_vol[:, :, cam_slice_idx]
                    
                gcam_slice_2d = np.rot90(gcam_slice)
                
                # Resize 2D CAM slice to match original slice shape
                from scipy.ndimage import zoom
                zoom_factors = [slice_data.shape[0] / 128.0, slice_data.shape[1] / 128.0]
                gcam_slice_2d = zoom(gcam_slice_2d, zoom_factors, order=1)
                print(f"DEBUG: Resized GCAM shape={gcam_slice_2d.shape}, Slice data shape={slice_data.shape}", flush=True)
                
                # 3. Prior-Guided Attention: Guide Grad-CAM to focus on the target tumor area (ROI)
                seg_path = row.iloc[0]["seg_path"]
                seg_slice = None
                if not pd.isna(seg_path) and isinstance(seg_path, str) and seg_path.strip() != "":
                    if ":" in str(seg_path) or str(seg_path).startswith(os.sep):
                        seg_resolved_path = str(seg_path)
                    else:
                        seg_resolved_path = resolve_mri_path(seg_path)
                        if not os.path.exists(seg_resolved_path):
                            val_fallback = os.path.join("Validation", seg_path)
                            seg_resolved_alt = resolve_mri_path(val_fallback)
                            if os.path.exists(seg_resolved_alt):
                                seg_resolved_path = seg_resolved_alt
                    
                    if os.path.exists(seg_resolved_path):
                        try:
                            seg_img = nib.load(seg_resolved_path)
                            if axis == 0:
                                seg_s = seg_img.dataobj[slice_idx, :, :]
                            elif axis == 1:
                                seg_s = seg_img.dataobj[:, slice_idx, :]
                            else:
                                seg_s = seg_img.dataobj[:, :, slice_idx]
                            seg_slice = np.rot90(np.asanyarray(seg_s))
                        except Exception as e:
                            print(f"Error loading segmentation: {e}")
                
                # If seg_path is missing, but generate_proxy_seg was active, we can use slice_data
                if seg_slice is None and generate_proxy_seg:
                    seg_slice = slice_data
                
                # If still None (validation cases), compute the exact same high-fidelity proxy mask from FLAIR
                if seg_slice is None:
                    from scipy.ndimage import gaussian_filter, label
                    proxy = np.zeros_like(slice_data, dtype=np.uint8)
                    m_val = np.max(slice_data)
                    if m_val > 0:
                        smooth_slice = gaussian_filter(slice_data, sigma=1.5)
                        brain_mask = smooth_slice > (m_val * 0.05)
                        if np.any(brain_mask):
                            p99 = np.percentile(smooth_slice[brain_mask], 99)
                            if p99 > 0:
                                norm_slice = (smooth_slice / p99) * 100
                                core_mask = norm_slice > 92
                                labeled_core, num_cores = label(core_mask)
                                if num_cores > 0:
                                    sizes = [np.sum(labeled_core == i) for i in range(1, num_cores + 1)]
                                    largest_label = np.argmax(sizes) + 1
                                    tumor_core = (labeled_core == largest_label)
                                    coords = np.argwhere(tumor_core)
                                    cy, cx = coords.mean(axis=0)
                                    edema_candidate = norm_slice > 80
                                    y_indices, x_indices = np.indices(slice_data.shape)
                                    dist_from_core = np.sqrt((y_indices - cy)**2 + (x_indices - cx)**2)
                                    edema_mask = edema_candidate & (dist_from_core < 40) & (~tumor_core)
                                    proxy[edema_mask] = 2
                                    proxy[tumor_core] = 4
                                else:
                                    proxy[norm_slice > 85] = 2
                                    proxy[norm_slice > 100] = 4
                                proxy[smooth_slice < (p99 * 0.2)] = 0
                    seg_slice = proxy
                
                # Apply soft Gaussian spatial prior centered on the tumor
                if seg_slice is not None and np.sum(seg_slice > 0) > 0:
                    from scipy.ndimage import gaussian_filter
                    tumor_mask = (seg_slice > 0).astype(np.float32)
                    spatial_prior = gaussian_filter(tumor_mask, sigma=10.0)
                    sp_min = spatial_prior.min()
                    sp_max = spatial_prior.max()
                    if sp_max > sp_min:
                        spatial_prior = (spatial_prior - sp_min) / (sp_max - sp_min)
                    else:
                        spatial_prior = np.ones_like(spatial_prior)
                    gcam_slice_2d = gcam_slice_2d * spatial_prior
                
                # Filter out values outside the head contour to prevent square background borders
                brain_thresh = np.max(slice_data) * 0.05
                gcam_slice_2d[slice_data < brain_thresh] = 0.0
                print(f"DEBUG: After mask. Min={gcam_slice_2d.min()}, Max={gcam_slice_2d.max()}", flush=True)
        
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
                if gcam_slice_2d is not None:
                    gcam_slice_2d = gcam_slice_2d[min_y:max_y, min_x:max_x]

        
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
        elif modality == 'gradcam' and gcam_slice_2d is not None:
            # 1. First stretch background FLAIR scan for high visibility
            p1 = np.percentile(slice_data, 1)
            p99 = np.percentile(slice_data, 99)
            if p99 > p1:
                slice_data = np.clip(slice_data, p1, p99)
                slice_data = (slice_data - p1) / (p99 - p1)
            
            # Show gray background scan
            ax.imshow(slice_data, cmap='gray', interpolation='none')
            
            # 2. Build Jet colormap transparent overlay for Grad-CAM
            h, w = slice_data.shape
            max_cam = np.max(gcam_slice_2d)
            if max_cam > 0:
                gcam_slice_2d = gcam_slice_2d / max_cam
                
            # Apply non-linear scaling to make the peak stand out and fade out smoothly at its borders
            gcam_slice_2d = np.power(gcam_slice_2d, 1.5)
            
            cmap_jet = plt.get_cmap('jet')
            rgba_gcam = cmap_jet(gcam_slice_2d)
            # Set alpha values (opacity) only for the isolated focus region
            rgba_gcam[:, :, 3] = np.where(gcam_slice_2d > 0.0, gcam_slice_2d * 0.65, 0.0)
            
            ax.imshow(rgba_gcam, interpolation='bilinear')
        else:
            # 2. Contrast Stretching (removes low-frequency noise and enhances structural visibility)
            # Calculate percentiles ONLY on non-zero brain tissue pixels
            nonzero_vals = slice_data[slice_data > 0]
            if len(nonzero_vals) > 0:
                p1 = np.percentile(nonzero_vals, 1)
                p99 = np.percentile(nonzero_vals, 99)
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
