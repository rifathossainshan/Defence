import numpy as np

def get_bbox_from_seg(seg):
    """
    Finds the 3D bounding box of the non-zero region in the segmentation mask.
    Returns: (min_h, max_h, min_w, max_w, min_d, max_d) or None if empty.
    """
    coords = np.argwhere(seg > 0)
    if coords.size == 0:
        return None
    
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0) + 1
    return (min_coords[0], max_coords[0], min_coords[1], max_coords[1], min_coords[2], max_coords[2])

def expand_bbox(bbox, margin, shape):
    """
    Expands the bounding box by a given margin while staying within image bounds.
    """
    h_min, h_max, w_min, w_max, d_min, d_max = bbox
    img_h, img_w, img_d = shape
    
    h_min = max(0, h_min - margin)
    h_max = min(img_h, h_max + margin)
    
    w_min = max(0, w_min - margin)
    w_max = min(img_w, w_max + margin)
    
    d_min = max(0, d_min - margin)
    d_max = min(img_d, d_max + margin)
    
    return (h_min, h_max, w_min, w_max, d_min, d_max)

def get_lesion_center_crop_coords(bbox, target_size, shape):
    """
    Calculates the start and end coordinates for a target_size crop centered on the bbox.
    target_size: (H, W, D) e.g., (128, 128, 128)
    """
    h_min, h_max, w_min, w_max, d_min, d_max = bbox
    img_h, img_w, img_d = shape
    th, tw, td = target_size
    
    # Calculate center of bbox
    c_h, c_w, c_d = (h_min + h_max) // 2, (w_min + w_max) // 2, (d_min + d_max) // 2
    
    # Calculate crop bounds
    h_start = c_h - th // 2
    w_start = c_w - tw // 2
    d_start = c_d - td // 2
    
    # Adjust if out of bounds (Shift the window)
    h_start = max(0, min(h_start, img_h - th))
    w_start = max(0, min(w_start, img_w - tw))
    d_start = max(0, min(d_start, img_d - td))
    
    return (h_start, h_start + th, w_start, w_start + tw, d_start, d_start + td)
