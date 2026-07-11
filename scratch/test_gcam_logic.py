import numpy as np
from scipy.ndimage import label

# Create dummy gcam_slice
gcam_slice_2d = np.zeros((10, 10))
gcam_slice_2d[2:5, 2:5] = 0.5
gcam_slice_2d[3, 3] = 1.0 # max point
gcam_slice_2d[8, 8] = 0.9

max_cam = np.max(gcam_slice_2d)
if max_cam > 0:
    gcam_slice_2d = gcam_slice_2d / max_cam
    
print("Before logic:", gcam_slice_2d)

threshold = 0.80
binary = gcam_slice_2d >= threshold

labeled, num_features = label(binary)
print("Labeled:", labeled)

if num_features > 0:
    max_idx = np.unravel_index(np.argmax(gcam_slice_2d), gcam_slice_2d.shape)
    print("max_idx:", max_idx)
    tumor_label = labeled[max_idx]
    print("tumor_label:", tumor_label)
    gcam_slice_2d[labeled != tumor_label] = 0.0
    
print("After logic:", gcam_slice_2d)
