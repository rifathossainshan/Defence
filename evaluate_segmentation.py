import numpy as np
import pandas as pd
import nibabel as nib
import os
from scipy.ndimage import gaussian_filter, label
import warnings
warnings.filterwarnings('ignore')

def compute_proxy_mask_3d(flair_vol):
    """
    আমাদের ড্যাশবোর্ডের on-the-fly প্রক্সি অ্যালগরিদমের ৩ডি ভার্সন,
    যা FLAIR সিকোয়েন্স থেকে টিউমার ডিটেক্ট করে।
    """
    proxy = np.zeros_like(flair_vol, dtype=np.uint8)
    m_val = np.max(flair_vol)
    if m_val > 0:
        # Gaussian smoothing
        smooth_vol = gaussian_filter(flair_vol, sigma=1.5)
        brain_mask = smooth_vol > (m_val * 0.05)
        if np.any(brain_mask):
            p99 = np.percentile(smooth_vol[brain_mask], 99)
            if p99 > 0:
                norm_vol = (smooth_vol / p99) * 100
                core_mask = norm_vol > 92
                labeled_core, num_cores = label(core_mask)
                if num_cores > 0:
                    sizes = [np.sum(labeled_core == i) for i in range(1, num_cores + 1)]
                    largest_label = np.argmax(sizes) + 1
                    tumor_core = (labeled_core == largest_label)
                    
                    coords = np.argwhere(tumor_core)
                    cz, cy, cx = coords.mean(axis=0)
                    
                    edema_candidate = norm_vol > 80
                    z_indices, y_indices, x_indices = np.indices(flair_vol.shape)
                    dist_from_core = np.sqrt((z_indices - cz)**2 + (y_indices - cy)**2 + (x_indices - cx)**2)
                    
                    edema_mask = edema_candidate & (dist_from_core < 40) & (~tumor_core)
                    
                    proxy[edema_mask] = 2
                    proxy[tumor_core] = 4
                else:
                    proxy[norm_vol > 85] = 2
                    proxy[norm_vol > 100] = 4
                
                proxy[smooth_vol < (p99 * 0.2)] = 0
    return proxy

def compute_dice(pred, target):
    """
    Whole Tumor (WT) এর জন্য Dice Similarity Coefficient হিসেব করে।
    """
    pred_wt = (pred > 0)
    target_wt = (target > 0)
    intersection = np.sum(pred_wt & target_wt)
    total = np.sum(pred_wt) + np.sum(target_wt)
    if total == 0: 
        return 1.0
    return (2. * intersection) / total

def evaluate_test_set():
    print("Loading metadata...")
    df = pd.read_csv('data/metadata/metadata_brats2021.csv')
    test_df = df[df['split'] == 'test']
    print(f"Found {len(test_df)} cases in the Testing set.\n")
    
    dice_scores = []
    
    print("Starting evaluation (this may take a few minutes depending on your CPU)...")
    for idx, row in test_df.iterrows():
        pid = row['patient_id']
        flair_path = str(row['flair_path'])
        seg_path = str(row['seg_path'])
        
        # Check if files actually exist
        if not os.path.exists(flair_path):
            continue
        if not os.path.exists(seg_path):
            print(f"Missing Ground Truth SEG for {pid}, skipping.")
            continue
            
        try:
            # Load FLAIR and Ground Truth Segmentations
            flair_vol = nib.load(flair_path).get_fdata()
            gt_vol = nib.load(seg_path).get_fdata()
            
            # Predict using our dashboard's logic
            pred_vol = compute_proxy_mask_3d(flair_vol)
            
            # Evaluate Accuracy
            dice = compute_dice(pred_vol, gt_vol)
            dice_scores.append(dice)
            print(f"Patient: {pid} | Dice Score: {dice:.4f} ({(dice*100):.1f}%)")
        except Exception as e:
            print(f"Error evaluating {pid}: {e}")
            
    if len(dice_scores) > 0:
        mean_dice = np.mean(dice_scores)
        print(f"\n==================================================")
        print(f"              FINAL EVALUATION REPORT             ")
        print(f"==================================================")
        print(f"Total Test Cases Evaluated: {len(dice_scores)}")
        print(f"Average Whole Tumor (WT) Dice Score: {mean_dice:.4f} ({(mean_dice*100):.2f}%)")
        print(f"==================================================")
        if mean_dice > 0.70:
            print("Status: Excellent! The on-the-fly proxy algorithm is highly accurate.")
        else:
            print("Status: Moderate accuracy. Consider fine-tuning intensity thresholds.")

if __name__ == '__main__':
    evaluate_test_set()
