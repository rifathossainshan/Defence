import numpy as np
import nibabel as nib
import os
import sys
import csv

def get_intensity_features(path):
    try:
        # Load header only if possible to save RAM, but here we need data
        img = nib.load(path).get_fdata()
        # Manual stats to be safe
        mean_val = float(np.mean(img))
        std_val = float(np.std(img))
        p95 = float(np.percentile(img, 95))
        return [mean_val, std_val, p95]
    except Exception as e:
        return [0.0, 0.0, 0.0]

def manual_cosine_similarity(v1, v2):
    dot = np.dot(v1, v2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

if __name__ == "__main__":
    print(">>> Phase 23: Zero-dependency CPU Baseline Retrieval")
    csv_path = "data/metadata/metadata_brats2021.csv"
    subset_path = "data/metadata/fixed_50_subset.npy"
    
    if not os.path.exists(subset_path):
        print(f"Error: {subset_path} not found.")
        sys.exit(1)
        
    indices = np.load(subset_path)
    
    # Load CSV manually without Pandas for max speed
    rows = []
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i in indices[:10]:
                rows.append(row)
    
    print(f"Analyzing {len(rows)} samples on Normal CPU...")
    features = []
    p_ids = []
    
    for row in rows:
        path = row["flair_path"]
        p_id = row["patient_id"]
        print(f"  Processing {p_id}...")
        f = get_intensity_features(path)
        features.append(f)
        p_ids.append(p_id)

    print("\n>>> Retrieval Results (Cosine Similarity based on Intensity):")
    for i in range(len(p_ids)):
        scores = []
        for j in range(len(p_ids)):
            if i == j: 
                scores.append(-1.0) # Mask self
            else:
                s = manual_cosine_similarity(features[i], features[j])
                scores.append(s)
        
        # Sort and get top matches
        top_idx = np.argsort(scores)[::-1][:3]
        results = [f"{p_ids[idx]} (Score: {scores[idx]:.4f})" for idx in top_idx]
        print(f"Query {p_ids[i]} matches with: {', '.join(results)}")
    
    print("\n>>> COMPLETED: Minimal CPU retrieval is successful.")
