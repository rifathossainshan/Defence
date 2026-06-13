import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def run_cross_dataset_check():
    # 1. Load Databases
    # Reference/Train: BraTS 2021
    ref_emb_path = "outputs/embeddings/hybrid_embeddings.npy"
    ref_meta_path = "outputs/embeddings/hybrid_metadata.csv"
    
    # Query/Test: TCGA
    test_emb_path = "outputs/embeddings/tcga_embeddings.npy"
    test_meta_path = "outputs/embeddings/tcga_metadata.csv"
    
    if not os.path.exists(ref_emb_path) or not os.path.exists(test_emb_path):
        print("Error: Missing embedding files.")
        return
        
    ref_embeddings = np.load(ref_emb_path)
    ref_meta = pd.read_csv(ref_meta_path)
    
    test_embeddings = np.load(test_emb_path)
    test_meta = pd.read_csv(test_meta_path)
    
    print(f"Reference Database (BraTS): {len(ref_embeddings)} cases")
    print(f"Query Database (TCGA): {len(test_embeddings)} cases")
    
    # 2. Similarity Search
    # Compute similarity between TCGA (rows) and BraTS (cols)
    sim_matrix = cosine_similarity(test_embeddings, ref_embeddings)
    
    # Showcase some matches
    test_cases = [0, 50, 102] # Mix of samples
    
    print("\n--- CROSS-DATASET RETRIEVAL (TCGA -> BraTS) ---")
    print(f"{'TCGA Query ID':<20} | {'BraTS Match ID':<20} | {'Similarity':<10}")
    print("-" * 65)
    
    for idx in test_cases:
        if idx >= len(test_embeddings): continue
        
        query_id = test_meta.iloc[idx]['patient_id']
        sims = sim_matrix[idx]
        
        # Sort similarities in descending order
        sorted_indices = np.argsort(sims)[::-1]
        
        # Find best match that is not the query patient itself
        match_id = None
        match_score = 0.0
        for m_idx in sorted_indices:
            candidate_id = ref_meta.iloc[m_idx]['patient_id']
            if candidate_id == query_id:
                continue
            match_id = candidate_id
            match_score = sims[m_idx]
            break
            
        print(f"{query_id:<20} | {match_id:<20} | {match_score:.4f}")

    print("\n[VERDICT] High similarity scores (>0.9) indicate successful knowledge transfer.")

if __name__ == "__main__":
    run_cross_dataset_check()
