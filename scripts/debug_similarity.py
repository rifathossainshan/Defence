import numpy as np
import pandas as pd
import faiss
import os

def debug_similarity(emb_path, meta_path):
    print("--- GATE CHECK 2.5: SIMILARITY DEBUG REPORT ---")
    
    # 1. Load Data
    if not os.path.exists(emb_path):
        print(f"Error: Embeddings not found at {emb_path}")
        return
        
    emb = np.load(emb_path)
    meta = pd.read_csv(meta_path)
    
    # 2. Basic Stats
    print(f"Embedding Shape: {emb.shape}")
    print(f"Min/Max/Mean: {emb.min():.6f} / {emb.max():.6f} / {emb.mean():.6f}")
    
    # 3. Numeric Spread (Variance per Dimension)
    variances = np.var(emb, axis=0)
    print(f"Avg Per-Dimension Variance: {np.mean(variances):.8f}")
    print(f"Max Per-Dimension Variance: {np.max(variances):.8f}")
    
    # 4. Pairwise Cosine Matrix (First 10)
    print("\nPairwise Cosine Similarity Matrix (First 5x5):")
    sample_emb = emb[:10].copy()
    norm = np.linalg.norm(sample_emb, axis=1, keepdims=True)
    sample_emb_norm = sample_emb / (norm + 1e-8)
    sim_matrix = np.matmul(sample_emb_norm, sample_emb_norm.T)
    print(np.round(sim_matrix[:5, :5], 6))

    # 5. Top-K Search Audit (with raw scores)
    print("\nTop-5 Search Audit (3 Cases):")
    index = faiss.IndexFlatIP(emb.shape[1])
    faiss.normalize_L2(emb)
    index.add(emb)
    
    for q_idx in [0, 25, 49]:
        q_id = meta.iloc[q_idx]["patient_id"]
        q_vec = emb[q_idx:q_idx+1]
        
        sims, ids = index.search(q_vec, 7) # Get 7 to see self and neighbors
        sims, ids = sims[0], ids[0]
        
        print(f"\nQuery Case: {q_id} (Index {q_idx})")
        print(f"Raw IDs returned: {ids}")
        print(f"Raw Scores returned: {sims}")
        
        # Verify self-match exclusion logic
        neighbors = []
        n_scores = []
        for i, idx in enumerate(ids):
            pid = meta.iloc[idx]["patient_id"]
            if pid == q_id:
                print(f"  [AUDIT] Self-match found at rank {i} (Correct)")
                continue
            neighbors.append(pid)
            n_scores.append(sims[i])
            if len(neighbors) >= 5: break
            
        print(f"  Filter Top-5 Neighbor Scores: {n_scores}")

    print("\n--- DEBUG COMPLETE ---")

if __name__ == "__main__":
    debug_similarity("outputs/embeddings/minieval_embeddings.npy", "outputs/embeddings/minieval_metadata.csv")
