import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import os

def verify_clinical_similarity():
    # 1. Load AI Results
    emb_path = "outputs/embeddings/tcga_embeddings.npy"
    meta_path = "outputs/embeddings/tcga_metadata.csv"
    
    if not os.path.exists(emb_path):
        print("Error: Embeddings not found.")
        return
        
    embeddings = np.load(emb_path)
    meta_df = pd.read_csv(meta_path)
    
    # 2. Load Radiomics Ground Truth
    gbm_csv = r"e:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-GBM\BraTS-TCGA-GBM\Pre-operative_TCGA_GBM_NIfTI_and_Segmentations\TCGA_GBM_radiomicFeatures.csv"
    lgg_csv = r"e:\Cse Engineering\11Defense\Testing\PKG - BraTS-TCGA-LGG\BraTS-TCGA-LGG\Pre-operative_TCGA_LGG_NIfTI_and_Segmentations\TCGA_LGG_radiomicFeatures.csv"
    
    gbm_rad = pd.read_csv(gbm_csv)
    lgg_rad = pd.read_csv(lgg_csv)
    rad_df = pd.concat([gbm_rad, lgg_rad], ignore_index=True)
    
    # Convert ID to match our patient_id (Handle suffixes if any)
    # The rad_df IDs are like TCGA-02-0006
    rad_df['ID'] = rad_df['ID'].astype(str).str.strip()
    meta_df['patient_id'] = meta_df['patient_id'].astype(str).str.strip()

    print(f"Total AI Embeddings: {len(embeddings)}")
    print(f"Total Radiomic Cases: {len(rad_df)}")
    
    # Calculate Similarity Matrix
    sim_matrix = cosine_similarity(embeddings)
    
    results = []
    # Pick some test cases (e.g. first 5)
    test_indices = [10, 50, 100, 150] # Mix of GBM and LGG
    
    print("\n--- CLINICAL SIMILARITY REPORT ---")
    print(f"{'Query ID':<15} | {'Match ID':<15} | {'Score':<6} | {'Vol % Diff (WT)':<12}")
    print("-" * 75)
    
    for idx in test_indices:
        if idx >= len(embeddings): continue
        
        query_id = meta_df.iloc[idx]['patient_id']
        
        # Get similarities for this query, exclude self
        sims = sim_matrix[idx].copy()
        sims[idx] = -1 # mask self
        
        match_idx = np.argmax(sims)
        match_score = sims[match_idx]
        match_id = meta_df.iloc[match_idx]['patient_id']
        
        # Lookup Clinical Data
        q_rad = rad_df[rad_df['ID'] == query_id]
        m_rad = rad_df[rad_df['ID'] == match_id]
        
        if len(q_rad) > 0 and len(m_rad) > 0:
            q_vol = q_rad.iloc[0]['VOLUME_WT']
            m_vol = m_rad.iloc[0]['VOLUME_WT']
            
            # Calculate % difference in volume
            if q_vol > 0:
                p_diff = abs(q_vol - m_vol) / q_vol * 100
                p_diff_str = f"{p_diff:.1f}%"
            else:
                p_diff_str = "N/A"
                
            print(f"{query_id:<15} | {match_id:<15} | {match_score:.3f} | {p_diff_str:<12}")
            results.append({
                "query": query_id,
                "match": match_id,
                "score": match_score,
                "vol_diff": p_diff_str
            })
        else:
            print(f"{query_id:<15} | {match_id:<15} | {match_score:.3f} | No Rad Data")

    print("\n[CONCLUSION] If Vol % Diff is low (< 30%), the retrieval is clinically consistent.")

if __name__ == "__main__":
    verify_clinical_similarity()
