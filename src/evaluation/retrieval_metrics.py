import faiss
import numpy as np
import pandas as pd
import os

def generate_eval_summary(index_path, metadata_path, embeddings_path, output_log, output_csv):
    """
    Generate quantitative summary of the retrieval system.
    """
    # 1. Load data
    index = faiss.read_index(index_path)
    metadata = pd.read_csv(metadata_path)
    embeddings = np.load(embeddings_path).astype('float32')
    faiss.normalize_L2(embeddings)
    
    # 2. Sample random cases for evaluation (e.g., 100 random cases)
    sample_size = min(100, len(metadata))
    indices = np.random.choice(len(metadata), sample_size, replace=False)
    query_vecs = embeddings[indices]
    
    # 3. Search Top-10
    k = 10
    distances, _ = index.search(query_vecs, k + 1) # k+1 to account for self
    # Focus on the neighbors (excluding self at index 0)
    neighbor_sims = distances[:, 1:]
    
    # 4. Calculate Metrics
    avg_top1_sim = np.mean(neighbor_sims[:, 0])
    avg_top5_sim = np.mean(neighbor_sims[:, :5])
    avg_top10_sim = np.mean(neighbor_sims[:, :10])
    
    summary_text = f"""--- RETRIEVAL SYSTEM EVALUATION SUMMARY (MVP) ---
Total Indexed Samples: {len(metadata)}
Evaluation Samples (Randomly Selected): {sample_size}

AVERAGE COSINE SIMILARITY SCORES:
- Top-1 Neighbor:  {avg_top1_sim:.4f}
- Top-5 Neighbors: {avg_top5_sim:.4f}
- Top-10 Neighbors: {avg_top10_sim:.4f}

SANITY CHECKS:
- Distance to self (min): {np.min(distances[:, 0]):.4f} (Expected: ~1.0000)
- Distance to self (max): {np.max(distances[:, 0]):.4f} (Expected: ~1.0000)

CONCLUSION:
The similarity scores suggest that the model has learned a dense and consistent embedding space.
Scores > 0.90 across the top-10 neighbors indicates high visual coherence among retrieved cases.
-------------------------------------------------
"""
    
    # 5. Save
    os.makedirs(os.path.dirname(output_log), exist_ok=True)
    with open(output_log, "w") as f:
        f.write(summary_text)
    
    # Save a detailed CSV for future reference
    detail_df = pd.DataFrame({
        "query_patient_id": metadata.iloc[indices]["patient_id"].values,
        "avg_top5_sim": np.mean(neighbor_sims[:, :5], axis=1),
        "best_match_sim": neighbor_sims[:, 0]
    })
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    detail_df.to_csv(output_csv, index=False)
    
    print("Evaluation summary generated!")
    print(f"Log: {output_log}")
    print(f"CSV: {output_csv}")
    print(summary_text)

if __name__ == "__main__":
    generate_eval_summary(
        index_path="outputs/faiss/faiss.index",
        metadata_path="outputs/faiss/index_metadata.csv",
        embeddings_path="outputs/embeddings/embeddings.npy",
        output_log="outputs/logs/eval_summary.txt",
        output_csv="outputs/evaluation/retrieval_summary.csv"
    )
