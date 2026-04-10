import faiss
import numpy as np
import pandas as pd
import os
import argparse
import sys
from pathlib import Path

def retrieve_topk(query_id, index_path, mapping_csv, embeddings_npy, output_csv, top_k=5, exclude_self=True):
    """
    Search for similar cases for a given patient_id.
    """
    # 1. Load Index and Metadata
    if not os.path.exists(index_path):
        print(f"Error: Index not found at {index_path}")
        return
    
    index = faiss.read_index(index_path)
    mapping_df = pd.read_csv(mapping_csv)
    embeddings = np.load(embeddings_npy).astype("float32")
    faiss.normalize_L2(embeddings)

    # 2. Find Query Embedding
    query_row = mapping_df[mapping_df["patient_id"] == query_id]
    if query_row.empty:
        print(f"Error: Patient ID {query_id} not found in index metadata.")
        return

    # In our index_metadata, 'faiss_index' column or the index itself maps to embeddings
    query_faiss_idx = query_row.iloc[0].name # row index in mapping_df
    query_vec = embeddings[query_faiss_idx:query_faiss_idx+1] # [1, 512]

    # 3. Search FAISS
    search_k = top_k + 1 if exclude_self else top_k
    similarities, indices = index.search(query_vec, search_k)
    similarities = similarities[0]
    indices = indices[0]

    # 4. Filter and Format Results
    results = []
    rank = 1
    for i, idx in enumerate(indices):
        patient_id = mapping_df.iloc[idx]["patient_id"]
        dataset = mapping_df.iloc[idx]["dataset"]
        score = similarities[i]

        if exclude_self and patient_id == query_id:
            continue
        
        results.append({
            "query_patient_id": query_id,
            "retrieved_patient_id": patient_id,
            "retrieved_dataset": dataset,
            "similarity_score": round(float(score), 4),
            "rank": rank
        })
        rank += 1
        if rank > top_k:
            break

    # 5. Save and Display
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)
        results_df.to_csv(output_csv, index=False)

        print(f"\nTop-{top_k} Retrieval Results for Query: {query_id}")
        print("="*70)
        print(results_df.to_string(index=False))
        print("="*70)
        print(f"Results saved to: {output_csv}")
    else:
        print("No results found.")

    return results_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve Top-K similar cases")
    parser.add_argument("--query_id", type=str, help="Patient ID to query")
    parser.add_argument("--index_path", type=str, default="outputs/faiss/faiss.index")
    parser.add_argument("--mapping_csv", type=str, default="outputs/faiss/index_metadata.csv")
    parser.add_argument("--embeddings_npy", type=str, default="outputs/embeddings/embeddings.npy")
    parser.add_argument("--output_csv", type=str, default="outputs/faiss/retrieval_results.csv")
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--include_self", action="store_true", help="Include the query case in results")
    
    args = parser.parse_args()
    
    # If no query_id provided, pick the first one from mapping for demo
    q_id = args.query_id
    if not q_id:
        if os.path.exists(args.mapping_csv):
            temp_df = pd.read_csv(args.mapping_csv)
            q_id = temp_df.iloc[0]["patient_id"]
        else:
            print("Error: No query_id provided and mapping file not found.")
            sys.exit(1)
            
    retrieve_topk(
        query_id=q_id,
        index_path=args.index_path,
        mapping_csv=args.mapping_csv,
        embeddings_npy=args.embeddings_npy,
        output_csv=args.output_csv,
        top_k=args.top_k,
        exclude_self=not args.include_self
    )
