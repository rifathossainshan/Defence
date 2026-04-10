import faiss
import numpy as np
import pandas as pd
import os
import argparse
from pathlib import Path

def build_faiss_index(embeddings_npy, embeddings_metadata, output_dir):
    """
    Builds a FAISS IndexFlatIP for cosine similarity search.
    """
    # 1. Load embeddings and metadata
    if not os.path.exists(embeddings_npy):
        print(f"Error: Embeddings file not found at {embeddings_npy}")
        return

    embeddings = np.load(embeddings_npy).astype('float32')
    metadata = pd.read_csv(embeddings_metadata)
    
    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # 2. L2 Normalize embeddings
    faiss.normalize_L2(embeddings)
    
    # 3. Create Index
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    
    # 4. Add embeddings to index
    index.add(embeddings)
    
    # 5. Save results
    os.makedirs(output_dir, exist_ok=True)
    
    index_path = os.path.join(output_dir, "faiss.index")
    faiss.write_index(index, index_path)
    
    mapping_df = metadata.copy()
    mapping_df["faiss_index"] = mapping_df.index
    mapping_df.to_csv(os.path.join(output_dir, "index_metadata.csv"), index=False)

    print(f"\n[SUCCESS] FAISS Index built with {index.ntotal} vectors.")
    print(f"Index saved to: {index_path}")
    print(f"Mapping saved to: {os.path.join(output_dir, 'index_metadata.csv')}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build FAISS Index")
    parser.add_argument("--embeddings_npy", type=str, default="outputs/embeddings/embeddings.npy")
    parser.add_argument("--embeddings_metadata", type=str, default="outputs/embeddings/embedding_metadata.csv")
    parser.add_argument("--output_dir", type=str, default="outputs/faiss")
    
    args = parser.parse_args()
    build_faiss_index(args.embeddings_npy, args.embeddings_metadata, args.output_dir)
