import faiss
import numpy as np
import pandas as pd
import os

def build_index(embeddings_path, metadata_path, output_index_path):
    print(f"--- BUILDING FAISS INDEX ---")
    
    if not os.path.exists(embeddings_path):
        print(f"Error: Embeddings file not found at {embeddings_path}")
        return

    # 1. Load Embeddings
    embeddings = np.load(embeddings_path).astype('float32')
    print(f"Loaded {embeddings.shape[0]} embeddings of dimension {embeddings.shape[1]}")

    # 2. Normalize (for Cosine Similarity using Inner Product)
    faiss.normalize_L2(embeddings)

    # 3. Create Index
    d = embeddings.shape[1]
    index = faiss.IndexFlatIP(d) # Inner Product index for cosine similarity
    index.add(embeddings)

    # 4. Save Index
    os.makedirs(os.path.dirname(output_index_path), exist_ok=True)
    faiss.write_index(index, output_index_path)
    print(f"[SUCCESS] FAISS Index saved to {output_index_path}")
    print(f"Total Vectors Indexed: {index.ntotal}")

if __name__ == "__main__":
    BUILD_CONFIG = {
        "embeddings_path": "outputs/embeddings/hybrid_embeddings.npy",
        "metadata_path": "outputs/embeddings/hybrid_metadata.csv",
        "output_index_path": "outputs/embeddings/hybrid_faiss.index"
    }
    build_index(**BUILD_CONFIG)
