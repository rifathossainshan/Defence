import sys
import os

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from retrieval.extract_embeddings_hybrid import extract_hybrid_full

if __name__ == "__main__":
    TCGA_CONFIG = {
        "model_path": "outputs/checkpoints/multibranch_hybrid_best.pth",
        "csv_path": "data/metadata/metadata_testing_tcga.csv", # New TCGA metadata
        "base_dir": ".", 
        "batch_size": 4,
        "output_size": 64,
        "sample_size": None, 
        "output_npy": "outputs/embeddings/tcga_embeddings.npy", # Separate output
        "output_csv": "outputs/embeddings/tcga_metadata.csv",
        "output_recon": "outputs/embeddings/tcga_recon_sample.npy"
    }
    
    print("Starting TCGA Embedding Extraction...")
    extract_hybrid_full(TCGA_CONFIG)
    print("TCGA Extraction Finished.")
