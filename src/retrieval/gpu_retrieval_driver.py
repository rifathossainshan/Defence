import os
import sys
import subprocess

# Add current dir to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_gpu_pipeline():
    print("==========================================")
    print("   GLIOMA MRI GPU RETRIEVAL PIPELINE      ")
    print("==========================================")
    
    # Path to our new high-performance environment
    python_exe = r"C:\gpu_env_311\Scripts\python.exe"
    
    if not os.path.exists(python_exe):
        print(f"Error: GPU environment NOT found at {python_exe}")
        return

    # 1. Extract Embeddings (Full Scale)
    print("\n[STEP 1] Extracting Embeddings for Full Dataset (GPU)...")
    extract_script = os.path.join("src", "retrieval", "extract_embeddings_hybrid.py")
    
    # Run extraction
    try:
        subprocess.run([python_exe, extract_script], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during extraction: {e}")
        return

    print("\n[STEP 2] Pipeline Complete!")
    print("Embeddings are saved in: outputs/embeddings/hybrid_embeddings.npy")
    print("You can now build the FAISS index or run similarity search.")

if __name__ == "__main__":
    run_gpu_pipeline()
