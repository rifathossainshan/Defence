import torch
import sys

def check_gpu():
    print(f"--- GPU DIAGNOSTIC REPORT ---")
    cuda_avail = torch.cuda.is_available()
    print(f"CUDA Available: {cuda_avail}")
    
    if cuda_avail:
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch CUDA Arch: {torch.cuda.get_arch_list()[:5]}...")
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device Name: {torch.cuda.get_device_name(0)}")
        
        # Memory Check
        t = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"Total Dedicated Memory: {t:.2f} GB")
    else:
        print("ALERT: CUDA NOT DETECTED BY PYTORCH.")
        print("Possible reasons: Driver mismatch or CPU-only PyTorch build.")

if __name__ == "__main__":
    check_gpu()
