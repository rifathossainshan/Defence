import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path("e:/Cse Engineering/11Defense/src")))

from datasets.ssl_dataset import BraTSSSLDataset
from preprocessing.simple_transforms import SimpleSSLTransform

def verify_phase5():
    CSV_PATH = "e:/Cse Engineering/11Defense/data/metadata/metadata_brats2021.csv"
    BASE_DIR = "e:/Cse Engineering/11Defense"

    transform = SimpleSSLTransform()
    
    print("Initializing Dataset...")
    dataset = BraTSSSLDataset(
        csv_file=CSV_PATH,
        base_dir=BASE_DIR,
        split="all",
        target_size=(128, 128, 128),
        transform=transform,
    )

    print("Initializing DataLoader (batch_size=4)...")
    loader = DataLoader(
        dataset, 
        batch_size=4, 
        shuffle=True, 
        num_workers=0,
        pin_memory=False
    )

    print("Getting first batch...")
    batch = next(iter(loader))
    
    view1_shape = batch["view1"].shape
    view2_shape = batch["view2"].shape
    ids = batch["id"]

    print("\n" + "="*30)
    print(" PHASE 5 CHECKPOINT ")
    print("="*30)
    print(f"batch['view1'].shape: {view1_shape}")
    print(f"batch['view2'].shape: {view2_shape}")
    print(f"Batch IDs: {ids}")
    print("="*30)

    # Basic assertion
    assert view1_shape == (4, 4, 128, 128, 128), f"Expected (4, 4, 128, 128, 128) but got {view1_shape}"
    print("\nPhase 5: SUCCESS! Loader is stable and shapes are correct.")

if __name__ == "__main__":
    verify_phase5()
