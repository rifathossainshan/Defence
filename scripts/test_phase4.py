import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(str(Path("e:/Cse Engineering/11Defense/src")))

from datasets.ssl_dataset import BraTSSSLDataset
from preprocessing.simple_transforms import SimpleSSLTransform

def test_phase4():
    CSV_PATH = "e:/Cse Engineering/11Defense/data/metadata/metadata_brats2021.csv"
    BASE_DIR = "e:/Cse Engineering/11Defense" # Paths in CSV are relative to root

    transform = SimpleSSLTransform(
        flip_prob=1.0, # Always flip for testing
        noise_prob=1.0, # Always noise for testing
        noise_std=0.01,
    )

    print("Initializing Dataset...")
    dataset = BraTSSSLDataset(
        csv_file=CSV_PATH,
        base_dir=BASE_DIR,
        split="all", # Use all cases for testing
        target_size=(128, 128, 128),
        use_seg=True,
        apply_crop=True,
        apply_normalize=True,
        transform=transform,
    )

    print(f"Dataset size: {len(dataset)}")
    
    print("Loading first sample...")
    sample = dataset[0]
    
    view1 = sample["view1"]
    view2 = sample["view2"]
    p_id = sample["id"]
    ds = sample["dataset"]
    seg = sample["seg"]

    print(f"\nVerification Results for Patient: {p_id} ({ds})")
    print(f"View 1 shape: {view1.shape}")
    print(f"View 2 shape: {view2.shape}")
    if seg is not None:
        print(f"Seg shape: {seg.shape}")
    
    # Assertions for Phase 4 completion
    assert view1.shape == (4, 128, 128, 128), f"Error: View 1 shape is {view1.shape}"
    assert view2.shape == (4, 128, 128, 128), f"Error: View 2 shape is {view2.shape}"
    assert not torch.equal(view1, view2), "Error: Both views are identical (augmentation failed)"
    
    print("\n[SUCCESS] Phase 4 Verification Passed!")

if __name__ == "__main__":
    try:
        test_phase4()
    except Exception as e:
        print(f"\n[FAILURE] Phase 4 Verification Failed: {str(e)}")
        sys.exit(1)
