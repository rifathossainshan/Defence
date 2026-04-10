import os
import sys
from typing import Dict, Optional, Tuple
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# Add src to path for internal imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from preprocessing.crop_roi import get_bbox_from_seg, get_lesion_center_crop_coords

class BraTSSSLDataset(Dataset):
    """
    BraTS SSL Dataset
    Returns:
        {
            "view1": Tensor [4, D, H, W],
            "view2": Tensor [4, D, H, W],
            "id": str,
            "dataset": str,
            "seg": Tensor [1, D, H, W] or None
        }
    """

    def __init__(
        self,
        csv_file: str,
        base_dir: str,
        split: str = "train",
        target_size: Tuple[int, int, int] = (128, 128, 128),
        use_seg: bool = True,
        apply_crop: bool = True,
        apply_normalize: bool = True,
        crop_mode: str = "whole",
        transform=None,
    ) -> None:
        self.df = pd.read_csv(csv_file)
        if split != "all":
            self.df = self.df[self.df["split"] == split].reset_index(drop=True)

        # valid rows only
        if "is_valid" in self.df.columns:
            self.df = self.df[self.df["is_valid"] == 1].reset_index(drop=True)

        if "has_all_modalities" in self.df.columns:
            self.df = self.df[self.df["has_all_modalities"] == 1].reset_index(drop=True)

        self.base_dir = Path(base_dir)
        self.target_size = target_size
        self.use_seg = use_seg
        self.apply_crop = apply_crop
        self.apply_normalize = apply_normalize
        self.crop_mode = crop_mode
        self.transform = transform

        self.modality_cols = ["t1_path", "t1ce_path", "t2_path", "flair_path"]

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        # 1. load modalities
        modalities = []
        for col in self.modality_cols:
            file_path = self.base_dir / row[col]
            img = self._load_nifti(str(file_path))
            modalities.append(img)

        # 2. load segmentation if available
        seg = None
        if self.crop_mode == "lesion" or self.use_seg:
            if "seg_path" in row and isinstance(row["seg_path"], str):
                seg_path = self.base_dir / row["seg_path"]
                if seg_path.exists():
                    seg = self._load_nifti(str(seg_path))
                elif self.crop_mode == "lesion":
                    print(f"Warning: Seg missing for {row['patient_id']}, falling back to whole-brain.")

        # 3. Handle Cropping (ROI/Central)
        if self.apply_crop:
            if self.crop_mode == "lesion" and seg is not None:
                bbox = get_bbox_from_seg(seg[0] if seg.ndim == 4 else seg)
                if bbox:
                    h_s, h_e, w_s, w_e, d_s, d_e = get_lesion_center_crop_coords(bbox, self.target_size, modalities[0].shape)
                else:
                    h_s, h_e, w_s, w_e, d_s, d_e = self._get_center_crop_coords(modalities[0].shape)
            else:
                h_s, h_e, w_s, w_e, d_s, d_e = self._get_center_crop_coords(modalities[0].shape)

            modalities = [img[h_s:h_e, w_s:w_e, d_s:d_e] for img in modalities]
            if seg is not None:
                # Handle [C, H, W, D] if already expanded
                if seg.ndim == 4:
                    seg = seg[:, h_s:h_e, w_s:w_e, d_s:d_e]
                else:
                    seg = seg[h_s:h_e, w_s:w_e, d_s:d_e]

        image = np.stack(modalities, axis=0)  # [4, H, W, D]
        if seg is not None:
            seg = np.expand_dims(seg, axis=0)  # [1, H, W, D]

        # 4. normalize
        if self.apply_normalize:
            image = self._zscore_normalize_multichannel(image)

        if self.apply_crop:
            image, seg = self._crop_foreground(image, seg)

        # 5. resize/pad to target size
        image = self._pad_or_crop_to_size(image, self.target_size)
        if seg is not None:
            seg = self._pad_or_crop_to_size(seg, self.target_size)

        # 6. make two SSL views
        view1 = image.copy()
        view2 = image.copy()

        if self.transform is not None:
            view1 = self.transform(view1)
            view2 = self.transform(view2)

        # 7. convert to tensor and transpose to [C, D, H, W] if needed
        # Current shape is [C, H, W, D]
        # PyTorch 3D Conv expects [C, D, H, W]
        view1 = torch.tensor(view1, dtype=torch.float32).permute(0, 3, 1, 2)
        view2 = torch.tensor(view2, dtype=torch.float32).permute(0, 3, 1, 2)

        if seg is not None:
            seg = torch.tensor(seg, dtype=torch.float32).permute(0, 3, 1, 2)

        sample = {
            "view1": view1,                # [4, D, H, W]
            "view2": view2,
            "id": row["patient_id"],
            "dataset": row["dataset"],
            "seg": seg if seg is not None else torch.zeros((1, *self.target_size))
        }
        return sample

    def _load_nifti(self, path: str) -> np.ndarray:
        img = nib.load(path)
        data = img.get_fdata().astype(np.float32)
        return data

    def _zscore_normalize_multichannel(self, image: np.ndarray) -> np.ndarray:
        out = np.zeros_like(image, dtype=np.float32)
        for c in range(image.shape[0]):
            x = image[c]
            mask = x > 0
            if mask.sum() > 0:
                mean = x[mask].mean()
                std = x[mask].std()
                out[c] = x
                out[c][mask] = (x[mask] - mean) / (std + 1e-8)
            else:
                out[c] = x
        return out

    def _crop_foreground(
        self,
        image: np.ndarray,
        seg: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        fg_mask = np.any(image > 0, axis=0)  # [H, W, D]
        coords = np.where(fg_mask)
        if len(coords[0]) == 0:
            return image, seg

        h_min, h_max = coords[0].min(), coords[0].max() + 1
        w_min, w_max = coords[1].min(), coords[1].max() + 1
        d_min, d_max = coords[2].min(), coords[2].max() + 1

        image = image[:, h_min:h_max, w_min:w_max, d_min:d_max]
        if seg is not None:
            seg = seg[:, h_min:h_max, w_min:w_max, d_min:d_max]
        return image, seg

    def _pad_or_crop_to_size(
        self,
        x: np.ndarray,
        target_size: Tuple[int, int, int]
    ) -> np.ndarray:
        c, h, w, d = x.shape
        th, tw, td = target_size

        pad_h = max(0, th - h)
        pad_w = max(0, tw - w)
        pad_d = max(0, td - d)

        if pad_h > 0 or pad_w > 0 or pad_d > 0:
            pad_before_h, pad_before_w, pad_before_d = pad_h // 2, pad_w // 2, pad_d // 2
            pad_after_h, pad_after_w, pad_after_d = pad_h - pad_before_h, pad_w - pad_before_w, pad_d - pad_before_d
            x = np.pad(x, ((0, 0), (pad_before_h, pad_after_h), (pad_before_w, pad_after_w), (pad_before_d, pad_after_d)), mode="constant", constant_values=0)

        _, h, w, d = x.shape
        start_h, start_w, start_d = max(0, (h - th) // 2), max(0, (w - tw) // 2), max(0, (d - td) // 2)
        x = x[:, start_h:start_h + th, start_w:start_w + tw, start_d:start_d + td]
        return x

    def _get_center_crop_coords(self, shape: Tuple[int, int, int]) -> Tuple[int, int, int, int, int, int]:
        h, w, d = shape
        th, tw, td = self.target_size
        
        h_start = max(0, (h - th) // 2)
        w_start = max(0, (w - tw) // 2)
        d_start = max(0, (d - td) // 2)
        
        return h_start, h_start + th, w_start, w_start + tw, d_start, d_start + td
