import os
from pathlib import Path
import pandas as pd

# Automatically locate the project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[2]

def clean_path_string(path_str):
    """Normalize path separators to match the current OS."""
    if pd.isna(path_str) or not isinstance(path_str, str):
        return None
    # Use Path to clean up slashes and resolve to OS standards
    return str(Path(path_str.strip()))

def resolve_mri_path(raw_path, dataset_name=None):
    """
    Resolves an MRI volume file path to be absolute and valid.
    Supports relative paths (BraTS2021) and absolute paths (TCGA).
    """
    if pd.isna(raw_path) or not isinstance(raw_path, str):
        return None
        
    cleaned = raw_path.strip().replace('\\', '/')
    
    # If it's already an absolute path and exists, return it
    p = Path(cleaned)
    if p.is_absolute():
        if p.exists():
            return str(p)
        # If it's absolute but doesn't exist directly (e.g. partition drive changed), try locating from root
        # Example: E:/Cse Engineering/11Defense/data/... -> PROJECT_ROOT / data/...
        rel_parts = p.parts[p.parts.index('data'):] if 'data' in p.parts else None
        if rel_parts:
            alt_path = PROJECT_ROOT.joinpath(*rel_parts)
            if alt_path.exists():
                return str(alt_path)
                
    # Otherwise, resolve relative to project root
    resolved = PROJECT_ROOT / cleaned
    if resolved.exists():
        return str(resolved)
        
    # fallback: search relative from 'data/' if it contains the word
    if 'data/' in cleaned:
        sub_path = cleaned.split('data/')[-1]
        resolved_alt = PROJECT_ROOT / 'data' / sub_path
        if resolved_alt.exists():
            return str(resolved_alt)
            
    return str(PROJECT_ROOT / cleaned)
