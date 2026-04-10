# Self-Supervised Multimodal 3D Glioma MRI Retrieval System

## Project Overview
This project implements a self-supervised, multimodal, case-based retrieval system for glioma MRI analysis. It learns transferable 3D representations from BraTS and TCGA datasets to enable similarity-based case searching and explainable retrieval.

## Project Structure
- `configs/`: Configuration files for models and training.
- `data/`:
    - `raw/`: Raw datasets (Training_Data, Testing, Validation).
    - `metadata/`: Master index CSVs and QC reports.
- `src/`:
    - `preprocessing/`: Scripts for standardization, QC, and on-the-fly transforms.
    - `datasets/`: PyTorch Dataset and DataLoader implementations.
    - `models/`: Architecture definitions (ResNet18, Multi-branch encoders).
    - `losses/`: SSL loss functions (SimCLR, BYOL, Hybrid).
    - `training/`: Training loops and engine logic.
    - `retrieval/`: FAISS index building and top-k search.
    - `evaluation/`: Visualization and metrics.
    - `utils/`: Common utilities.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `outputs/`: 
    - `checkpoints/`: Model weights.
    - `embeddings/`: Extracted feature vectors.
    - `faiss/`: Built search indexes.
    - `figures/`: Training curves and visualization plots.
    - `logs/`: Training logs.

## Setup
Environment is configured with:
- PyTorch
- MONAI
- Nibabel
- FAISS-CPU
- SimpleITK
- Scikit-Learn
- Pandas/Numpy
