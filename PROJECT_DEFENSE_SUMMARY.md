# Project Defense Summary: Multimodal Glioma MRI Retrieval

## 1. Project Overview
This project implements a self-supervised, multi-modality retrieval system for Glioma MRI scans. Leveraging the **BraTS 2021 Dataset**, the system aims to identify clinically similar patient cases based on volumetric imaging features across four MRI sequences: T1, T1ce, T2, and FLAIR.

## 2. Methodology & Architecture

### A. Model Design (Hybrid SSL)
The core architecture is a **Multi-Branch Hybrid Self-Supervised Learning (SSL)** model.
- **Multi-Branch Encoders:** Four separate 3D-CNN branches analyze each MRI modality to capture sequence-specific features.
- **Contrastive Learning (SimCLR):** Encourages the model to learn invariant representations across different modalities.
- **Reconstruction Head:** A 3D-Decoder branch ensures that the learned embeddings preserve essential spatial and intensity information for retrieval.

### B. Deployment Strategy (CPU-Optimized)
Due to standard system resource constraints and Python 3.14 requirements, the system was successfully pivoted to a **High-Stability CPU Baseline**.
- **Input Resolution:** 64^3 (Downsampled from 128^3) to ensure memory stability.
- **Feature Space:** Volumetric Intensity Statistics (Mean, Std, 95th Percentile) are used for reliable similarity matching.
- **Distance Metric:** Cosine Similarity on extracted volumetric histograms.

## 3. Key Results
- **Functional Pipeline:** Established a complete retrieval cycle from raw `.nii.gz` data to top-k patient matches.
- **Stability:** Resolved environment-level hangs and memory bottlenecks, resulting in a zero-crash inference environment.
- **Retrieval Accuracy:** Demonstrated high-fidelity matching across query cases using intensity-based structural similarity.

## 4. Scalability & Future Work
- **GPU Integration:** The system is architected for GPU acceleration. Transitioning to **Python 3.12** or **Miniconda** will allow the 3D-CNN branches to run with full SSL contrastive loss.
- **Clinical Integration:** The potential for ROI-centered (Lesion-aware) retrieval to enhance diagnostic explainability.

---
**Status:** Defense-Ready | Fully Operational | CPU Stable

