# Research Tables Summary

## Table 1: Dataset Summary
| Dataset    |   Cases | Modalities              | Role                  |
|:-----------|--------:|:------------------------|:----------------------|
| BraTS 2021 |    1389 | 4 (T1, T1ce, T2, FLAIR) | Training / Validation |
| TCGA-GBM   |     102 | 4 (T1, T1ce, T2, FLAIR) | External Testing      |
| TCGA-LGG   |      65 | 4 (T1, T1ce, T2, FLAIR) | External Testing      |

## Table 2: Main Retrieval Performance
| Metric                     | Proposed Hybrid Model   |
|:---------------------------|:------------------------|
| Average Similarity (Top-1) | 0.9917                  |
| Average Similarity (Top-5) | 0.9898                  |
| Perfect Index Match Rate   | 100%                    |
| Zero-Shot Gen. Rate        | >98%                    |

## Table 3: Cross-Dataset Evaluation
| Query Source Cohort   |   Mean Similarity (L2 Normalized) |
|:----------------------|----------------------------------:|
| BraTS (Internal)      |                          0.990811 |
| TCGA (External)       |                          0.997358 |

## Table 4: Ablation Study
| Component           | Configuration       |   Retrieval Sim. |
|:--------------------|:--------------------|-----------------:|
| Preprocessing       | Whole-Brain         |           0.9894 |
| Preprocessing       | Lesion-Centered     |           0.989  |
| Architecture        | Single-Modality     |           0.945  |
| Architecture (Ours) | Multi-Branch Hybrid |           0.9921 |
