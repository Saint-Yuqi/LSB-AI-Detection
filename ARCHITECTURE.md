# Project Overview

A modular Python package for detecting Low Surface Brightness (LSB) features in astronomical images using SAM2/SAM3 (Segment Anything Model).

---

## Directory Structure (Level 2)

```
LSB-AI-Detection/
├── configs/                          # YAML configuration files
│   ├── unified_data_prep.yaml        #   4-phase data pipeline
│   ├── eval_sam2.yaml                #   SAM2 evaluation
│   ├── eval_sam3.yaml                #   SAM3 type-aware evaluation
│   ├── noise_profiles.yaml           #   Forward noise model SNR profiles
│   └── sam3_dataset_split.yaml       #   Galaxy-level train/val split
├── data/                             # Raw and processed datasets
├── docs/                             # Documentation (API and Datasets)
├── scripts/                          # CLI entry points (data / eval / cluster / viz / analysis)
│   ├── data/                         #   Dataset build, noise FITS, splits
│   ├── eval/                         #   SAM2/SAM3 eval, local batch bash
│   ├── cluster/                      #   Slurm templates + sbatch launcher (site-specific)
│   ├── viz/                          #   QA grids and metric figures
│   └── analysis/                     #   Mask stats and recall curves
├── src/                              # Source code package
│   ├── analysis/                     #   Mask geometric metrics
│   ├── data/                         #   Data loading & preprocessing
│   ├── evaluation/                   #   IoU metrics & SAM3 type-aware eval
│   ├── inference/                    #   SAM2 automask & SAM3 prompt runners
│   ├── noise/                        #   Forward observation noise model
│   ├── pipelines/unified_dataset/    #   Modular dataset pipeline subpackage
│   ├── postprocess/                  #   Prior filter, grouping, representative selection
│   ├── utils/                        #   COCO utils, logger, geometry helpers
│   └── visualization/               #   Overlay & QA visualization
├── tests/                            # Unit tests (pytest)
├── tools/
│   └── githooks/                     # Pre-commit AST sync hooks
├── logs/                             # Runtime logs (auto-generated)
├── notebooks/                        # Jupyter notebooks
├── CHANGELOG.md                      # Version history
└── requirements.txt                  # Python dependencies
```

---

## Doc Index (Progressive Loading Protocol Pathing)
- [Data Module (`src/data`)](src/data/MODULE_DOC.md)
- [Pipeline Scripts (`scripts`)](scripts/MODULE_DOC.md)
- [Pipelines Module (`src/pipelines`)](src/pipelines/MODULE_DOC.md)
- [Inference Module (`src/inference`)](src/inference/MODULE_DOC.md)
- [Noise Module (`src/noise`)](src/noise/MODULE_DOC.md)
- [Postprocess Module (`src/postprocess`)](src/postprocess/MODULE_DOC.md)
- [Analysis Module (`src/analysis`)](src/analysis/MODULE_DOC.md)
- [Evaluation Module (`src/evaluation`)](src/evaluation/MODULE_DOC.md)
- [Visualization Module (`src/visualization`)](src/visualization/MODULE_DOC.md)
- [Utilities Module (`src/utils`)](src/utils/MODULE_DOC.md)

---

## Configuration Reference

| Config File              | Process              | Output                               |
| :----------------------- | :------------------- | :----------------------------------- |
| `unified_data_prep.yaml` | 4-Phase Pipeline     | Canonical GT + SAM2 link + SAM3 COCO |
| `noise_profiles.yaml`    | Forward Noise Gen    | Noisy FITS per SNR profile           |
| `sam3_dataset_split.yaml`| Galaxy-Level Split   | Train/val COCO annotations           |
| `eval_sam2.yaml`         | SAM2 Eval            | Model metrics                        |
| `eval_sam3.yaml`         | SAM3 Eval            | Type-aware metrics JSON              |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AST Versioned Git Hooks
git config core.hooksPath tools/githooks

# 1. Unified Pipeline (Recommended)
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml

# 2. Noise augmentation
python scripts/data/render_noisy_fits.py --config configs/unified_data_prep.yaml
python scripts/data/build_noise_augmented_annotations.py --config configs/unified_data_prep.yaml

# 3. Train/val split
python scripts/data/split_annotations.py --config configs/sam3_dataset_split.yaml

# 4. Evaluate
python scripts/eval/evaluate_sam2.py --config configs/eval_sam2.yaml
python scripts/eval/evaluate_sam3.py --config configs/eval_sam3.yaml
```
