# Project Overview

A modular Python package for detecting Low Surface Brightness (LSB) features in astronomical images using SAM2/SAM3 (Segment Anything Model).

---

## Directory Structure (Level 2)

```
LSB-AI-Detection/
├── configs/                          # Configuration files
├── data/                             # Raw and processed datasets
├── docs/                             # Documentation (API and Datasets)
├── scripts/                          # Entry point scripts for pipelines and evaluation
├── src/                              # Source code package
├── tools/
│   └── githooks/                     # Versioned git hooks containing pre-commit AST sync
├── logs/                             # Runtime logs (auto-generated)
├── notebooks/                        # Jupyter notebooks
├── CHANGELOG.md                      # Version history
└── requirements.txt                  # Python dependencies
```

---

## Doc Index (Progressive Loading Protocol Pathing)
- [Data Module (`src/data`)](src/data/MODULE_DOC.md)
- [Pipeline Scripts (`scripts`)](scripts/MODULE_DOC.md)
- [Inference Module (`src/inference`)](src/inference/MODULE_DOC.md)
- [Postprocess Module (`src/postprocess`)](src/postprocess/MODULE_DOC.md)
- [Analysis Module (`src/analysis`)](src/analysis/MODULE_DOC.md)
- [Evaluation Module (`src/evaluation`)](src/evaluation/MODULE_DOC.md)
- [Visualization Module (`src/visualization`)](src/visualization/MODULE_DOC.md)
- [Utilities Module (`src/utils`)](src/utils/MODULE_DOC.md)

---

## Configuration Reference

| Config File              | Process          | Output                               |
| :----------------------- | :--------------- | :----------------------------------- |
| `unified_data_prep.yaml` | 4-Phase Pipeline | Canonical GT + SAM2 link + SAM3 COCO |
| `data_prep_sam2.yaml`    | Legacy           | Folder-based (1072x1072)             |
| `data_prep_sam3.yaml`    | Legacy           | COCO JSON (1024x1024)                |
| `eval_sam2.yaml`         | Eval             | Model metrics                        |
| `automask_sweep.yaml`    | Sweep            | AutoMask ranking                     |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Configure AST Versioned Git Hooks
git config core.hooksPath tools/githooks

# Unified Pipeline (Recommended)
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml

# Evaluate model
python scripts/eval_model.py --checkpoint /path/to/model.pt --save_vis
```
