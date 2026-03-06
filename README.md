# LSB-AI-Detection

Automated detection and segmentation of **Low Surface Brightness (LSB)** features — stellar streams and satellite galaxies — in astronomical images using fine-tuned SAM2/SAM3 (Segment Anything Model).

Built on the [FIREbox-DR1](https://fire.northwestern.edu/) cosmological simulation data (surface-brightness maps in FITS format).

---

## Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [End-to-End Workflow](#end-to-end-workflow)
  - [Phase 0: Mask Statistics](#phase-0-mask-statistics-analysis)
  - [Phase 1–4: Unified Data Preparation](#unified-data-preparation-pipeline)
  - [Model Evaluation](#model-evaluation)
- [Configuration Reference](#configuration-reference)
- [Evaluation Metrics](#evaluation-metrics)
- [Visualization](#visualization)

---

## Installation

```bash
# Clone
git clone <repo-url> && cd LSB-AI-Detection

# Create conda env (SAM2/SAM3 require separate installation)
conda create -n lsb python=3.12
conda activate lsb
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

git clone https://github.com/facebookresearch/sam3.git
cd sam3
pip install -e .
pip install -e ".[notebooks]"

# For development
pip install -e ".[train,dev]"

git clone https://github.com/facebookresearch/sam2.git && cd sam2
pip install -e .
pip install -e ".[notebooks]"
# Core dependencies
pip install -r requirements.txt

# SAM2 / SAM3 — install from their repos into the same env
# e.g. pip install -e /path/to/sam2   and   pip install -e /path/to/sam3

# (Optional) Versioned git hooks
git config core.hooksPath tools/githooks
```

**Key dependencies** (`requirements.txt`):

| Category | Packages |
|----------|----------|
| Data | `numpy`, `astropy`, `opencv-python`, `pycocotools`, `Pillow`, `scikit-image` |
| DL | `torch>=2.0`, `torchvision>=0.15` |
| Config | `PyYAML`, `tqdm` |
| Viz | `matplotlib` |

---

## Project Structure

```
LSB-AI-Detection/
├── configs/                        # YAML configs for all pipelines
│   ├── unified_data_prep.yaml      # 4-phase unified pipeline (primary)
│   ├── eval_sam3.yaml              # SAM3 evaluation config
│   ├── eval_sam2.yaml              # SAM2 evaluation config
│   └── noise_profiles.yaml         # Forward observation model SNR profiles
│
├── scripts/                        # Entry-point scripts
│   ├── prepare_unified_dataset.py  # ★ Main data pipeline (4-phase)
│   ├── evaluate_model.py           # SAM3 type-aware evaluation
│   ├── analyze_mask_stats.py       # GT instance statistics → filter thresholds
│   ├── generate_noisy_fits.py      # Noise injection (SNR profiles)
│   ├── render_noisy_fits.py        # Render noise-injected FITS → PNG
│   ├── plot_mask_stats.py          # Boxplots for shape metrics
│   ├── visualize_sam3.py           # 4-column grid visualization
│   └── run_batch_eval.sh           # Batch evaluation shell wrapper
│
├── src/                            # Python package
│   ├── data/                       # FITS I/O, preprocessing (asinh, linear_mag, multi_exposure)
│   ├── noise/                      # ForwardObservationModel (SB → flux → Poisson → readout)
│   ├── inference/                  # SAM2 AutoMask runner, SAM3 text-prompt runner
│   ├── postprocess/                # Type-aware filtering pipeline
│   │   ├── satellite_prior_filter.py   # Area/solidity/aspect rules
│   │   ├── streams_sanity_filter.py    # Area bounds, edge-touch fraction
│   │   ├── core_exclusion_filter.py    # Centroid radius exclusion
│   │   ├── candidate_grouping.py       # Union-Find centroid clustering
│   │   └── representative_selection.py # Best mask per group
│   ├── analysis/                   # Per-mask geometry metrics
│   ├── evaluation/                 # Pixel + instance metrics, SAM3 eval orchestration
│   ├── visualization/              # Overlay, 3-panel comparison plotting
│   └── utils/                      # COCO RLE, logger, geometry helpers
│
├── data/
│   ├── 01_raw/                     # Raw FIREbox FITS + masks
│   └── 02_processed/               # Pipeline outputs (renders, GT, exports)
│
├── outputs/                        # Evaluation results, mask stats, overlays
├── requirements.txt
└── ARCHITECTURE.md
```

---

## End-to-End Workflow

The typical workflow has three stages:

```
[0] analyze_mask_stats  →  filter thresholds (one-time)
[1–4] prepare_unified_dataset  →  renders + GT + inference + export
[5] evaluate_model  →  Dice / IoU / Recall metrics
```

### Phase 0: Mask Statistics Analysis

Computes per-instance geometric statistics from canonical GT masks. The output `mask_stats_summary.json` drives all downstream filter thresholds (satellite prior filter, streams sanity filter).

```bash
python scripts/analyze_mask_stats.py \
    --gt_root data/02_processed/gt_canonical/current \
    --output_dir outputs/mask_stats
```

**Outputs:**

| File | Content |
|------|---------|
| `mask_instance_stats.csv` | Per-instance row: area, solidity, aspect ratios, curvature |
| `mask_stats_summary.json` | Quantile distributions + `filter_recommendations` section |

The `filter_recommendations` block provides data-driven thresholds:

```json
{
  "streams": {
    "min_area": 42,
    "max_area": 98304,
    "min_solidity": 0.12,
    "aspect_sym_moment_max": 8.5
  },
  "satellites": {
    "min_area": 15,
    "max_area": 5200,
    "min_solidity": 0.65,
    "aspect_sym_moment_max": 3.2
  }
}
```

---

### Unified Data Preparation Pipeline

The main data pipeline (`scripts/prepare_unified_dataset.py`) converts raw FITS data into training-ready datasets through **4 phases**:

```
FITS (SB maps)
  │
  ▼
Phase 1: RENDER ─── FITS → RGB PNGs (per preprocessing variant)
  │
  ▼
Phase 2: GT ──────── SB masks → streams_instance_map.npy (canonical GT)
  │
  ▼
Phase 3: INFERENCE ─ SAM2 AutoMask or SAM3 text-prompt → filter → merge/evaluate
  │
  ▼
Phase 4: EXPORT ──── SAM2 symlinks + SAM3 COCO annotations.json
```

#### Basic Usage

```bash
# Run full pipeline (all 4 phases)
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml

# Run a single phase
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase render
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase gt
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase inference
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase export

# Subset of galaxies (comma-separated IDs)
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --galaxies 11,13,19

# Force rebuild all outputs
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --force

# Force rebuild specific preprocessing variants only
python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml \
    --force-variants asinh_stretch,multi_exposure
```

#### Phase Details

**Phase 1 — Render**: Loads `.fits.gz` surface-brightness maps and applies preprocessing variants to produce RGB PNGs. Each variant generates a separate directory tree.

Available preprocessors (defined in `src/data/preprocessing.py`):

| Variant | Description |
|---------|-------------|
| `asinh_stretch` | Arcsinh stretch with configurable nonlinearity + zeropoint |
| `linear_magnitude` | Linear mapping in magnitude space (global min/max) |
| `multi_exposure` | Composite of linear + asinh channels (3-channel RGB) |

Output: `data/02_processed/renders/current/{variant}/{galaxy_id}_{orientation}/0000.png`

**Phase 2 — GT**: Loads SB-threshold masks (e.g. SB=32 mag/arcsec²), resizes with nearest-neighbor interpolation to preserve instance IDs, saves as `.npy`.

Output: `data/02_processed/gt_canonical/current/{base_key}/streams_instance_map.npy`

**Phase 3 — Inference**: Runs the selected engine on rendered images, applies type-aware post-processing filters, then either merges (SAM2) or evaluates (SAM3).

- **SAM2 engine** (`inference_phase.engine: "sam2"`):
  AutoMask → `append_metrics` → centroid grouping → representative selection → satellite prior filter → core exclusion → merge with streams GT → `instance_map_uint8.png`

- **SAM3 engine** (`inference_phase.engine: "sam3"`):
  Text-prompt inference (e.g. "stellar stream", "satellite galaxy") → type-aware filter fork (streams → `StreamsSanityFilter`, satellites → `SatellitePriorFilter` + `CoreExclusionFilter`) → save raw/post JSON + QA overlay

**Phase 4 — Export**: Generates training-ready formats:

- **SAM2**: Folder-based symlinks (`img_folder/` + `gt_folder/`)
- **SAM3**: COCO-format `annotations.json` with RLE-encoded masks + `images/` symlinks

---

### Model Evaluation

After training, evaluate models with `scripts/evaluate_model.py`. This script is **type-aware**: it splits predictions into streams and satellites, computing metrics for each type independently and combined.

```bash
# SAM3 evaluation (default config)
python scripts/evaluate_model.py --config configs/eval_sam3.yaml

# Override paths via CLI
python scripts/evaluate_model.py --config configs/eval_sam3.yaml \
    --render-dir data/02_processed/renders/current/asinh_stretch \
    --gt-dir data/02_processed/gt_canonical/current \
    --output-dir outputs/eval_sam3 \
    --save-overlays

# Limit samples for debugging
python scripts/evaluate_model.py --config configs/eval_sam3.yaml --max-samples 5

# Per-galaxy aggregation (merge eo+fo)
python scripts/evaluate_model.py --config configs/eval_sam3.yaml --per-galaxy

# Tag with SNR tier (for noise-injected evaluations)
python scripts/evaluate_model.py --config configs/eval_sam3.yaml --snr-tag snr10
```

**Evaluation pipeline:**

1. **Discover pairs**: Match `{render_dir}/{base_key}/0000.png` ↔ `{gt_dir}/{base_key}/` GT files
2. **Inference**: Run SAM3 with text prompts (per-prompt confidence thresholds)
3. **Filter**: Apply `StreamsSanityFilter` on stream predictions (post layer)
4. **Metrics**: Compute pixel-level and instance-level metrics per type (raw + post)
5. **Aggregate**: Macro-average across images, output JSON

**Output** (`outputs/eval_sam3/eval_YYYYMMDD_HHMMSS.json`):

```
{
  "config": { ... },
  "summary": {
    "overall": {
      "streams":    { "raw": {...}, "post": {...} },
      "satellites": { "raw": {...}, "post": {...} },
      "combined":   { "raw": {...}, "post": {...} }
    }
  },
  "per_image": [ ... ]
}
```

---

## Configuration Reference

### `configs/unified_data_prep.yaml`

```yaml
paths:
  firebox_root: "data/01_raw/LSB_and_Satellites/FIREbox-DR1"
  output_root: "data/02_processed"

data_selection:
  galaxy_ids: [11, 13, 19, ...]       # FIREbox galaxy IDs
  orientations: ["eo", "fo"]          # Edge-on, face-on
  canonical_sb_threshold: 32.0        # mag/arcsec² for GT masks

processing:
  target_size: [1024, 1024]           # Resize target (W, H)

preprocessing_variants:               # Each produces a render directory
  - name: "asinh_stretch"
    params: { nonlinearity: 200.0, zeropoint: 22.5, clip_percentile: 99.5 }
  - name: "linear_magnitude"
    params: { global_mag_min: 20.0, global_mag_max: 35.0 }

inference_phase:
  engine: "sam2"                      # "sam2" or "sam3"
  input_image_variant: "linear_magnitude"
  sam3:
    checkpoint: "/path/to/sam3/checkpoint.pt"
    prompts:
      - { text: "stellar stream", type_label: "streams" }
      - { text: "satellite galaxy", type_label: "satellites" }

satellites:                           # SAM2 AutoMask pipeline config
  checkpoint: "/path/to/sam2/checkpoint.pt"
  generator:
    points_per_side: 64
    pred_iou_thresh: 0.65
    stability_score_thresh: 0.95
  prior:
    stats_json: "outputs/mask_stats/mask_stats_summary.json"
  core_exclusion:
    radius_frac: 0.08
```

### `configs/eval_sam3.yaml`

```yaml
paths:
  render_dir: "data/02_processed/renders/current/asinh_stretch"
  gt_dir: "data/02_processed/gt_canonical/current"
  output_dir: "outputs/eval_sam3"

sam3:
  checkpoint: "/path/to/checkpoint.pt"
  bpe_path: "/path/to/bpe_simple_vocab_16e6.txt.gz"

prompts:
  - { text: "stellar stream", type_label: "streams", confidence_threshold: 0.55 }
  - { text: "satellite galaxy", type_label: "satellites", confidence_threshold: 0.45 }

match_iou_thresh: 0.5

post_filter:
  stats_json: "outputs/mask_stats/mask_stats_summary.json"
  edge_touch_frac: 0.8
```

---

## Evaluation Metrics

The evaluation system (`src/evaluation/metrics.py`) computes two levels of metrics, reported for both **raw** (unfiltered) and **post** (filtered) prediction layers:

### Pixel-Level Metrics

| Metric | Formula | Empty-mask handling |
|--------|---------|---------------------|
| **Dice** | 2·TP / (2·TP + FP + FN) | `null` if both empty; `0.0` if one empty |
| **Precision** | TP / (TP + FP) | `null` if no pred pixels |
| **Recall** | TP / (TP + FN) | `null` if no GT pixels |
| **Hausdorff95** | Symmetric 95th-percentile boundary distance | `null` if both empty; image diagonal if one empty |

### Instance-Level Metrics

| Metric | Description |
|--------|-------------|
| **Matched IoU** | Mean IoU of valid matches (Hungarian optimal 1:1 assignment on IoU matrix) |
| **Instance Recall** | Fraction of GT instances with a valid match (IoU ≥ threshold) |

Aggregation modes:
- **Macro**: Mean ± std across images (null values skipped)
- **Micro**: Global TP/FP/FN sums → single Dice/Precision/Recall

---

## Visualization

```bash
# SAM3 dataset: 4-column grid (Original, Streams, Satellites, Combined)
python scripts/visualize_sam3.py

# Mask statistics boxplots (streams vs satellites)
python scripts/plot_mask_stats.py

# Evaluation metrics across SNR tiers
python scripts/visualize_eval_metrics.py

# Overlay GT masks on rendered images
python scripts/overlay_masks_on_streams.py
```

---

## Additional Scripts

| Script | Purpose |
|--------|---------|
| `generate_noisy_fits.py` | Inject Poisson + readout noise at specified SNR profiles |
| `render_noisy_fits.py` | Render noise-injected FITS to PNG |
| `run_batch_eval.sh` | Shell wrapper for batch evaluation across SNR tiers |

---

## Data Layout

After running the full pipeline, the processed data directory looks like:

```
data/02_processed/
├── renders/current/
│   ├── asinh_stretch/{base_key}/0000.png
│   └── linear_magnitude/{base_key}/0000.png
├── gt_canonical/current/{base_key}/
│   ├── streams_instance_map.npy      # Phase 2: streams-only GT
│   ├── instance_map_uint8.png        # Phase 3: merged streams + satellites
│   ├── instances.json                # Instance ID → type mapping
│   ├── manifest.json                 # Provenance metadata
│   └── overlay.png                   # QA visualization
├── sam2_prepared/                     # Phase 4 export
│   ├── img_folder/{variant_key}/0000.png → (symlink)
│   └── gt_folder/{variant_key}/0000.png  → (symlink)
└── sam3_prepared/                     # Phase 4 export
    ├── images/{variant_key}.png → (symlink)
    └── annotations.json              # COCO format with RLE masks
```

Where `{base_key}` = `{galaxy_id:05d}_{orientation}` (e.g. `00011_eo`) and `{variant_key}` = `{base_key}_{preprocessing}` (e.g. `00011_eo_asinh_stretch`).
