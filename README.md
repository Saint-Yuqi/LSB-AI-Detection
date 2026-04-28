<p align="center">
  <img src="docs/assets/render_multi_exposure.jpg" width="180" alt="Multi-exposure render"/>
  <img src="docs/assets/gt_instance_overlay.jpg" width="180" alt="Ground truth overlay"/>
  <img src="docs/assets/eval_overlay.jpg" width="180" alt="Evaluation overlay"/>
</p>

<h1 align="center">LSB-AI-Detection</h1>

<p align="center">
  <b>Automated detection of Low Surface Brightness features in astronomical images<br/>using fine-tuned SAM3 (Segment Anything Model 3)</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12-blue?logo=python&logoColor=white" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/PyTorch-2.7-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch 2.7"/>
  <img src="https://img.shields.io/badge/SAM3-Meta_AI-4267B2?logo=meta&logoColor=white" alt="SAM3"/>
  <img src="https://img.shields.io/badge/data-FIREbox--DR1_%7C_PNbody-orange" alt="FIREbox-DR1 | PNbody"/>
</p>

---

Detects and segments **tidal features** (stellar streams, shells, plumes) and **satellite galaxies** — the faintest structures around galaxies — from cosmological surface-brightness maps (FITS format). Sources include [FIREbox-DR1](https://fire.northwestern.edu/), the FIREbox Gold Satellites benchmark, and PNbody 24-line-of-sight renders. The pipeline covers raw FITS preprocessing, noise-augmented training data generation, SAM3 fine-tuning, and benchmark-aware evaluation.

> **Taxonomy.** The active dataset (`tidal_v1`) uses a **3-class** scheme: `tidal_features`, `satellites`, `inner_galaxy`. The legacy 2-class (`streams` / `satellites`) labels are aliased through `src/pipelines/unified_dataset/taxonomy.py`. The `inner_galaxy` class is produced by the prior filter relabelling hard-center satellite candidates rather than dropping them.

> **Model.** Only **SAM3** is supported — the legacy SAM2 stack was removed. The single evaluation entrypoint is `scripts/eval/evaluate_checkpoint.py` driven by `configs/eval_checkpoint.yaml`.

<br/>

## Visual Overview

### Preprocessing Variants

Two preprocessing strategies convert raw surface-brightness (mag/arcsec²) FITS data into model-ready RGB images. A third multi-exposure composite is implemented but currently disabled in the active pipeline.

<table>
  <tr>
    <td align="center"><b>Asinh Stretch</b></td>
    <td align="center"><b>Linear Magnitude</b></td>
    <td align="center"><b>Multi-Exposure (3-ch)</b></td>
  </tr>
  <tr>
    <td><img src="docs/assets/render_asinh_stretch.jpg" width="250"/></td>
    <td><img src="docs/assets/render_linear_magnitude.jpg" width="250"/></td>
    <td><img src="docs/assets/render_multi_exposure.jpg" width="250"/></td>
  </tr>
  <tr>
    <td>Nonlinear arcsinh mapping<br/>emphasizing faint features</td>
    <td>Linear mapping in magnitude<br/>space (global min/max) — <i>active default</i></td>
    <td>R=linear, G=asinh, B=gamma<br/>composite (currently disabled)</td>
  </tr>
</table>

### Ground Truth & Predictions

<table>
  <tr>
    <td align="center"><b>Ground Truth Instances</b></td>
    <td align="center"><b>SAM3 Evaluation Overlay</b></td>
  </tr>
  <tr>
    <td><img src="docs/assets/gt_instance_overlay.jpg" width="380"/></td>
    <td><img src="docs/assets/eval_overlay.jpg" width="380"/></td>
  </tr>
  <tr>
    <td>Merged tidal_features + satellites instance map<br/>Color-coded per-instance overlay</td>
    <td>GT contours (white = tidal features, yellow = satellites)<br/>Predictions (blue = tidal features, green = satellites)</td>
  </tr>
</table>

### Noise Robustness

A forward observation noise model (SB → flux → counts → Poisson → readout → magnitude) simulates realistic observing conditions across SNR tiers:

<table>
  <tr>
    <td align="center"><b>Clean (no noise)</b></td>
    <td align="center"><b>SNR ≈ 50</b></td>
    <td align="center"><b>SNR ≈ 20</b></td>
  </tr>
  <tr>
    <td><img src="docs/assets/render_asinh_stretch.jpg" width="250"/></td>
    <td><img src="docs/assets/render_noisy_snr50.jpg" width="250"/></td>
    <td><img src="docs/assets/render_noisy_snr20.jpg" width="250"/></td>
  </tr>
  <tr>
    <td>Ideal simulation output</td>
    <td>Faint structures partially visible</td>
    <td>Only brightest features survive</td>
  </tr>
</table>

### SAM3 Dataset Visualization

4-column grid: **Original → Tidal features → Satellites → Combined** across preprocessing variants.

<p align="center">
  <img src="docs/assets/sam3_dataset_grid.jpg" width="90%" alt="SAM3 4-column grid visualization"/>
</p>

<details>
<summary>More examples</summary>

<p align="center">
  <img src="docs/assets/sam3_dataset_grid_2.jpg" width="90%" alt="SAM3 grid — Galaxy 00019"/>
</p>

</details>

---

## Evaluation Results

### Metrics Across SNR Tiers

Performance degrades gracefully as noise increases.

<table>
  <tr>
    <td align="center"><b>Combined Metrics by SNR</b></td>
    <td align="center"><b>Degradation Curves</b></td>
  </tr>
  <tr>
    <td><img src="docs/assets/metrics_bar_combined.png" width="450"/></td>
    <td><img src="docs/assets/degradation_curves.png" width="450"/></td>
  </tr>
</table>

### Mask Shape Statistics

Data-driven filter thresholds derived from ground truth geometry (tidal features vs satellites).

<p align="center">
  <img src="docs/assets/mask_stats_distributions.png" width="85%" alt="Mask statistics distributions"/>
</p>

---

## Installation

```bash
git clone <repo-url> && cd LSB-AI-Detection

# Conda environment
conda create -n sam3 python=3.12
conda activate sam3
pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt

# SAM3 (install into same env)
git clone https://github.com/facebookresearch/sam3.git
cd sam3 && pip install -e ".[notebooks]" && cd ..

# (Optional) Versioned git hooks
git config core.hooksPath tools/githooks
```

| Category | Packages |
|----------|----------|
| Data | `numpy`, `astropy`, `opencv-python`, `pycocotools`, `Pillow`, `scikit-image` |
| DL | `torch>=2.0`, `torchvision>=0.15` |
| Config | `PyYAML`, `tqdm` |
| Viz | `matplotlib` |

---

## Quick Start

```bash
# 1. Full unified pipeline (Render → GT → SAM3 inference → Export)
conda run --no-capture-output -n sam3 python scripts/data/prepare_unified_dataset.py \
    --config configs/unified_data_prep.yaml

# 2. Noise augmentation (forward observation model)
python scripts/data/generate_noisy_fits.py --config configs/noise_profiles.yaml
python scripts/data/render_noisy_fits.py     --config configs/unified_data_prep.yaml
python scripts/data/build_noise_augmented_annotations.py --config configs/unified_data_prep.yaml

# 3. Galaxy-level train/val split
python scripts/data/split_annotations.py --config configs/sam3_dataset_split.yaml

# 4. Evaluate a checkpoint on one of the three benchmarks
python scripts/eval/evaluate_checkpoint.py --config configs/eval_checkpoint.yaml
```

---

## End-to-End Pipeline

```
                    ┌─────────────────────────────────────────┐
                    │     FITS Surface-Brightness Maps         │
                    │  (FIREbox-DR1 / Fbox-Gold / PNbody)      │
                    └────────────────┬────────────────────────┘
                                     │
          ┌──────────────────────────┼──────────────────────────┐
          ▼                          ▼                          ▼
   ┌─────────────┐          ┌──────────────┐          ┌──────────────┐
   │ Phase 0     │          │ Phase 1      │          │ Phase 2      │
   │ Mask Stats  │          │ RENDER       │          │ GT (3-class) │
   │             │          │              │          │              │
   │ GT geometry │          │ FITS → RGB   │          │ FIREbox SB-  │
   │ → filter    │          │ PNG images   │          │ threshold +  │
   │ thresholds  │          │ (per variant)│          │ instance map │
   └─────┬───────┘          └──────┬───────┘          └──────┬───────┘
         │                         │                         │
         │                         ▼                         │
         │                 ┌────────────────┐                │
         │                 │ Phase 3        │                │
         │                 │ SAM3 Inference │                │
         │                 │                │                │
         └────────────────►│ "satellite     │◄───────────────┘
                           │  galaxy" prompt│
                           │ → Score gate   │
                           │ → Prior filter │
                           │   (relabel     │
                           │   inner_galaxy)│
                           └───────┬────────┘
                                   ▼
                           ┌──────────────┐
                           │ Phase 4      │
                           │ SAM3 Export  │
                           │              │
                           │ COCO JSON    │
                           │ + RLE masks  │
                           │ (3 classes)  │
                           └──────┬───────┘
                                  ▼
                  ┌──────────────────────────────────────┐
                  │       Noise Augmentation              │
                  │  Forward Observation Model (Poisson   │
                  │  + readout noise) at SNR 5/10/20/50   │
                  └──────────────┬───────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────┐
                  │   Galaxy-Level Train / Val Split      │
                  └──────────────┬───────────────────────┘
                                 ▼
                  ┌──────────────────────────────────────┐
                  │   Checkpoint Evaluation               │
                  │   (3 benchmarks, 3 post layers)       │
                  │   Dice · Precision · Recall · HD95    │
                  │   Matched IoU · Instance Recall       │
                  └──────────────────────────────────────┘
```

---

## Detailed Workflow

### Phase 0 — Mask Statistics (one-time)

Computes per-instance geometric statistics from canonical GT masks. Output drives all downstream filter thresholds.

```bash
python scripts/analysis/analyze_mask_stats.py \
    --gt_root data/02_processed/gt_canonical_tidal_v1/current \
    --output_dir outputs/mask_stats
```

<details>
<summary>Output files & filter recommendations</summary>

| File | Content |
|------|---------|
| `mask_instance_stats.csv` | Per-instance: area, solidity, aspect ratios, curvature |
| `mask_stats_summary.json` | Quantile distributions + `filter_recommendations` |

```json
{
  "tidal_features": {
    "min_area": 42, "max_area": 98304,
    "min_solidity": 0.12, "aspect_sym_moment_max": 8.5
  },
  "satellites": {
    "min_area": 14, "max_area": 5200,
    "min_solidity": 0.8852, "aspect_sym_moment_max": 2.6731
  }
}
```

Regenerate the distribution plots:

```bash
python scripts/analysis/plot_mask_stats.py
```

</details>

### Phases 1–4 — Unified Data Preparation

The main pipeline (`scripts/data/prepare_unified_dataset.py`) converts raw FITS data into training-ready datasets. The active GT subdirectory is `gt_canonical_tidal_v1` (legacy `gt_canonical` is left untouched for compatibility).

```bash
# Full pipeline (all 4 phases)
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml

# Run individual phases
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase render
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase gt
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase inference
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --phase export

# Subset of galaxies
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --galaxies 11,13,19

# Force rebuild
python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml --force
```

| Phase | Action | Output |
|-------|--------|--------|
| **1 — Render** | FITS → RGB PNGs per preprocessing variant | `renders/current/{variant}/{base_key}/0000.png` |
| **2 — GT** | SB-threshold masks → 3-class instance maps | `gt_canonical_tidal_v1/current/{base_key}/instance_map_uint8.png` |
| **3 — Inference** | SAM3 `"satellite galaxy"` prompt → score gate → prior filter (relabel hard-center → `inner_galaxy`) | `sam3_predictions_*.json`, overlay, manifest |
| **4 — Export** | SAM3 COCO `annotations.json` (3 categories) | `sam3_prepared_tidal_v1/` |

Tidal features come from FIREbox SB31.5 FITS via `gt.py` rather than from a SAM3 stream prompt — only the satellite prompt runs at inference time on the active path.

### Noise Augmentation

Inject realistic observation noise using a forward model (SB → flux → counts → Poisson → readout → magnitude):

```bash
# Generate noisy FITS at multiple SNR tiers
python scripts/data/generate_noisy_fits.py --config configs/noise_profiles.yaml

# Render noisy FITS → PNG
python scripts/data/render_noisy_fits.py --config configs/unified_data_prep.yaml

# Build noise-augmented COCO annotations
python scripts/data/build_noise_augmented_annotations.py --config configs/unified_data_prep.yaml
```

### Train/Val Split

Galaxy-level splitting ensures no data leakage between train and validation:

```bash
python scripts/data/split_annotations.py --config configs/sam3_dataset_split.yaml
```

### Checkpoint Evaluation

A single CLI evaluates a SAM3 checkpoint across three benchmarks at a fixed 1024×1024 working grid:

| Benchmark mode | Content | ROI / framing |
|---|---|---|
| `fbox_gold_satellites` | 132 satellite samples | ROI `[277:747, 277:747]` (470×470) |
| `firebox_dr1_streams`  | 72 streams samples at SBlim31.5 | full-frame |
| `gt_canonical`         | 70 streams + satellites samples (post-retrain use) | full-frame |

Each sample runs through up to three layers — `raw`, `post_pred_only` (5 prediction-only stages), and `post_gt_aware` (only populated for `gt_canonical`).

```bash
# Default benchmark (set in the YAML)
python scripts/eval/evaluate_checkpoint.py --config configs/eval_checkpoint.yaml

# Regenerate overlays only (no re-inference)
python scripts/eval/evaluate_checkpoint.py --config configs/eval_checkpoint.yaml --overlays-only
```

PR-curve / threshold-sweep utilities live under `scripts/eval/`:

```bash
python scripts/eval/compute_auc_pr.py          # AUC-PR per type
python scripts/eval/auc_pr_threshold_table.py  # Threshold table
python scripts/eval/plot_auc_pr.py             # PR curves
python scripts/eval/post_policy_pr_sweep.py    # Post-policy sweep
```

### AI Verifier Protocol — *experimental, partial*

A separate human-in-the-loop verification track lives under `src/review/`, `scripts/review/`, and `configs/review/`. It is being built to feed corrected labels back into a "Shadow GT" via a hash-validated correction flow. **Status:** rendering / silver-label / example-builder scripts are in place and only a subset of inputs has been prepared so far — whether we run a full verifier loop end-to-end is still being decided. Treat this track as exploratory; none of the main training/evaluation flows depend on it.

```bash
# Render review assets (crops + context + EV stamps)
python scripts/review/render_review_assets.py

# Derive silver labels
python scripts/review/generate_silver_labels.py

# Build verifier examples (business JSONL → chat JSONL via run_etl.py)
python scripts/review/build_verifier_examples.py
python scripts/review/run_etl.py
```

---

## Visualization Scripts

| Script | Purpose | Output |
|--------|---------|--------|
| `scripts/viz/visualize_sam3.py` | 4-column grid: Original / Tidal / Satellites / Combined | `sam3_prepared*/visualizations_grid/*.jpg` |
| `scripts/analysis/plot_mask_stats.py` | Tidal vs satellites shape distribution boxplots | `outputs/mask_stats/*.png` |
| `scripts/analysis/plot_checkpoint_satellite_tradeoff.py` | Checkpoint-level precision/recall trade-off | `outputs/eval_*/tradeoff.png` |

---

## Evaluation Metrics

Two levels of metrics, reported for `raw` (unfiltered), `post_pred_only`, and (where applicable) `post_gt_aware` prediction layers:

### Pixel-Level

| Metric | Formula | Empty-mask handling |
|--------|---------|---------------------|
| **Dice** | 2·TP / (2·TP + FP + FN) | `null` if both empty; `0.0` if one empty |
| **Precision** | TP / (TP + FP) | `null` if no predicted pixels |
| **Recall** | TP / (TP + FN) | `null` if no GT pixels |
| **Hausdorff95** | Symmetric 95th-percentile boundary distance | `null` if both empty; image diagonal if one empty |

### Instance-Level

| Metric | Description |
|--------|-------------|
| **Matched IoU** | Mean IoU of valid matches (Hungarian 1:1 on IoU matrix) |
| **Instance Recall** | Fraction of GT instances matched (IoU ≥ threshold) |

### Satellite Diagnostics (Phase 1)

`src/evaluation/satellite_diagnostics.py` adds a GT-driven, no-IoU-threshold error-typing layer on top of Hungarian matching. Each raw satellite candidate is classified into one of: `compact_complete`, `diffuse_core`, `reject_unmatched`, `reject_low_purity`. Enabled when `diagnostics.enabled: true` in `configs/eval_checkpoint.yaml`.

Aggregation: **Macro** (mean ± std across images) and **Micro** (global TP/FP/FN sums).

---

## Project Structure

```
LSB-AI-Detection/
├── configs/                            # YAML configuration files
│   ├── unified_data_prep.yaml          #   Main 4-phase unified pipeline (FIREbox-DR1)
│   ├── unified_data_prep_pnbody.yaml   #   PNbody variant
│   ├── eval_checkpoint.yaml            #   Single SAM3 checkpoint-eval entrypoint config
│   ├── noise_profiles*.yaml            #   Forward noise model SNR profiles
│   ├── sam3_dataset_split.yaml         #   Galaxy-level train/val split
│   ├── pnbody/                         #   PNbody-specific configs
│   ├── review/                         #   AI Verifier Protocol V1.1 (experimental)
│   │   ├── render_spec_v1.yaml
│   │   ├── prompt_registry_v1.yaml
│   │   └── silver_policy_v1.yaml
│   └── archive/                        #   Archived migration-only configs
│
├── scripts/                            # CLI entry points (by concern)
│   ├── MODULE_DOC.md
│   ├── data/                           #   Dataset build, noise FITS, splits, PNbody
│   │   ├── prepare_unified_dataset.py  #     Main 4-phase pipeline
│   │   ├── generate_noisy_fits.py      #     Forward noise FITS
│   │   ├── render_noisy_fits.py        #     Noisy FITS → PNG
│   │   ├── build_noise_augmented_annotations.py
│   │   ├── split_annotations.py        #     Galaxy-level train/val split
│   │   ├── build_training_dataset.py   #     Merge / symlink COCO sources
│   │   ├── build_fbox_gold_benchmark.py
│   │   ├── generate_pnbody_fits.py     #     PNbody → 24-LOS VIS2 FITS
│   │   ├── generate_pnbody_noisy_fits.py
│   │   └── rebuild_sam3_dataset.py
│   ├── eval/                           #   Single checkpoint-eval entrypoint + PR utilities
│   │   ├── evaluate_checkpoint.py      #     The eval CLI (3 benchmarks, 3 post layers)
│   │   ├── compute_auc_pr.py
│   │   ├── auc_pr_threshold_table.py
│   │   ├── plot_auc_pr.py
│   │   └── post_policy_pr_sweep.py
│   ├── cluster/                        #   Slurm job bodies (site-specific)
│   │   ├── generate_pnbody_fits.slurm
│   │   ├── generate_pnbody_noisy_fits.slurm
│   │   ├── render_pnbody_noisy_fits.slurm
│   │   ├── regen_gt_sam3.slurm
│   │   └── run_pnbody_pseudo_gt_h100.slurm
│   ├── review/                         #   AI Verifier Protocol CLI scripts (experimental)
│   │   ├── render_review_assets.py
│   │   ├── generate_silver_labels.py
│   │   ├── build_verifier_examples.py
│   │   ├── run_etl.py
│   │   ├── import_corrections.py
│   │   ├── assemble_round.py
│   │   ├── edit_authoritative_gt.py
│   │   ├── bootstrap_shadow_gt.py
│   │   └── migrate_satellite_overrides.py
│   ├── viz/                            #   Figures / QA grids
│   │   └── visualize_sam3.py           #     4-column grid visualization
│   └── analysis/                       #   Stats from masks / eval JSON
│       ├── analyze_mask_stats.py       #     GT instance statistics → thresholds
│       ├── plot_mask_stats.py          #     Shape metric boxplots
│       └── plot_checkpoint_satellite_tradeoff.py
│
├── src/                                # Python source package
│   ├── data/                           #   FITS I/O, preprocessing (asinh, linear, multi-exposure)
│   ├── noise/                          #   ForwardObservationModel (SB → flux → Poisson → readout)
│   ├── inference/                      #   SAM3 text-prompt runner
│   │   └── sam3_prompt_runner.py
│   ├── postprocess/                    #   Type-aware filtering pipeline
│   │   ├── satellite_prior_filter.py   #     Area/solidity/aspect rules + inner_galaxy relabel
│   │   ├── streams_sanity_filter.py    #     Area bounds, edge-touch fraction
│   │   ├── core_exclusion_filter.py    #     Centroid radius exclusion (legacy)
│   │   ├── satellite_score_gate.py     #     Tier-based score thresholds (small/medium/large)
│   │   ├── satellite_pipeline.py       #     Composed satellite stages
│   │   ├── candidate_grouping.py       #     Union-Find centroid clustering
│   │   ├── representative_selection.py #     Best mask per group
│   │   ├── stream_satellite_conflict_filter.py
│   │   └── satellite_conflict_resolver.py
│   ├── pipelines/unified_dataset/      #   Modular pipeline subpackage
│   │   ├── taxonomy.py                 #     Canonical 3-class scheme + alias mapping
│   │   ├── inference_sam3.py           #     SAM3 inference orchestration
│   │   └── compose.py / gt.py / render.py / export.py / split.py / noise_aug.py / artifacts.py / ...
│   ├── analysis/                       #   Per-mask geometry metrics
│   ├── evaluation/                     #   Pixel + instance metrics, checkpoint eval orchestration
│   │   ├── checkpoint_eval.py          #     Eval pipeline driver
│   │   ├── metrics.py                  #     Hungarian, primary_gt_match, purity/completeness
│   │   └── satellite_diagnostics.py    #     A/B/C/D classifier (compact/diffuse/reject)
│   ├── review/                         #   AI Verifier (experimental, partial)
│   ├── visualization/                  #   Overlay, grid, comparison plotting
│   ├── models/                         #   Model wrappers
│   └── utils/                          #   COCO RLE, logger, geometry helpers
│
├── data/
│   ├── 01_raw/                         # Raw FITS (FIREbox-DR1, Fbox-Gold-Satellites, PNbody)
│   ├── 02_processed/                   # Pipeline outputs (renders, gt_canonical_tidal_v1, exports)
│   └── 04_noise/                       # Noise-injected FITS
│
├── tests/                              # Unit tests (pytest)
├── docs/                               # Documentation & assets
├── outputs/                            # Evaluation results & plots
├── notebooks/                          # Jupyter notebooks
├── requirements.txt
├── ARCHITECTURE.md                     # Detailed architecture docs
└── CHANGELOG.md                        # Version history
```

---

## Configuration Reference

### `configs/unified_data_prep.yaml`

<details>
<summary>Full config structure (tidal_v1 path)</summary>

```yaml
paths:
  firebox_root: "data/01_raw/LSB_and_Satellites/FIREbox-DR1"
  output_root:  "data/02_processed"
  gt_subdir:        "gt_canonical_tidal_v1"
  pseudo_gt_subdir: "pseudo_gt_canonical_tidal_v1"

data_selection:
  galaxy_ids: [11, 13, 19, ...]
  views: ["eo", "fo"]
  canonical_sb_threshold: 31.5

processing:
  target_size: [1024, 1024]
  infer_preprocessing: "linear_magnitude"

preprocessing_variants:
  - name: "asinh_stretch"
    params: { nonlinearity: 200.0, zeropoint: 22.5, clip_percentile: 99.5 }
  - name: "linear_magnitude"
    params: { global_mag_min: 20.0, global_mag_max: 35.0 }

inference_phase:
  engine: "sam3"           # only 'sam3' is supported
  run_mode: "evaluate"
  input_image_variant: "linear_magnitude"
  sam3:
    checkpoint: "/path/to/sam3/checkpoint.pt"
    bpe_path:   "/path/to/bpe_simple_vocab_16e6.txt.gz"
    confidence_threshold: 0.18
    resolution: 1008
    # tidal_v1: only the satellite prompt runs at inference time.
    # Tidal features come from FIREbox SB31.5 FITS via gt.py.
    prompts:
      - { text: "satellite galaxy", type_label: "satellites",
          confidence_threshold: 0.18 }
    score_gate:
      small_area_max_px: 200
      medium_area_max_px: 600
      small_min_score: 0.60
      medium_min_score: 0.20
      large_min_score: 0.18
    prior_filter:
      area_min: 14
      solidity_min: 0.8852
      aspect_sym_max: 2.6731
      hard_center_radius_frac: 0.03
      hard_center_action: "relabel_inner_galaxy"
    conflict_policy:
      enabled: false
```

</details>

### `configs/eval_checkpoint.yaml`

<details>
<summary>Single checkpoint-eval entrypoint</summary>

```yaml
checkpoint: "scratch/.../checkpoints/checkpoint.pt"

benchmark:
  mode: "fbox_gold_satellites"   # | firebox_dr1_streams | gt_canonical
  allow_instance_drop: false
  fbox:
    manifest:   "data/01_raw/.../Fbox_Gold_Satellites/dataset_manifest.json"
    masks_root: "data/01_raw/.../Fbox_Gold_Satellites/MASKS"
    roi:        "data/01_raw/.../Fbox_Gold_Satellites/roi_definition.json"
  dr1:
    root: "data/01_raw/LSB_and_Satellites/FIREbox-DR1"
    sb_threshold: 31.5
  canonical:
    gt_dir: "data/02_processed/gt_canonical_tidal_v1/current"

render:
  root: "data/02_processed/renders_eval"
  condition: "current"          # current | noisy
  variant: "linear_magnitude"
  noise_profile: null

target_size: [1024, 1024]

sam3:
  bpe_path: "/path/to/bpe_simple_vocab_16e6.txt.gz"
  resolution: 1008
  confidence_threshold: 0.18
  device: "cuda"

prompts:
  fbox_gold_satellites:
    - { text: "satellite galaxy", type_label: "satellites",     confidence_threshold: 0.18 }
  firebox_dr1_streams:
    - { text: "stellar stream",   type_label: "tidal_features", confidence_threshold: 0.30 }
  gt_canonical:
    - { text: "stellar stream",   type_label: "tidal_features", confidence_threshold: 0.30 }
    - { text: "satellite galaxy", type_label: "satellites",     confidence_threshold: 0.18 }

post:
  pred_only:
    enable_streams_sanity: true
    enable_score_gate:     true
    enable_prior_filter:   true
    enable_core_policy:    false
    enable_cross_type_conflict: false
    streams_sanity:
      stats_json: "outputs/mask_stats/mask_stats_summary.json"
      edge_touch_frac: 0.8
    score_gate: { ... }
    prior_filter:
      area_min: 14
      solidity_min: 0.8852
      aspect_sym_max: 2.6731
      hard_center_radius_frac: 0.03
      hard_center_action: "drop"   # eval keeps drop semantics
  gt_aware:
    enable_gt_stream_conflict: false
```

</details>

### `configs/noise_profiles.yaml`

<details>
<summary>Noise model config</summary>

```yaml
paths:
  output_root: "data/04_noise"

noise_model:
  zeropoint: 22.5
  sky_level: 200
  read_noise: 5
  signal_quantile: 0.90
  background_quantile: 0.20

profiles:
  snr05: { target_snr: 5 }
  snr10: { target_snr: 10 }
  snr20: { target_snr: 20 }
  snr50: { target_snr: 50 }
```

</details>

---

## Data Layout

After running the full pipeline:

```
data/02_processed/
├── renders/current/                          # Phase 1 output
│   ├── asinh_stretch/{base_key}/0000.png
│   └── linear_magnitude/{base_key}/0000.png
├── renders/noisy/                            # Noise-augmented renders
│   └── {variant}/snr{05,10,20,50}/{base_key}/0000.png
├── gt_canonical_tidal_v1/current/{base_key}/ # Phase 2+3 output (tidal_v1, 3 classes)
│   ├── tidal_features_instance_map.npy
│   ├── instance_map_uint8.png                #   Combined 3-class instance map
│   ├── instances.json                        #   Instance ID → type mapping
│   ├── manifest.json                         #   Provenance metadata
│   └── overlay.png                           #   QA visualization
├── renders_eval/                             # Eval-only renders (Fbox-Gold, DR1-streams)
│   ├── fbox_gold_satellites/{condition}/{variant}/...
│   └── firebox_dr1_streams/{condition}/{variant}/...
└── sam3_prepared_tidal_v1/                   # Phase 4: SAM3 export
    ├── images/{variant_key}.png → (symlink)
    ├── annotations.json                      #   COCO format with RLE masks (3 categories)
    └── visualizations_grid/                  #   QA grid images
```

Where `{base_key}` = `{galaxy_id:05d}_{view}` (e.g. `00011_eo`).

---

## Module Documentation

Each source module has its own `MODULE_DOC.md` with detailed API docs:

| Module | Description | Docs |
|--------|-------------|------|
| `src/data/` | FITS I/O, preprocessing | [MODULE_DOC.md](src/data/MODULE_DOC.md) |
| `src/noise/` | Forward observation model | [MODULE_DOC.md](src/noise/MODULE_DOC.md) |
| `src/inference/` | SAM3 prompt runner | [MODULE_DOC.md](src/inference/MODULE_DOC.md) |
| `src/postprocess/` | Type-aware filter pipeline | [MODULE_DOC.md](src/postprocess/MODULE_DOC.md) |
| `src/pipelines/` | Unified dataset pipeline (3-class taxonomy) | [MODULE_DOC.md](src/pipelines/MODULE_DOC.md) |
| `src/analysis/` | Mask geometry metrics | [MODULE_DOC.md](src/analysis/MODULE_DOC.md) |
| `src/evaluation/` | Pixel + instance metrics, satellite diagnostics | [MODULE_DOC.md](src/evaluation/MODULE_DOC.md) |
| `src/visualization/` | Overlay & QA plotting | [MODULE_DOC.md](src/visualization/MODULE_DOC.md) |
| `src/utils/` | COCO RLE, geometry, logger | [MODULE_DOC.md](src/utils/MODULE_DOC.md) |
| `scripts/` | CLI entry points | [MODULE_DOC.md](scripts/MODULE_DOC.md) |

---

## Acknowledgements

- **[FIREbox-DR1](https://fire.northwestern.edu/)** — Cosmological simulation data
- **[SAM3](https://github.com/facebookresearch/sam3)** — Segment Anything Model 3 (Meta AI)
