# Module: scripts

## Responsibilities
- Serve as the executable entry points (CLI) orchestrating the complete LSB AI Detection pipeline.
- Tie together discrete structural modules into end-to-end workflows.
- Manage dataset generation, model evaluation, and visual diagnostics.
- Run folder-based SAM3 evaluation over rendered/noisy image sets with reproducible JSON outputs.
- Generate PNbody multi-view FITS datasets and metadata manifests from halo HDF5 inputs.

## Non-goals
- **No Core Logic:** Scripts must defer complex math, astrophysical conversions, filtering logic, and metric calculations to downstream modules.
- **No Hardcoding Paths:** Scripts must rely on externally provided configuration schemas for resolving dependencies.

## Layout
- `scripts/data/` — Unified pipeline, noise FITS, renders, COCO splits, training merge, PNbody FITS.
- `scripts/eval/` — SAM2/SAM3 evaluation CLIs, deprecated `eval_model`/`evaluate_model` shims, local batch/bash sweep helpers.
- `scripts/cluster/` — Slurm job bodies and `sbatch` launchers (partition, modules, mail are site-specific; edit before use).
- `scripts/viz/` — SAM2/SAM3 grids, eval metric charts, mask overlays.
- `scripts/analysis/` — Mask statistics CSV/JSON analysis and plotting helpers.

## Inputs / Outputs

### Dataset Generator (`data/prepare_unified_dataset.py`)
- **Input:** `--config: Path` pointing to a `.yaml` file (e.g. `configs/unified_data_prep.yaml`).
  - `--phase`: `render | gt | inference | export | all`
  - `--galaxies`: Comma-separated galaxy IDs subset
  - `--force` / `--force-variants`: Force rebuild
- **Output:**
  - Phase 1: `renders/current/{variant}/{base_key}/0000.png`
  - Phase 2: `gt_canonical/current/{base_key}/streams_instance_map.npy`
  - Phase 3: `gt_canonical/current/{base_key}/instance_map_uint8.png` (SAM2) or `sam3_predictions_*.json` (SAM3)
  - Phase 4: `sam2_prepared/` symlinks + `sam3_prepared/annotations.json`

### PNbody 24-View FITS Generator (`data/generate_pnbody_fits.py`)
- **Input:** `--config: Path` pointing to a `.yaml` file (e.g. `configs/pnbody/firebox_pnbody_24los.yaml`).
  - `--galaxies`: Optional comma-separated galaxy ID subset override for smoke tests
  - `--dry-run` (flag): Log `mockimgs_sb_compute_images` commands without executing them
- **YAML schema (`--config`):**
  - Required keys:
    - `halo_root: str`
    - `halo_pattern: str` with `{galaxy_id}` formatting placeholder
    - `galaxy_ids: List[int]`
    - `los_file: str`
    - `distance: float`
    - `instrument_file: str`
    - `output_root: str`
    - `metadata_root: str`
  - Optional keys:
    - `rsp_opts.rsp_mode: str` (defaults to `"None"`)
    - `rsp_opts.rsp_fac: float` (defaults to `0.6`)
- **Output:**
  - `{output_root}/magnitudes-Fbox-{galaxy_id}-los{00..23}-VIS2.fits.gz`
  - `{metadata_root}/views.csv` — rows with columns:
    - `galaxy_id: int`
    - `view_id: str`
    - `los_x: float`
    - `los_y: float`
    - `los_z: float`
    - `source_hdf5: str`
    - `output_fits: str`
  - `{metadata_root}/generation_manifest.json` — top-level keys:
    - `generated_at: str`
    - `config: str`
    - `distance_mpc: float`
    - `rsp_opts: Dict`
    - `instrument_file: str`
    - `n_galaxies: int`
    - `galaxies: List[Dict]` with per-galaxy `halo_file`, `n_views`, and `views`

### SAM2 Evaluation (`eval/evaluate_sam2.py`, formerly `eval/eval_model.py`)
- **Input:** `--config: Path` (`configs/eval_sam2.yaml` by default)
- **Output:** `{output_dir}/iou_results_{timestamp}.json`
- Note: `eval/eval_model.py` is a deprecated thin wrapper that emits `FutureWarning`.

### Galaxy-Level COCO Split (`data/split_annotations.py`)
- **Input:** `--annotations: Path` (`data/02_processed/sam3_prepared/annotations.json`)
- **Output:** `annotations_train.json`, `annotations_val.json`, `split_manifest.json`

### Folder-Based SAM3 Evaluation (`eval/evaluate_sam3.py`, formerly `eval/evaluate_model.py`)
- **Input:** `--config: Path` (`configs/eval_sam3.yaml` by default), plus optional overrides:
  - `--render-dir: Path`
  - `--gt-dir: Path`
  - `--output-dir: Path`
  - `--max-samples: int`
  - `--per-galaxy` (flag)
  - `--snr-tag: str`
  - `--save-overlays` (flag)
- **YAML schema (`--config`):**
  - Required keys:
    - `sam3.checkpoint: str`
    - `sam3.bpe_path: str`
    - `paths.render_dir: str`
    - `paths.gt_dir: str`
    - `paths.output_dir: str`
  - Optional keys:
    - `sam3.resolution: int`
    - `target_size: List[int]` (`[H, W]`, defaults to `[1024, 1024]`)
    - `prompts: List[Prompt]`
      - `Prompt.text: str`
      - `Prompt.type_label: str`
      - `Prompt.confidence_threshold: float`
    - `match_iou_thresh: float`
    - `post_filter.min_area: int`
    - `post_filter.max_area_frac: float`
    - `post_filter.edge_touch_frac: float`
- **Output:**
  - `{output_dir}/eval_results_{timestamp}.json` — top-level keys:
    - `config: Dict`
    - `summary: Dict` (`overall` and optional `per_galaxy`)
    - `per_image: List[Dict]` (type-aware metrics: `streams`, `satellites`, `combined` with `raw`/`post`)
    - `created_at: str` (ISO datetime)
  - Optional QA overlays when `--save-overlays` is set:
    - `{output_dir}/overlays/{base_key}_overlay.png`

### Analysis & Plotting (`analysis/analyze_mask_stats.py`, `analysis/plot_*.py`, `viz/visualize_*.py`)
- **Input:** `--stats_csv: Path` — CSV file with EXACT columns: [`image_id: str`, `total_masks: int`, `core_hits: int`, `passed_prior: int`, `runtime_ms: float`]. Or `--masks_dir: Path` containing per-sample `.json` files.
  - Per-sample JSON: `List[MaskDictSerialized]`, each element (TypedDict, STRICTLY CLOSED):
    - `segmentation: Dict` — RLE-encoded mask with keys `{"counts": str, "size": List[int]}` ([H, W])
    - `area: int`
    - `bbox: List[int]` ([x0, y0, w, h])
    - `predicted_iou: float`
    - `point_coords: List[List[float]]`
    - `stability_score: float`
    - `crop_box: List[int]`
    - `area_clean: int`
    - `solidity: float`
    - `aspect_sym_moment: float`
    - `dist_to_center: float`
    - `group_id: int`
    - `reject_reason: str`
- **Output:**
  - `mask_stats_summary.json` — schema:
    - `streams`:
      - `count: int`
      - `area_mean: float`
      - `solidity_mean: float`
      - `aspect_mean: float`
    - `satellites`: (same sub-schema as `streams`)
  - One `.png` figure per metric × feature_type combination. Metric set (EXHAUSTIVE): [`area_clean`, `solidity`, `aspect_sym_moment`, `aspect_sym_boundary`, `curvature_ratio`, `dist_to_center`]. Feature types: [`streams`, `satellites`]. File naming: `{metric}_{feature_type}.png`, written as 3-channel RGB at configurable DPI resolution (`int`). Total output cardinality: `6 metrics × 2 feature_types = 12 files`.

## Invariants
- **Reproducibility:** Evaluation runs serialize the exact `.yaml` config snapshot alongside output artifacts in the same directory.
- **Dependency Inversion:** All path resolution is driven by CLI arguments; no paths are hardcoded in script bodies.
- **PNbody View Cardinality:** PNbody FITS generation requires exactly 24 LOS vectors and emits canonical `los00`..`los23` outputs.
- **CLI Fallback:** PNbody FITS generation prefers `mockimgs_sb_compute_images` on `PATH`, but may fall back to the repo-local script entry point.

## Produced Artifacts
- Configured inference datasets.
- Evaluation run matrices resolving to static JSON/CSV artifacts.
- Performance plots evaluating validation baselines.
- Type-aware SAM3 evaluation JSON reports and optional overlay visualizations.
- PNbody FITS cubes plus LOS metadata tables and generation manifests.

## Failure Modes
- `FileNotFoundError`: Raised when any CLI-provided path (`--config`, `--dataset_dir`, `--checkpoint`, `--masks_dir`) does not exist on the filesystem.
- `yaml.YAMLError`: Raised when the `.yaml` config file cannot be parsed (syntax error).
- `KeyError`: Raised when the parsed config dict is missing a required key (e.g., `output_dir`, `sb_thresholds`, `model_cfg`, `iou_threshold`).
- `RuntimeError`: Raised when GPU memory is exhausted during inference (OOM).
- `ValueError`: Raised by `data/generate_pnbody_fits.py` when the LOS table does not contain exactly 24 vectors.
- `subprocess.CalledProcessError`: Raised by `data/generate_pnbody_fits.py` when `mockimgs_sb_compute_images` exits non-zero for a halo/view job.
 
