# Module: scripts

## Responsibilities
- Serve as the executable entry points (CLI) orchestrating the complete LSB AI Detection pipeline.
- Tie together discrete structural modules into end-to-end workflows.
- Manage dataset generation, hyperparameter sweeping, model evaluation, and visual diagnostics.

## Non-goals
- **No Core Logic:** Scripts must defer complex math, astrophysical conversions, filtering logic, and metric calculations to downstream modules.
- **No Hardcoding Paths:** Scripts must rely on externally provided configuration schemas for resolving dependencies.

## Inputs / Outputs

### Dataset Generators (`build_dataset.py`, `prepare_unified_dataset.py`)
- **Input:** `--config: Path` pointing to a `.yaml` file.
  - YAML schema:
    - Required keys:
      - `output_dir: str`
      - `sb_thresholds: List[float]`
      - `galaxy_ids: List[int]`
      - `orientations: List[str]`
      - `preprocessor: str`
      - `preprocessor_kwargs`:
        - `target_size: int`
        - `nonlinearity: float`
        - `zeropoint: float`
    - Optional keys:
      - `preprocessor_kwargs.b_mode: str` — defaults to `'add'` if omitted
- **Output (per sample):**
  - `{output_dir}/img_folder/{sample_name}/0000.png` — `np.ndarray (H, W, 3)`, dtype `np.uint8`
  - `{output_dir}/gt_folder/{sample_name}/0000.png` — `np.ndarray (H, W)`, dtype `np.uint8`

### Inference & Evaluation (`sweep_automask_configs.py`, `eval_model.py`)
- **Input:** `--dataset_dir: Path`, `--checkpoint: Path`, `--config: Path`.
  - YAML schema (all keys required):
    - `model_cfg: str`
    - `device: str`
    - `iou_threshold: float`
    - `param_grid: List[ParamGridEntry]`
      - `ParamGridEntry` (TypedDict, all keys required):
        - `points_per_side: int`
        - `pred_iou_thresh: float`
        - `stability_score_thresh: float`
        - `min_mask_region_area: int`
- **Output:**
  - `per_image_metrics.csv` — columns: `sample_name: str`, `recall: float`, `binary_iou: float`, `mean_instance_iou: float`, `num_gt_instances: int`, `num_pred_masks: int`
  - `summary.json` — schema:
    - `mean_recall: float`
    - `mean_binary_iou: float`
    - `num_samples: int`

### Analysis & Plotting (`analyze_mask_stats.py`, `plot_*.py`, `visualize_*.py`)
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

## Produced Artifacts
- Configured inference datasets.
- Evaluation run matrices resolving to static JSON/CSV artifacts.
- Performance plots evaluating validation baselines.

## Failure Modes
- `FileNotFoundError`: Raised when any CLI-provided path (`--config`, `--dataset_dir`, `--checkpoint`, `--masks_dir`) does not exist on the filesystem.
- `yaml.YAMLError`: Raised when the `.yaml` config file cannot be parsed (syntax error).
- `KeyError`: Raised when the parsed config dict is missing a required key (e.g., `output_dir`, `sb_thresholds`, `model_cfg`, `iou_threshold`).
- `RuntimeError`: Raised when GPU memory is exhausted during inference (OOM).
