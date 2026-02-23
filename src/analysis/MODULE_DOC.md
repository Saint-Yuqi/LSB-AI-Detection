# Module: analysis

## Responsibilities
- Compute mathematically rigorous geometric metrics for binary masks.
- Aggregate and rank configuration experiments during hyperparameter sweeps using custom scalar metrics.

## Non-goals
- **No Mask Filtering:** Does not actively drop or reject any masks.
- **No Inference:** Does not run neural network passes.

## Type Aliases

### `MaskDict` (TypedDict, STRICTLY CLOSED)
- `segmentation: np.ndarray` (boolean, HxW)
- `area: int`
- `bbox: List[int]` ([x0, y0, w, h])
- `predicted_iou: float`
- `point_coords: List[List[float]]`
- `stability_score: float`
- `crop_box: List[int]`

## Inputs / Outputs

### `compute_mask_metrics(seg, H, W, compute_hull)`
- **Input:**
  - `seg: np.ndarray` — shape `(H, W)`, dtype `np.uint8`
  - `H: int`, `W: int`
  - `compute_hull: bool`
- **Output:** dict with exactly these keys (all required, `Optional` where noted):
  - `area_raw: int`
  - `area_clean: int`
  - `bbox_w: int`
  - `bbox_h: int`
  - `aspect_sym_moment: float`
  - `aspect_sym_boundary: Optional[float]` — `None` when `compute_hull=False` or contour fit fails
  - `curvature_ratio: Optional[float]` — `None` when skeleton cannot be computed
  - `aspect_sym: float`
  - `solidity: Optional[float]` — `None` when convex hull area is zero
  - `centroid_xy: Tuple[float, float]`
  - `centroid_x: float`
  - `centroid_y: float`
  - `dist_to_center: float`

### `append_metrics_to_masks(masks, H, W, compute_hull)`
- **Input:** `masks: List[MaskDict]`
  - Schema per Dict (STRICTLY CLOSED):
    - `segmentation: np.ndarray` (boolean, HxW)
    - `area: int`
    - `bbox: List[int]` ([x0, y0, w, h])
    - `predicted_iou: float`
    - `point_coords: List[List[float]]`
    - `stability_score: float`
    - `crop_box: List[int]`
- **Output:** `None` — mutates each dict in-place, appending exactly the keys listed under `compute_mask_metrics`.

### `summarise_config(config_dir)`
- **Input:** `config_dir: Path`
- **Output:** dict with exactly:
  - `n_images: int`
  - `mean_time_ms: float`
  - `time_p50: float`
  - `time_p95: float`
  - `mean_N_raw: float`
  - `mean_N_groups: float`
  - `mean_N_reps: float`
  - `mean_N_dup_rejected: float`
  - `mean_dup_reject_rate: float`
  - `mean_N_keep_prior: float`
  - `mean_N_ambiguous: float`
  - `mean_ambiguous_rate: float`
  - `mean_N_core_rejected: float`
  - `mean_core_rate: float`
  - `mean_N_keep_final: float`
  - `std_N_keep_final: float`
  - `CV_N_keep_final: float`
  - `mean_stability_score_of_kept: float`
  - `score: float`

### `aggregate_and_rank(output_root)`
- **Input:** `output_root: Path`
- **Output:** `List` sorted descending by `score`, each element containing the same keys as `summarise_config` plus:
  - `config_name: str`

## Invariants
- **Data Decoding:** Input `seg` arrays must be fully decoded spatial binaries (`np.uint8`); the module does not decode RLE or compressed formats.
- **Single-Component Contract:** Outputs a single set of scalar shape metrics per `segmentation` input; zero-foreground masks return `None` for those keys.
- **Numeric Stability:** Aspect ratios are bounded to `[0.0, 1.0]` and are rotation-invariant; division-by-zero conditions yield `None` instead of raising.
- **Zero-Foreground Boundary:** When the input mask contains zero foreground pixels, metric keys documented as `Optional` return `None`.

## Produced Artifacts
- **In-Memory:** Enriches existing mask dictionaries with new geometric scalar keys.
- **On-Disk:** Generates `summary.json` per `config_dir` and a global `ranking.json` under `output_root`.

## Failure Modes
- `ImportError`: Raised when `curvature_ratio` or `aspect_sym_boundary` is requested but `skimage` or `cv2` is not installed.
- `ValueError`: Raised when `seg.dtype` is not `np.uint8`, or when `H`/`W` do not match `seg.shape`.
- `KeyError`: Raised by `append_metrics_to_masks` when a mask dict is missing the `segmentation` key.
