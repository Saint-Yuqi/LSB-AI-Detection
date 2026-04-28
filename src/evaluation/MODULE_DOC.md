# Module: evaluation

## Responsibilities
- Calculate rigorous Intersection over Union (IoU) scores for binary and instance segmentation tasks.
- Perform robust ground-truth-to-prediction bipartite matching to evaluate Instance Recall accurately.
- Aggregate type-aware SAM3 evaluation results and serialize reproducible run artifacts.

## Non-goals
- **No Sweep Management:** Does not orchestrate hyperparameter sweeps or score ranking.
- **No Mask Generation:** Does not interface with models.

## Type Aliases

### `MaskDict` (TypedDict, STRICTLY CLOSED)
- `segmentation: np.ndarray` (boolean, HxW)
- `area: int`
- `bbox: List[int]` ([x0, y0, w, h])
- `score: float`
- `point_coords: List[List[float]]`
- `stability_score: float`
- `crop_box: List[int]`

## Inputs / Outputs

### `calculate_iou(pred_mask, gt_mask)`
- **Input:**
  - `pred_mask: np.ndarray` — shape `(H, W)`, dtype `np.uint8`
  - `gt_mask: np.ndarray` — shape `(H, W)`, dtype `int` or `np.uint8`
- **Output:** `float` — bounded in `[0.0, 1.0]`

### `calculate_instance_iou(pred_masks, gt_mask)`
- **Input:**
  - `pred_masks: List[MaskDict]`
  - `gt_mask: np.ndarray` — shape `(H, W)`, dtype `int`
- **Output:** `InstanceIoUResult` (TypedDict, all keys required):
  - `binary_iou: float`
  - `mean_instance_iou: float`
  - `num_gt_instances: int`
  - `num_pred_masks: int`
  - `instance_ious: List[float]`

### `calculate_matched_metrics(pred_masks, gt_mask, iou_threshold)`
- **Input:**
  - `pred_masks: List[MaskDict]`
  - `gt_mask: np.ndarray` — shape `(H, W)`, dtype `int`
  - `iou_threshold: float`
- **Output:** `MatchedMetricsResult` (TypedDict, all keys required):
  - `recall: float`
  - `matched_iou: float`
  - `num_detected: int`
  - `num_gt_instances: int`
  - `per_instance_details: List[InstanceDetail]`
    - `InstanceDetail` (TypedDict, all keys required):
      - `gt_instance_id: int`
      - `best_pred_idx: int`
      - `best_iou: float`
      - `detected: bool`
      - `gt_area: int`
      - `gt_bbox: List[int]` — `[x_min, y_min, width, height]`

### `calculate_pixel_metrics(pred_mask, gt_mask)`
- **Input:**
  - `pred_mask: np.ndarray` — shape `(H, W)`, dtype `bool` or binary integer
  - `gt_mask: np.ndarray` — shape `(H, W)`, dtype `bool` or binary integer
- **Output:** `Dict[str, Any]` with keys:
  - `dice: Optional[float]`
  - `precision: Optional[float]`
  - `recall: Optional[float]`
  - `capped_hausdorff95: Optional[float]`
  - `tp: int`, `fp: int`, `fn: int`

### `calculate_optimal_instance_metrics(pred_masks, gt_instance_map, iou_threshold)`
- **Input:**
  - `pred_masks: List[MaskDict]`
  - `gt_instance_map: np.ndarray` — shape `(H, W)`, instance IDs (`0` = background)
  - `iou_threshold: float`
- **Output:** `Dict[str, Any]` with keys:
  - `instance_recall: Optional[float]`
  - `matched_iou: Optional[float]`
  - `unmatched_iou: Optional[float]`
  - `num_gt: int`, `num_detected: int`, `num_pred: int`
  - `per_instance_details: List[Dict[str, Any]]`

### Pred-centric primitives (no IoU threshold)
These are reusable building blocks for the diagnostic layer (and any
notebook / experiment that asks "what GT does this prediction primarily
hit?"). They do NOT touch `calculate_optimal_instance_metrics`, which
remains the Hungarian-based GT-centric evaluator.

#### `primary_gt_match(pred_bin, gt_instance_map)`
- **Input:**
  - `pred_bin: np.ndarray` — `(H, W)` bool, a single prediction.
  - `gt_instance_map: np.ndarray` — `(H, W)` int, 0 = bg, positive = instance ID.
- **Output:** `Dict[str, Any]`:
  - `matched_gt_id: Optional[int]` — arg-max positive ID, `None` iff zero overlap.
  - `overlap_px: int`
  - `pred_area: int`
  - `matched_gt_area: Optional[int]` — `None` iff `matched_gt_id is None`.
- **Tie-break:** first-argmax (smallest positive GT ID wins).

#### `derive_purity_completeness(overlap_px, pred_area, matched_gt_area)`
- Pure scalar helper. Single-prediction, no global dependency.
- **Output:** `Dict[str, Optional[float]]`:
  - `purity: float` — `overlap_px / pred_area` (0.0 when `pred_area == 0`).
  - `completeness: Optional[float]` — `None` iff `matched_gt_area is None`.
  - `seed_gt_ratio: Optional[float]` — `None` iff `matched_gt_area is None`.

#### `compute_one_to_one_flags(pred_bins, gt_instance_map)`
- Batch helper. Builds the `(N_pred × N_gt)` overlap matrix once.
- **Output:** `List[bool]`, same length as `pred_bins`. `True` iff the
  prediction is the arg-max-overlap prediction for its primary GT.
- **Tie-break:** first-argmax — lowest list index wins on ties.

### Satellite diagnostic taxonomy (`satellite_diagnostics.py`)
Phase-1, GT-driven, offline. Classifies raw satellite predictions from
matched-GT overlap, purity/completeness, and a one-to-one seed/GT ratio
guard (no IoU threshold). Four labels:
`compact_complete`, `diffuse_core`, `reject_unmatched`, `reject_low_purity`.
`reject_host_background` is reserved for Phase 2 (needs host-support
loader).

- `DiagnosticCfg` — frozen dataclass with `min_purity_for_match`,
  `completeness_complete`, `complete_one_to_one_min_completeness`,
  `complete_one_to_one_max_seed_ratio`, `annulus_r_in_frac`,
  `annulus_r_out_frac`, `radial_n_rings`. Populated from
  `diagnostics.satellites` in `configs/eval_checkpoint.yaml`.
- `CandidateRow` (TypedDict, STRICTLY CLOSED) — one row per raw satellite
  prediction, keyed by `(raw_index, candidate_id)`. Fields:
  - `raw_index: int`, `candidate_id: str` — stable source-raw identity,
    matches `predictions_raw.json`.
  - `seed_area: int`, `confidence_score: float` — `confidence_score`
    replaces SAM3's mask `score` at the diagnostic boundary to avoid
    confusion with "prediction vs GT IoU".
  - `matched_gt_id: Optional[int]`, `matched_gt_area: Optional[int]`,
    `overlap_px: int`.
  - `purity: float`, `completeness: Optional[float]`,
    `seed_gt_ratio: Optional[float]`.
  - `is_one_to_one: bool`.
  - `host_background_frac: Optional[float]` — always `None` in Phase 1.
  - `intersects_roi: bool` — legacy field name for ROI membership. The
    membership rule is mask-centroid-in-ROI, so masks that only touch the ROI
    boundary from outside are not counted in `counts_*_roi`.
  - `annulus_excess: Optional[float]`, `radial_monotonicity: Optional[float]`.
  - `taxonomy_label: str`, `label_reason: str`.
- `SatelliteDiagnosticReport` (TypedDict) — `per_candidate`,
  `counts_by_label`, `counts_by_label_roi`, `counts_post_by_label`,
  `counts_post_by_label_roi`, `thresholds_used`, `host_support_available`.
- `build_candidate_table(raw_sats, gt_sat_map, render_signal, H, W, cfg, roi_bbox=None, host_support=None, post_sats=None) -> SatelliteDiagnosticReport`
  - `render_signal` MUST be `(H, W) float32` — the same intensity array
    SAM3 saw at inference. No magnitude-to-flux inversion. The eval
    script handles RGB → grayscale conversion once per sample.
  - `per_candidate` rows cover EVERY raw satellite prediction.
  - When `post_sats` is supplied, post-filtered candidates are classified
    with the same taxonomy and exposed as aggregate `counts_post_*`
    fields in `diagnostics.json` and `diagnostics_summary`.
- `classify(matched_gt_id, purity, completeness, seed_gt_ratio, is_one_to_one, cfg) -> (label, reason)`
  — decision order:
  `reject_unmatched` -> one-to-one near-complete ratio guard ->
  `reject_low_purity` -> `compact_complete` -> `diffuse_core`.
- `aggregate_diagnostics(per_sample_rows: list[list[CandidateRow]]) -> dict`
  — global counts + quantile summaries of `purity`, `completeness`,
  `seed_gt_ratio`, `annulus_excess`, `radial_monotonicity` per label.

### `compute_sample_report(..., render_signal=None)` — report schema update
- Returns a **two-tuple** `(report, diag_report)`:
  - `report` is the existing per-sample dict, now with an optional
    `report["diagnostics"]["satellites_raw"]` block that holds ONLY a
    `summary` and a relative `per_candidate_path` (default
    `"diagnostics.json"`). Per-candidate rows are **never** embedded in
    `report`.
  - `diag_report` is the full `SatelliteDiagnosticReport`, or `None` when
    the benchmark is streams-only, `render_signal` is `None`, or
    `diagnostics.enabled` is `False`.
- The caller writes `sample_dir/diagnostics.json` from `diag_report` and
  feeds per-sample `diag_report["per_candidate"]` lists plus optional
  `counts_post_*` summaries to `aggregate_diagnostics(...)`.
  `aggregate(reports)` signature is unchanged — typed-block summaries only,
  no sidecar reads.

### Checkpoint Evaluation Utilities (`checkpoint_eval.py`)
Single SAM3 evaluation entrypoint backing `scripts/eval/evaluate_checkpoint.py`.
Drives three benchmarks (`fbox_gold_satellites`, `firebox_dr1_streams`,
`gt_canonical`) at a fixed 1024×1024 working grid with three layers:
`raw`, `post_pred_only`, and (where applicable) `post_gt_aware`.

- `Sample` (TypedDict) — one render/GT pair plus benchmark/ROI metadata,
  rasterized GT maps, and the optional `gt_rles_by_type` / `gt_path_version`
  fields used on the tidal_v1 path.
- `compute_sample_report(sample, raw_masks, po_pair, ga_pair, iou_thresh, render_signal=None) -> (report, diag_report)`
  — per-sample report builder. Returns `(report, diag_report)` where
  `diag_report` is `None` for streams-only benchmarks or when diagnostics
  are disabled.
- `aggregate(reports) -> dict` — typed-block macro/micro summaries.
- The legacy SAM3 utility module `sam3_eval.py` (and its
  `discover_pairs / run_and_evaluate / aggregate_results / save_results /
  save_eval_overlay` helpers) has been removed; checkpoint_eval is the
  single replacement.

## Invariants
- **Missing Annotations:** GT instances with pixel ID 0 (background) are excluded from recall calculation.
- **Shape Contract:** `pred_masks[i]['segmentation']` and `gt_mask` must share identical `(H, W)`; mismatched shapes raise `ValueError` before any computation.
- **Empty GT Boundary:** When `gt_mask` contains no positive-label instances (all-zero ground truth), `recall` returns `0.0` and `num_gt_instances` returns `0`.

## Produced Artifacts
- In-memory dicts containing per-instance evaluation details and aggregate scalar metrics.
- Serialized SAM3 evaluation JSON reports and optional type-aware QA overlays.

## Failure Modes
- `ValueError`: Raised when `pred_mask.shape != gt_mask.shape`.
- `KeyError`: Raised when any element of `pred_masks` is missing the `segmentation` key.

## Tidal_v1 (3-class) Additions

### `metrics.calculate_optimal_instance_metrics_rle(pred_rle_list, gt_rle_list, iou_threshold)`
- RLE-aware variant of `calculate_optimal_instance_metrics`. Same return shape; operates on COCO RLE dicts directly so within-class overlap is preserved end-to-end.

### `metrics.rasterize_per_class_rles(rle_list, H, W)`
- Pack per-class RLEs into an `(H, W)` int32 raster with last-wins overlap. Used by `_typed_blocks_tidal_v1` to keep the legacy taxonomy + pixel-metric paths working unchanged.

### `Sample` (extended)
- New fields: `gt_rles_by_type: dict[str, list[dict]] | None` (per-class GT RLE lists on the new path) and `gt_path_version: str` (`"legacy"` default or `"tidal_v1"`).

### `checkpoint_eval._build_tidal_v1_sample(subdir, key_match, render_cfg, gt_dir_root)`
- Build a `Sample` from a per-key new-path GT subdir. Decodes tidal_features RLEs from the per-class `.npy`, pulls satellites + inner_galaxy RLEs from `sam3_predictions_post.json` keyed by `raw_index`. Fail-closed: raises `KeyError` when a row carrying `source: "sam3_post"` is missing from the predictions JSON.

### `checkpoint_eval._typed_blocks_tidal_v1(stream_masks, sat_masks, sample, iou_thresh, diag_cfg)`
- Emits 3 type blocks on the new path: `tidal_features` (RLE Hungarian), `satellites` (hybrid: taxonomy + pixel from rastered GT plus an `instance_rle` RLE Hungarian sub-block), `inner_galaxy` (RLE Hungarian only). Dispatched from `_typed_blocks` when `sample.gt_path_version == "tidal_v1"`.

### `checkpoint_eval._compute_slice_block_rle(pred_masks, gt_rles, iou_thresh, H, W)`
- RLE-aware variant of `_compute_slice_block`. Same dict shape (detection / pixel / per_instance / `is_empty_trivial`) so callers see a consistent block layout.

### `checkpoint_eval._rle_typed_entry(pred_masks, gt_rles, roi_bbox, iou_thresh, H, W)`
- `{full_frame, roi}` block. ROI restriction uses centroid-in-roi for predictions and any-pixel-in-roi for GT RLEs.

## Failure Modes (tidal_v1)
- `KeyError`: `_build_tidal_v1_sample` raises when a row carries `source: "sam3_post"` but its `raw_index` is missing from `sam3_predictions_post.json` — refusing to silently fall back to a per-class map decode that would collapse within-class overlap.
