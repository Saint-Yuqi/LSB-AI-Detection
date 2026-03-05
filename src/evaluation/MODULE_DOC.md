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
- `predicted_iou: float`
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

### SAM3 Evaluation Utilities (`sam3_eval.py`)
- `discover_pairs(render_dir, gt_dir, max_samples)` → `List[Dict]` of render/GT pair metadata.
- `run_and_evaluate(runner, pair, prompts, H_work, W_work, match_iou_thresh, streams_filter)` → per-image type-aware metrics dict.
- `aggregate_results(per_image, group_by)` → macro/micro summaries (`overall` or `galaxy` grouping).
- `save_results(output_dir, config, summary, per_image)` → writes `eval_results_{timestamp}.json`.
- `save_eval_overlay(path, render_path, gt_streams_map, gt_satellites_map, stream_masks, satellite_masks)` → QA PNG overlay.

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
