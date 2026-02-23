# Module: evaluation

## Responsibilities
- Calculate rigorous Intersection over Union (IoU) scores for binary and instance segmentation tasks.
- Perform robust ground-truth-to-prediction bipartite matching to evaluate Instance Recall accurately.

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

## Invariants
- **Missing Annotations:** GT instances with pixel ID 0 (background) are excluded from recall calculation.
- **Shape Contract:** `pred_masks[i]['segmentation']` and `gt_mask` must share identical `(H, W)`; mismatched shapes raise `ValueError` before any computation.
- **Empty GT Boundary:** When `gt_mask` contains no positive-label instances (all-zero ground truth), `recall` returns `0.0` and `num_gt_instances` returns `0`.

## Produced Artifacts
- In-memory dicts containing per-instance evaluation details and aggregate scalar metrics.

## Failure Modes
- `ValueError`: Raised when `pred_mask.shape != gt_mask.shape`.
- `KeyError`: Raised when any element of `pred_masks` is missing the `segmentation` key.
