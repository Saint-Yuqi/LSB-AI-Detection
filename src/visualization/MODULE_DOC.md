# Module: visualization

## Responsibilities
- Generate diagnostic overlay images drawing colored contours on top of RGB physical data.
- Create multi-panel comparative plots (Input vs. Prediction vs. Ground Truth).

## Non-goals
- **No Evaluation:** Plotting functions do not calculate or return any metrics.
- **No Image Processing:** Assumes the input image is already a normalized, plottable RGB array.

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

### `save_overlay(image, kept, core_rejected, prior_rejected, duplicate_rejected, ambiguous, out_path, draw_prior, draw_duplicate, draw_ambiguous)`
- **Input:**
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `kept: List[MaskDict]`
  - `core_rejected: List[MaskDict]`
  - `prior_rejected: List[MaskDict]`
  - `duplicate_rejected: List[MaskDict]`
  - `ambiguous: List[MaskDict]`
  - `out_path: Union[str, Path]`
  - `draw_prior: bool`
  - `draw_duplicate: bool`
  - `draw_ambiguous: bool`
- **Output:** `None` — writes a 3-channel RGB PNG file to `out_path`.

### `save_visualization(image, pred_masks, gt_mask, output_path, dpi)`
- **Input:**
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `pred_masks: List[MaskDict]`
  - `gt_mask: np.ndarray` — shape `(H, W)`, dtype `int`
  - `output_path: Path`
  - `dpi: int`
- **Output:** `None` — writes a multi-panel PNG figure to `output_path`.

## Invariants
- **Output Format Contract:** All written files are 8-bit RGB PNG. Input arrays with incompatible dtype or shape raise `ValueError` before any file is created.

## Produced Artifacts
- Colored contour overlay `.png` files and multi-panel comparison `.png` figures written to caller-specified paths.

## Failure Modes
- `ValueError`: Raised by `save_overlay` when `image.shape[:2] != segmentation.shape` for any mask in any input list.
- `ValueError`: Raised by `save_visualization` when `image.shape[:2] != pred_masks[i].segmentation.shape` or `!= gt_mask.shape`.
- `KeyError`: Raised by `save_overlay` when any mask dict is missing the `segmentation` key.
- `KeyError`: Raised by `save_visualization` when any pred mask dict is missing `segmentation`, `area`, `predicted_iou`, or `stability_score`.
- `OSError`: Raised when `out_path` / `output_path` is not writable (permissions error or invalid directory).
