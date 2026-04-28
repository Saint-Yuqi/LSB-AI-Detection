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
- `score: float`
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

### `save_evaluation_overlay(path, image, gt_streams, predictions, gt_satellites=None)`
- **Input:**
  - `path: Path`
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `gt_streams: np.ndarray | List[MaskDict]` — GT stream instance map or GT mask dict list
  - `predictions: List[dict]` — each with `segmentation`, optional `type_label`, `score`, `bbox`
  - `gt_satellites: np.ndarray | List[MaskDict] | None` — optional GT satellite instance map or GT mask dict list
- **Output:** `None` — writes contour-only QA overlay PNG with GT contours, post prediction contours, score labels, and a legend with a semi-transparent dark backdrop.

### `save_pseudo_label_overlay(path, image, predictions)`
- **Input:**
  - `path: Path`
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `predictions: List[dict]` — each with `segmentation`, optional `type_label`, `score`, `bbox`
- **Output:** `None` — writes prediction-only QA overlay PNG for pseudo-label workflows (no GT contour layer).

### `save_instance_overlay(path, image, instance_map)`
- **Input:**
  - `path: Path`
  - `image: np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`
  - `instance_map: np.ndarray` — shape `(H, W)`, integer instance IDs (0 = background)
- **Output:** `None` — writes colored-instance overlay PNG.

## Invariants
- **Output Format Contract:** All written files are 8-bit RGB PNG. Input arrays with incompatible dtype or shape raise `ValueError` before any file is created.
- **Type-Aware Palette:** Evaluation and pseudo-label overlays use separate stream/satellite colors and annotate scores when `bbox` metadata is available.

## Produced Artifacts
- Colored contour overlay `.png` files and multi-panel comparison `.png` figures written to caller-specified paths.
- Evaluation overlays can render either GT instance maps or GT mask dict lists and optionally include satellite GT contours.
- Prediction-only pseudo-label QA overlays written to caller-specified paths.

## Failure Modes
- `ValueError`: Raised by `save_overlay` when `image.shape[:2] != segmentation.shape` for any mask in any input list.
- `ValueError`: Raised by `save_visualization` when `image.shape[:2] != pred_masks[i].segmentation.shape` or `!= gt_mask.shape`.
- `KeyError`: Raised by `save_overlay` when any mask dict is missing the `segmentation` key.
- `KeyError`: Raised by `save_visualization` when any pred mask dict is missing `segmentation`, `area`, `score`, or `stability_score`.
- `OSError`: Raised when `out_path` / `output_path` is not writable (permissions error or invalid directory).

## Tidal_v1 (3-class) Additions

### `overlay._is_tidal_feature_label(tl)`
- Helper that recognises both the legacy `"streams"` / `"stellar stream"` labels and the new `"tidal_features"` label so existing call-sites keep coloring them with the same stream palette.

### Inner-galaxy palette
- New `_COLOUR_INNER_GALAXY_CONTOUR = (200, 100, 255)` (purple) used by `save_evaluation_overlay` and `save_pseudo_label_overlay` for masks with `type_label == "inner_galaxy"`. Tidal_features and legacy streams keep their existing blue palette; satellites keep their orange palette.
