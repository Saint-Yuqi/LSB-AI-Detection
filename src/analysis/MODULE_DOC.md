# Module: analysis

## Responsibilities
- Compute mathematically rigorous geometric metrics for binary masks.

## Non-goals
- **No Mask Filtering:** Does not actively drop or reject any masks.
- **No Inference:** Does not run neural network passes.

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

### `annulus_excess(signal, seg, r_in_frac=1.2, r_out_frac=2.0)`
- **Input:**
  - `signal: np.ndarray` — shape `(H, W)`, `float32`. Single-channel
    intensity in whatever units the model saw at inference. No
    magnitude-to-flux inversion happens anywhere.
  - `seg: np.ndarray` — shape `(H, W)`, binary mask.
  - `r_in_frac, r_out_frac: float` — annulus radii in units of `r_eq`
    (equivalent-circle radius `sqrt(area_clean / π)`). Must satisfy
    `0 < r_in_frac < r_out_frac`.
- **Output:** `Optional[float]` — mean signal in the evaluation annulus
  `[r_in_frac·r_eq, r_out_frac·r_eq]` minus mean signal in the reference
  ring `[2·r_out_frac·r_eq, 3·r_out_frac·r_eq]`. Returns `None` when
  `area_clean < MIN_AREA_FOR_HULL` or when either annulus has zero
  on-frame pixels.
- **Failure modes:** `ValueError` on shape mismatch or non-2D signal.

### `radial_monotonicity(signal, seg, n_rings=6, r_out_frac=2.0)`
- **Input:**
  - `signal: np.ndarray` — shape `(H, W)`, `float32`.
  - `seg: np.ndarray` — shape `(H, W)`, binary mask.
  - `n_rings: int` — number of equal-width rings from `r=0` to
    `r_out_frac · r_eq`. `n_rings < 2` returns `None`.
  - `r_out_frac: float` — outer radius of the last ring in units of `r_eq`.
- **Output:** `Optional[float]` in `[0, 1]` — fraction of `(n_rings − 1)`
  consecutive pairs where the azimuthally-averaged signal strictly
  decreases with radius. `1.0` = strictly monotone decay from centroid
  outward. Returns `None` when `area_clean < MIN_AREA_FOR_HULL` or when
  any ring has zero on-frame pixels.
- **Failure modes:** `ValueError` on shape mismatch or `r_out_frac <= 0`.

These two helpers are NOT added to `append_metrics_to_masks` — they
require the render signal alongside the mask, and only the satellite
diagnostics module calls them today.

### `append_metrics_to_masks(masks, H, W, compute_hull)`
- **Input:** `masks: List[MaskDict]`
  - Schema per Dict (STRICTLY CLOSED):
    - `segmentation: np.ndarray` (boolean, HxW)
    - `area: int`
    - `bbox: List[int]` ([x0, y0, w, h])
    - `score: float`
    - `point_coords: List[List[float]]`
    - `stability_score: float`
    - `crop_box: List[int]`
- **Output:** `None` — mutates each dict in-place, appending exactly the keys listed under `compute_mask_metrics`.

## Invariants
- **Data Decoding:** Input `seg` arrays must be fully decoded spatial binaries (`np.uint8`); the module does not decode RLE or compressed formats.
- **Single-Component Contract:** Outputs a single set of scalar shape metrics per `segmentation` input; zero-foreground masks return `None` for those keys.
- **Numeric Stability:** Aspect ratios are bounded to `[0.0, 1.0]` and are rotation-invariant; division-by-zero conditions yield `None` instead of raising.
- **Zero-Foreground Boundary:** When the input mask contains zero foreground pixels, metric keys documented as `Optional` return `None`.

## Produced Artifacts
- **In-Memory:** Enriches existing mask dictionaries with new geometric scalar keys.

## Failure Modes
- `ImportError`: Raised when `curvature_ratio` or `aspect_sym_boundary` is requested but `skimage` or `cv2` is not installed.
- `ValueError`: Raised when `seg.dtype` is not `np.uint8`, or when `H`/`W` do not match `seg.shape`.
- `KeyError`: Raised by `append_metrics_to_masks` when a mask dict is missing the `segmentation` key.
