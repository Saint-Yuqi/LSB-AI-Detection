# Module: postprocess

## Responsibilities
- Filter raw predicted masks using astrophysical priors.
- Identify and reject masks located in the excluded core region of the main galaxy.
- Cluster duplicate masks pointing to the same physical object.
- Select the single best representative mask per cluster.

## Non-goals
- **No Mask Generation:** Does not invoke SAM2 or generate masks.
- **No Mask Metrics Extraction:** Expects metrics to be pre-calculated.

## Type Aliases

### `MaskDict` (TypedDict, STRICTLY CLOSED)
- `segmentation: np.ndarray` (boolean, HxW)
- `area: int`
- `bbox: List[int]` ([x0, y0, w, h])
- `predicted_iou: float`
- `point_coords: List[List[float]]`
- `stability_score: float`
- `crop_box: List[int]`

### `RepCfg` (TypedDict, all keys optional)
- `iou_weight: float`
- `stability_weight: float`
- `area_weight: float`

## Inputs / Outputs

### `group_by_centroid(masks)`
- **Input:** `masks: List[MaskDict]`
  - Each element additionally requires:
    - `centroid_xy: Tuple[float, float]`
    - `dist_px: float`
- **Output:** `None` — mutates each dict in-place, adding:
  - `group_id: int`

### `CoreExclusionFilter.filter(masks, H, W)`
- **Input:**
  - `masks: List[MaskDict]`
  - `H: int`, `W: int`
- **Output:** `Tuple[List[MaskDict], List[MaskDict], CoreStats]`
  - Element 0: kept mask dicts (retains full `MaskDict` schema)
  - Element 1: core-rejected mask dicts (retains full `MaskDict` schema)
  - Element 2: `CoreStats` (TypedDict, all keys required):
    - `R_exclude: float`
    - `dist_p05: float`
    - `dist_p50: float`
    - `dist_p95: float`
    - `core_area_min: int`
    - `core_area_max: int`
    - `core_area_mean: float`
    - `core_solidity_mean: float`

### `select_representatives(masks, cfg)`
- **Input:**
  - `masks: List[MaskDict]`
    - Each element additionally requires:
      - `group_id: int`
      - `area_clean: int`
      - `aspect_sym_moment: float`
      - `aspect_sym: float`
  - `cfg: Optional[RepCfg]`
- **Output:** `Tuple[List[MaskDict], List[MaskDict]]`
  - Element 0: representative kept dicts, each retains the full input schema plus:
    - `rep_score: float`
  - Element 1: rejected dicts, each retains the full input schema plus:
    - `reject_reason: str`

### `SatellitePriorFilter.filter(masks)`
- **Input:** `masks: List[MaskDict]`
  - Each element additionally requires:
    - `area_clean: int`
    - `solidity: float`
    - `aspect_sym_moment: float`
    - `aspect_sym: float`
    - `dist_to_center: float`
- **Output:** `Tuple[List[MaskDict], List[MaskDict], List[MaskDict]]`
  - Element 0 (kept): each dict retains the full input schema.
  - Element 1 (prior-rejected): each dict retains the full input schema plus:
    - `reject_reason: str` — rejection reason string tag (e.g., `'prior_rejected'`)
  - Element 2 (ambiguous): each dict retains the full input schema plus:
    - `reject_reason: str` — rejection reason string tag (e.g., `'core_rejected'`)

## Invariants
- **Uniqueness:** The kept list from `select_representatives` contains at most one dict per distinct `group_id`.
- **Ambiguous Zone:** The ambiguous output list contains masks that belong to neither the kept nor the rejected partition; each carries a `reject_reason: str` tag.

## Produced Artifacts
- Partitioned in-memory mask lists; scalar stats dict from `CoreExclusionFilter`.

## Failure Modes
- `KeyError`: Raised by `group_by_centroid` if any mask dict is missing `centroid_xy` or `dist_px`.
- `KeyError`: Raised by `select_representatives` if any mask dict is missing `group_id`, `stability_score`, `predicted_iou`, `area_clean`, `aspect_sym_moment`, or `aspect_sym`.
- `KeyError`: Raised by `SatellitePriorFilter.filter` if any mask dict is missing `area_clean`, `solidity`, `aspect_sym_moment`, `aspect_sym`, or `dist_to_center`.
