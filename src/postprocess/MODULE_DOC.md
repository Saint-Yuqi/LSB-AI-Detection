# Module: postprocess

## Responsibilities
- Filter raw predicted masks using astrophysical priors.
- Apply lightweight sanity filtering for stream masks before metric aggregation.
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
- `score: float`
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

### `SatellitePriorFilter.filter(masks)` (slim, v2)
- **Input:** `masks: List[MaskDict]`
  - Each element additionally requires:
    - `area_clean: int`
    - `solidity: float`
    - `aspect_sym_moment: float` (or legacy `aspect_sym` alias)
    - `dist_to_center: float` (optional; computed from `segmentation` when missing)
- **Output:** `Tuple[List[MaskDict], List[MaskDict], List[MaskDict]]`
  - Element 0 (kept): each dict retains the full input schema.
  - Element 1 (prior-rejected): each dict retains the full input schema plus:
    - `reject_reason: str` in `{'prior_hard_center', 'prior_area_low', 'prior_solidity', 'prior_aspect'}`
  - Element 2 (ambiguous): **always an empty list** (slot kept for backward-compatible signature; DR1 v4 removed the ambiguous zone and `area_max` rejection).
  - Optional hard-center rule: when `hard_center_radius_frac` is configured, masks with
    `dist_to_center / min(H, W) < hard_center_radius_frac` are rejected here before
    `core_policy` runs.

### `SatelliteScoreGate.decide(area_clean_px, score)`
- **Input:** `area_clean_px: int`, `score: float`
- **Output:** `Tuple[str, str]` — `(decision, reason)` where
  - `decision ∈ {'pass', 'drop'}`
  - `reason ∈ {'pass_small','drop_small_score','pass_medium','drop_medium_score','pass_large','drop_large_score'}`
- **Thresholds (static):** `small_area_max_px=200`, `medium_area_max_px=600`, `small_min_score=0.60`, `medium_min_score=0.20`, `large_min_score=0.18`.

### `SatelliteCorePolicy.decide(dist_to_center_frac, area_clean_px, score, solidity, aspect_sym_moment)`
- **Output:** `Tuple[str, str]` — `(decision, reason)` where
  - `decision ∈ {'pass', 'rescue', 'drop'}`
  - `reason ∈ {'drop_hard_core','drop_soft_core','rescue_soft_core','pass_outside_core'}`
- **Thresholds:** `hard_core_radius_frac=0.03`, `soft_core_radius_frac=0.08`, `rescue_area_min_px=600`, `rescue_solidity_min=0.90`, `rescue_aspect_max=1.80`, `rescue_score_min=0.18`.

### `SatelliteConflictResolver.match_stream(sat_seg, streams_instance_map)`
- **Input:** binary `sat_seg`, int `streams_instance_map` (positive IDs are real stream instances, 0 is background).
- **Output:** `Tuple[Optional[int], int, float, float]` — `(matched_stream_id, overlap_px, overlap_ratio_satellite, overlap_ratio_stream)`.
- `matched_stream_id` is the real instance ID of the stream with the largest overlap, or `None` when no overlap.

### `SatelliteConflictResolver.decide(matched_stream_id, overlap_ratio_satellite, area_clean_px, solidity, aspect_sym_moment)`
- **Output:** `Tuple[str, str, dict]` — `(decision, reason, extras)` where
  - `decision ∈ {'pass', 'win', 'drop'}`
  - `reason ∈ {'pass_no_stream_conflict','satellite_wins','area_under_600_swallowed_by_stream','not_compact_enough_to_win','lost_to_stream_area_ge_600'}`
  - `extras["matched_stream_id"]` is always populated (may be `None`).

### `SatellitePipelineRunner.run(raw_sats, streams_gt_map, H, W, base_key=None)`
- Drives each raw satellite mask dict through an 8-stage fixed-order state machine
  (`raw_retrieval` → `metrics_completion` → `size_aware_score_gate` → `satellite_prior_filter`
  → `core_exclusion_or_soft_core_rescue` → `stream_conflict_resolution` → `final_gt_write`
  → `diagnostics_emit`).
- **Output:** `SatellitePipelineResult`
  - `final_sats: list[MaskDict]`
  - `candidates: list[{candidate_id: str, candidate_rle_sha1: str, final_status: str, matched_stream_id: Optional[int], history: list[StageEvent]}]`
  - `image_summary: dict` (stage drop counts, final-status counts, threshold versions)
- Each `StageEvent` carries strict fields: `stage`, `input_state`, `rule_name`, `threshold_version`,
  `threshold_values`, `decision`, `reason`, `output_state`, `metrics_snapshot_thin`.
- `metrics_snapshot_thin` is enforced to only contain whitelisted scalar keys:
  `score`, `area_clean_px`, `solidity`, `aspect_sym_moment`,
  `dist_to_center_px`, `dist_to_center_frac`, `overlap_ratio_satellite`, `overlap_ratio_stream`.
  Any other key (bbox, segmentation, RLE, contours, raw area, etc.) is explicitly forbidden.

### `load_streams_cfg(stats_json)`
- **Input:**
  - `stats_json: Path | str` (default: `outputs/mask_stats/mask_stats_summary.json`)
- **Output:** `dict[str, Any]` with keys:
  - `min_area: int`
  - `max_area_px: int | None`
  - `edge_touch_frac: float`
- **Failure:** Warns and returns safe defaults if file missing or malformed.

### `StreamsSanityFilter.__init__(min_area, max_area_frac, edge_touch_frac, max_area_px)`
- **Input:**
  - `min_area: int`
  - `max_area_frac: float`
  - `edge_touch_frac: float`
  - `max_area_px: int | None` — absolute upper-area bound from GT stats; takes priority over `max_area_frac` when set.
- **Output:** Initialized lightweight stream-filter object.

### `StreamsSanityFilter.filter(masks, H, W)`
- **Input:**
  - `masks: List[MaskDict]` (requires `segmentation`; uses `area` if provided)
  - `H: int`, `W: int`
- **Output:** `Tuple[List[MaskDict], List[MaskDict]]`
  - Element 0: kept masks
  - Element 1: rejected masks with `reject_reason` in:
    - `sanity_area_low`
    - `sanity_area_high` (uses `max_area_px` when set, else `max_area_frac * H * W`)
    - `sanity_edge`

## Invariants
- **Uniqueness:** The kept list from `select_representatives` contains at most one dict per distinct `group_id`.
- **Ambiguous Zone:** The ambiguous output list contains masks that belong to neither the kept nor the rejected partition; each carries a `reject_reason: str` tag.

## Produced Artifacts
- Partitioned in-memory mask lists; scalar stats dict from `CoreExclusionFilter`.

## Failure Modes
- `KeyError`: Raised by `group_by_centroid` if any mask dict is missing `centroid_xy` or `dist_px`.
- `KeyError`: Raised by `select_representatives` if any mask dict is missing `group_id`, `stability_score`, `score`, `area_clean`, `aspect_sym_moment`, or `aspect_sym`.
- `KeyError`: Raised by `SatellitePriorFilter.filter` if any mask dict is missing `segmentation`. Missing `solidity` / `aspect_sym_moment` are computed on-demand from the segmentation.
- `KeyError`: Raised by `StreamsSanityFilter.filter` if any mask dict is missing `segmentation`.
- `AssertionError`: Raised by `SatellitePipelineRunner` (via `build_thin`) if a `metrics_snapshot_thin` ever contains a key outside the whitelist.

## Tidal_v1 (3-class) Additions

### `SatellitePriorFilter.decide_with_target(mask)`
- **Input:** mask dict (segmentation + optional `dist_to_center`, `area_clean`, `solidity`, `aspect_sym_moment`).
- **Output:** `Tuple[str, str, Optional[str]]` — `(decision, reason, target_type)` where `decision` is `"pass"`, `"drop"`, or `"relabel"`.
- When `cfg["hard_center_action"] == "relabel_inner_galaxy"` and the mask centroid lies within `hard_center_radius_frac`, returns `("relabel", "prior_hard_center", "inner_galaxy")`. Otherwise the third slot is `None` and `decide()` semantics are preserved (still surfaces the legacy 2-tuple shape via `SatellitePriorFilter.decide()`).

### `SatellitePriorFilter.filter(masks)` (return-shape change)
- Still returns a 3-tuple `(kept, rejected, relabeled)`. On the legacy `hard_center_action: "drop"` setting `relabeled` is always `[]`, preserving today's pnbody caller signature. On `"relabel_inner_galaxy"`, relabeled masks carry `mask["type_label"] = "inner_galaxy"` and `mask["relabel_target"] = "inner_galaxy"`.

### `SatellitePipelineRunner.__init__` (extended)
- Accepts `core_policy: SatelliteCorePolicy | None = None` and `conflict_resolver: SatelliteConflictResolver | None = None` (both optional now). New keyword flags `enable_core_policy: bool = True` and `enable_conflict_resolution: bool = True`.
- When a stage is disabled, the runner emits `StageEvent("skipped", reason="core_disabled" | "conflict_disabled")` for each in-flight candidate and passes through unchanged.
- `SatellitePipelineResult` carries an additional `final_inner_galaxy: list[dict]` bucket. Image summary gains `n_final_inner_galaxy` and reports `"disabled"` for `core_policy` / `conflict_policy` threshold versions when those stages are off.

### `satellite_core_policy` (deprecated)
- Module retained for legacy import compatibility; new (tidal_v1) configs should construct `SatellitePipelineRunner(..., core_policy=None, enable_core_policy=False)`. Hard-center handling is done by `SatellitePriorFilter.decide_with_target` instead.

### `StreamSatelliteConflictFilter` (`stream_satellite_conflict_filter.py`)
- Cross-class conflict resolver between predicted streams and predicted satellites in the `post_pred_only` chain. Decides per-pair via overlap ratio, area, and shape priors and emits drop/keep decisions with a `THRESHOLD_VERSION`-tagged reason string. Disabled on the tidal_v1 path (`enable_cross_type_conflict: false` in `configs/eval_checkpoint.yaml`).

### `SatelliteConflictResolver` — `THRESHOLD_VERSION = "conflict_policy_v1_dr1"`
### `SatelliteScoreGate` — `THRESHOLD_VERSION = "score_gate_v1_static"`
- Both classes export their threshold-set version string for inclusion in eval/diagnostics manifests so policy changes are traceable across runs.
