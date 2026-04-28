# Module: src/pipelines

## Responsibilities
- Contain the core logic for multi-phase data preparation pipelines.
- Provide testable, importable functions that scripts/ entry points delegate to.
- Preserve config-driven path resolution across legacy DR1 and new PNbody view conventions.
- Materialize pseudo-label artifacts that can substitute for GT in PNbody-style workflows.

## Sub-packages

### `unified_dataset/`
4-Phase pipeline for preparing the LSB training dataset:
- `keys.py` — `BaseKey`, `VariantKey` dataclasses
- `paths.py` — `PathResolver` plus pattern formatting for legacy `{orientation}` and new `{view_id}`
- `config.py` — `load_config`, `_resolve_views`, `generate_base_keys`
- `preprocessor_factory.py` — `create_preprocessor` (single source of truth)
- `fs_utils.py` — `sha1_file` and shared filesystem helpers
- `render.py` — Phase 1: FITS -> RGB renders
- `gt.py` — Phase 2: SB mask -> streams instance map
- `inference.py` — Phase 3 dispatcher (routes to SAM2 or SAM3)
- `inference_sam2.py` — SAM2 AutoMask -> filter -> merge -> cache
- `inference_sam3.py` — SAM3 text-prompt -> filter -> evaluate or pseudo_label
- `artifacts.py` — Prediction JSON, pseudo-GT rasterization, satellites cache, instance merging, `assign_stable_ids`
- `export.py` — Phase 4: SAM2 symlinks + SAM3 COCO annotations
- `split.py` — Galaxy-level COCO train/val split (hash-stable)
- `noise_aug.py` — Noise augmentation (train-only): symlinks noisy renders, copies GT annotations
- `compose.py` — Compose training dataset from multiple annotation sources (symlink or merge)

### Layered Dataset Pipeline (post-export)

Canonical ordering: `export.py` → `split.py` → `noise_aug.py` → `compose.py`

```
annotations.json (canonical clean)
  ├─ split → annotations_train.json + annotations_val.json
  │           + split_manifest.json
  ├─ noise_aug → annotations_train_noise_augmented.json
  │               + noise_aug_manifest.json
  └─ compose → annotations_train_active.json (training target)
                + compose_manifest.json
```

Shared dataset root contract: all annotation files use `file_name` paths
relative to the same dataset root (`sam3_prepared/`). The compose layer
only merges JSON — image symlinks are created by upstream producers.

## Non-goals
- No CLI argument parsing (that lives in `scripts/`).
- No direct I/O to stdout (use logging).

## Inputs / Outputs

### `load_config(path)`
- **Input:** `path: Path` to a pipeline YAML config.
- **Output:** `dict` loaded via `yaml.safe_load`.

### `_resolve_views(config)`
- **Input:** `config: dict` with `data_selection`.
- **Output:** `List[str]` canonical view IDs.
- **Compatibility Contract:** Prefers `data_selection.views`; falls back to legacy `data_selection.orientations` with `DeprecationWarning`.

### `generate_base_keys(config, galaxy_filter=None)`
- **Input:** Pipeline config plus optional `List[int]` galaxy subset.
- **Output:** `List[BaseKey]` covering every selected `galaxy_id × view_id` combination.

### `_format_pattern(pattern, key, **extra)`
- **Input:** Template string plus `BaseKey`.
- **Output:** `str` formatted using `{view_id}` when present, otherwise legacy `{orientation}`.

### `PathResolver.get_mask_path(key, sb_threshold)`
- **Input:** `key: BaseKey`, `sb_threshold: float`.
- **Output:** `Optional[Path]`.
- **Nullable Contract:** Returns `None` when mask generation is intentionally disabled or a view-specific mask subdir is not configured.

### `rasterize_pseudo_gt(masks, H, W)`
- **Input:** Post-filtered SAM3 masks with `segmentation` and optional `type_label`.
- **Output:** Tuple `(instance_map_uint8, instances_list)`.
- **Schema:** `instances_list` items are `{"id": int, "type": str}`.

### `save_pseudo_gt(gt_dir, masks, H, W)`
- **Input:** GT artifact directory plus post-filtered masks.
- **Output:** Tuple `(instance_map_uint8, instances_list)`.
- **Side Effects:** Writes `instance_map_uint8.png` and `instances.json` to `gt_dir`.

### `run_inference_sam3(config, base_keys, logger, force=False, force_variants=None)`
- **Input:** Unified-dataset config, `List[BaseKey]`, logger, and rebuild controls.
- **Mode Contract:** `config["inference_phase"]["run_mode"]` may be `evaluate` or `pseudo_label`.
- **Purity Contract:** Standard `evaluate` runs are override-free; reviewed exceptions are migrated later through Shadow GT tooling, not injected at runtime.
- **Output:**
  - `evaluate`: `sam3_predictions_raw.json`, `sam3_predictions_post.json`, `sam3_eval_overlay.png`, manifest update with `gt_source: "streams_instance_map"`
    `sam3_eval_overlay.png` is the standard pipeline post-filter QA overlay with streams GT only; compare-only reference-GT overlays are produced by `scripts/eval/run_gt_refresh_compare.py`, not by the pipeline itself.
  - `pseudo_label`: `sam3_predictions_raw.json`, `sam3_predictions_post.json`, `instance_map_uint8.png`, `instances.json`, `sam3_pseudo_label_overlay.png`, manifest update with `gt_source: "none"`

## Invariants
- `BaseKey.view_id` is the canonical view identifier in-memory, even when configs still use the legacy `orientation` spelling.
- Path formatting remains config-driven; no dataset-specific filenames are hardcoded outside pattern templates.
- Pseudo-label GT rasterization uses `uint8` instance IDs and therefore caps per-image instance count at 255.
- SAM3 pseudo-label completion is defined by six artifacts: raw predictions, post predictions, pseudo GT image, `instances.json`, overlay, and `manifest.json`.
- **Stable prediction identity** — `assign_stable_ids(masks)` in
  `unified_dataset/artifacts.py` stamps `raw_index` (global ordinal over
  the full mask list, streams + satellites together) and `candidate_id`
  (per-type ordinal `{sat|stream}_NNNN`). Once stamped, these IDs are
  the cross-layer source-raw identity keys: a candidate that survives
  into `predictions_post_pred_only.json` or `predictions_post_gt_aware.json`
  carries the same `raw_index` it had in `predictions_raw.json`. As a
  result, `raw_index` values in post-layer JSONs are non-consecutive —
  that is correct. Readers that want "row index inside this file" should
  index the `predictions` array directly. `save_predictions_json` reuses
  pre-stamped IDs when present and falls back to local file ordinals only
  when they are absent.

## Produced Artifacts
- Render/GT/inference/export directory trees for the unified dataset pipeline.
- Pseudo-label GT artifacts: `instance_map_uint8.png`, `instances.json`, and `sam3_pseudo_label_overlay.png`.
- Manifest metadata carrying `run_mode`, `gt_source`, and SAM3 count/timing summaries.

## Failure Modes
- `KeyError`: Raised when neither `data_selection.views` nor legacy `data_selection.orientations` is present.
- `AssertionError`: Raised by `rasterize_pseudo_gt` / `save_pseudo_gt` when more than 255 instances would overflow `uint8`.
- `FileNotFoundError`: May be raised downstream when YAML-referenced paths or stats files are missing.
