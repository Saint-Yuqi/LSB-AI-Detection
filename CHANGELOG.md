# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Satellite diagnostic taxonomy (pred-centric, no-IoU-threshold, Phase 1).**
  A GT-driven error-typing layer on top of the existing Hungarian
  evaluator. Runs on raw satellite predictions when the benchmark has
  satellites (`fbox_gold_satellites`, `gt_canonical`) and
  `diagnostics.enabled=true` in `configs/eval_checkpoint.yaml`.
  - New reusable primitives in `src/evaluation/metrics.py`:
    `primary_gt_match(pred_bin, gt_instance_map)` (arg-max match via
    `np.bincount`, no IoU threshold), `derive_purity_completeness(...)`
    (pure scalar), and `compute_one_to_one_flags(pred_bins, gt_instance_map)`
    (batch helper, first-argmax tie-break).
  - New image-evidence helpers in `src/analysis/mask_metrics.py`:
    `annulus_excess(signal, seg, r_in_frac, r_out_frac)` and
    `radial_monotonicity(signal, seg, n_rings, r_out_frac)`. Both operate
    on `(H, W) float32` intensity — the same ``render_signal`` SAM3 saw
    at inference — with **no** magnitude-to-flux inversion anywhere.
  - New module `src/evaluation/satellite_diagnostics.py` exposes
    `DiagnosticCfg`, `CandidateRow` (TypedDict, closed),
    `SatelliteDiagnosticReport`, `classify(...)`, `build_candidate_table(...)`,
    and `aggregate_diagnostics(per_sample_rows)`. Classifier is A/B/C/D
    with four labels — `compact_complete`, `diffuse_core`,
    `reject_unmatched`, `reject_low_purity`. `reject_host_background`
    is reserved for Phase 2 (needs a host-support loader).
  - `compute_sample_report(..., render_signal=None)` now returns
    `(report, diag_report)`. `report["diagnostics"]["satellites_raw"]`
    carries ONLY a summary and a relative `per_candidate_path` — full
    per-candidate rows go to `sample_dir/diagnostics.json` so
    `report.json` stays small. `aggregate(reports)` signature is
    unchanged (typed-block summaries only); diagnostic aggregation lives
    in `satellite_diagnostics.aggregate_diagnostics`.
  - `scripts/eval/evaluate_checkpoint.py` now calls
    `assign_stable_ids(raw_masks)` once on the full list before splitting
    by `type_label`, converts `render_rgb → render_signal` once per sample,
    writes `sample_dir/diagnostics.json`, and emits `report.json` with
    both `summary` and `diagnostics_summary` top-level blocks.
- **Stable cross-layer prediction identity (`raw_index` / `candidate_id`).**
  `src/pipelines/unified_dataset/artifacts.py` gains
  `assign_stable_ids(masks)`, factored out of `save_predictions_json`.
  Callers stamp IDs once on the full raw mask list; downstream layers
  (`post_pred_only`, `post_gt_aware`, `diagnostics.json`) preserve them.
  **Schema shift:** `raw_index` is now "source-raw ordinal across layers",
  not "row position inside this file". Readers that want file-local row
  numbers should index the JSON array directly. `save_predictions_json`
  reuses pre-stamped IDs when present; legacy local-ordinal behaviour is
  the fallback when they are absent.
- **Checkpoint evaluation overlays now include predictions + scores** — the
  eval loop in `scripts/eval/evaluate_checkpoint.py` writes three (or four,
  for `gt_canonical`) overlay PNGs per sample under `overlays/`:
  `gt_contour.png` (GT-only, type-aware colours + optional ROI box),
  `raw_overlay.png`, `post_pred_only_overlay.png`, and
  `post_gt_aware_overlay.png` (only when `ga_pair is not None`). The new
  `src/visualization/overlay.save_eval_prediction_overlay` draws GT +
  optional ROI + per-prediction contours coloured by a deterministic
  per-type palette (seed 42; streams cool / satellites warm) plus a
  compact `s:0.xx` / `c:0.xx` confidence label per mask. Overlay writing
  is now isolated in a `_write_sample_overlays(sample_dir, sample,
  render_rgb, raw_masks, po_pair, ga_pair, sample_label)` helper and
  sources `post_gt_aware` strictly from `ga_pair`, decoupling it from
  the `cfg.output.save_post_predictions` JSON branch.
- **`--overlays-only` on `scripts/eval/evaluate_checkpoint.py`**
  — regenerates `overlays/*.png` from on-disk
  `predictions_raw.json` / `predictions_post_pred_only.json` /
  (optional) `predictions_post_gt_aware.json` with no SAM3 inference
  and no `report.json` write.
- **Checkpoint evaluation pipeline (3 benchmarks, 1024-only)** — new
  single-entrypoint SAM3 evaluation stack:
  - `configs/eval_checkpoint.yaml` + `scripts/eval/evaluate_checkpoint.py`
    drive `fbox_gold_satellites`, `firebox_dr1_streams`, and `gt_canonical`
    from one CLI with per-benchmark prompt lists and a unified 3-layer post
    config (`raw`, `post_pred_only`, `post_gt_aware`).
  - `src/evaluation/checkpoint_eval.py` hosts the 3 benchmark loaders, a
    fixed 2051→1024 downsample with a positive-ID-set preservation guard
    (`allow_instance_drop` opt-in), the frozen 1024 ROI
    `[277:747, 277:747]` for Fbox with `any_pixel_intersects` instance
    membership, and a dual-layer post (`apply_post_pred_only` +
    `apply_post_gt_aware`).
  - `post_pred_only` runs the 5 prediction-only stages for every benchmark
    (`streams_sanity`, `score_gate`, `prior_filter`, `core_policy`,
    `cross_type_conflict` via `StreamSatelliteConflictFilter` with
    re-filter hooks disabled). `post_gt_aware` only runs on `gt_canonical`
    and layers `SatelliteConflictResolver` on top of `post_pred_only`.
  - `src/visualization/overlay.save_gt_contour_only_overlay` draws
    yellow-only GT contours with a compact top-left label and no
    prediction fills, matching the eval QA overlay spec.
- **Multi-benchmark noise + render scripts** — generalized
  `scripts/data/generate_pnbody_noisy_fits.py` (new `source.filename_suffix`,
  `source.sb_maps_layout`, `source.enumeration.mode` = grid / manifest /
  mask_glob), plus new `configs/noise_profiles_fbox_gold.yaml`,
  `configs/noise_profiles_dr1_streams.yaml`, and
  `scripts/data/render_eval_benchmark.py` which writes 1024x1024 PNGs to
  `data/02_processed/renders_eval/{benchmark}/{current|noisy}/[{profile}/]{variant}/{base_key}/0000.png`.

### Fixed

- **Checkpoint eval overlay: "??" glyphs in the top-left label.**
  `_label_for_overlay` in `scripts/eval/evaluate_checkpoint.py` used
  Unicode middle-dots (`·`, U+00B7) which OpenCV's Hershey font renders
  as `?`. Replaced with ASCII `|` separators.
- **Checkpoint eval overlay: all GT drawn in yellow.**
  `save_gt_contour_only_overlay` in `src/visualization/overlay.py`
  accepted `gt_type_of_id` but ignored it; now streams are drawn in
  white and satellites in yellow (ids missing from the mapping fall
  back to yellow with a warning log). Fixes
  `firebox_dr1_streams/*/gt_contour.png` showing streams as yellow and
  `gt_canonical/*/gt_contour.png` collapsing both types to yellow.
- **Checkpoint eval overlay: missing ROI box for
  `fbox_gold_satellites`.** Added an optional `roi_bbox` argument to
  `save_gt_contour_only_overlay` using the same `(y0, x0, y1, x1)`
  contract as `Sample.roi_bbox_1024`; the (y, x) → (x, y) swap required
  by `cv2.rectangle` is localized to a single helper. The Fbox ROI
  (`FBOX_ROI_1024 = (277, 277, 747, 747)`) is now drawn in yellow and
  annotated in the legend.

### Removed

- **Legacy SAM2 stack (aggressive purge)** — deleted
  `scripts/eval/evaluate_sam2.py`, `scripts/eval/eval_model.py`,
  `configs/eval_sam2.yaml`,
  `src/pipelines/unified_dataset/inference_sam2.py`,
  `src/inference/sam2_automask_runner.py`, `scripts/viz/visualize_sam2.py`,
  `scripts/analysis/plot_recall_curve.py`, and
  `scripts/viz/overlay_masks_on_streams.py`. The `satellites:` block and
  `engine: sam2` branch were dropped from `configs/unified_data_prep.yaml`;
  `PathResolver.sam2_root` / `get_sam2_dir` and the SAM2 symlink leg of the
  export phase were removed. `_load_reject_candidates_sam2` in
  `src/review/example_builder.py` was deleted and
  `scripts/review/build_verifier_examples.py` now uses the SAM3 loader only.
- **Legacy SAM3 evaluators** — deleted
  `scripts/eval/evaluate_sam3.py`, `scripts/eval/evaluate_sam3_artifacts.py`,
  `scripts/eval/evaluate_model.py`, `scripts/eval/run_gt_refresh_compare.py`,
  `scripts/eval/run_sweep_eval.sh`, `scripts/eval/run_batch_eval.sh`,
  `scripts/eval/run_batch_eval_type_aware.sh`,
  `scripts/cluster/launch_eval_sweep.sh`, `scripts/cluster/eval_sweep.slurm`,
  `scripts/viz/visualize_eval_metrics.py`, `src/evaluation/sam3_eval.py`,
  `configs/eval_sam3.yaml`, `configs/eval_sam3_conf03.yaml`, and the
  associated tests (`tests/test_gt_refresh_compare.py`,
  `tests/test_eval_type_aware.py`). Use
  `scripts/eval/evaluate_checkpoint.py` instead.

### Other added notes from prior batches


- **Shadow GT migration workflow for reviewed satellites** — introduces an
  explicit two-step flow replacing the runtime satellite override hook:
  1. `scripts/review/bootstrap_shadow_gt.py` scaffolds a standalone
     `scratch/gt_shadow/gt_canonical/current` tree by symlinking renders and
     copying authoritative `streams_instance_map.npy` / `manifest.json` only —
     authoritative satellite artifacts (`instance_map_uint8.png`,
     `instances.json`, any `sam3_predictions_*.json`, overlays, diagnostics)
     are never copied, so a subsequent pure DR1 SAM3 evaluate run writes clean
     satellite artifacts into the shadow tree.
  2. `scripts/review/migrate_satellite_overrides.py` translates the archived
     `configs/archive/sam3_satellite_overrides_legacy.yaml` into concrete
     `adopt_raw_candidate` and `delete_authoritative_instance` calls against
     the shadow GT. `force_keep` adopts from the shadow's native
     `sam3_predictions_raw.json` unless the entry's `candidate_rle_sha1`
     appears in `inject_from_json`, in which case the external probe JSON is
     used. `force_drop` resolves the targeted surviving shadow instance by
     replaying the configured `satellite_sort_policy` against the shadow
     `sam3_predictions_post.json` / `streams_instance_map.npy`, resolves that
     candidate to a concrete `instance_id`, then calls
     `delete_authoritative_instance(instance_id=...)`. Supports
     `--dry-run` and `--base-keys` filtering; a JSON summary is written to
     `_migration_logs/migration_summary.json`.
- **`--candidate-rle-sha1` selector** — `adopt_raw_candidate` and
  `scripts/review/edit_authoritative_gt.py adopt-raw` accept the RLE SHA1 as a
  stable selector for review adoption. The selector is resolved first and
  cross-checked against `candidate_id` / `raw_index` when all three are
  supplied. This is the preferred selector for the Shadow GT migration flow.
- **`--gt-reference-root` for `scripts/eval/run_gt_refresh_compare.py`** —
  the orchestrator now accepts a GT reference root used for base-key
  discovery, variant scaffolding (`streams_instance_map.npy`, `manifest.json`),
  and compare overlay backfill. Defaults to the current canonical GT root for backward
  compatibility; pass a Shadow GT root to compare against post-migration GT.
- **Compare-only legacy eval overlay backfill** — `scripts/eval/run_gt_refresh_compare.py`
  now runs a per-variant post-pass that writes
  `artifacts_root/gt_canonical/current/{base_key}/sam3_legacy_eval_overlay.png`,
  reproducing the old `evaluate_sam3.py` visual semantics (selected GT reference
  contours + raw prediction fills on the scratch render). Helpers:
  `_split_predictions_by_type`, `_render_legacy_eval_overlay_for_base_key`,
  `_render_fullgt_eval_overlay_for_base_key`, `generate_legacy_eval_overlays`.
  The same post-pass also writes
  `artifacts_root/gt_canonical/current/{base_key}/sam3_fullgt_eval_overlay.png`
  for post-filter predictions against the selected GT reference root. Post-pass runs even
  under `--skip-inference` (backfill) and is skipped for variants whose
  inference subprocess failed. Missing `sam3_predictions_raw.json` or
  `sam3_predictions_post.json` is a skip for that specific artifact; missing
  reference GT/render when a corresponding prediction JSON exists is a hard
  error. Compare-only artifacts are intentionally NOT produced by
  `src/pipelines/unified_dataset/inference_sam3.py` or `_evaluate_complete(...)`.
  Overlay artifact semantics are now explicit:
  - `sam3_raw_overlay.png` = raw predictions, no GT
  - `sam3_eval_overlay.png` = post-filter predictions + streams GT only
  - `sam3_legacy_eval_overlay.png` = raw predictions + selected GT reference root
  - `sam3_fullgt_eval_overlay.png` = post-filter predictions + selected GT reference root
  New tests: `tests/test_gt_refresh_compare.py`, `tests/test_visualization_overlay.py`.
- **SAM3 environment contract hardening** — added `src/utils/runtime_env.py`
  and wired the current SAM3 render/eval/review CLIs to enforce
  `CONDA_DEFAULT_ENV=sam3` unless `IGNORE_ENV_CONTRACT=1` is set for tests/CI.
  `scripts/eval/run_gt_refresh_compare.py` now launches inference subprocesses
  with `conda run --no-capture-output -n <env> python ...`, accepts
  `--conda-env`, and the bash wrappers
  (`run_batch_eval.sh`, `run_batch_eval_type_aware.sh`, `run_sweep_eval.sh`)
  self-reexec into `sam3` with `exec conda run --no-capture-output -n sam3 ...`.
- **DR1 Satellite Pipeline v4** — explicit 8-stage state machine for DR1 canonical evaluate
  - `src/postprocess/satellite_pipeline.py` — `SatellitePipelineRunner`, `SatelliteCandidateState`, `StageEvent`, `SatellitePipelineResult`; thin-metrics whitelist enforced
  - `src/postprocess/satellite_score_gate.py` — static three-tier size-aware score gate (`*_px` / `*_score` unit-safe keys)
  - `src/postprocess/satellite_core_policy.py` — `hard_core_radius_frac=0.03` / `soft_core_radius_frac=0.08` with strict soft-core rescue
  - `src/postprocess/satellite_conflict_resolver.py` — GT-aware conflict resolver that records real `matched_stream_id` from `streams_instance_map.npy`
  - DR1 evaluate emits new artifacts per base key:
    - `sam3_raw_overlay.png` (contour-only)
    - `sam3_satellite_diagnostics.json` (`image_summary` + `candidates` with thin history only; never stores bbox / contours / RLE / segmentation bodies)
  - `configs/unified_data_prep.yaml` adds `inference_phase.sam3.score_gate`, `core_policy`, `conflict_policy` blocks
  - `sam3_satellite_pipeline_version` field in manifest (`v4` for evaluate, `legacy` for pseudo-label)
  - New unit tests: `tests/test_satellite_score_gate.py`, `tests/test_satellite_core_policy.py`, `tests/test_satellite_conflict_resolver.py`, `tests/test_satellite_pipeline.py`

### Changed

- **SAM3 DR1 evaluate is now a pure, override-free flow** — the standard
  inference path writes `sam3_predictions_raw.json`, `sam3_raw_overlay.png`,
  `sam3_predictions_post.json`, and `sam3_satellite_diagnostics.json` strictly
  from native checkpoint output and deterministic post-filters. Reviewed
  exceptions are no longer applied at inference time; they are migrated
  explicitly into a Shadow GT (see Added).

### Removed

- **Legacy runtime satellite override hook** — removed
  `inference_phase.sam3.satellite_overrides_path` from
  `configs/unified_data_prep.yaml` and added a fail-fast guard in
  `src/pipelines/unified_dataset/inference_sam3.py` that aborts if the key is
  reintroduced. Deleted `_load_satellite_overrides`,
  `_inject_external_satellite_masks`, `_mask_rle_sha1`, and every
  `external_satellite_injections` diagnostic write.
- **`manual_override` post-processing stage** — removed from
  `src/postprocess/satellite_pipeline.py`: `SatellitePipelineRunner.__init__`
  no longer accepts `manual_overrides`, the `manual_override` entry is gone
  from `STAGE_ORDER`, and `_stage_manual_override`, `_find_override_match`,
  `_override_threshold_values`, and all `manual_force_keep` /
  `manual_force_drop` behaviors are deleted.
- **`analyzer.py` scratch script** — deleted; depended on the removed
  `_mask_rle_sha1` helper.

- **DR1 Satellite overlays are now contour-only** — `save_evaluation_overlay` no longer does alpha fills; added `save_raw_overlay` helper for pre-filter raw visualisation
- **Evaluation overlay API widened** — `save_evaluation_overlay(...)` now accepts GT instance maps or GT mask dict lists, supports optional full-GT satellite contours, and draws its legend on a semi-transparent dark backing box so the top-left labels stay readable over bright structure.
- **`SatellitePriorFilter` slimmed down** — removed `area_max` / `ambiguous_factor` / `ambiguous_max` branches; ambiguous return slot is always `[]` (backward-compatible 3-tuple). DR1, pnbody, and SAM2 flows no longer get GT-derived satellite area rejection (consistent with the decision to de-couple satellite area thresholds from `mask_stats_summary.json`).
- **`_evaluate_complete` / force-rebuild cleanup** updated to include new DR1 artifacts (`sam3_raw_overlay.png`, `sam3_eval_overlay.png`, `sam3_satellite_diagnostics.json`).

- **Review: GT-Driven EV Refactor (V1.2)**
  - `satellite_ev` label space tightened to `{confirm_complete, add_missing, confirm_empty}` (removed `remove_fp`, `redraw`)
  - `satellite_ev` and `stream_ev` silver generation switched from pred-vs-GT exhaustivity to deterministic GT-driven synthetic variants (`gt_complete`, `gt_empty`, `drop_N`)
  - `EvAssetRefs` extended with `synthetic_variant_id`, `hidden_instance_ids`, `visible_instance_ids`
  - `compute_revision_hash_ev` accepts `synthetic_variant_id` (backward-compat: `None` omitted from hash)
  - `ensure_ev_image` gains `state_key` for multi-variant filename disambiguation
  - `_build_ev_examples` iterates all `image:*` silver records; fragment hints built from visible instances only
  - `generate_silver_labels.py`: `--pred-dir` optional for EV families
  - `correction.py`: synthetic write-back guard rejects `drop_*` variants
  - `spot_check_silver.py`: variant distribution + visible/hidden disjointness checks
  - `prompt_registry_v1.yaml`: `sat_ev_v1` updated; `stream_ev_v1` intentionally unchanged
  - Old `label_satellite_ev` / `label_stream_ev` marked deprecated (still present as migration stubs)
- **Scripts layout:** CLI entry points grouped under `scripts/data/`, `scripts/eval/`, `scripts/viz/`, and `scripts/analysis/` (update any local commands or job scripts that still used flat `scripts/*.py` paths).
- **HPC:** Slurm eval sweep moved to `scripts/cluster/` (`launch_eval_sweep.sh`, `eval_sweep.slurm`); launcher exports `REPO_ROOT` / `CONDA_ENV`; batch script no longer hardcodes a single machine path (still edit `#SBATCH` and `module load` for your cluster).

### Added

- **Forward Observation Noise Model** for controlled SNR degradation of SB maps
  - `src/noise/forward_observation.py` — `ForwardObservationModel` class
    - Physics chain: SB(mag) → flux → counts → +sky → Poisson → +read_noise → −sky → mag
    - Quantile-based SNR regions (signal_quantile, background_quantile) — GT-free
    - Analytic `expected_snr()` via variance decomposition (no Monte Carlo)
    - `from_target_snr()` factory: bisection on analytic SNR to auto-tune signal_scale
    - Negative flux → NaN (compatible with existing `LSBPreprocessor` nan_to_num)
    - Isolated `np.random.Generator` per instance (multi-process safe)
  - `configs/noise_profiles.yaml` — 4 target profiles (SNR 5/10/20/50)
  - `scripts/data/generate_noisy_fits.py` — Batch CLI with `--galaxies` and `--profiles` filters
  - Output: `data/04_noise/{profile}/magnitudes-Fbox-{gid}-{orient}-VIS2.fits.gz`
  - FITS headers annotated: NOISSNR, NOISSCL, NOISSKY, NOISRDN, NOISMSN

- **Noise-Augmented Training Dataset**
  - `scripts/data/render_noisy_fits.py` — Render noisy FITS → PNG images per SNR profile
  - `scripts/data/build_noise_augmented_annotations.py` — Generate COCO annotations for noise-augmented images, reusing GT masks from clean images
  - `src/pipelines/unified_dataset/noise_aug.py` — Core noise augmentation logic (render + annotation generation)
  - Output: `annotations_train_noise_augmented.json` with `_snr{N}` image variants

- **Galaxy-Level Train/Val Split**
  - `scripts/data/split_annotations.py` — Split COCO annotations by galaxy ID (no data leakage)
  - `configs/sam3_dataset_split.yaml` — Split configuration (train/val ratio, seed)
  - `src/pipelines/unified_dataset/split.py` — Split logic with galaxy-level grouping

- **SAM3 Evaluation Pipeline**
  - `scripts/eval/evaluate_sam3.py` — End-to-end SAM3 evaluation CLI
  - `src/evaluation/sam3_eval.py` — Type-aware IoU evaluation (streams vs satellites)
  - `src/inference/sam3_prompt_runner.py` — SAM3 prompt-based inference runner
  - `scripts/eval/evaluate_sam2.py` — Standalone SAM2 evaluation script
  - `scripts/eval/run_batch_eval.sh`, `scripts/eval/run_batch_eval_type_aware.sh` — Batch evaluation shell wrappers

- **Unified Dataset Pipeline Modularization** (`src/pipelines/unified_dataset/`)
  - Split monolithic pipeline into focused submodules: `config`, `paths`, `keys`, `fs_utils`, `render`, `gt`, `inference`, `inference_sam2`, `inference_sam3`, `compose`, `export`, `artifacts`, `noise_aug`, `split`, `preprocessor_factory`

- **Geometry Utility** — `src/utils/geometry.py`
  - `discrete_convex_area()`: Pixel-corner-aware convex hull area to fix solidity > 1.0 on compact shapes

- **Visualization** — Two new overlay functions in `src/visualization/overlay.py`
  - `save_evaluation_overlay()`: QA overlay with GT white contours + prediction semi-transparent fills
  - `save_instance_overlay()`: Colored instance map overlay for merged GT QA

- **Tests** — 7 new test modules
  - `test_artifacts.py`, `test_cli_compat.py`, `test_compose.py`, `test_dataset_keys.py`, `test_galaxy_split.py`, `test_gt_phase.py`, `test_noise_aug.py`

### Changed

- **SAM3 Visualization**: Completely rewrote `scripts/viz/visualize_sam3.py` to support the refactored data pipeline outputs.
  - Adapted to unified dataset annotations, visualizing variants like `asinh_stretch` vs `linear_magnitude`.
  - Utilized `multiprocessing` to generate visualizations for all galaxies concurrently.
  - Implemented fully vectorized RLE decoding (`np.repeat`) and mask broadcasting.
  - Outputs a new 4-column grid layout (Original, Streams, Satellites, Combined) under `visualizations_grid/`.
- **Solidity computation**: `mask_metrics.py` and `satellite_prior_filter.py` now use `discrete_convex_area` (pixel-corner expansion) instead of raw `ConvexHull` on pixel centers.
- **Robust config loading**: `load_filter_cfg`, `load_area_target`, `load_streams_cfg` all use 3-tier guards (file → JSON parse → key presence) with `warnings.warn` instead of silent fallback.
- **Streams filter**: `StreamsSanityFilter` supports absolute `max_area_px` threshold from GT stats, taking priority over fractional `max_area_frac`.

### Removed

- `src/analysis/sweep_scoring.py` — Sweep aggregation logic superseded by direct evaluation pipeline.
- `scripts/sweep_automask_configs.py` — AutoMask sweep script replaced by integrated pipeline.
- `scripts/build_dataset.py` — Replaced by `scripts/data/prepare_unified_dataset.py` + submodules.
- `configs/automask_sweep.yaml`, `configs/automask_sweep_2.yaml`, `configs/data_prep_sam2.yaml`, `configs/data_prep_sam3.yaml`, `configs/sweep_subset_5.yaml` — Obsolete configs.

## [0.4.1] - 2026-02-18

### Changed

- **Aspect ratio: dual-method + curvature metric**
  - `aspect_sym_moment`: covariance-eigenvalue axis ratio (rotation-invariant, global shape)
  - `aspect_sym_boundary`: `cv2.fitEllipse` axis ratio (boundary-aware)
  - `curvature_ratio`: skeleton_length / ellipse_major (>1 = curved/bent structure)
  - `aspect_sym`: kept as backward-compat alias for `aspect_sym_moment`
  - Updated in `src/analysis/mask_metrics.py`, `scripts/analysis/analyze_mask_stats.py`, `src/postprocess/satellite_prior_filter.py`
  - Filter reads `aspect_sym_moment` first, falls back to `aspect_sym` for legacy compat
  - New deps: `opencv-python-headless`, `scikit-image` (lazy-imported)

## [0.4.0] - 2026-02-09

### Added

- **Unified Data Pipeline** for canonical ground truth generation
  - `scripts/data/prepare_unified_dataset.py` - 4-phase pipeline (Render → GT → Satellites → Export)
  - `configs/unified_data_prep.yaml` - Centralized configuration for all phases
  - **Phased Architecture**:
    - **Render**: Caches preprocessed images (linear, asinh, multi-exposure)
    - **GT**: Preserves original streams IDs from FITS masks
    - **Satellites**: AutoMask integration with `satellites_cache.npz` (RLE encoded)
    - **Export**: Generates SAM2 symlinks and SAM3 COCO attributes from single source

- **Traceability & Reproducibility**
  - `manifest.json` per sample: Configs, processing stats, inference time, overlap rates
  - `id_map.json`: Explicit mapping of instance IDs (streams vs satellites)
  - `overlay.png`: Visual QA for every generated ground truth
  - **Deterministic Sorting**: Logic to ensure stable instance IDs across runs

- **Satellite Integration Features**
  - `overlap_policy`: Support for "keep_streams" (satellites don't overwrite streams)
  - `satellite_sort_policy`: Configurable sorting (e.g., area descending) before ID assignment
  - **Merged GT**: `instance_map_uint8.png` combining streams + satellites

### Changed

- **SAM3 Output Format**: Enforced COCO-only structure (no `masks/` folder) for compatibility
- **AutoMask Workflow**: Integrated directly into data prep pipeline (replacing standalone sweep for production generation)


## [0.3.0] - 2026-02-04

### Added

- **Multi-Exposure Preprocessor** for 3-channel feature encoding
  - `MultiExposurePreprocessor` class in `src/data/preprocessing.py`
  - R channel: Linear magnitude mapping (global range [20, 35])
  - G channel: Asinh stretch (mag → flux → asinh)
  - B channel: Gamma from G (g^γ, γ=0.5 boosts faint features)
  - `get_params_dict()` for metadata export (thesis reproducibility)
  - `last_stats` dict with per-image `vmax_ref_flux`, `finite_pixel_ratio`, channel stats

- **Preview Mode** for rendering comparison
  - `--preview` CLI flag in `scripts/build_dataset.py`
  - Generates 2×3 panel PNG: R/G/B (gray) + RGB/old_linear/old_asinh
  - Saves metadata JSON with all rendering params + computed values
  - Prints channel stats to console (min/max/mean, vmax_ref_flux)

- **Configuration updates**
  - `preprocessing_method: "multi_exposure"` option in config files
  - `gamma` parameter for B channel (default: 0.5)

### Changed

- `configs/data_prep_sam2.yaml` - Added multi_exposure method docs and gamma parameter

## [0.2.0] - 2026-02-02

### Added

- **Dual preprocessing method support** for dataset preparation
  - `LinearMagnitudePreprocessor` - Global linear magnitude normalization [20, 35] mag/arcsec²
  - Config-driven method selection via `preprocessing_method` field
  - Direct magnitude-to-pixel mapping for cross-galaxy photometric consistency
  - Simple pipeline: Clean → Clip → Linear Normalize → 8-bit RGB

- **Configuration enhancements**
  - `preprocessing_method` field in all config files ("asinh_stretch" or "linear_magnitude")
  - Method-specific parameters clearly documented for both approaches
  - Interpolation mode selection for linear method ("cubic", "linear", "nearest")

- **Test scripts**
  - `test_linear_preprocessing.py` - Validates LinearMagnitudePreprocessor pipeline stages

### Changed

- `src/data/preprocessing.py` - Added `LinearMagnitudePreprocessor` class alongside `LSBPreprocessor`
- `scripts/build_dataset.py` - Dynamic preprocessor instantiation based on config method
- `configs/data_prep_sam2.yaml` - Added preprocessing method selection with documentation
- `configs/data_prep_sam3.yaml` - Added preprocessing method selection with documentation

### Technical Notes

- **Preprocessing method comparison**:
  - `asinh_stretch` (default): Max dynamic range, per-image adaptive, Mag→Flux→Asinh
  - `linear_magnitude`: Cross-galaxy consistency, global fixed range [20,35], direct mapping
- **Backward compatibility**: Default method remains "asinh_stretch" for existing workflows

## [0.1.0] - 2026-01-31

### Added

- **Modular project structure** following Cookiecutter Data Science standards
  - `src/data/` - Data loading and preprocessing modules
  - `src/evaluation/` - IoU metric calculations
  - `src/utils/` - Logger and COCO utilities
  - `src/visualization/` - Plotting functions

- **Configuration system** with task-specific YAML files
  - `configs/data_prep_sam2.yaml` - SAM2 folder-based video format (1072×1072)
  - `configs/data_prep_sam3.yaml` - SAM3 COCO JSON format (1024×1024)
  - `configs/eval_sam2.yaml` - Model evaluation settings

- **Entry scripts** for streamlined execution
  - `scripts/build_dataset.py` - Dataset preparation with config switching
  - `scripts/eval_model.py` - Model evaluation with CLI override support

- **Core utilities**
  - `src/utils/logger.py` - Rotating file logger (`logs/*.log`)
  - `src/utils/coco_utils.py` - RLE encoding, bbox extraction, COCO annotations
  - `src/data/io.py` - FITS/image loading with broken symlink detection
  - `src/data/preprocessing.py` - `LSBPreprocessor` class for magnitude normalization

- **Documentation**
  - `docs/datasets/datareadme.md` - Dataset format comparison
  - `docs/api/` - Sphinx-generated API documentation
  - `PROJECT_CONTEXT.md` - Project structure and code skeleton

### Changed

- Refactored monolithic scripts into modular package
  - `prepare_sam3_data_new.py` → `scripts/build_dataset.py` + `src/data/*`
  - `sam2_inference_iou.py` → `scripts/eval_model.py` + `src/evaluation/*`

### Technical Notes

- **SAM2 vs SAM3 format differences**:
  - SAM2: Folder-based `img_folder/{sample}/0000.png`, log transform, 1072×1072
  - SAM3: Flat `images/*.png` + COCO JSON, linear magnitude scaling, 1024×1024
  
- **CLI precedence**: Command-line arguments override YAML config values

- **Reproducibility**: All configs include `random_seed` (default: 42)
