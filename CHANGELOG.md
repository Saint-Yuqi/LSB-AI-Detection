# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  - `scripts/generate_noisy_fits.py` — Batch CLI with `--galaxies` and `--profiles` filters
  - Output: `data/04_noise/{profile}/magnitudes-Fbox-{gid}-{orient}-VIS2.fits.gz`
  - FITS headers annotated: NOISSNR, NOISSCL, NOISSKY, NOISRDN, NOISMSN

- **Noise-Augmented Training Dataset**
  - `scripts/render_noisy_fits.py` — Render noisy FITS → PNG images per SNR profile
  - `scripts/build_noise_augmented_annotations.py` — Generate COCO annotations for noise-augmented images, reusing GT masks from clean images
  - `src/pipelines/unified_dataset/noise_aug.py` — Core noise augmentation logic (render + annotation generation)
  - Output: `annotations_train_noise_augmented.json` with `_snr{N}` image variants

- **Galaxy-Level Train/Val Split**
  - `scripts/split_annotations.py` — Split COCO annotations by galaxy ID (no data leakage)
  - `configs/sam3_dataset_split.yaml` — Split configuration (train/val ratio, seed)
  - `src/pipelines/unified_dataset/split.py` — Split logic with galaxy-level grouping

- **SAM3 Evaluation Pipeline**
  - `scripts/evaluate_sam3.py` — End-to-end SAM3 evaluation CLI
  - `src/evaluation/sam3_eval.py` — Type-aware IoU evaluation (streams vs satellites)
  - `src/inference/sam3_prompt_runner.py` — SAM3 prompt-based inference runner
  - `scripts/evaluate_sam2.py` — Standalone SAM2 evaluation script
  - `scripts/run_batch_eval.sh`, `scripts/run_batch_eval_type_aware.sh` — Batch evaluation shell wrappers

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

- **SAM3 Visualization**: Completely rewrote `scripts/visualize_sam3.py` to support the refactored data pipeline outputs.
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
- `scripts/build_dataset.py` — Replaced by `scripts/prepare_unified_dataset.py` + submodules.
- `configs/automask_sweep.yaml`, `configs/automask_sweep_2.yaml`, `configs/data_prep_sam2.yaml`, `configs/data_prep_sam3.yaml`, `configs/sweep_subset_5.yaml` — Obsolete configs.

## [0.4.1] - 2026-02-18

### Changed

- **Aspect ratio: dual-method + curvature metric**
  - `aspect_sym_moment`: covariance-eigenvalue axis ratio (rotation-invariant, global shape)
  - `aspect_sym_boundary`: `cv2.fitEllipse` axis ratio (boundary-aware)
  - `curvature_ratio`: skeleton_length / ellipse_major (>1 = curved/bent structure)
  - `aspect_sym`: kept as backward-compat alias for `aspect_sym_moment`
  - Updated in `src/analysis/mask_metrics.py`, `scripts/analyze_mask_stats.py`, `src/postprocess/satellite_prior_filter.py`
  - Filter reads `aspect_sym_moment` first, falls back to `aspect_sym` for legacy compat
  - New deps: `opencv-python-headless`, `scikit-image` (lazy-imported)

## [0.4.0] - 2026-02-09

### Added

- **Unified Data Pipeline** for canonical ground truth generation
  - `scripts/prepare_unified_dataset.py` - 4-phase pipeline (Render → GT → Satellites → Export)
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
