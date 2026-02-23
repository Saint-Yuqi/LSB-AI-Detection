# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
