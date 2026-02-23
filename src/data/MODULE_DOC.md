# Module: data

## Responsibilities
- Handle file I/O operations for astronomical data formats (`.fits.gz`, image files, mask files).
- Parse sample and metadata naming conventions consistently.
- Perform astrophysical-standard preprocessing to convert Surface Brightness arrays (mag/arcsec²) into machine learning-ready RGB format.
- Load and parse satellite instance metadata from serialization (`.pkl`).

## Non-goals
- **No Evaluation:** This module does not evaluate any models or analyze masks.
- **No Iteration:** It does not iterate over directories. Batch processing is handled externally.

## Inputs / Outputs

### `load_fits_gz(path)`
- **Input:** `path: Path`
- **Output:** `np.ndarray` — shape `(H, W)`, dtype `np.float32`

### `load_image(path)`
- **Input:** `path: Path`
- **Output:** `np.ndarray` — shape `(H, W, 3)`, dtype `np.uint8`

### `load_mask(path)`
- **Input:** `path: Path`
- **Output:** `np.ndarray` — shape `(H, W)`, dtype `np.uint8`

### `parse_sample_name(name)`
- **Input:** `name: str`
- **Output:** `Optional` dict with all keys present or `None` if parsing fails:
  - `galaxy_id: int`
  - `orientation: str`
  - `sb_threshold: float`
  - `type: str`

### `SatelliteInstance.from_dict(galaxy_id, data, load_seg_map)`
- **Input:**
  - `galaxy_id: str`
  - `data` dict with required keys:
    - `instance_id: int`
    - `area: int`
    - `bbox: List[int]` — `[x_min, y_min, width, height]`
    - `centroid: List[float]` — `[x, y]`
    - `seg_map: np.ndarray` — shape `(H, W)`, dtype `np.uint8` (Optional, loaded only if `load_seg_map=True`)
  - `load_seg_map: bool`
- **Output:** `SatelliteInstance`

### `SatelliteDataLoader.get_satellites(galaxy_id, orientation, sb_threshold, load_seg_map)`
- **Input:** `galaxy_id: int`, `orientation: str`, `sb_threshold: float`, `load_seg_map: bool`
- **Output:** `GalaxySatellites`

### Preprocessors `process(sb_map)` — (`LSBPreprocessor`, `LinearMagnitudePreprocessor`, `MultiExposurePreprocessor`)
- **Input:** `sb_map: np.ndarray` — shape `(H, W)`, dtype `np.float32`
- **Output:** `np.ndarray` — shape `(H_target, W_target, 3)`, dtype `np.uint8`

### Preprocessors `resize_mask(mask)`
- **Input:** `mask: np.ndarray` — shape `(H, W)`, dtype `np.uint8`
- **Output:** `np.ndarray` — shape `(H_target, W_target)`, dtype `np.uint8`

## Invariants
- **Photometry:** Lower magnitude value corresponds to higher physical flux (brighter source).
- **Mask Integrity:** Integer instance IDs are preserved exactly through resizing; no interpolation is applied.

## Produced Artifacts
- Returns in-memory strongly-typed instances (`np.ndarray`, `SatelliteInstance`, `GalaxySatellites`).

## Failure Modes
- `FileNotFoundError`: Raised when `path` does not exist or is a broken symlink (applies to all `load_*` functions and `.pkl` loading).
- `ValueError`: Raised by `load_fits_gz` when the file is not a valid gzip-compressed FITS file.
- `ValueError`: Raised by any Preprocessor `__init__` when `b_mode` is missing or holds an unrecognized string value.
- `KeyError`: Raised by `SatelliteInstance.from_dict` when `data` is missing any required key (`instance_id`, `area`, `bbox`, `centroid`).
