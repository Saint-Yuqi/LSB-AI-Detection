# Project Context Summary

> Auto-generated project structure and code skeleton for LSB-AI-Detection.
> Last updated: 2026-02-09

## Project Overview

A modular Python package for detecting Low Surface Brightness (LSB) features in astronomical images using SAM2/SAM3 (Segment Anything Model).

---

## Directory Structure

```
LSB-AI-Detection/
├── configs/                          # Configuration files
│   ├── data_prep_sam2.yaml           # SAM2 dataset config (folder-based, 1072×1072)
│   ├── data_prep_sam3.yaml           # SAM3 dataset config (COCO JSON, 1024×1024)
│   ├── eval_sam2.yaml                # Model evaluation config
│   └── automask_sweep.yaml           # AutoMask config sweep definitions
│
├── data/
│   ├── 01_raw/                       # Raw FITS data (symlinked)
│   │   └── LSB_and_Satellites/       # FIREbox-DR1 + fbox data
│   └── 02_processed/                 # Generated training data
│       └── sam3_unified/             # COCO format (images/, masks/, annotations.json)
│       └── sam2_prepared/            # folder-based format (images/, masks/)
│
├── docs/
│   ├── api/                          # Sphinx API documentation
│   │   ├── conf.py
│   │   ├── index.rst
│   │   └── _build/html/              # Generated HTML docs
│   └── datasets/                     # Dataset documentation
│       ├── datareadme.md             # SAM2 vs SAM3 format comparison
│       └── FIREbox-DR1_analysis.md
│
├── scripts/                          # Entry point scripts
│   ├── build_dataset.py              # Dataset preparation (SAM2/SAM3)
│   ├── eval_model.py                 # Model evaluation
│   ├── sweep_automask_configs.py     # AutoMask config sweep & ranking
│   ├── analyze_mask_stats.py         # GT mask statistics analysis
│   └── visualize_sam3.py             # Visualization for SAM3 dataset
│
├── src/                              # Source code package
│   ├── data/                         # Data loading & preprocessing
│   │   ├── io.py
│   │   └── preprocessing.py
│   ├── inference/                    # Model inference wrappers
│   │   └── sam2_automask_runner.py   # SAM2 AutoMask generator wrapper
│   ├── postprocess/                  # Mask post-processing filters
│   │   ├── satellite_prior_filter.py # Area/solidity/aspect_sym rules
│   │   └── core_exclusion_filter.py  # Centre-radius exclusion
│   ├── analysis/                     # Metrics & scoring
│   │   ├── mask_metrics.py           # Per-mask geometry computation
│   │   └── sweep_scoring.py          # Config ranking & aggregation
│   ├── evaluation/                   # Evaluation metrics
│   │   └── metrics.py
│   ├── utils/                        # Utilities
│   │   ├── coco_utils.py
│   │   └── logger.py
│   └── visualization/                # Plotting
│       ├── plotting.py
│       └── overlay.py                # Multi-layer contour overlays
│
├── logs/                             # Runtime logs (auto-generated)
├── notebooks/                        # Jupyter notebooks
├── CHANGELOG.md                      # Version history
└── requirements.txt                  # Python dependencies
```

---

## Code Skeleton

### `src/data/io.py`
```python
def load_fits_gz(filepath: Path) -> np.ndarray:
    """Load gzipped FITS file with symlink validation."""

def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array (H, W, 3)."""

def load_mask(path: Path) -> np.ndarray:
    """Load mask as grayscale numpy array (H, W)."""

def parse_sample_name(name: str) -> Optional[dict]:
    """Parse '{galaxy_id}_{orient}_SB{threshold}_{type}' format."""

# --- Satellite Data Structures ---

@dataclass
class SatelliteInstance:
    """Single satellite from PKL: x, y, seg_map (2051×2051), seg_ids, area, etc."""
    def get_binary_mask(self, shape: tuple) -> np.ndarray: ...

@dataclass  
class GalaxySatellites:
    """All satellites for one galaxy at one SB threshold."""
    def get_combined_mask(self, shape: tuple) -> np.ndarray: ...

class SatelliteDataLoader:
    """Loader for props_gals_Fbox_new.pkl (lazy-loading, ~13GB)."""
    def get_satellites(self, galaxy_id, orientation, sb_threshold) -> GalaxySatellites: ...
```

#### PKL Structure: `props_gals_Fbox_new.pkl`
```
dict['{galaxy_id}, {orient}']     # e.g., '11, eo' (132 galaxies)
  └─ dict['SBlim{threshold}']     # e.g., 'SBlim27', 'SBlim27.5'
       └─ dict['id{N}']           # e.g., 'id1', 'id2', 'id3'
            ├─ x, y: float64              # Centroid coordinates
            ├─ geo-x, geo-y: float64      # Geometric centroid
            ├─ seg_map: ndarray (2051, 2051) uint8  # Full mask
            ├─ seg_ids: ndarray (N, 2) int64        # [y, x] coords
            ├─ area: uint64               # Pixel count
            ├─ axis_ratio, gini: float64  # Morphology
            └─ mag_r, sb_fltr: float64    # Photometry
```

---

### `src/data/preprocessing.py`
```python
class LSBPreprocessor:
    """Asinh stretch preprocessing for LSB astronomical data.
    
    Pipeline: Magnitude → Flux → Asinh Stretch → 8-bit RGB
    """
    
    def __init__(
        self,
        zeropoint: float = 22.5,         # Mag zeropoint for flux conversion
        nonlinearity: float = 10.0,      # Asinh stretch parameter (10-50)
        clip_percentile: float = 99.5,   # Percentile for vmax reference
        target_size: Tuple[int, int] = (1024, 1024)
    ): ...
    
    def mag_to_flux(self, mag: np.ndarray) -> np.ndarray:
        """Convert mag/arcsec² to flux: flux = 10^((zp - mag) / 2.5)"""
    
    def asinh_stretch(self, flux: np.ndarray) -> np.ndarray:
        """Apply Asinh stretch: linear for faint, log for bright."""
    
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """Full pipeline: clean → mag→flux → clip → asinh → 8-bit RGB."""
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask with nearest-neighbor interpolation."""
```
```python
class LinearMagnitudePreprocessor:
    """Linear stretch preprocessing for astronomical data.
    
    Pipeline: Magnitude → Flux → Linear Stretch → 8-bit RGB
    """
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """Full pipeline: clean → mag→flux → clip → linear → 8-bit RGB."""
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask with nearest-neighbor interpolation."""
```
```python
class MultiExposurePreprocessor:
    """3-channel multi-exposure rendering for LSB visualization.
    
    Channels:
        R: Linear magnitude mapping (global_mag_min → 255, global_mag_max → 0)
        G: Asinh stretch (mag → flux → asinh)
        B: Gamma from G (g^gamma, gamma < 1 boosts faint)
    
    Attributes:
        last_stats: Per-image computed values (vmax_ref_flux, finite_pixel_ratio, etc.)
    """
    def __init__(
        self,
        global_mag_min: float = 20.0,
        global_mag_max: float = 35.0,
        zeropoint: float = 22.5,
        nonlinearity: float = 300.0,
        clip_percentile: float = 99.5,
        gamma: float = 0.5,
        target_size: Tuple[int, int] = (1024, 1024),
    ): ...
    
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """Returns (H, W, 3) uint8 RGB: [R=linear, G=asinh, B=gamma]"""
    
    def get_params_dict(self) -> Dict[str, Any]:
        """Get all rendering params + per-image stats for metadata logging."""
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask with nearest-neighbor interpolation."""
```

### `src/evaluation/metrics.py`
```python
def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Binary IoU between prediction and ground truth."""

def calculate_instance_iou(pred_masks: List[Dict], gt_mask: np.ndarray) -> Dict:
    """Instance IoU metrics.
    Returns: {'binary_iou': float, 'mean_instance_iou': float, 
              'num_gt_instances': int, 'num_pred_masks': int}
    """

def calculate_matched_metrics(pred_masks: List[Dict], gt_mask: np.ndarray, iou_threshold: float = 0.5) -> Dict:
    """Instance-level Recall with per-GT matching."""
```

---

### `src/inference/sam2_automask_runner.py`
```python
class AutoMaskRunner:
    """Wrapper for SAM2AutomaticMaskGenerator with CUDA sync timing.
    
    Args:
        checkpoint: Path to SAM2 checkpoint (.pt)
        model_cfg: SAM2 config yaml (e.g., 'configs/sam2.1/sam2.1_hiera_b+.yaml')
        device: 'cuda' or 'cpu'
    """
    def run(self, image: np.ndarray, config: Dict | None = None) -> Tuple[List[Dict], float]:
        """Generate masks. Returns (masks, time_ms)."""
    
    def warmup(self, image: np.ndarray, n: int = 2):
        """JIT warmup passes."""
```

---

### `src/postprocess/satellite_prior_filter.py`
```python
def load_filter_cfg(stats_json: Path, feature_type: str = "satellites") -> Dict:
    """Load thresholds from mask_stats_summary.json (data-driven)."""

class SatellitePriorFilter:
    """Filter masks by area/solidity/aspect_sym rules."""
    def filter(self, masks: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Returns (kept, rejected)."""
```

---

### `src/postprocess/core_exclusion_filter.py`
```python
class CoreExclusionFilter:
    """Exclude masks with centroid inside R_exclude = radius_frac * min(H,W)."""
    def filter(self, masks: List[Dict], H: int, W: int) -> Tuple[List[Dict], List[Dict], Dict]:
        """Returns (kept, core_hits, diagnostics)."""
```

---

### `src/analysis/mask_metrics.py`
```python
def compute_mask_metrics(seg: np.ndarray, H: int, W: int, compute_hull: bool = True) -> Dict:
    """Vectorised per-mask geometry: area, bbox, aspect_sym, solidity, centroid, dist_to_center."""

def append_metrics_to_masks(masks: List[Dict], H: int, W: int, compute_hull: bool = True):
    """In-place add geometry metrics to each mask dict."""
```

---

### `src/analysis/sweep_scoring.py`
```python
def summarise_config(config_dir: Path) -> Dict:
    """Aggregate per_image_metrics.csv → summary with CV, core_rate, stability, score."""

def aggregate_and_rank(output_root: Path) -> List[Dict]:
    """Walk config dirs, compute summaries, write ranking.json."""
```

---

### `src/visualization/overlay.py`
```python
def save_overlay(
    image: np.ndarray,
    kept: List[Dict],
    core_rejected: List[Dict] | None = None,
    prior_rejected: List[Dict] | None = None,
    out_path: Path = "overlay.png",
    draw_prior: bool = False,
):
    """Multi-layer contours: kept (green), core (red), prior (gray)."""
```

---

### `src/utils/logger.py`
```python
def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: int = logging.INFO
) -> logging.Logger:
    """Create logger with console + rotating file handlers.
    File output: logs/{name}_YYYY-MM-DD.log
    """
```

---

### `src/utils/coco_utils.py`
```python
def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask to COCO RLE format."""

def get_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Extract [x, y, width, height] from mask."""

def create_categories(thresholds: List[float]) -> Tuple[List[Dict], Dict]:
    """Generate COCO categories for stream/satellite at each SB threshold."""

def process_mask_to_annotations(
    mask_data: np.ndarray,
    image_id: int,
    category_id: int,
    ann_id: int,
    min_area: int = 10
) -> Tuple[List[Dict], int]:
    """Convert instance mask to COCO annotation dicts.
    Filters instances smaller than min_area pixels."""
```

---

### `src/visualization/plotting.py`
```python
def save_visualization(
    image: np.ndarray,
    pred_masks: List[Dict],
    gt_mask: np.ndarray,
    output_path: Path
) -> None:
    """3-panel comparison: input | predictions | ground truth."""
```

---

## Configuration Reference

| Config File           | Format       | Image Size | Processing               |
| --------------------- | ------------ | ---------- | ------------------------ |
| `data_prep_sam2.yaml` | Folder-based | 1072×1072  | FITS → Flux → Asinh      |
| `data_prep_sam3.yaml` | COCO JSON    | 1024×1024  | FITS → Flux → Asinh      |
| `eval_sam2.yaml`      | -            | -          | Model evaluation         |
| `automask_sweep.yaml` | Config list  | -          | AutoMask sweep & ranking |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Build SAM3 dataset
python scripts/build_dataset.py --config configs/data_prep_sam3.yaml

# Build SAM2 dataset (switch config)
python scripts/build_dataset.py --config configs/data_prep_sam2.yaml

# Evaluate model
python scripts/eval_model.py --checkpoint /path/to/model.pt --save_vis
```
