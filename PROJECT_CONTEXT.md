# Project Context Summary

> Auto-generated project structure and code skeleton for LSB-AI-Detection.
> Last updated: 2026-04-22

## Project Overview

A modular Python package for detecting Low Surface Brightness (LSB) features
in astronomical images using SAM3 (Segment Anything Model 3). The legacy
SAM2 stack was removed; the single evaluation entrypoint is
`scripts/eval/evaluate_checkpoint.py` (+ `configs/eval_checkpoint.yaml`),
backed by `src/evaluation/checkpoint_eval.py`. Three benchmarks are
supported on a fixed 1024x1024 working grid:

- `fbox_gold_satellites` — 132 satellite samples, ROI `[277:747, 277:747]`,
  renders under `data/02_processed/renders_eval/fbox_gold_satellites/...`.
- `firebox_dr1_streams` — 72 streams samples at SBlim31.5, full-frame,
  renders under `data/02_processed/renders_eval/firebox_dr1_streams/...`.
- `gt_canonical` — 70 streams+satellites samples, full-frame, renders
  reused from `data/02_processed/renders/{condition}/{variant}/...`.

Post-processing has two layers: `post_pred_only` (runs for every
benchmark; 5 prediction-only stages ending in cross-type conflict) and
`post_gt_aware` (only populated for `gt_canonical`; adds the GT-stream
conflict resolver).

---

## Directory Structure

```
LSB-AI-Detection/
├── configs/                          # Configuration files
│   ├── unified_data_prep.yaml        # Main unified pipeline config (4-phase)
│   ├── archive/                      # Archived migration-only configs
│   │   └── sam3_satellite_overrides_legacy.yaml
│   ├── noise_profiles.yaml           # Forward observation noise profiles
│   ├── sam3_dataset_split.yaml       # Galaxy-level train/val split
│   ├── eval_sam2.yaml                # SAM2 evaluation config
│   ├── eval_sam3.yaml                # SAM3 type-aware evaluation config
│   └── review/                       # AI Verifier Protocol V1.1
│       ├── render_spec_v1.yaml       # Visual specs per family
│       ├── prompt_registry_v1.yaml   # Prompt templates per family
│       └── silver_policy_v1.yaml     # Silver derivation thresholds
│
├── data/
│   ├── 01_raw/                       # Raw FITS data (symlinked)
│   │   └── LSB_and_Satellites/       # FIREbox-DR1 + fbox data
│   ├── 02_processed/                 # Generated training data
│   │   ├── gt_canonical/             # Canonical Ground Truth (instance_map_uint8.png)
│   │   ├── renders/                  # Cached intermediate renders
│   │   ├── sam2_prepared/            # SAM2 symlinks (img_folder, gt_folder)
│   │   └── sam3_prepared/            # SAM3 COCO (images, annotations*.json)
│   └── 04_noise/                     # Noise-injected FITS (forward observation model)
│       ├── snr05/                    # SNR≈5 profiles
│       ├── snr10/                    # SNR≈10 profiles
│       ├── snr20/                    # SNR≈20 profiles
│       └── snr50/                    # SNR≈50 profiles
│
├── docs/
│   ├── api/                          # Sphinx API documentation
│   └── datasets/                     # Dataset documentation
│       ├── datareadme.md             # SAM2 vs SAM3 format comparison
│       └── FIREbox-DR1_analysis.md
│
├── scripts/                          # CLI entry points (data / eval / review / viz / analysis)
│   ├── MODULE_DOC.md
│   ├── data/                         # Dataset build, noise, splits
│   │   ├── prepare_unified_dataset.py
│   │   ├── render_noisy_fits.py
│   │   ├── build_noise_augmented_annotations.py
│   │   ├── build_training_dataset.py
│   │   ├── split_annotations.py
│   │   ├── generate_noisy_fits.py
│   │   └── generate_pnbody_fits.py
│   ├── eval/                         # Evaluation + local batch bash
│   │   ├── evaluate_sam2.py
│   │   ├── evaluate_sam3.py
│   │   ├── run_batch_eval.sh
│   │   ├── run_batch_eval_type_aware.sh
│   │   └── run_sweep_eval.sh
│   ├── cluster/                      # Slurm jobs (site-specific headers)
│   │   ├── launch_eval_sweep.sh
│   │   └── eval_sweep.slurm
│   ├── viz/
│   │   ├── visualize_sam3.py
│   │   ├── visualize_eval_metrics.py
│   │   ├── visualize_sam2.py
│   │   └── overlay_masks_on_streams.py
│   ├── review/                       # AI Verifier Protocol CLI scripts
│   │   ├── generate_silver_labels.py # Silver label derivation
│   │   ├── render_review_assets.py   # Render crop + context + EV assets
│   │   ├── build_verifier_examples.py # Pipeline artifacts -> business JSONL
│   │   ├── run_etl.py                # Business -> chat JSONL
│   │   ├── import_corrections.py     # Correction import with hash validation
│   │   ├── assemble_round.py         # Round artifact assembly
│   │   ├── edit_authoritative_gt.py  # adopt-raw / delete-instance CLI (supports --candidate-rle-sha1)
│   │   ├── bootstrap_shadow_gt.py    # Scaffold Shadow GT (streams + manifest, no satellite artifacts)
│   │   └── migrate_satellite_overrides.py # Legacy override YAML -> explicit Shadow GT adopt/delete
│   └── analysis/
│       ├── analyze_mask_stats.py
│       ├── plot_mask_stats.py
│       └── plot_recall_curve.py
│
├── src/                              # Source code package
│   ├── data/                         # Data loading & preprocessing
│   │   ├── io.py
│   │   └── preprocessing.py
│   ├── noise/                        # Forward observation noise model
│   │   └── forward_observation.py    # SB→flux→counts→Poisson→read→mag
│   ├── inference/                    # Model inference wrappers
│   │   ├── sam2_automask_runner.py   # SAM2 AutoMask generator wrapper
│   │   └── sam3_prompt_runner.py     # SAM3 prompt-based inference runner
│   ├── postprocess/                  # Mask post-processing filters
│   │   ├── satellite_prior_filter.py # Area/solidity/aspect_sym rules (slim v2)
│   │   ├── satellite_score_gate.py   # DR1 static 3-tier size-aware score gate
│   │   ├── satellite_core_policy.py  # DR1 hard/soft core + soft-core rescue
│   │   ├── satellite_conflict_resolver.py # DR1 GT-aware stream-vs-sat resolver
│   │   ├── satellite_pipeline.py     # DR1 8-stage pipeline runner + state/events
│   │   ├── core_exclusion_filter.py  # Centre-radius exclusion (legacy, pnbody)
│   │   ├── stream_satellite_conflict_filter.py # Legacy conflict filter (pnbody)
│   │   ├── streams_sanity_filter.py  # Streams false-positive guard
│   │   ├── candidate_grouping.py     # Centroid-based mask clustering
│   │   └── representative_selection.py # Best-mask-per-group selection
│   ├── analysis/                     # Metrics & scoring
│   │   └── mask_metrics.py           # Per-mask geometry computation
│   ├── evaluation/                   # Evaluation metrics
│   │   ├── metrics.py                # Binary/instance IoU, matched metrics
│   │   └── sam3_eval.py              # Type-aware SAM3 evaluation pipeline
│   ├── pipelines/                    # Pipeline core logic
│   │   └── unified_dataset/          # Modular dataset prep subpackage
│   │       ├── config.py             # Config loading & validation
│   │       ├── paths.py              # Path resolution
│   │       ├── keys.py               # Galaxy key parsing
│   │       ├── fs_utils.py           # Filesystem helpers
│   │       ├── preprocessor_factory.py # Preprocessor instantiation
│   │       ├── render.py             # Phase 1: Image rendering
│   │       ├── gt.py                 # Phase 2: GT generation
│   │       ├── inference.py          # Phase 3: Inference dispatcher
│   │       ├── inference_sam2.py     # Phase 3: SAM2 AutoMask inference
│   │       ├── inference_sam3.py     # Phase 3: SAM3 prompt inference
│   │       ├── compose.py            # Phase 3: Merge streams + satellites
│   │       ├── export.py             # Phase 4: SAM2/SAM3 export
│   │       ├── artifacts.py          # Manifest & QA artifact generation; assign_stable_ids + save_predictions_json (cross-layer raw_index)
│   │       ├── noise_aug.py          # Noise augmentation logic
│   │       └── split.py              # Galaxy-level train/val split
│   ├── utils/                        # Utilities
│   │   ├── coco_utils.py             # RLE encoding, COCO annotation helpers
│   │   ├── geometry.py               # Pixel-corner-aware convex hull area
│   │   ├── logger.py                 # Rotating file logger
│   │   └── runtime_env.py            # Strict conda env contract for SAM3 CLIs
│   ├── review/                       # AI Verifier Protocol V1.1
│   │   ├── MODULE_DOC.md
│   │   ├── schemas.py                # Frozen dataclasses, enums, label spaces
│   │   ├── key_adapter.py            # BaseKey <-> sample_id/halo_id mapping
│   │   ├── render_spec.py            # Versioned render specification registry
│   │   ├── prompt_registry.py        # Fixed prompt template registry
│   │   ├── review_render.py          # Crop/context/EV rendering
│   │   ├── asset_manager.py          # Dedup, crop cache, asset manifests
│   │   ├── candidate_matcher.py      # Pred-centric matching for silver labeling
│   │   ├── silver_labeler.py         # DR1 auto-derived silver labels
│   │   ├── example_builder.py        # Pipeline artifacts -> business JSONL
│   │   ├── etl.py                    # Business JSONL -> chat JSONL transform
│   │   ├── correction.py             # Redraw closed-loop, revision tracking
│   │   ├── holdout.py                # Group-wise holdout splitting
│   │   └── round_manager.py          # Round artifact assembly
│   └── visualization/                # Plotting
│       ├── plotting.py               # Multi-panel comparison plots
│       └── overlay.py                # Multi-layer contour & QA overlays
│
├── tests/                            # Unit tests (pytest)
│   ├── test_artifacts.py
│   ├── test_cli_compat.py
│   ├── test_coco_rle.py
│   ├── test_compose.py
│   ├── test_dataset_keys.py
│   ├── test_eval_type_aware.py
│   ├── test_galaxy_split.py
│   ├── test_gt_phase.py
│   ├── test_matched_metrics.py
│   ├── test_noise_aug.py
│   ├── test_review_schemas.py        # Review schema invariants + key adapter
│   ├── test_candidate_matcher.py     # Pred-centric matching correctness
│   ├── test_review_render.py         # Rendering determinism + visual contracts
│   ├── test_etl.py                   # ETL round-trip, hash stability
│   ├── test_silver_labeler.py        # Silver policy compliance
│   ├── test_correction.py            # Revision hash, redraw closed-loop
│   ├── test_example_builder.py       # Dual-source example building
│   └── test_holdout.py               # Group-wise split, no leakage
│
├── tools/
│   └── githooks/                     # Pre-commit AST sync hooks
├── logs/                             # Runtime logs (auto-generated)
├── notebooks/                        # Jupyter notebooks
├── CHANGELOG.md                      # Version history
└── requirements.txt                  # Python dependencies
```

---

## Code Skeleton

### `scripts/data/prepare_unified_dataset.py`
```python
def run_render_phase(config, base_keys, logger):
    """Phase 1: Generate & cache rendered images (linear, asinh, etc.)."""

def run_gt_phase(config, base_keys, logger):
    """Phase 2: Generate streams_instance_map.npy (preserve original IDs)."""

def run_satellites_phase(config, base_keys, logger):
    """Phase 3: AutoMask inference → Filter → Cache (.npz) → Merge.
    Outputs: instance_map_uint8.png (Training GT), instances.json, overlay.png."""

def run_export_phase(config, base_keys, logger):
    """Phase 4: Generate SAM2 symlinks & SAM3 COCO annotations from canonical GT."""
```

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
    def __init__(self, zeropoint=22.5, nonlinearity=10.0,
                 clip_percentile=99.5, target_size=(1024, 1024)): ...
    def mag_to_flux(self, mag: np.ndarray) -> np.ndarray: ...
    def asinh_stretch(self, flux: np.ndarray) -> np.ndarray: ...
    def process(self, sb_map: np.ndarray) -> np.ndarray: ...
    def resize_mask(self, mask: np.ndarray) -> np.ndarray: ...

class LinearMagnitudePreprocessor:
    """Linear stretch: Magnitude → Flux → Linear Stretch → 8-bit RGB"""
    def process(self, sb_map: np.ndarray) -> np.ndarray: ...
    def resize_mask(self, mask: np.ndarray) -> np.ndarray: ...

class MultiExposurePreprocessor:
    """3-channel: R=linear magnitude, G=asinh stretch, B=gamma from G."""
    def __init__(self, global_mag_min=20.0, global_mag_max=35.0, zeropoint=22.5,
                 nonlinearity=300.0, clip_percentile=99.5, gamma=0.5,
                 target_size=(1024, 1024)): ...
    def process(self, sb_map: np.ndarray) -> np.ndarray: ...
    def get_params_dict(self) -> Dict[str, Any]: ...
    def resize_mask(self, mask: np.ndarray) -> np.ndarray: ...
```

### `src/noise/forward_observation.py`
```python
class ForwardObservationModel:
    """Forward observation noise: SB(mag) → flux → counts → Poisson → read → −sky → mag.
    Quantile-based SNR. Analytic variance: Var ≈ (counts_bkg + sky) + read_noise².
    Negative flux → NaN (LSBPreprocessor nan_to_num handles downstream).
    """
    def __init__(self, zeropoint, signal_scale, sky_level, read_noise,
                 signal_quantile, background_quantile, seed): ...
    def inject(self, sb_map: np.ndarray) -> np.ndarray: ...
    def expected_snr(self, sb_map: np.ndarray) -> float: ...
    @staticmethod
    def from_target_snr(target_snr, sb_map, ...) -> 'ForwardObservationModel': ...
```

### `src/inference/sam2_automask_runner.py`
```python
class AutoMaskRunner:
    """Wrapper for SAM2AutomaticMaskGenerator with CUDA sync timing."""
    def run(self, image: np.ndarray, config: Dict | None = None) -> Tuple[List[Dict], float]: ...
    def warmup(self, image: np.ndarray, n: int = 2): ...
```

### `src/inference/sam3_prompt_runner.py`
```python
class SAM3PromptRunner:
    """SAM3 prompt-based inference: loads model, runs per-annotation prompts."""
```

### `src/evaluation/metrics.py`
```python
def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float: ...
def calculate_instance_iou(pred_masks: List[Dict], gt_mask: np.ndarray) -> Dict: ...
def calculate_matched_metrics(pred_masks: List[Dict], gt_mask: np.ndarray,
                              iou_threshold: float = 0.5) -> Dict: ...
# Pred-centric primitives (no IoU threshold) — reusable by diagnostics.
def primary_gt_match(pred_bin: np.ndarray, gt_instance_map: np.ndarray) -> dict: ...
def derive_purity_completeness(overlap_px: int, pred_area: int,
                               matched_gt_area: int | None) -> dict: ...
def compute_one_to_one_flags(pred_bins: list[np.ndarray],
                             gt_instance_map: np.ndarray) -> list[bool]: ...
```

### `src/evaluation/satellite_diagnostics.py`
```python
@dataclass(frozen=True)
class DiagnosticCfg:
    min_purity_for_match: float = 0.50
    completeness_complete: float = 0.50
    annulus_r_in_frac: float = 1.2
    annulus_r_out_frac: float = 2.0
    radial_n_rings: int = 6

class CandidateRow(TypedDict):
    """One row per raw satellite prediction. Keyed by (raw_index, candidate_id).
    confidence_score replaces SAM3's mask score at the diagnostic boundary.
    host_background_frac is reserved for Phase 2."""

def build_candidate_table(raw_sats, gt_sat_map, render_signal, H, W, cfg,
                          roi_bbox=None, host_support=None) -> SatelliteDiagnosticReport: ...
def classify(matched_gt_id, purity, completeness, cfg) -> tuple[str, str]: ...
def aggregate_diagnostics(per_sample_rows) -> dict: ...
```

### `src/evaluation/sam3_eval.py`
```python
def discover_pairs(pred_dir: Path, gt_dir: Path, ...) -> list[dict]: ...
def run_and_evaluate(pair: dict, ...) -> dict: ...
def aggregate_results(results: list[dict], ...) -> dict: ...
def save_results(results: dict, output_dir: Path, ...): ...
def save_eval_overlay(path: Path, image: np.ndarray, ...): ...
```

---

### `src/postprocess/satellite_prior_filter.py`
```python
def load_filter_cfg(stats_json: Path, feature_type: str = "satellites") -> Dict:
    """Load slim thresholds (area_min / solidity_min / aspect_sym_max) from stats (3-tier guards)."""

class SatellitePriorFilter:
    """Filter masks by area_min / solidity / aspect_sym rules. No area_max, no ambiguous zone."""
    def filter(self, masks: List[Dict]) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Returns (kept, rejected, []). Third slot always empty for backward-compat."""
```

### `src/postprocess/satellite_score_gate.py`
```python
class SatelliteScoreGate:
    """Static three-tier size-aware score gate (DR1 v4)."""
    def decide(self, area_clean_px: int, score: float) -> Tuple[str, str]:
        """(decision, reason). decision ∈ {'pass','drop'}."""
```

### `src/postprocess/satellite_core_policy.py`
```python
class SatelliteCorePolicy:
    """Hard-core reject + soft-core strict rescue (DR1 v4)."""
    def decide(self, dist_to_center_frac, area_clean_px, score,
               solidity, aspect_sym_moment) -> Tuple[str, str]:
        """(decision, reason). decision ∈ {'pass','rescue','drop'}."""
```

### `src/postprocess/satellite_conflict_resolver.py`
```python
class SatelliteConflictResolver:
    """Resolve stream-vs-satellite conflicts using real GT stream IDs."""
    def match_stream(self, sat_seg, streams_instance_map) -> Tuple[Optional[int], int, float, float]:
        """Pick max-overlap stream instance id; returns (matched_id, overlap_px, r_sat, r_stream)."""
    def decide(self, matched_stream_id, overlap_ratio_satellite,
               area_clean_px, solidity, aspect_sym_moment) -> Tuple[str, str, dict]:
        """(decision, reason, extras). decision ∈ {'pass','win','drop'}."""
```

### `src/postprocess/satellite_pipeline.py`
```python
STAGE_ORDER = ('raw_retrieval', 'metrics_completion', 'size_aware_score_gate',
               'satellite_prior_filter', 'core_exclusion_or_soft_core_rescue',
               'stream_conflict_resolution', 'final_gt_write', 'diagnostics_emit')

@dataclass
class StageEvent:
    stage: str; input_state: str; rule_name: str
    threshold_version: str; threshold_values: dict
    decision: str; reason: str; output_state: str
    metrics_snapshot_thin: dict  # whitelisted scalar keys only

@dataclass
class SatelliteCandidateState:
    candidate_id: str; candidate_rle_sha1: str; mask: dict
    status: str = 'alive'; matched_stream_id: Optional[int] = None
    history: List[StageEvent] = field(default_factory=list)

@dataclass
class SatellitePipelineResult:
    final_sats: List[dict]; candidates: List[dict]; image_summary: dict

class SatellitePipelineRunner:
    def run(self, raw_sats, streams_gt_map, H, W, base_key=None) -> SatellitePipelineResult: ...
```

### `src/postprocess/core_exclusion_filter.py`
```python
class CoreExclusionFilter:
    """Exclude masks with centroid inside R_exclude = radius_frac * min(H,W). (pnbody-only in v4)"""
    def filter(self, masks, H, W) -> Tuple[List[Dict], List[Dict], Dict]: ...
```

### `src/postprocess/streams_sanity_filter.py`
```python
def load_streams_cfg(stats_json: Path) -> dict[str, Any]:
    """Load streams filter thresholds from mask_stats_summary.json (3-tier guards)."""

class StreamsSanityFilter:
    """Lightweight false-positive guard for streams masks."""
    def __init__(self, min_area=50, max_area_frac=0.5,
                 edge_touch_frac=0.8, max_area_px=None): ...
    def filter(self, masks, H, W) -> Tuple[List[Dict], List[Dict]]: ...
```

### `src/postprocess/candidate_grouping.py`
```python
def group_by_centroid(masks: List[Dict], merge_radius: float) -> None:
    """Union-Find clustering by centroid proximity. Mutates in-place: adds group_id."""
```

### `src/postprocess/representative_selection.py`
```python
def load_area_target(stats_json: Path, key: str = "satellites_global") -> float:
    """Load median area from mask_stats_summary.json (3-tier guards)."""

def select_representatives(masks, cfg=None) -> Tuple[List[Dict], List[Dict]]:
    """Pick best mask per group_id by composite rep_score."""
```

---

### `src/analysis/mask_metrics.py`
```python
def compute_mask_metrics(seg: np.ndarray, H: int, W: int,
                         compute_hull: bool = True) -> Dict:
    """Vectorised per-mask geometry: area, bbox, aspect_sym, solidity, centroid, dist_to_center."""

def append_metrics_to_masks(masks: List[Dict], H: int, W: int,
                            compute_hull: bool = True):
    """In-place add geometry metrics to each mask dict."""
```

### `src/utils/geometry.py`
```python
def discrete_convex_area(coords: np.ndarray) -> float:
    """Convex hull area with pixel-corner expansion (fixes solidity > 1.0)."""
```

### `src/utils/coco_utils.py`
```python
def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]: ...
def get_bbox_from_mask(binary_mask: np.ndarray) -> List[float]: ...
def create_categories(thresholds: List[float]) -> Tuple[List[Dict], Dict]: ...
def process_mask_to_annotations(mask_data, image_id, category_id,
                                ann_id, min_area=10) -> Tuple[List[Dict], int]: ...
```

### `src/utils/logger.py`
```python
def setup_logger(name: str, log_dir: Optional[Path] = None,
                 level: int = logging.INFO) -> logging.Logger:
    """Create logger with console + rotating file handlers."""
```

### `src/visualization/overlay.py`
```python
def save_overlay(image, kept, core_rejected=None, prior_rejected=None,
                 duplicate_rejected=None, ambiguous=None, out_path="overlay.png",
                 draw_prior=False, draw_duplicate=False, draw_ambiguous=False): ...

def save_evaluation_overlay(path, image, streams_map, predictions): ...
    """QA overlay: GT white contours + prediction semi-transparent fills."""

def save_instance_overlay(path, image, instance_map): ...
    """Colored instance map overlay for merged GT QA."""
```

### `src/visualization/plotting.py`
```python
def save_visualization(image, pred_masks, gt_mask, output_path, dpi=150):
    """3-panel comparison: input | predictions | ground truth."""
```

---

## Configuration Reference

| Config File              | Process              | Output                               |
| :----------------------- | :------------------- | :----------------------------------- |
| `unified_data_prep.yaml` | 4-Phase Pipeline     | Canonical GT + SAM2 link + SAM3 COCO |
| `noise_profiles.yaml`    | Forward Noise Gen    | Noisy FITS per SNR profile           |
| `sam3_dataset_split.yaml`| Galaxy-Level Split   | Train/val COCO annotations           |
| `eval_sam2.yaml`         | SAM2 Eval            | Model metrics                        |
| `eval_sam3.yaml`         | SAM3 Eval            | Type-aware metrics JSON              |

---

## Quick Start

```bash
# Install dependencies inside the SAM3 runtime env
conda run --no-capture-output -n sam3 pip install -r requirements.txt

# Configure AST Versioned Git Hooks
git config core.hooksPath tools/githooks

# Current SAM3 render/eval/review flows assume the `sam3` conda env.

# 1. Unified Pipeline (Recommended)
conda run --no-capture-output -n sam3 python scripts/data/prepare_unified_dataset.py --config configs/unified_data_prep.yaml

# 2. Noise augmentation
conda run --no-capture-output -n sam3 python scripts/data/render_noisy_fits.py --config configs/unified_data_prep.yaml
conda run --no-capture-output -n sam3 python scripts/data/build_noise_augmented_annotations.py --config configs/unified_data_prep.yaml

# 3. Train/val split
conda run --no-capture-output -n sam3 python scripts/data/split_annotations.py --config configs/sam3_dataset_split.yaml

# 4. Evaluate
conda run --no-capture-output -n sam3 python scripts/eval/evaluate_sam2.py --config configs/eval_sam2.yaml
conda run --no-capture-output -n sam3 python scripts/eval/evaluate_sam3.py --config configs/eval_sam3.yaml
```
