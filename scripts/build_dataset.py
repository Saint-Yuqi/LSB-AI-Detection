#!/usr/bin/env python3
"""
Dataset Builder for SAM2/SAM3 Training Data

Generates training data in either:
- folder_based (SAM2): img_folder/{sample}/0000.png + gt_folder/{sample}/0000.png
- coco_instances (SAM3): images/*.png + masks/*.png + annotations.json

Usage:
    python scripts/build_dataset.py --config configs/data_prep_sam3.yaml
    python scripts/build_dataset.py --config configs/data_prep_sam2.yaml
"""

import argparse
import json
import random
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_fits_gz, SatelliteDataLoader
from src.data.preprocessing import LSBPreprocessor, LinearMagnitudePreprocessor, MultiExposurePreprocessor
from src.utils.logger import setup_logger
from src.utils.coco_utils import mask_to_rle, get_bbox_from_mask


# =============================================================================
# DATA KEY: The single source of truth for sample identity
# =============================================================================

@dataclass(frozen=True)
class SampleKey:
    """
    Immutable key for a unique sample.
    Used to ensure perfect alignment between images and masks.
    """
    galaxy_id: int
    orientation: str
    sb_threshold: float
    feature_type: str  # 'streams' or 'satellites'
    
    def __str__(self) -> str:
        return f"Gal{self.galaxy_id}_{self.orientation}_SB{self.sb_threshold}_{self.feature_type}"


# =============================================================================
# IMAGE KEY: For grouping samples that share the same base image
# =============================================================================

@dataclass(frozen=True)
class ImageKey:
    """
    Key for a unique base image (without SB threshold).
    Multiple SampleKeys may share the same ImageKey.
    """
    galaxy_id: int
    orientation: str
    feature_type: str
    
    @classmethod
    def from_sample_key(cls, sample_key: SampleKey) -> 'ImageKey':
        return cls(
            galaxy_id=sample_key.galaxy_id,
            orientation=sample_key.orientation,
            feature_type=sample_key.feature_type
        )


# =============================================================================
# PATH RESOLVER
# =============================================================================

class PathResolver:
    """Resolves paths for FITS files based on config."""
    
    def __init__(self, config: Dict[str, Any]):
        self.firebox_root = Path(config['paths']['firebox_root'])
        self.fbox_root = Path(config['paths']['fbox_root'])
        self.data_sources = config['data_sources']
    
    def get_image_path(self, key: ImageKey) -> Path:
        """Get FITS image path for an ImageKey."""
        if key.feature_type == 'streams':
            source = self.data_sources['streams']
            root = self.firebox_root
        else:  # satellites
            source = self.data_sources['satellites']
            root = self.fbox_root
        
        subdir = source['image_subdir']
        pattern = source['image_pattern']
        filename = pattern.format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation
        )
        return root / subdir / filename
    
    def get_stream_mask_path(self, key: SampleKey) -> Path:
        """Get FITS mask path for a streams sample."""
        source = self.data_sources['streams']
        
        # Select correct mask subdirectory based on orientation
        if key.orientation == 'eo':
            subdir = source['mask_subdir_eo']
        else:
            subdir = source['mask_subdir_fo']
        
        pattern = source['mask_pattern']
        filename = pattern.format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation,
            threshold=key.sb_threshold
        )
        return self.firebox_root / subdir / filename


# =============================================================================
# OUTPUT WRITER: Strategy pattern for SAM2 vs SAM3
# =============================================================================

class OutputWriter:
    """Base class for output writers."""
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        self.output_root = Path(config['paths']['output_root'])
        self.output_format = config['output_format']
    
    def setup(self) -> None:
        """Create output directories."""
        raise NotImplementedError
    
    def write_sample(
        self,
        key: SampleKey,
        image: np.ndarray,
        mask: np.ndarray,
        image_key: ImageKey
    ) -> None:
        """Write a single sample."""
        raise NotImplementedError
    
    def finalize(self) -> None:
        """Finalize output (e.g., write annotations.json)."""
        pass


class FolderBasedWriter(OutputWriter):
    """SAM2 folder-based output: img_folder/{sample}/0000.png"""
    
    def setup(self) -> None:
        img_folder = self.output_root / self.output_format['image_folder']
        gt_folder = self.output_root / self.output_format['gt_folder']
        img_folder.mkdir(parents=True, exist_ok=True)
        gt_folder.mkdir(parents=True, exist_ok=True)
        self.logger.info(f"SAM2 output: {self.output_root}")
    
    def write_sample(
        self,
        key: SampleKey,
        image: np.ndarray,
        mask: np.ndarray,
        image_key: ImageKey
    ) -> None:
        # Format folder name using template
        template = self.output_format['folder_template']
        folder_name = template.format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation,
            threshold=key.sb_threshold,
            type=key.feature_type
        )
        
        frame_filename = self.output_format['frame_filename']
        
        # Create sample directories
        img_dir = self.output_root / self.output_format['image_folder'] / folder_name
        gt_dir = self.output_root / self.output_format['gt_folder'] / folder_name
        img_dir.mkdir(parents=True, exist_ok=True)
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Write files
        img_path = img_dir / frame_filename
        mask_path = gt_dir / frame_filename
        
        cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(mask_path), mask)


class CocoInstancesWriter(OutputWriter):
    """SAM3 COCO format: images/*.png + annotations.json"""
    
    def __init__(self, config: Dict[str, Any], logger):
        super().__init__(config, logger)
        self.annotations: List[Dict] = []
        self.images: List[Dict] = []
        self.categories: List[Dict] = []
        self.category_map: Dict[str, int] = {}
        self.image_id_map: Dict[ImageKey, int] = {}
        self.next_image_id = 1
        self.next_ann_id = 1
        self.written_images: set = set()  # Track which images already written
        self.min_instance_area = config.get('annotation', {}).get('min_instance_area', 10)
    
    def setup(self) -> None:
        img_folder = self.output_root / self.output_format['image_folder']
        mask_folder = self.output_root / self.output_format['mask_folder']
        img_folder.mkdir(parents=True, exist_ok=True)
        mask_folder.mkdir(parents=True, exist_ok=True)
        
        # Build category list from data selection
        self._build_categories()
        self.logger.info(f"SAM3 output: {self.output_root}")
        self.logger.info(f"Categories: {len(self.categories)}")
    
    def _build_categories(self) -> None:
        """
        Build COCO categories with semantic names for SAM3 text prompts.
        
        SAM3 uses category 'name' as the text query for detection.
        We use clean semantic names like 'stellar stream' and 'satellite galaxy'
        for better text-based detection performance.
        
        Category structure:
        - id: unique integer ID
        - name: semantic name used as text prompt (e.g., 'stellar stream')
        - supercategory: parent category (e.g., 'low surface brightness')
        - sb_threshold: optional, for reference (NOT used as prompt)
        """
        data_sel = self.config['data_selection']
        feature_types = data_sel['feature_types']
        thresholds = data_sel['sb_thresholds']
        
        # Semantic name mapping for SAM3 text prompts
        # These are the actual text queries used during detection
        semantic_names = {
            'streams': 'stellar stream',
            'satellites': 'satellite galaxy',
            'stellar_stream': 'stellar stream',
            'satellite': 'satellite galaxy'
        }
        
        # Check if we should include SB threshold in category name
        # Default: separate categories per threshold for fine-grained detection
        annotation_cfg = self.config.get('annotation', {})
        include_threshold_in_name = annotation_cfg.get('include_threshold_in_name', False)
        
        cat_id = 1
        for ftype in feature_types:
            # Get semantic name for this feature type
            semantic_name = semantic_names.get(ftype, ftype.replace('_', ' '))
            
            for thresh in thresholds:
                # Category name used as SAM3 text prompt
                if include_threshold_in_name:
                    # Include SB threshold: "stellar stream at SB 27"
                    name = f"{semantic_name} at SB {thresh}"
                else:
                    # Clean semantic name only: "stellar stream"
                    name = semantic_name
                
                self.categories.append({
                    'id': cat_id,
                    'name': name,  # This is the SAM3 text prompt
                    'supercategory': 'low surface brightness',
                    'feature_type': ftype,  # Original feature type
                    'sb_threshold': thresh   # SB threshold for reference
                })
                self.category_map[(ftype, thresh)] = cat_id
                cat_id += 1
    
    def write_sample(
        self,
        key: SampleKey,
        image: np.ndarray,
        mask: np.ndarray,
        image_key: ImageKey
    ) -> None:
        # Get or create image_id
        if image_key not in self.image_id_map:
            image_id = self.next_image_id
            self.next_image_id += 1
            self.image_id_map[image_key] = image_id
            
            # Write image file (only once per ImageKey)
            template = self.output_format['image_template']
            img_filename = template.format(
                galaxy_id=image_key.galaxy_id,
                orientation=image_key.orientation,
                type=image_key.feature_type
            )
            img_path = self.output_root / self.output_format['image_folder'] / img_filename
            
            if image_key not in self.written_images:
                cv2.imwrite(str(img_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                self.written_images.add(image_key)
            
            # Add image entry
            h, w = image.shape[:2]
            self.images.append({
                'id': image_id,
                'file_name': img_filename,
                'width': w,
                'height': h
            })
        else:
            image_id = self.image_id_map[image_key]
        
        # Write mask file
        mask_template = self.output_format['mask_template']
        mask_filename = mask_template.format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation,
            threshold=key.sb_threshold,
            type=key.feature_type
        )
        mask_path = self.output_root / self.output_format['mask_folder'] / mask_filename
        cv2.imwrite(str(mask_path), mask)
        
        # Create annotations for each instance
        category_id = self.category_map[(key.feature_type, key.sb_threshold)]
        self._add_annotations(mask, image_id, category_id)
    
    def _add_annotations(self, mask: np.ndarray, image_id: int, category_id: int) -> None:
        """Extract instance annotations from mask."""
        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]  # Skip background
        
        for inst_id in instance_ids:
            binary_mask = (mask == inst_id).astype(np.uint8)
            area = int(binary_mask.sum())
            
            if area < self.min_instance_area:
                continue
            
            rle = mask_to_rle(binary_mask)
            bbox = get_bbox_from_mask(binary_mask)
            
            self.annotations.append({
                'id': self.next_ann_id,
                'image_id': image_id,
                'category_id': category_id,
                'segmentation': rle,
                'area': area,
                'bbox': bbox,
                'iscrowd': 0
            })
            self.next_ann_id += 1
    
    def finalize(self) -> None:
        """Write annotations.json."""
        annotations_file = self.output_format['annotations_file']
        output_path = self.output_root / annotations_file
        
        coco_format = {
            'info': {
                'description': 'LSB-AI-Detection Dataset',
                'date_created': datetime.now().isoformat(),
                'version': '1.0'
            },
            'images': self.images,
            'annotations': self.annotations,
            'categories': self.categories
        }
        
        with open(output_path, 'w') as f:
            json.dump(coco_format, f, indent=2)
        
        self.logger.info(f"Written: {output_path}")
        self.logger.info(f"  Images: {len(self.images)}")
        self.logger.info(f"  Annotations: {len(self.annotations)}")
        self.logger.info(f"  Categories: {len(self.categories)}")


# =============================================================================
# MAIN BUILDER
# =============================================================================

class DatasetBuilder:
    """
    Main dataset builder coordinating all components.
    
    Uses SatelliteDataLoader as the single source of truth for satellite masks.
    Uses LSBPreprocessor for all image processing.
    """
    
    def __init__(self, config: Dict[str, Any], logger):
        self.config = config
        self.logger = logger
        
        # Initialize preprocessor based on config method selection
        proc_cfg = config['processing']
        method = proc_cfg.get('preprocessing_method', 'asinh_stretch')
        
        if method == 'asinh_stretch':
            # ASTROPHYSICS PIPELINE: Magnitude → Flux → Asinh Stretch
            # Preserves maximum dynamic range
            from src.data.preprocessing import LSBPreprocessor
            self.preprocessor = LSBPreprocessor(
                zeropoint=proc_cfg['zeropoint'],
                nonlinearity=proc_cfg['nonlinearity'],
                clip_percentile=proc_cfg['clip_percentile'],
                target_size=tuple(proc_cfg['target_size'])
            )
            self.logger.info(f"Preprocessing: Asinh stretch (zp={proc_cfg['zeropoint']}, β={proc_cfg['nonlinearity']})")
        
        elif method == 'linear_magnitude':
            # GLOBAL LINEAR NORMALIZATION: Direct magnitude mapping
            # Preserves cross-galaxy photometric consistency
            from src.data.preprocessing import LinearMagnitudePreprocessor
            
            # Map interpolation string to cv2 constant
            interp_str = proc_cfg.get('interpolation', 'cubic').lower()
            interp_map = {
                'linear': cv2.INTER_LINEAR,
                'cubic': cv2.INTER_CUBIC,
                'nearest': cv2.INTER_NEAREST
            }
            interp_mode = interp_map.get(interp_str, cv2.INTER_CUBIC)
            
            self.preprocessor = LinearMagnitudePreprocessor(
                global_mag_min=proc_cfg.get('global_mag_min', 20.0),
                global_mag_max=proc_cfg.get('global_mag_max', 35.0),
                target_size=tuple(proc_cfg['target_size']),
                interpolation=interp_mode
            )
            self.logger.info(
                f"Preprocessing: Linear magnitude "
                f"[{proc_cfg.get('global_mag_min', 20.0)}, {proc_cfg.get('global_mag_max', 35.0)}] mag/arcsec²"
            )
        
        elif method == 'multi_exposure':
            # MULTI-EXPOSURE 3-CHANNEL: Different stretch per channel
            # R=linear, G=asinh, B=configurable mode
            b_mode = proc_cfg.get('b_mode', 'gamma')
            zscale_contrast = proc_cfg.get('zscale_contrast', 0.25)
            r_gain = proc_cfg.get('r_gain', 1.0)
            b_gain = proc_cfg.get('b_gain', 1.0)
            self.preprocessor = MultiExposurePreprocessor(
                global_mag_min=proc_cfg.get('global_mag_min', 20.0),
                global_mag_max=proc_cfg.get('global_mag_max', 35.0),
                zeropoint=proc_cfg.get('zeropoint', 22.5),
                nonlinearity=proc_cfg.get('nonlinearity', 300.0),
                clip_percentile=proc_cfg.get('clip_percentile', 99.5),
                gamma=proc_cfg.get('gamma', 0.5),
                b_mode=b_mode,
                zscale_contrast=zscale_contrast,
                r_gain=r_gain,
                b_gain=b_gain,
                target_size=tuple(proc_cfg['target_size']),
            )
            self.logger.info(
                f"Preprocessing: Multi-exposure RGB "
                f"(b_mode={b_mode}, γ={proc_cfg.get('gamma', 0.5)}, zsc={zscale_contrast}, r×{r_gain}, b×{b_gain})"
            )
        
        else:
            raise ValueError(
                f"Unknown preprocessing_method: {method}. "
                f"Expected 'asinh_stretch', 'linear_magnitude', or 'multi_exposure'"
            )
        
        self.path_resolver = PathResolver(config)
        
        # Initialize satellite data loader (lazy)
        pkl_path = Path(config['paths']['satellite_pickle'])
        self.satellite_loader = SatelliteDataLoader(pkl_path, lazy_load=True)
        
        # Initialize writer based on format type
        format_type = config['output_format']['format_type']
        if format_type == 'folder_based':
            self.writer = FolderBasedWriter(config, logger)
        elif format_type == 'coco_instances':
            self.writer = CocoInstancesWriter(config, logger)
        else:
            raise ValueError(f"Unknown format_type: {format_type}")
        
        # Data selection from config
        self.galaxy_ids = config['data_selection']['galaxy_ids']
        self.orientations = config['data_selection']['orientations']
        self.sb_thresholds = config['data_selection']['sb_thresholds']
        self.feature_types = config['data_selection']['feature_types']
        
        # Statistics
        self.stats = {
            'processed': 0,
            'skipped_no_image': 0,
            'skipped_no_mask': 0,
            'skipped_empty_mask': 0
        }
    
    def build(self) -> None:
        """Run the full build process."""
        self.logger.info("=" * 60)
        self.logger.info("Dataset Build Started")
        self.logger.info(f"Format: {self.config['output_format']['format_type']}")
        self.logger.info(f"Galaxies: {len(self.galaxy_ids)}")
        self.logger.info(f"Thresholds: {len(self.sb_thresholds)}")
        self.logger.info("=" * 60)
        
        self.writer.setup()
        
        # Generate all sample keys
        sample_keys = self._generate_sample_keys()
        self.logger.info(f"Total samples to process: {len(sample_keys)}")
        
        # Group by ImageKey for efficient processing
        image_groups = self._group_by_image_key(sample_keys)
        self.logger.info(f"Unique images: {len(image_groups)}")
        
        # Process each image group
        for image_key, keys in image_groups.items():
            self._process_image_group(image_key, keys)
        
        self.writer.finalize()
        
        # Report statistics
        self.logger.info("=" * 60)
        self.logger.info("Build Complete")
        self.logger.info(f"  Processed: {self.stats['processed']}")
        self.logger.info(f"  Skipped (no image): {self.stats['skipped_no_image']}")
        self.logger.info(f"  Skipped (no mask): {self.stats['skipped_no_mask']}")
        self.logger.info(f"  Skipped (empty mask): {self.stats['skipped_empty_mask']}")
        self.logger.info("=" * 60)
    
    def _generate_sample_keys(self) -> List[SampleKey]:
        """Generate all sample keys from config."""
        keys = []
        for gal_id in self.galaxy_ids:
            for orient in self.orientations:
                for thresh in self.sb_thresholds:
                    for ftype in self.feature_types:
                        keys.append(SampleKey(
                            galaxy_id=gal_id,
                            orientation=orient,
                            sb_threshold=thresh,
                            feature_type=ftype
                        ))
        return keys
    
    def _group_by_image_key(self, sample_keys: List[SampleKey]) -> Dict[ImageKey, List[SampleKey]]:
        """Group sample keys by their base image."""
        groups: Dict[ImageKey, List[SampleKey]] = {}
        for key in sample_keys:
            img_key = ImageKey.from_sample_key(key)
            if img_key not in groups:
                groups[img_key] = []
            groups[img_key].append(key)
        return groups
    
    def _process_image_group(self, image_key: ImageKey, sample_keys: List[SampleKey]) -> None:
        """Process all samples sharing the same base image."""
        # Load and preprocess image once
        image_path = self.path_resolver.get_image_path(image_key)
        
        if not image_path.resolve().exists():
            self.logger.warning(f"Image not found: {image_path}")
            self.stats['skipped_no_image'] += len(sample_keys)
            return
        
        try:
            sb_map = load_fits_gz(image_path)
            image = self.preprocessor.process(sb_map)
        except Exception as e:
            self.logger.error(f"Failed to load image {image_path}: {e}")
            self.stats['skipped_no_image'] += len(sample_keys)
            return
        
        # Process each sample key
        for key in sample_keys:
            mask = self._get_mask(key, sb_map.shape)
            
            if mask is None:
                self.stats['skipped_no_mask'] += 1
                continue
            
            if mask.max() == 0:
                self.stats['skipped_empty_mask'] += 1
                continue
            
            # Resize mask to match image
            mask = self.preprocessor.resize_mask(mask)
            
            # Write sample
            self.writer.write_sample(key, image, mask, image_key)
            self.stats['processed'] += 1
    
    def _get_mask(self, key: SampleKey, original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get mask for a sample key.
        
        Uses SatelliteDataLoader for satellites (single source of truth).
        Uses load_fits_gz for streams.
        """
        if key.feature_type == 'streams':
            return self._get_stream_mask(key)
        else:  # satellites
            return self._get_satellite_mask(key, original_shape)
    
    def _get_stream_mask(self, key: SampleKey) -> Optional[np.ndarray]:
        """Load stream mask from FITS file."""
        mask_path = self.path_resolver.get_stream_mask_path(key)
        
        if not mask_path.resolve().exists():
            self.logger.debug(f"Stream mask not found: {mask_path}")
            return None
        
        try:
            # Preserve instance labels from the FITS mask
            # Original masks contain unique integer labels for each stream instance
            mask_data = load_fits_gz(mask_path)
            mask = np.round(mask_data).astype(np.uint8)
            return mask
        except Exception as e:
            self.logger.warning(f"Failed to load stream mask {mask_path}: {e}")
            return None
    
    def _get_satellite_mask(self, key: SampleKey, original_shape: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Get satellite mask from SatelliteDataLoader.
        
        This is the SINGLE SOURCE OF TRUTH for satellite data.
        """
        try:
            satellites = self.satellite_loader.get_satellites(
                galaxy_id=key.galaxy_id,
                orientation=key.orientation,
                sb_threshold=key.sb_threshold,
                load_seg_map=False  # Use seg_ids for efficiency
            )
            
            if satellites.count == 0:
                return None
            
            # Generate combined instance mask
            mask = satellites.get_combined_mask(shape=original_shape)
            return mask
            
        except Exception as e:
            self.logger.warning(f"Failed to get satellite mask for {key}: {e}")
            return None


# =============================================================================
# CLI
# =============================================================================

def set_random_seeds(seed: Optional[int]) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        try:
            import torch
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        except ImportError:
            pass


# =============================================================================
# PREVIEW MODE
# =============================================================================

def run_preview_mode(
    config: Dict[str, Any],
    galaxy_id: int,
    orientation: str,
    feature_type: str,
    sb_threshold: Optional[float],
    logger
) -> None:
    """
    Generate 2x3 preview panel comparing rendering methods.
    
    Panel layout:
        Row 1: R channel (gray), G channel (gray), B channel (gray)
        Row 2: RGB composite, old method (3× dup), old asinh RGB
    
    Outputs:
        - preview_{sample_id}.png: 2x3 panel image
        - preview_{sample_id}.json: Rendering metadata
    """
    proc_cfg = config['processing']
    target_size = tuple(proc_cfg['target_size'])
    
    # Resolve paths
    path_resolver = PathResolver(config)
    image_key = ImageKey(
        galaxy_id=galaxy_id,
        orientation=orientation,
        feature_type=feature_type
    )
    fits_path = path_resolver.get_image_path(image_key)
    
    if not fits_path.exists():
        logger.error(f"FITS file not found: {fits_path}")
        return
    
    # Load raw magnitude map
    sb_map = load_fits_gz(fits_path)
    logger.info(f"Loaded: {fits_path} | shape={sb_map.shape}")
    
    # Build sample ID for filename
    sb_str = f"_SB{sb_threshold}" if sb_threshold else ""
    sample_id = f"Gal{galaxy_id:05d}_{orientation}_{feature_type}{sb_str}"
    
    # 1. Multi-exposure preprocessor
    b_mode = proc_cfg.get('b_mode', 'gamma')
    zscale_contrast = proc_cfg.get('zscale_contrast', 0.25)
    r_gain = proc_cfg.get('r_gain', 1.0)
    b_gain = proc_cfg.get('b_gain', 1.0)
    multi_proc = MultiExposurePreprocessor(
        global_mag_min=proc_cfg.get('global_mag_min', 20.0),
        global_mag_max=proc_cfg.get('global_mag_max', 35.0),
        zeropoint=proc_cfg.get('zeropoint', 22.5),
        nonlinearity=proc_cfg.get('nonlinearity', 300.0),
        clip_percentile=proc_cfg.get('clip_percentile', 99.5),
        gamma=proc_cfg.get('gamma', 0.5),
        b_mode=b_mode,
        zscale_contrast=zscale_contrast,
        r_gain=r_gain,
        b_gain=b_gain,
        target_size=target_size,
    )
    rgb_multi = multi_proc.process(sb_map)
    
    # Extract individual channels (before resize, use last_stats)
    r_gray = rgb_multi[:, :, 0]
    g_gray = rgb_multi[:, :, 1]
    b_gray = rgb_multi[:, :, 2]
    
    # 2. Old method: Linear magnitude (3× duplicate)
    linear_proc = LinearMagnitudePreprocessor(
        global_mag_min=proc_cfg.get('global_mag_min', 20.0),
        global_mag_max=proc_cfg.get('global_mag_max', 35.0),
        target_size=target_size,
    )
    rgb_linear = linear_proc.process(sb_map)
    
    # 3. Old method: Asinh (3× duplicate)
    asinh_proc = LSBPreprocessor(
        zeropoint=proc_cfg.get('zeropoint', 22.5),
        nonlinearity=proc_cfg.get('nonlinearity', 300.0),
        clip_percentile=proc_cfg.get('clip_percentile', 99.5),
        target_size=target_size,
    )
    rgb_asinh = asinh_proc.process(sb_map)
    
    # Print channel stats to console
    stats = multi_proc.last_stats
    print("\n" + "=" * 60)
    print(f"Preview Stats: {sample_id}")
    print("=" * 60)
    print(f"  vmax_ref_flux      : {stats.get('vmax_ref_flux', 0):.6f}")
    print(f"  raw_finite_ratio   : {stats.get('raw_finite_pixel_ratio', 0):.4f}")
    print(f"  R channel          : min={stats.get('r_min')}, max={stats.get('r_max')}, mean={stats.get('r_mean', 0):.2f}")
    print(f"  G channel          : min={stats.get('g_min')}, max={stats.get('g_max')}, mean={stats.get('g_mean', 0):.2f}")
    print(f"  B channel ({b_mode:12s}): min={stats.get('b_min')}, max={stats.get('b_max')}, mean={stats.get('b_mean', 0):.2f}")
    print("=" * 60 + "\n")
    
    # Build 2x3 panel
    # Row 1: R, G, B (as grayscale → RGB for display)
    # Row 2: RGB composite, linear (old), asinh (old)
    def gray_to_rgb(gray: np.ndarray) -> np.ndarray:
        return np.stack([gray, gray, gray], axis=-1)
    
    row1 = np.concatenate([
        gray_to_rgb(r_gray),
        gray_to_rgb(g_gray),
        gray_to_rgb(b_gray)
    ], axis=1)
    
    row2 = np.concatenate([
        rgb_multi,
        rgb_linear,
        rgb_asinh
    ], axis=1)
    
    panel = np.concatenate([row1, row2], axis=0)
    
    # Add labels (using cv2.putText) - B label reflects actual b_mode from config
    b_label = f"B: {b_mode.capitalize()}" if b_mode != "zscale_asinh" else "B: ZScale+Asinh"
    labels = [
        ("R: Linear Mag", (10, 30)),
        ("G: Asinh", (target_size[0] + 10, 30)),
        (b_label, (2 * target_size[0] + 10, 30)),
        ("RGB Composite", (10, target_size[1] + 30)),
        ("Old: Linear (3x)", (target_size[0] + 10, target_size[1] + 30)),
        ("Old: Asinh (3x)", (2 * target_size[0] + 10, target_size[1] + 30)),
    ]
    for text, pos in labels:
        cv2.putText(
            panel, text, pos,
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
        )
    
    # Create output directory
    output_root = Path(config['paths']['output_root'])
    preview_dir = output_root / 'preview'
    preview_dir.mkdir(parents=True, exist_ok=True)
    
    # Save panel PNG
    panel_path = preview_dir / f"preview_{sample_id}.png"
    cv2.imwrite(str(panel_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
    logger.info(f"Saved panel: {panel_path}")
    
    # Save metadata JSON
    metadata = {
        'galaxy_id': galaxy_id,
        'orientation': orientation,
        'feature_type': feature_type,
        'sb_threshold': sb_threshold,
        'fits_path': str(fits_path),
        **multi_proc.get_params_dict(),
    }
    json_path = preview_dir / f"preview_{sample_id}.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {json_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Build SAM2/SAM3 training dataset from FITS files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/build_dataset.py --config configs/data_prep_sam3.yaml
  python scripts/build_dataset.py --config configs/data_prep_sam2.yaml
  
  # Preview mode: generate 2x3 comparison panel
  python scripts/build_dataset.py --config configs/data_prep_sam2.yaml \\
      --preview --galaxy-id 11 --orientation eo --feature-type streams
        """
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        required=True,
        help='Path to YAML configuration file'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without writing files'
    )
    parser.add_argument(
        '--preview',
        action='store_true',
        help='Generate 2x3 preview panel comparing rendering methods'
    )
    parser.add_argument(
        '--galaxy-id',
        type=int,
        help='Galaxy ID for preview mode'
    )
    parser.add_argument(
        '--orientation',
        type=str,
        choices=['eo', 'fo'],
        help='Orientation for preview mode (eo or fo)'
    )
    parser.add_argument(
        '--feature-type',
        type=str,
        choices=['streams', 'satellites'],
        default='streams',
        help='Feature type for preview mode (default: streams)'
    )
    parser.add_argument(
        '--sb-threshold',
        type=float,
        default=None,
        help='SB threshold for preview metadata (optional)'
    )
    
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set random seed
    seed = config.get('reproducibility', {}).get('random_seed')
    set_random_seeds(seed)
    
    # Setup logger
    logger = setup_logger(
        name='build_dataset',
        log_dir=PROJECT_ROOT / 'logs'
    )
    logger.info(f"Config: {config_path}")
    
    # Preview mode
    if args.preview:
        if args.galaxy_id is None or args.orientation is None:
            print("Error: --preview requires --galaxy-id and --orientation")
            sys.exit(1)
        run_preview_mode(
            config=config,
            galaxy_id=args.galaxy_id,
            orientation=args.orientation,
            feature_type=args.feature_type,
            sb_threshold=args.sb_threshold,
            logger=logger
        )
        return
    
    # Build dataset
    builder = DatasetBuilder(config, logger)
    
    if args.dry_run:
        logger.info("DRY RUN - no files will be written")
        sample_keys = builder._generate_sample_keys()
        logger.info(f"Would process {len(sample_keys)} samples")
    else:
        builder.build()


if __name__ == '__main__':
    main()

