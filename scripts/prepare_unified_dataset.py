#!/usr/bin/env python3
"""
Unified Dataset Preparation Pipeline

4-Phase Architecture:
    render     → renders/current/{preprocessing}/{BaseKey}/0000.png
    gt         → gt_canonical/.../streams_instance_map.npy
    satellites → satellites_cache.npz + instance_map_uint8.png + instances.json
    export     → SAM2 symlinks + SAM3 annotations.json

Usage:
    python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml
    python scripts/prepare_unified_dataset.py --config ... --phase render
    python scripts/prepare_unified_dataset.py --config ... --galaxies 11,13

Env:
    CUDA, PyTorch with bf16 support for satellites phase.
"""
from __future__ import annotations

import argparse
import json
import hashlib
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

# Project imports
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_fits_gz
from src.data.preprocessing import (
    LSBPreprocessor,
    LinearMagnitudePreprocessor,
    MultiExposurePreprocessor,
)
from src.utils.coco_utils import mask_to_rle


# =============================================================================
# DATA KEYS
# =============================================================================

@dataclass(frozen=True)
class BaseKey:
    """Immutable key for a unique galaxy+orientation."""
    galaxy_id: int
    orientation: str
    
    def __str__(self) -> str:
        return f"{self.galaxy_id:05d}_{self.orientation}"


@dataclass(frozen=True)
class VariantKey:
    """Extends BaseKey with preprocessing variant."""
    base_key: BaseKey
    preprocessing: str
    
    def __str__(self) -> str:
        return f"{self.base_key}_{self.preprocessing}"


# =============================================================================
# PATH RESOLVER
# =============================================================================

class PathResolver:
    """Resolves input/output paths based on config."""
    
    def __init__(self, config: dict[str, Any]):
        self.firebox_root = Path(config["paths"]["firebox_root"])
        self.output_root = Path(config["paths"]["output_root"])
        self.data_sources = config["data_sources"]
        self.target_size = tuple(config["processing"]["target_size"])
    
    # Input paths
    def get_fits_path(self, key: BaseKey) -> Path:
        src = self.data_sources["streams"]
        pattern = src["image_pattern"].format(
            galaxy_id=key.galaxy_id, orientation=key.orientation
        )
        return self.firebox_root / src["image_subdir"] / pattern
    
    def get_mask_path(self, key: BaseKey, sb_threshold: float) -> Path:
        src = self.data_sources["streams"]
        subdir = src["mask_subdir_eo"] if key.orientation == "eo" else src["mask_subdir_fo"]
        pattern = src["mask_pattern"].format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation,
            threshold=int(sb_threshold) if sb_threshold == int(sb_threshold) else sb_threshold,
        )
        return self.firebox_root / subdir / pattern
    
    # Output paths
    def get_render_dir(self, preprocessing: str, key: BaseKey) -> Path:
        return self.output_root / "renders" / "current" / preprocessing / str(key)
    
    def get_gt_dir(self, key: BaseKey) -> Path:
        return self.output_root / "gt_canonical" / "current" / str(key)
    
    def get_sam2_dir(self) -> Path:
        return self.output_root / "sam2_prepared"
    
    def get_sam3_dir(self) -> Path:
        return self.output_root / "sam3_prepared"


# =============================================================================
# PREPROCESSOR FACTORY
# =============================================================================

def create_preprocessor(name: str, params: dict, target_size: tuple[int, int]):
    """Factory for preprocessors matching src/data/preprocessing.py."""
    if name == "asinh_stretch":
        return LSBPreprocessor(
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 50.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            target_size=target_size,
        )
    elif name == "linear_magnitude":
        return LinearMagnitudePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            target_size=target_size,
        )
    elif name == "multi_exposure":
        return MultiExposurePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 300.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            gamma=params.get("gamma", 0.5),
            b_mode=params.get("b_mode", "gamma"),
            target_size=target_size,
        )
    else:
        raise ValueError(f"Unknown preprocessor: {name}")


# =============================================================================
# PHASE 1: RENDER
# =============================================================================

def run_render_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Render FITS → RGB for each preprocessing variant."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RENDER")
    logger.info("=" * 60)
    
    resolver = PathResolver(config)
    variants = config["preprocessing_variants"]
    target_size = tuple(config["processing"]["target_size"])
    
    # Build preprocessors
    preprocessors = {}
    for v in variants:
        preprocessors[v["name"]] = create_preprocessor(
            v["name"], v.get("params", {}), target_size
        )
    
    stats = {"rendered": 0, "skipped_exists": 0, "skipped_no_fits": 0}
    
    for key in base_keys:
        fits_path = resolver.get_fits_path(key)
        
        if not fits_path.exists():
            logger.warning(f"FITS not found: {fits_path}")
            stats["skipped_no_fits"] += 1
            continue
        
        # Load FITS once
        sb_map = load_fits_gz(fits_path)
        
        for name, proc in preprocessors.items():
            out_dir = resolver.get_render_dir(name, key)
            out_path = out_dir / "0000.png"
            
            # Force rebuild if variant is in force list
            if out_path.exists() and force_variants and name in force_variants:
                out_path.unlink()
                logger.info(f"Force rebuild: {name}/{key}")
            elif out_path.exists():
                stats["skipped_exists"] += 1
                continue
            
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Process and save
            rgb = proc.process(sb_map)
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            stats["rendered"] += 1
    
    logger.info(f"Rendered: {stats['rendered']}, Skipped (exists): {stats['skipped_exists']}, Skipped (no FITS): {stats['skipped_no_fits']}")


# =============================================================================
# PHASE 2: GT (Streams)
# =============================================================================

def run_gt_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Load SB32 streams mask → streams_instance_map.npy."""
    logger.info("=" * 60)
    logger.info("PHASE 2: GT (Streams)")
    logger.info("=" * 60)
    
    resolver = PathResolver(config)
    sb_threshold = config["data_selection"]["canonical_sb_threshold"]
    target_size = tuple(config["processing"]["target_size"])
    
    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_mask": 0}
    
    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        npy_path = gt_dir / "streams_instance_map.npy"
        
        if npy_path.exists():
            stats["skipped_exists"] += 1
            continue
        
        mask_path = resolver.get_mask_path(key, sb_threshold)
        
        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            stats["skipped_no_mask"] += 1
            continue
        
        gt_dir.mkdir(parents=True, exist_ok=True)
        
        # Load mask (preserve original IDs)
        mask_data = load_fits_gz(mask_path)
        instance_map = np.round(mask_data).astype(np.int32)
        
        # Resize with NEAREST to preserve IDs
        instance_map_resized = cv2.resize(
            instance_map, target_size, interpolation=cv2.INTER_NEAREST
        )
        
        # Save
        np.save(npy_path, instance_map_resized)
        
        # Write partial manifest
        manifest = {
            "sb_threshold_used": sb_threshold,
            "source_mask": str(mask_path),
            "source_mask_sha1": _sha1_file(mask_path),
            "max_stream_id": int(instance_map_resized.max()),
            "n_stream_instances": int(len(np.unique(instance_map_resized)) - 1),
            "target_size": list(target_size),
            "created_at": datetime.now().isoformat(),
        }
        (gt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        
        stats["processed"] += 1
    
    logger.info(f"Processed: {stats['processed']}, Skipped (exists): {stats['skipped_exists']}, Skipped (no mask): {stats['skipped_no_mask']}")


def _sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA1 of file (first 1MB for large files)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(chunk_size))
    except Exception:
        return ""
    return h.hexdigest()


# =============================================================================
# PHASE 3: SATELLITES (AutoMask + Merge)
# =============================================================================

def run_satellites_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Run AutoMask → filter → cache → merge."""
    logger.info("=" * 60)
    logger.info("PHASE 3: SATELLITES (AutoMask + Merge)")
    logger.info("=" * 60)
    
    sat_cfg = config.get("satellites", {})
    if not sat_cfg.get("enabled", True):
        logger.info("Satellites disabled in config, skipping.")
        return
    
    resolver = PathResolver(config)
    input_variant = sat_cfg.get("input_image_variant", "linear_magnitude")
    target_size = tuple(config["processing"]["target_size"])
    
    # Import inference modules
    from src.inference.sam2_automask_runner import AutoMaskRunner
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    from src.postprocess.core_exclusion_filter import CoreExclusionFilter
    from src.postprocess.candidate_grouping import group_by_centroid
    from src.postprocess.representative_selection import select_representatives
    from src.analysis.mask_metrics import append_metrics_to_masks
    
    # Initialize runner
    runner = AutoMaskRunner()
    
    # Type-convert generator config (YAML may load as strings)
    raw_gen_cfg = sat_cfg.get("generator", {})
    gen_cfg = {}
    int_keys = {"points_per_side", "points_per_batch", "min_mask_region_area", "crop_n_layers", "crop_n_points_downscale_factor"}
    float_keys = {"pred_iou_thresh", "stability_score_thresh", "stability_score_offset", "box_nms_thresh", "crop_nms_thresh"}
    for k, v in raw_gen_cfg.items():
        if k in int_keys:
            gen_cfg[k] = int(v)
        elif k in float_keys:
            gen_cfg[k] = float(v)
        else:
            gen_cfg[k] = v
    grouping_cfg = sat_cfg.get("grouping", {})
    prior_cfg_entry = sat_cfg.get("prior", {})
    core_cfg = sat_cfg.get("core_exclusion", {})
    sort_policy = sat_cfg.get("satellite_sort_policy", ["area_desc"])
    overlap_policy = sat_cfg.get("overlap_policy", "keep_streams")
    
    # Load prior filter config
    stats_json = Path(prior_cfg_entry.get("stats_json", "outputs/mask_stats/mask_stats_summary.json"))
    prior_cfg = load_filter_cfg(stats_json)
    prior_cfg["ambiguous_factor"] = prior_cfg_entry.get("ambiguous_factor", 0.25)
    prior_cfg["core_radius_frac"] = core_cfg.get("radius_frac", 0.08)
    
    prior_flt = SatellitePriorFilter(prior_cfg)
    core_flt = CoreExclusionFilter(radius_frac=core_cfg.get("radius_frac", 0.08))
    
    # Warm up
    first_key = base_keys[0] if base_keys else None
    if first_key:
        warmup_path = resolver.get_render_dir(input_variant, first_key) / "0000.png"
        if warmup_path.exists():
            warmup_img = np.array(Image.open(warmup_path).convert("RGB"))
            runner.warmup(warmup_img, n=2)
    
    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_render": 0, "skipped_no_streams": 0}
    
    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        cache_path = gt_dir / "satellites_cache.npz"
        final_map_path = gt_dir / "instance_map_uint8.png"
        
        # Check if already done (respect force flag)
        if final_map_path.exists() and cache_path.exists():
            if force_variants:
                # Force rebuild: remove stale outputs
                final_map_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                (gt_dir / "instances.json").unlink(missing_ok=True)
                (gt_dir / "id_map.json").unlink(missing_ok=True)
                (gt_dir / "overlay.png").unlink(missing_ok=True)
                logger.info(f"Force rebuild satellites: {key}")
            else:
                stats["skipped_exists"] += 1
                continue
        
        # Load render
        render_path = resolver.get_render_dir(input_variant, key) / "0000.png"
        if not render_path.exists():
            logger.warning(f"Render not found: {render_path}")
            stats["skipped_no_render"] += 1
            continue
        
        # Load streams instance map
        streams_npy = gt_dir / "streams_instance_map.npy"
        if not streams_npy.exists():
            logger.warning(f"Streams map not found: {streams_npy}")
            stats["skipped_no_streams"] += 1
            continue
        
        streams_map = np.load(streams_npy)
        max_stream_id = int(streams_map.max())
        
        # Load image
        image = np.array(Image.open(render_path).convert("RGB"))
        H, W = image.shape[:2]
        
        # Run AutoMask
        masks, time_ms = runner.run(image, gen_cfg)
        
        # Pipeline: metrics → grouping → selection → prior → core
        if masks:
            append_metrics_to_masks(masks, H, W, compute_hull=False)
            group_by_centroid(masks, dist_px=grouping_cfg.get("centroid_dist_px", 10.0))
            reps, dups = select_representatives(masks, {})
            append_metrics_to_masks(reps, H, W, compute_hull=True)
            kept_prior, rej_prior, ambig = prior_flt.filter(reps)
            kept_final, core_hits, _ = core_flt.filter(kept_prior, H, W)
        else:
            kept_final, ambig, dups, rej_prior, core_hits = [], [], [], [], []
        
        # Sort kept_final deterministically
        kept_final = _sort_masks(kept_final, sort_policy)
        
        # Cache all buckets (RLE encoded)
        _save_satellites_cache(
            cache_path,
            kept=kept_final,
            ambiguous=ambig,
            dup_rejected=dups,
            prior_rejected=rej_prior,
            core_rejected=core_hits,
            input_image_sha1=_sha1_file(render_path),
            H=H, W=W,
        )
        
        # Merge: streams + satellites
        instance_map, instances_list, id_map, overlap_stats = _merge_instances(
            streams_map, kept_final, max_stream_id, overlap_policy
        )
        
        # Assert uint8 range
        max_final_id = int(instance_map.max())
        assert 0 <= max_final_id <= 255, f"max_id={max_final_id} exceeds uint8"
        assert instance_map.dtype == np.uint8, f"dtype={instance_map.dtype}"
        
        # Save outputs
        cv2.imwrite(str(final_map_path), instance_map)
        (gt_dir / "instances.json").write_text(json.dumps(instances_list, indent=2))
        (gt_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))
        
        # Update manifest
        manifest_path = gt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update({
            "automask_config_name": sat_cfg.get("automask_config_name", "unknown"),
            "input_image_variant": input_variant,
            "satellite_sort_policy": sort_policy,
            "overlap_policy": overlap_policy,
            "n_raw_masks": len(masks) if masks else 0,
            "n_satellites_kept": len(kept_final),
            "n_ambiguous": len(ambig),
            "max_stream_id": max_stream_id,
            "max_final_id": max_final_id,
            "overlap_px": overlap_stats.get("overlap_px", 0),
            "overlap_rate": overlap_stats.get("overlap_rate", 0.0),
            "inference_time_ms": round(time_ms, 2),
            "updated_at": datetime.now().isoformat(),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))
        
        # Generate overlay
        _save_overlay(gt_dir / "overlay.png", image, instance_map)
        
        stats["processed"] += 1
        
        if (stats["processed"]) % 10 == 0:
            logger.info(f"Progress: {stats['processed']}/{len(base_keys)}")
    
    logger.info(f"Processed: {stats['processed']}, Skipped (exists): {stats['skipped_exists']}")


def _sort_masks(masks: list[dict], policy: list[str]) -> list[dict]:
    """Sort masks deterministically by policy."""
    def sort_key(m):
        keys = []
        for p in policy:
            if p == "area_desc":
                keys.append(-m.get("area", 0))
            elif p == "area_asc":
                keys.append(m.get("area", 0))
            elif p == "centroid_x_asc":
                keys.append(m.get("centroid_x", 0))
            elif p == "centroid_y_asc":
                keys.append(m.get("centroid_y", 0))
        return tuple(keys)
    return sorted(masks, key=sort_key)


def _save_satellites_cache(
    path: Path,
    kept: list[dict],
    ambiguous: list[dict],
    dup_rejected: list[dict],
    prior_rejected: list[dict],
    core_rejected: list[dict],
    input_image_sha1: str,
    H: int, W: int,
) -> None:
    """Save cache with RLE-encoded masks and metadata."""
    def encode_bucket(masks):
        encoded = []
        for m in masks:
            seg = m.get("segmentation", np.zeros((H, W), dtype=bool))
            rle = _mask_to_rle(seg)
            encoded.append({
                "rle": rle,
                "area": m.get("area", 0),
                "centroid_x": m.get("centroid_x", 0),
                "centroid_y": m.get("centroid_y", 0),
                "stability_score": m.get("stability_score", 0),
                "predicted_iou": m.get("predicted_iou", 0),
            })
        return encoded
    
    cache = {
        "kept": encode_bucket(kept),
        "ambiguous": encode_bucket(ambiguous),
        "dup_rejected": encode_bucket(dup_rejected),
        "prior_rejected": encode_bucket(prior_rejected),
        "core_rejected": encode_bucket(core_rejected),
        "metadata": {
            "input_image_sha1": input_image_sha1,
            "H": H, "W": W,
            "created_at": datetime.now().isoformat(),
        }
    }
    np.savez_compressed(path, cache=cache)


def _mask_to_rle(binary_mask: np.ndarray) -> dict:
    """Convert binary mask to COCO-compatible RLE format."""
    return mask_to_rle(binary_mask.astype(np.uint8))


def _merge_instances(
    streams_map: np.ndarray,
    satellites: list[dict],
    max_stream_id: int,
    overlap_policy: str,
) -> tuple[np.ndarray, list[dict], dict, dict]:
    """Merge streams + satellites into final instance map."""
    instance_map = streams_map.copy().astype(np.int32)
    
    # Build instances list from streams
    stream_ids = sorted([int(x) for x in np.unique(streams_map) if x > 0])
    instances_list = [{"id": sid, "type": "streams"} for sid in stream_ids]
    
    # ID mapping
    id_map = {
        "streams": {str(sid): sid for sid in stream_ids},
        "satellites": {},
    }
    
    overlap_px = 0
    total_sat_px = 0
    
    for i, sat in enumerate(satellites):
        new_id = max_stream_id + i + 1
        seg = sat.get("segmentation", np.zeros_like(streams_map, dtype=bool))
        
        if overlap_policy == "keep_streams":
            # Only fill where streams==0
            mask = seg & (instance_map == 0)
        else:
            mask = seg
        
        overlap_px += int((seg & (streams_map > 0)).sum())
        total_sat_px += int(seg.sum())
        
        instance_map[mask] = new_id
        instances_list.append({"id": new_id, "type": "satellites"})
        id_map["satellites"][str(i)] = new_id
    
    overlap_rate = overlap_px / total_sat_px if total_sat_px > 0 else 0.0
    
    # Convert to uint8
    assert instance_map.max() <= 255, f"Too many instances: {instance_map.max()}"
    instance_map = instance_map.astype(np.uint8)
    
    return instance_map, instances_list, id_map, {"overlap_px": overlap_px, "overlap_rate": overlap_rate}


def _save_overlay(path: Path, image: np.ndarray, instance_map: np.ndarray) -> None:
    """Generate QA overlay with colored instances."""
    overlay = image.copy()
    
    # Generate colors for each instance
    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]
    
    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(256, 3))
    
    for inst_id in unique_ids:
        mask = instance_map == inst_id
        color = colors[int(inst_id) % 256]
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)
    
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# =============================================================================
# PHASE 4: EXPORT (SAM2 + SAM3)
# =============================================================================

def run_export_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Generate SAM2 symlinks and SAM3 annotations.json."""
    logger.info("=" * 60)
    logger.info("PHASE 4: EXPORT (SAM2 + SAM3)")
    logger.info("=" * 60)
    
    resolver = PathResolver(config)
    variants = config["preprocessing_variants"]
    target_size = tuple(config["processing"]["target_size"])
    
    sam2_dir = resolver.get_sam2_dir()
    sam3_dir = resolver.get_sam3_dir()
    
    # Create directories
    (sam2_dir / "img_folder").mkdir(parents=True, exist_ok=True)
    (sam2_dir / "gt_folder").mkdir(parents=True, exist_ok=True)
    (sam3_dir / "images").mkdir(parents=True, exist_ok=True)
    
    # SAM3 annotations accumulator
    sam3_images = []
    sam3_annotations = []
    sam3_categories = [
        {"id": 1, "name": "stellar stream", "supercategory": "lsb"},
        {"id": 2, "name": "satellite galaxy", "supercategory": "lsb"},
    ]
    image_id = 0
    ann_id = 0
    
    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        instance_map_path = gt_dir / "instance_map_uint8.png"
        instances_path = gt_dir / "instances.json"
        
        if not instance_map_path.exists():
            continue
        
        instances = json.loads(instances_path.read_text()) if instances_path.exists() else []
        instance_map = np.array(Image.open(instance_map_path))
        H, W = instance_map.shape
        
        for variant in variants:
            vname = variant["name"]
            variant_key = VariantKey(key, vname)
            
            render_path = resolver.get_render_dir(vname, key) / "0000.png"
            if not render_path.exists():
                continue
            
            # SAM2: Create symlinks
            sam2_img_dir = sam2_dir / "img_folder" / str(variant_key)
            sam2_gt_dir = sam2_dir / "gt_folder" / str(variant_key)
            sam2_img_dir.mkdir(parents=True, exist_ok=True)
            sam2_gt_dir.mkdir(parents=True, exist_ok=True)
            
            sam2_img_link = sam2_img_dir / "0000.png"
            sam2_gt_link = sam2_gt_dir / "0000.png"
            
            # Force rebuild symlinks if variant forced
            if force_variants and vname in force_variants:
                if sam2_img_link.exists() or sam2_img_link.is_symlink():
                    sam2_img_link.unlink()
                if sam2_gt_link.exists() or sam2_gt_link.is_symlink():
                    sam2_gt_link.unlink()
            
            if not sam2_img_link.exists():
                sam2_img_link.symlink_to(render_path.resolve())
            if not sam2_gt_link.exists():
                sam2_gt_link.symlink_to(instance_map_path.resolve())
            
            # SAM3: Flat image copy + annotations
            sam3_img_name = f"{variant_key}.png"
            sam3_img_path = sam3_dir / "images" / sam3_img_name
            
            # Force rebuild SAM3 symlink
            if force_variants and vname in force_variants:
                if sam3_img_path.exists() or sam3_img_path.is_symlink():
                    sam3_img_path.unlink()
            
            if not sam3_img_path.exists():
                sam3_img_path.symlink_to(render_path.resolve())
            
            # Add image entry
            image_id += 1
            sam3_images.append({
                "id": image_id,
                "file_name": f"images/{sam3_img_name}",
                "width": W,
                "height": H,
            })
            
            # Add annotations for each instance
            for inst in instances:
                inst_id = inst["id"]
                inst_type = inst["type"]
                category_id = 1 if inst_type == "streams" else 2
                
                binary_mask = (instance_map == inst_id).astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue
                
                rle = _mask_to_rle(binary_mask.astype(bool))
                bbox = _get_bbox(binary_mask)
                area = int(binary_mask.sum())
                
                ann_id += 1
                sam3_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })
    
    # Write SAM3 annotations.json
    coco = {
        "info": {
            "description": "LSB-AI-Detection Unified Dataset",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
        },
        "images": sam3_images,
        "annotations": sam3_annotations,
        "categories": sam3_categories,
    }
    (sam3_dir / "annotations.json").write_text(json.dumps(coco, indent=2))
    
    logger.info(f"SAM2: {len(base_keys) * len(variants)} symlinks")
    logger.info(f"SAM3: {len(sam3_images)} images, {len(sam3_annotations)} annotations")


def _get_bbox(binary_mask: np.ndarray) -> list[int]:
    """Get [x, y, w, h] bounding box from binary mask."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [int(x1), int(y1), int(x2 - x1 + 1), int(y2 - y1 + 1)]


# =============================================================================
# MAIN
# =============================================================================

def load_config(path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def generate_base_keys(config: dict, galaxy_filter: list[int] | None = None) -> list[BaseKey]:
    """Generate BaseKeys from config."""
    galaxy_ids = config["data_selection"]["galaxy_ids"]
    if galaxy_filter:
        galaxy_ids = [g for g in galaxy_ids if g in galaxy_filter]
    orientations = config["data_selection"]["orientations"]
    
    return [
        BaseKey(gid, ori)
        for gid in galaxy_ids
        for ori in orientations
    ]


def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Preparation")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Config YAML path")
    parser.add_argument("--phase", type=str, choices=["render", "gt", "satellites", "export", "all"], default="all")
    parser.add_argument("--galaxies", type=str, default=None, help="Comma-separated galaxy IDs subset")
    parser.add_argument("--force", action="store_true", help="Force rebuild all variants in selected phase")
    parser.add_argument("--force-variants", type=str, default=None,
                        help="Comma-separated variant names to force rebuild (e.g. asinh_stretch,multi_exposure)")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)
    
    config = load_config(args.config)
    
    # Parse galaxy filter
    galaxy_filter = None
    if args.galaxies:
        galaxy_filter = [int(g.strip()) for g in args.galaxies.split(",")]
    
    # Parse force variants
    force_variants: set[str] | None = None
    if args.force:
        # Force all variants defined in config
        force_variants = {v["name"] for v in config["preprocessing_variants"]}
        logger.info(f"Force rebuild ALL variants: {force_variants}")
    elif args.force_variants:
        force_variants = {v.strip() for v in args.force_variants.split(",")}
        logger.info(f"Force rebuild variants: {force_variants}")
    
    base_keys = generate_base_keys(config, galaxy_filter)
    logger.info(f"Processing {len(base_keys)} BaseKeys")
    
    phases = ["render", "gt", "satellites", "export"] if args.phase == "all" else [args.phase]
    
    for phase in phases:
        if phase == "render":
            run_render_phase(config, base_keys, logger, force_variants)
        elif phase == "gt":
            run_gt_phase(config, base_keys, logger, force_variants)
        elif phase == "satellites":
            run_satellites_phase(config, base_keys, logger, force_variants)
        elif phase == "export":
            run_export_phase(config, base_keys, logger, force_variants)
    
    logger.info("Done.")


if __name__ == "__main__":
    main()
