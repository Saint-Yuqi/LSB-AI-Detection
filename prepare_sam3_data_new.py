#!/usr/bin/env python3
"""
Unified SAM3 Data Preparation Script (Best Quality Version)

Combines:
1. The Structure of Unified Script (Streams + Satellites, COCO JSON, Instance Masks)
2. The Image Quality of Script B (Linear Magnitude Scaling, Global Normalization)

Author: Yuqi
Date: 2026-02-02
"""

import gzip
import json
import pickle
import re
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from astropy.io import fits
import cv2
from pycocotools import mask as mask_util
from tqdm import tqdm

# === Configuration ===
SCRIPT_DIR = Path(__file__).parent
FIREBOX_ROOT = SCRIPT_DIR / "LSB_and_Satellites" / "FIREbox-DR1"
FBOX_ROOT = SCRIPT_DIR / "LSB_and_Satellites" / "fbox"

# Data sources
SB_MAPS_DIR = FIREBOX_ROOT / "SB_maps"
MASKS_EO_DIR = FIREBOX_ROOT / "MASKS_EO"
MASKS_FO_DIR = FIREBOX_ROOT / "MASKS_FO"
SATELLITE_PICKLE = Path("/shares/feldmann.ics.mnf.uzh/Lucas/pNbody/satellites/fbox/props_gals_Fbox_new.pkl")

# Output
OUTPUT_ROOT = SCRIPT_DIR / "sam3_prepared_unified_v2"

# Target resolution
TARGET_SIZE = (1024, 1024)

# Surface brightness thresholds (mag/arcsec²)
SB_THRESHOLDS = [27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32]
GALAXY_IDS = [11, 13, 19, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 56, 63, 64, 66, 68, 72]
ORIENTATIONS = ["eo", "fo"]

# === IMAGE PROCESSING CONFIG (From Script B) ===
# Global normalization range (mag/arcsec²)
# This is the secret sauce for good details!
GLOBAL_MAG_MIN = 20.0  # Brightest (Saturation point)
GLOBAL_MAG_MAX = 35.0  # Faintest (Black point)


def load_fits_gz(filepath: Path) -> np.ndarray:
    """Load a gzipped FITS file."""
    with gzip.open(filepath, 'rb') as f:
        with fits.open(f) as hdul:
            data = hdul[0].data
    return data


def process_image_physics_based(sb_map: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """
    [The "Script B" Logic]
    Directly normalizes Magnitude maps using global physical limits.
    This preserves the faint streams exactly how you like them.
    
    Args:
        sb_map: Surface brightness map (mag/arcsec²)
        target_size: (width, height)
    
    Returns:
        8-bit RGB image
    """
    # 1. Clean Data (Handle NaNs)
    # Any NaN or Inf is treated as background (Faintest)
    sb_clean = np.nan_to_num(sb_map, nan=GLOBAL_MAG_MAX, posinf=GLOBAL_MAG_MAX, neginf=GLOBAL_MAG_MIN)
    
    # 2. Clip to Global Range [20, 35]
    # This ensures consistency across ALL galaxies.
    sb_clipped = np.clip(sb_clean, GLOBAL_MAG_MIN, GLOBAL_MAG_MAX)
    
    # 3. Normalize to [0, 1]
    # Formula: (Max - Val) / (Max - Min)
    # Because Lower Mag = Brighter, we invert the subtraction.
    # Mag 35 -> 0.0 (Black)
    # Mag 20 -> 1.0 (White)
    img_norm = (GLOBAL_MAG_MAX - sb_clipped) / (GLOBAL_MAG_MAX - GLOBAL_MAG_MIN)
    
    # 4. Convert to 8-bit (Standard for SAM inputs)
    # Script B used 16-bit, but SAM usually needs 8-bit RGB. 
    # Since we used the exact same curve, the details will be preserved in 8-bit too.
    img_8bit = (img_norm * 255).astype(np.uint8)
    
    # 5. Convert to 3-channel RGB (Gray duplicated)
    img_rgb = np.stack([img_8bit] * 3, axis=-1)
    
    # 6. Resize (Cubic for sharpness)
    h, w = img_rgb.shape[:2]
    if (w, h) != target_size:
        img_rgb = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_CUBIC)
        
    return img_rgb


def resize_mask(mask: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """Resize mask using Nearest Neighbor to preserve IDs."""
    h, w = mask.shape
    if (w, h) != target_size:
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
    return mask


def mask_to_rle(binary_mask: np.ndarray) -> Dict:
    """Convert binary mask to RLE."""
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def get_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """Get bbox [x, y, w, h]."""
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    if not rows.any() or not cols.any():
        return [0, 0, 0, 0]
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


def process_mask_to_annotations(mask_data: np.ndarray, image_id: int, category_id: int, ann_id: int) -> Tuple[List[Dict], int]:
    """Convert instance mask to annotations."""
    annotations = []
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels > 0]
    
    for label in unique_labels:
        binary_mask = (mask_data == label).astype(np.uint8)
        area = int(binary_mask.sum())
        if area < 10: continue # Filter noise
        
        annotations.append({
            "id": ann_id,
            "image_id": image_id,
            "category_id": category_id,
            "segmentation": mask_to_rle(binary_mask),
            "bbox": get_bbox_from_mask(binary_mask),
            "area": area,
            "iscrowd": 0
        })
        ann_id += 1
    return annotations, ann_id


def extract_satellite_mask(sat_data: Any, image_shape: Tuple[int, int]) -> Optional[np.ndarray]:
    """Extract mask from satellite data."""
    if isinstance(sat_data, np.ndarray):
        mask = (sat_data > 0).astype(np.uint8)
    elif isinstance(sat_data, dict) and 'seg_ids' in sat_data:
        coords = sat_data['seg_ids']
        if coords is None or len(coords) == 0: return None
        coords = np.array(coords)
        seg_shape = sat_data.get('seg_map', np.zeros(image_shape)).shape
        mask = np.zeros(seg_shape, dtype=np.uint8)
        if coords.ndim == 2 and coords.shape[1] >= 2:
            y_coords = np.clip(coords[:, 0], 0, seg_shape[0]-1).astype(int)
            x_coords = np.clip(coords[:, 1], 0, seg_shape[1]-1).astype(int)
            mask[y_coords, x_coords] = 1
    else:
        return None

    if mask.shape != image_shape:
        mask = cv2.resize(mask, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_NEAREST)
    return mask


def create_categories(thresholds: List[float]) -> Tuple[List[Dict], Dict[Tuple[str, float], int]]:
    """Create simplified COCO categories."""
    # Simplified Logic: Only 2 Main Categories to prevent confusion?
    # Or keep granular? Let's stick to granular for now as requested.
    categories = []
    mapping = {}
    cat_id = 1
    for t in thresholds:
        t_str = str(int(t)) if t == int(t) else str(t)
        # Stream
        categories.append({"id": cat_id, "name": f"stellar_stream_SB{t_str}", "supercategory": "stellar_stream"})
        mapping[("stream", t)] = cat_id
        cat_id += 1
        # Satellite
        categories.append({"id": cat_id, "name": f"satellite_SB{t_str}", "supercategory": "satellite"})
        mapping[("satellite", t)] = cat_id
        cat_id += 1
    return categories, mapping


def process_galaxy(galaxy_id, orientation, satellite_data, images_dir, masks_dir, image_id, ann_id, cat_mapping):
    """Process single galaxy - creates separate images for streams and satellites."""
    
    all_image_infos = []
    all_annotations = []
    mask_dir_src = MASKS_EO_DIR if orientation == "eo" else MASKS_FO_DIR
    
    # === PART 1: STREAMS ===
    # Load VIS2 image from FIREBOX_ROOT for streams
    streams_sb_map_path = SB_MAPS_DIR / f"magnitudes-Fbox-{galaxy_id}-{orientation}-VIS2.fits.gz"
    if streams_sb_map_path.exists():
        try:
            sb_map_streams = load_fits_gz(streams_sb_map_path)
            img_rgb_streams = process_image_physics_based(sb_map_streams, TARGET_SIZE)
            
            img_filename_streams = f"Fbox-{galaxy_id}-{orientation}_streams.png"
            cv2.imwrite(str(images_dir / img_filename_streams), cv2.cvtColor(img_rgb_streams, cv2.COLOR_RGB2BGR))
            
            image_info_streams = {"id": image_id, "file_name": img_filename_streams, "height": TARGET_SIZE[1], "width": TARGET_SIZE[0]}
            all_image_infos.append(image_info_streams)
            
            # Process stream masks
            for t in SB_THRESHOLDS:
                cat_id = cat_mapping.get(("stream", t))
                mask_path = mask_dir_src / f"ark_features-{galaxy_id}-{orientation}-SBlim{t}.fits.gz"
                if mask_path.exists():
                    mask_data = load_fits_gz(mask_path)
                    mask_data = resize_mask(mask_data, TARGET_SIZE)
                    new_anns, ann_id = process_mask_to_annotations(mask_data, image_id, cat_id, ann_id)
                    all_annotations.extend(new_anns)
                    # Debug mask
                    if len(new_anns) > 0:
                        cv2.imwrite(str(masks_dir / f"Fbox-{galaxy_id}-{orientation}_SB{t}_streams.png"), (mask_data>0).astype(np.uint8)*255)
            
            image_id += 1
        except Exception as e:
            print(f"Error processing streams image {streams_sb_map_path}: {e}")
    
    # === PART 2: SATELLITES ===
    # Load VIS image from FBOX_ROOT for satellites
    satellites_sb_map_path = FBOX_ROOT / "sb_maps" / f"magnitudes-Fbox-{galaxy_id}-{orientation}-VIS.fits.gz"
    if satellite_data and satellites_sb_map_path.exists():
        # Match keys
        keys = [f"{galaxy_id}, {orientation}", f"{galaxy_id},{orientation}", (galaxy_id, orientation)]
        gal_data = next((satellite_data[k] for k in keys if k in satellite_data), None)
        
        if gal_data:
            try:
                sb_map_satellites = load_fits_gz(satellites_sb_map_path)
                img_rgb_satellites = process_image_physics_based(sb_map_satellites, TARGET_SIZE)
                
                img_filename_satellites = f"Fbox-{galaxy_id}-{orientation}_satellites.png"
                cv2.imwrite(str(images_dir / img_filename_satellites), cv2.cvtColor(img_rgb_satellites, cv2.COLOR_RGB2BGR))
                
                image_info_satellites = {"id": image_id, "file_name": img_filename_satellites, "height": TARGET_SIZE[1], "width": TARGET_SIZE[0]}
                all_image_infos.append(image_info_satellites)
                
                # Process satellite masks
                for sblim_key, sblim_data in gal_data.items():
                    # Parse key "SBlim27" -> 27.0
                    try: 
                        val = float(re.findall(r"[\d\.]+", str(sblim_key))[0])
                    except: continue
                    
                    if val not in SB_THRESHOLDS: continue
                    cat_id = cat_mapping.get(("satellite", val))
                    
                    # Combine satellites for this threshold
                    combined_mask = np.zeros((TARGET_SIZE[1], TARGET_SIZE[0]), dtype=np.uint8)
                    inst_id = 1
                    
                    if isinstance(sblim_data, dict):
                        for _, sat_blob in sblim_data.items():
                            mask = extract_satellite_mask(sat_blob, (TARGET_SIZE[1], TARGET_SIZE[0]))
                            if mask is not None and mask.sum() > 10:
                                combined_mask[mask > 0] = inst_id
                                # Annotation
                                all_annotations.append({
                                    "id": ann_id, "image_id": image_id, "category_id": cat_id,
                                    "segmentation": mask_to_rle(mask), "bbox": get_bbox_from_mask(mask),
                                    "area": int(mask.sum()), "iscrowd": 0
                                })
                                ann_id += 1
                                inst_id += 1
                    
                    if inst_id > 1:
                        cv2.imwrite(str(masks_dir / f"Fbox-{galaxy_id}-{orientation}_SB{val}_satellites.png"), (combined_mask>0).astype(np.uint8)*255)
                
                image_id += 1
            except Exception as e:
                print(f"Error processing satellites image {satellites_sb_map_path}: {e}")
    
    return all_image_infos, all_annotations, image_id, ann_id


def main():
    if OUTPUT_ROOT.exists(): shutil.rmtree(OUTPUT_ROOT)
    (OUTPUT_ROOT / "images").mkdir(parents=True)
    (OUTPUT_ROOT / "masks").mkdir(parents=True)
    
    # Load Satellites
    try:
        with open(SATELLITE_PICKLE, 'rb') as f: sat_data = pickle.load(f)
    except:
        sat_data = None
        print("Warning: No satellite pickle found.")

    categories, cat_mapping = create_categories(SB_THRESHOLDS)
    coco = {"images": [], "annotations": [], "categories": categories}
    
    img_id, ann_id = 0, 0
    for gid in tqdm(GALAXY_IDS):
        for orient in ORIENTATIONS:
            infos, anns, img_id, ann_id = process_galaxy(gid, orient, sat_data, OUTPUT_ROOT/"images", OUTPUT_ROOT/"masks", img_id, ann_id, cat_mapping)
            if infos:
                coco["images"].extend(infos)
                coco["annotations"].extend(anns)
                
    with open(OUTPUT_ROOT / "annotations.json", 'w') as f:
        json.dump(coco, f)
    print(f"Done! Saved to {OUTPUT_ROOT}")

if __name__ == "__main__":
    main()