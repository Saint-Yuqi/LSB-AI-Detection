"""
COCO format utilities.

Provides functions for converting masks to COCO-compatible RLE format,
extracting bounding boxes, and generating COCO annotations.
"""

from typing import Any, Dict, List, Tuple

import numpy as np


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """
    Convert a binary mask to COCO compressed RLE format.
    
    Args:
        binary_mask: Binary mask array (0 = background, 1 = foreground)
        
    Returns:
        RLE dict {'size': [H,W], 'counts': str}.
        counts is a compressed byte-string decoded to UTF-8 for JSON serialization.
        
    Raises:
        ImportError: If pycocotools is not installed.
    """
    from pycocotools import mask as mask_util  # lazy import
    mask_fortran = np.asfortranarray(binary_mask.astype(np.uint8))
    rle = mask_util.encode(mask_fortran)
    rle['counts'] = rle['counts'].decode('utf-8')
    return rle


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """
    Decode a COCO RLE dict back to a binary mask.
    
    Args:
        rle: RLE dict {'size': [H,W], 'counts': str}
        
    Returns:
        Binary mask array (H, W), dtype uint8.
        
    Raises:
        ImportError: If pycocotools is not installed.
    """
    from pycocotools import mask as mask_util  # lazy import
    if isinstance(rle['counts'], str):
        rle_input = {'size': rle['size'], 'counts': rle['counts'].encode('utf-8')}
    else:
        rle_input = rle
    return mask_util.decode(rle_input)


def get_bbox_from_mask(binary_mask: np.ndarray) -> List[float]:
    """
    Extract bounding box [x, y, width, height] from a binary mask.
    
    Args:
        binary_mask: Binary mask array
        
    Returns:
        Bounding box in COCO format [x, y, width, height].
        Returns [0, 0, 0, 0] for empty masks.
    """
    rows = np.any(binary_mask, axis=1)
    cols = np.any(binary_mask, axis=0)
    
    if not rows.any() or not cols.any():
        return [0.0, 0.0, 0.0, 0.0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [
        float(x_min),
        float(y_min),
        float(x_max - x_min + 1),
        float(y_max - y_min + 1)
    ]


def create_categories(
    thresholds: List[float]
) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, float], int]]:
    """
    Create COCO category definitions for stream and satellite classes.
    
    Generates categories for each surface brightness threshold, with both
    'stellar_stream' and 'satellite' variants.
    
    Args:
        thresholds: List of surface brightness thresholds
        
    Returns:
        Tuple of:
        - List of category dictionaries for COCO JSON
        - Mapping dict: (type, threshold) -> category_id
        
    Example:
        >>> categories, mapping = create_categories([27.0, 30.0])
        >>> mapping[("stream", 27.0)]  # Returns category ID for SB27 streams
    """
    categories: List[Dict[str, Any]] = []
    mapping: Dict[Tuple[str, float], int] = {}
    cat_id = 1
    
    for threshold in thresholds:
        # Format threshold string (remove .0 for integers)
        t_str = str(int(threshold)) if threshold == int(threshold) else str(threshold)
        
        # Stream category
        categories.append({
            "id": cat_id,
            "name": f"stellar_stream_SB{t_str}",
            "supercategory": "stellar_stream"
        })
        mapping[("stream", threshold)] = cat_id
        cat_id += 1
        
        # Satellite category
        categories.append({
            "id": cat_id,
            "name": f"satellite_SB{t_str}",
            "supercategory": "satellite"
        })
        mapping[("satellite", threshold)] = cat_id
        cat_id += 1
    
    return categories, mapping


def process_mask_to_annotations(
    mask_data: np.ndarray,
    image_id: int,
    category_id: int,
    ann_id: int,
    min_area: int = 10
) -> Tuple[List[Dict[str, Any]], int]:
    """
    Convert an instance mask to COCO annotation dictionaries.
    
    Args:
        mask_data: Instance mask with unique labels for each instance
        image_id: COCO image ID
        category_id: COCO category ID
        ann_id: Starting annotation ID
        min_area: Minimum pixel area to include (filters noise)
        
    Returns:
        Tuple of:
        - List of annotation dictionaries
        - Updated annotation ID counter
    """
    annotations: List[Dict[str, Any]] = []
    
    unique_labels = np.unique(mask_data)
    unique_labels = unique_labels[unique_labels > 0]  # Exclude background
    
    for label in unique_labels:
        binary_mask = (mask_data == label).astype(np.uint8)
        area = int(binary_mask.sum())
        
        if area < min_area:
            continue
        
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
