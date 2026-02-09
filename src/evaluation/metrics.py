"""
Segmentation evaluation metrics.

Provides IoU (Intersection over Union) calculations for binary and
instance segmentation tasks.
"""

from typing import Any, Dict, List

import numpy as np

# Try importing regionprops; fallback to manual bbox extraction
try:
    from skimage.measure import regionprops
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False


def _extract_bbox_area_numpy(mask: np.ndarray) -> tuple:
    """Extract bbox and area using pure numpy (fallback when skimage unavailable)."""
    coords = np.argwhere(mask > 0)
    if len(coords) == 0:
        return (0, 0, 0, 0), 0
    minr, minc = coords.min(axis=0)
    maxr, maxc = coords.max(axis=0)
    area = int(np.sum(mask > 0))
    return (int(minr), int(minc), int(maxr) + 1, int(maxc) + 1), area


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate IoU between predicted and ground truth masks.
    
    Both masks are treated as binary (0 = background, >0 = foreground).
    
    Args:
        pred_mask: Predicted mask array
        gt_mask: Ground truth mask array
        
    Returns:
        IoU score in range [0, 1]
    """
    pred_binary = (pred_mask > 0).astype(bool)
    gt_binary = (gt_mask > 0).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        # Both masks are empty
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def calculate_instance_iou(
    pred_masks: List[Dict],
    gt_mask: np.ndarray
) -> Dict[str, float]:
    """
    Calculate IoU metrics for instance segmentation.
    
    Args:
        pred_masks: List of predicted masks from SAM2.
            Each dict should have a 'segmentation' key with a binary mask.
        gt_mask: Ground truth mask with instance IDs (0=bg, 1,2,3...=instances)
        
    Returns:
        Dictionary containing:
        - binary_iou: Overall coverage IoU
        - mean_instance_iou: Mean IoU across GT instances
        - num_gt_instances: Number of ground truth instances
        - num_pred_masks: Number of predicted masks
        - instance_ious: List of per-instance IoU values
    """
    # Combine all predicted masks into one binary mask
    if len(pred_masks) == 0:
        combined_pred = np.zeros_like(gt_mask, dtype=bool)
    else:
        combined_pred = np.zeros_like(gt_mask, dtype=bool)
        for mask_data in pred_masks:
            combined_pred = np.logical_or(combined_pred, mask_data['segmentation'])
    
    # Binary IoU (overall coverage)
    binary_iou = calculate_iou(combined_pred.astype(np.uint8), gt_mask)
    
    # Instance-level analysis
    gt_instances = np.unique(gt_mask)
    gt_instances = gt_instances[gt_instances > 0]  # Remove background
    
    instance_ious: List[float] = []
    for inst_id in gt_instances:
        inst_mask = (gt_mask == inst_id)
        inst_iou = calculate_iou(
            combined_pred.astype(np.uint8),
            inst_mask.astype(np.uint8)
        )
        instance_ious.append(inst_iou)
    
    return {
        'binary_iou': binary_iou,
        'mean_instance_iou': float(np.mean(instance_ious)) if instance_ious else 0.0,
        'num_gt_instances': len(gt_instances),
        'num_pred_masks': len(pred_masks),
        'instance_ious': instance_ious
    }


def calculate_matched_metrics(
    pred_masks: List[Dict],
    gt_mask: np.ndarray,
    iou_threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Instance-based Recall: Match each GT instance to best prediction.
    
    Resolves "incomplete annotation → false low Precision" by focusing on
    whether each GT object is successfully detected.
    
    Args:
        pred_masks: List of SAM2 predicted masks (each with 'segmentation' key)
        gt_mask: Ground truth mask (0=bg, 1,2,3...=instance IDs)
        iou_threshold: Minimum IoU to count as detected
        
    Returns:
        recall: Fraction of GT instances with max_iou > threshold
        matched_iou: Mean IoU of successfully matched instances
        num_detected: Count of detected GT instances
        num_gt_instances: Total GT instances
        per_instance_details: List[Dict] with shape/matching data per GT instance
    """
    # Extract GT instances
    gt_instances = np.unique(gt_mask)
    gt_instances = gt_instances[gt_instances > 0]  # Remove background
    num_gt = len(gt_instances)
    
    if num_gt == 0:
        return {
            'recall': 0.0,
            'matched_iou': 0.0,
            'num_detected': 0,
            'num_gt_instances': 0,
            'per_instance_details': []
        }
    
    # Combine all pred masks into binary array (H x W x N_pred)
    if len(pred_masks) == 0:
        pred_binary_stack = np.zeros(
            (gt_mask.shape[0], gt_mask.shape[1], 1), dtype=bool
        )
    else:
        pred_binary_stack = np.stack(
            [mask_data['segmentation'] for mask_data in pred_masks], axis=-1
        )
    
    # Extract GT instance properties
    gt_labeled = gt_mask.astype(int)
    
    if HAS_SKIMAGE:
        props = regionprops(gt_labeled)
    else:
        # Fallback: manual bbox/area extraction using numpy
        props = []
        for inst_id in gt_instances:
            inst_mask_binary = (gt_mask == inst_id)
            bbox, area = _extract_bbox_area_numpy(inst_mask_binary)
            # Create mock regionprops object
            class MockProp:
                def __init__(self, label, bbox, area):
                    self.label = label
                    self.bbox = bbox  # (minr, minc, maxr, maxc)
                    self.area = area
            props.append(MockProp(inst_id, bbox, area))
    
    per_instance_details = []
    
    # Vectorized computation of IoU for ALL GT instances against ALL pred masks
    # Flatten HW to speed up matrix operations
    # HW = H * W
    HW = gt_mask.shape[0] * gt_mask.shape[1]
    
    # Bbox/Area props
    gt_props = props
    
    # Convert pred_masks to a single (HW, N_pred) bitmask
    if len(pred_masks) > 0:
        # (N, H, W) -> (N, HW) -> (HW, N)
        pred_masks_hw = pred_binary_stack.reshape(HW, -1).astype(np.float32)
        pred_areas = pred_masks_hw.sum(axis=0)  # (N_pred,)
    else:
        pred_masks_hw = None
        pred_areas = None

    detected_count = 0
    matched_ious = []

    for prop in gt_props:
        inst_id = prop.label
        inst_mask_hw = (gt_mask == inst_id).reshape(HW, 1).astype(np.float32)
        gt_area = float(prop.area)
        
        best_iou = 0.0
        best_idx = -1
        
        if pred_masks_hw is not None:
            # Intersection: (1, HW) @ (HW, N) -> (1, N)
            intersections = (inst_mask_hw.T @ pred_masks_hw).flatten()
            
            # Union: area_gt + area_pred - intersection (broadcasted)
            unions = gt_area + pred_areas - intersections
            
            # IoU
            ious = np.where(unions > 0, intersections / unions, 0.0)
            
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
        
        detected = best_iou >= iou_threshold
        if detected:
            detected_count += 1
            matched_ious.append(best_iou)
        
        minr, minc, maxr, maxc = prop.bbox
        per_instance_details.append({
            'gt_instance_id': int(inst_id),
            'best_pred_idx': best_idx,
            'best_iou': best_iou,
            'detected': detected,
            'gt_area': int(gt_area),
            'gt_bbox': [int(minc), int(minr), int(maxc), int(maxr)],
        })
    
    recall = detected_count / num_gt if num_gt > 0 else 0.0
    avg_matched_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    
    return {
        'recall': recall,
        'matched_iou': avg_matched_iou,
        'num_detected': detected_count,
        'num_gt_instances': num_gt,
        'per_instance_details': per_instance_details
    }
