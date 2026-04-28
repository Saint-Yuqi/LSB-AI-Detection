"""
Segmentation evaluation metrics.

Provides IoU, Dice, Precision, Recall, capped Hausdorff95, and
optimal-matching instance metrics for binary and instance segmentation.

New functions (v5):
    calculate_pixel_metrics     – Dice, Precision, Recall, capped_hausdorff95
    calculate_optimal_instance_metrics – Hungarian 1:1 matched IoU / recall

Empty-mask conventions:
    Both empty  → null (None) for all pixel metrics
    One empty   → 0.0 for Dice; null for Precision/Recall; diagonal for HD95
    See docstrings for rationale.
"""

from typing import Any, Dict, List, Optional

import math

import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.optimize import linear_sum_assignment

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
        pred_masks: List of predicted masks.
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
        pred_masks: List of predicted masks (each with 'segmentation' key)
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


# =========================================================================== #
#  v5 metrics: pixel-level + optimal-matched instance-level
# =========================================================================== #


def _boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """Extract boundary pixels: mask XOR erode(mask, 1px)."""
    if mask.sum() == 0:
        return mask
    eroded = binary_erosion(mask, iterations=1)
    return (mask & ~eroded).astype(bool)


def calculate_pixel_metrics(
    pred_binary: np.ndarray,
    gt_binary: np.ndarray,
) -> Dict[str, Optional[float]]:
    """
    Pixel-level segmentation metrics with null-aware empty-mask handling.

    Args:
        pred_binary: (H, W) bool/uint8 predicted foreground mask.
        gt_binary:   (H, W) bool/uint8 ground-truth foreground mask.

    Returns:
        dice:               2·TP / (2·TP + FP + FN).  null if both empty.
        precision:          TP / (TP + FP).            null if no pred pixels.
        recall:             TP / (TP + FN).            null if no GT pixels.
        capped_hausdorff95: symmetric 95th-pct boundary distance.
                            null if both empty; diagonal if one empty.
        tp, fp, fn:         raw pixel counts (int) for downstream micro-avg.
    """
    pred = pred_binary.astype(bool)
    gt = gt_binary.astype(bool)

    tp = int((pred & gt).sum())
    fp = int((pred & ~gt).sum())
    fn = int((~pred & gt).sum())

    pred_empty = pred.sum() == 0
    gt_empty = gt.sum() == 0

    # --- Dice ---
    if pred_empty and gt_empty:
        dice: Optional[float] = None
    else:
        denom = 2 * tp + fp + fn
        dice = (2.0 * tp / denom) if denom > 0 else 0.0

    # --- Precision ---
    if pred_empty:
        precision: Optional[float] = None
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # --- Recall ---
    if gt_empty:
        recall: Optional[float] = None
    else:
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # --- capped_hausdorff95 ---
    H, W = pred.shape
    diagonal = math.sqrt(H * H + W * W)

    if pred_empty and gt_empty:
        hausdorff95: Optional[float] = None
    elif pred_empty or gt_empty:
        hausdorff95 = diagonal
    else:
        pred_boundary = _boundary_pixels(pred)
        gt_boundary = _boundary_pixels(gt)

        if pred_boundary.sum() == 0 or gt_boundary.sum() == 0:
            # Degenerate: 1-pixel masks have no boundary after erosion
            hausdorff95 = diagonal
        else:
            # EDT from gt_boundary → distances at pred_boundary positions
            dt_gt = distance_transform_edt(~gt_boundary)
            d_pred_to_gt = dt_gt[pred_boundary]

            # EDT from pred_boundary → distances at gt_boundary positions
            dt_pred = distance_transform_edt(~pred_boundary)
            d_gt_to_pred = dt_pred[gt_boundary]

            hausdorff95 = float(max(
                np.percentile(d_pred_to_gt, 95),
                np.percentile(d_gt_to_pred, 95),
            ))

    return {
        'dice': dice,
        'precision': precision,
        'recall': recall,
        'capped_hausdorff95': hausdorff95,
        'tp': tp,
        'fp': fp,
        'fn': fn,
    }


# =========================================================================== #
#  Pred-centric primitives (reusable by diagnostics, notebooks, experiments)
#
#  These functions do NOT use an IoU threshold. They answer:
#      "For a single prediction, what GT instance does it primarily hit,
#       and how much of it lands inside / covers that GT?"
#
#  Split by scope so the single-prediction helpers have no hidden global
#  dependency on the rest of the prediction list.
# =========================================================================== #


def primary_gt_match(
    pred_bin: np.ndarray,
    gt_instance_map: np.ndarray,
) -> Dict[str, Any]:
    """
    O(|P|) arg-max match of one prediction against an instance map.

    Counts positive GT IDs under the predicted foreground via
    ``np.bincount(gt_instance_map[pred_bin])``, then picks the most-overlapped
    positive ID. No IoU threshold.

    Tie-break: ``np.argmax`` returns the first occurrence, i.e. the smallest
    positive GT ID wins when two IDs have identical overlap. This is the
    documented behaviour.

    Args:
        pred_bin:         (H, W) bool — single prediction foreground.
        gt_instance_map:  (H, W) int — positive IDs are GT instances, 0 = bg.

    Returns:
        dict with keys:
            matched_gt_id:    int | None   — arg-max positive ID; None when
                                              there is zero overlap with any
                                              positive ID.
            overlap_px:       int          — |P ∩ G*|.
            pred_area:        int          — |P|.
            matched_gt_area:  int | None   — |G*|; None iff matched_gt_id
                                              is None.

    Raises:
        ValueError: if ``pred_bin.shape != gt_instance_map.shape``.
    """
    if pred_bin.shape != gt_instance_map.shape:
        raise ValueError(
            f"Shape mismatch: pred_bin {pred_bin.shape} vs "
            f"gt_instance_map {gt_instance_map.shape}"
        )
    pred_bool = pred_bin.astype(bool)
    pred_area = int(pred_bool.sum())
    if pred_area == 0:
        return {
            'matched_gt_id': None,
            'overlap_px': 0,
            'pred_area': 0,
            'matched_gt_area': None,
        }
    ids_under_pred = gt_instance_map[pred_bool]
    # bincount requires non-negative integers; gt_instance_map is int with 0=bg.
    if ids_under_pred.min() < 0:
        raise ValueError(
            "gt_instance_map contains negative values; expected 0 = bg and "
            "positive instance IDs only."
        )
    counts = np.bincount(ids_under_pred.astype(np.int64))
    if counts.size <= 1 or counts[1:].max() == 0:
        return {
            'matched_gt_id': None,
            'overlap_px': 0,
            'pred_area': pred_area,
            'matched_gt_area': None,
        }
    counts[0] = 0
    gid = int(counts.argmax())
    overlap_px = int(counts[gid])
    matched_gt_area = int((gt_instance_map == gid).sum())
    return {
        'matched_gt_id': gid,
        'overlap_px': overlap_px,
        'pred_area': pred_area,
        'matched_gt_area': matched_gt_area,
    }


def derive_purity_completeness(
    overlap_px: int,
    pred_area: int,
    matched_gt_area: Optional[int],
) -> Dict[str, Optional[float]]:
    """
    Pure scalar helper. Single-prediction, no global dependency.

    Args:
        overlap_px:       |P ∩ G*|.
        pred_area:        |P|.
        matched_gt_area:  |G*|, or None when there is no matched GT.

    Returns:
        dict with keys:
            purity:         overlap_px / pred_area        (0.0 when pred_area == 0)
            completeness:   overlap_px / matched_gt_area  (None iff matched_gt_area is None)
            seed_gt_ratio:  pred_area / matched_gt_area   (None iff matched_gt_area is None)
    """
    purity = (overlap_px / pred_area) if pred_area > 0 else 0.0
    if matched_gt_area is None:
        return {
            'purity': float(purity),
            'completeness': None,
            'seed_gt_ratio': None,
        }
    if matched_gt_area <= 0:
        raise ValueError(
            f"matched_gt_area must be > 0 when provided; got {matched_gt_area}"
        )
    return {
        'purity': float(purity),
        'completeness': float(overlap_px / matched_gt_area),
        'seed_gt_ratio': float(pred_area / matched_gt_area),
    }


def compute_one_to_one_flags(
    pred_bins: List[np.ndarray],
    gt_instance_map: np.ndarray,
) -> List[bool]:
    """
    Batch helper. Builds the (N_pred × N_gt) overlap matrix once and returns
    ``True`` for prediction p iff p is the arg-max-overlap prediction for its
    primary GT (and that primary GT has positive overlap with p).

    Tie-break: ``np.argmax`` takes the first occurrence — the prediction at
    the lowest list index wins when two predictions have identical overlap
    with the same GT.

    Args:
        pred_bins:        list of (H, W) bool arrays, all sharing the image
                          shape of ``gt_instance_map``. Predictions in this
                          list should belong to a single type channel (the
                          caller decides — e.g. satellites only).
        gt_instance_map:  (H, W) int, 0 = bg.

    Returns:
        list[bool] of length ``len(pred_bins)``.

    Raises:
        ValueError: if any pred shape mismatches ``gt_instance_map``.
    """
    n_pred = len(pred_bins)
    if n_pred == 0:
        return []

    gt_ids = np.unique(gt_instance_map)
    gt_ids = gt_ids[gt_ids > 0]
    if gt_ids.size == 0:
        # No GT at all: no prediction is "the best hit" for any GT.
        return [False] * n_pred

    H, W = gt_instance_map.shape
    for idx, p in enumerate(pred_bins):
        if p.shape != (H, W):
            raise ValueError(
                f"pred_bins[{idx}].shape={p.shape} does not match "
                f"gt_instance_map shape {(H, W)}"
            )

    # Primary GT per prediction (first-argmax tie-break).
    primary_gt = np.full(n_pred, -1, dtype=np.int64)
    overlap_with_primary = np.zeros(n_pred, dtype=np.int64)
    for i, p in enumerate(pred_bins):
        info = primary_gt_match(p, gt_instance_map)
        if info['matched_gt_id'] is None:
            continue
        primary_gt[i] = info['matched_gt_id']
        overlap_with_primary[i] = info['overlap_px']

    # For each GT id, find the pred with the largest overlap with that GT
    # (first-argmax tie-break). A pred is one-to-one iff it is that winner
    # AND its overlap with that GT is > 0 AND the GT is its own primary.
    best_pred_for_gt: Dict[int, int] = {}
    # Compute overlap(p, g) only for (p, g) pairs where g == primary_gt[p];
    # that is the only comparison we need for the "mutual primary" check.
    per_gt_candidates: Dict[int, List[tuple]] = {}
    for i in range(n_pred):
        gid = int(primary_gt[i])
        if gid < 0:
            continue
        per_gt_candidates.setdefault(gid, []).append((i, int(overlap_with_primary[i])))

    for gid, cands in per_gt_candidates.items():
        # first-argmax: pick the candidate with the largest overlap; on tie,
        # the one encountered first (lowest i) wins — matches np.argmax.
        best_i = cands[0][0]
        best_ov = cands[0][1]
        for i, ov in cands[1:]:
            if ov > best_ov:
                best_i = i
                best_ov = ov
        if best_ov > 0:
            best_pred_for_gt[gid] = best_i

    flags = [False] * n_pred
    for i in range(n_pred):
        gid = int(primary_gt[i])
        if gid < 0:
            continue
        if best_pred_for_gt.get(gid) == i:
            flags[i] = True
    return flags


def calculate_optimal_instance_metrics(
    pred_masks: List[Dict],
    gt_instance_map: np.ndarray,
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """
    Optimal 1:1 instance matching via Hungarian algorithm.

    Builds IoU matrix (N_gt × N_pred), solves assignment with
    scipy.optimize.linear_sum_assignment on cost = 1 − IoU.
    Pairs with IoU < iou_threshold are discarded post-assignment.

    Args:
        pred_masks:       list of mask dicts with 'segmentation' (H, W) bool.
        gt_instance_map:  (H, W) int array, 0 = background, 1..N = instance IDs.
        iou_threshold:    minimum IoU for a valid match.

    Returns:
        instance_recall:  fraction of GT instances with a valid match.
        matched_iou:      mean IoU of valid matched pairs.
        unmatched_iou:    mean best-IoU of unmatched GT instances.
        num_gt:           total GT instances.
        num_detected:     GT instances with valid match.
        num_pred:         total pred masks.
        per_instance_details: list of dicts per GT instance.
    """
    gt_ids = np.unique(gt_instance_map)
    gt_ids = gt_ids[gt_ids > 0]
    num_gt = len(gt_ids)
    num_pred = len(pred_masks)

    if num_gt == 0:
        return {
            'instance_recall': None,
            'matched_iou': None,
            'unmatched_iou': None,
            'num_gt': 0,
            'num_detected': 0,
            'num_pred': num_pred,
            'per_instance_details': [],
        }

    if num_pred == 0:
        details = []
        for gid in gt_ids:
            gt_area = int((gt_instance_map == gid).sum())
            details.append({
                'gt_instance_id': int(gid),
                'matched_pred_idx': None,
                'iou': 0.0,
                'detected': False,
                'gt_area': gt_area,
            })
        return {
            'instance_recall': 0.0,
            'matched_iou': 0.0,
            'unmatched_iou': 0.0,
            'num_gt': num_gt,
            'num_detected': 0,
            'num_pred': 0,
            'per_instance_details': details,
        }

    # Build binary masks: (N_gt, HW) and (N_pred, HW)
    HW = gt_instance_map.size
    gt_masks_flat = np.stack(
        [(gt_instance_map == gid).ravel() for gid in gt_ids],
        axis=0,
    ).astype(np.float32)  # (N_gt, HW)

    pred_masks_flat = np.stack(
        [m['segmentation'].ravel() for m in pred_masks],
        axis=0,
    ).astype(np.float32)  # (N_pred, HW)

    # IoU matrix: (N_gt, N_pred)
    # intersection = gt @ pred.T  (matrix product on flattened binary)
    intersection = gt_masks_flat @ pred_masks_flat.T  # (N_gt, N_pred)
    gt_areas = gt_masks_flat.sum(axis=1, keepdims=True)     # (N_gt, 1)
    pred_areas = pred_masks_flat.sum(axis=1, keepdims=True).T  # (1, N_pred)
    union = gt_areas + pred_areas - intersection
    iou_matrix = np.where(union > 0, intersection / union, 0.0)  # (N_gt, N_pred)

    # Hungarian optimal assignment on cost = 1 − IoU
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)

    # Build assignment map: gt_idx → (pred_idx, iou)
    assignment: Dict[int, tuple] = {}
    for r, c in zip(row_idx, col_idx):
        assignment[r] = (c, float(iou_matrix[r, c]))

    # Classify matches
    matched_ious = []
    unmatched_ious = []
    details = []
    detected = 0

    for gi, gid in enumerate(gt_ids):
        gt_area = int(gt_masks_flat[gi].sum())

        if gi in assignment:
            pred_idx, iou_val = assignment[gi]
            is_match = iou_val >= iou_threshold
        else:
            # More GTs than preds — unassigned
            best_iou = float(iou_matrix[gi].max()) if num_pred > 0 else 0.0
            pred_idx = None
            iou_val = best_iou
            is_match = False

        if is_match:
            detected += 1
            matched_ious.append(iou_val)
        else:
            # Best overlap for unmatched (may differ from assigned pair)
            best = float(iou_matrix[gi].max()) if num_pred > 0 else 0.0
            unmatched_ious.append(best)

        details.append({
            'gt_instance_id': int(gid),
            'matched_pred_idx': int(pred_idx) if pred_idx is not None and is_match else None,
            'iou': float(iou_val),
            'detected': is_match,
            'gt_area': gt_area,
        })

    instance_recall = detected / num_gt
    avg_matched = float(np.mean(matched_ious)) if matched_ious else 0.0
    avg_unmatched = float(np.mean(unmatched_ious)) if unmatched_ious else 0.0

    return {
        'instance_recall': instance_recall,
        'matched_iou': avg_matched,
        'unmatched_iou': avg_unmatched,
        'num_gt': num_gt,
        'num_detected': detected,
        'num_pred': num_pred,
        'per_instance_details': details,
    }


def calculate_optimal_instance_metrics_rle(
    pred_rle_list: List[Dict],
    gt_rle_list: List[Dict],
    iou_threshold: float = 0.5,
) -> Dict[str, Any]:
    """RLE-aware variant of ``calculate_optimal_instance_metrics``.

    Operates directly on lists of COCO RLE dicts (one per instance) rather
    than packing instances into a single int32 raster. This preserves
    within-class overlap that would otherwise collapse during raster
    packing (e.g. PNbody tidal_features, satellites under the new GT).

    Returns the same metric tuple shape as the integer-raster variant so
    surrounding plumbing in ``checkpoint_eval`` stays unchanged. The
    ``gt_instance_id`` slot in ``per_instance_details`` is replaced by a
    1-based positional index because RLEs carry no intrinsic ID.
    """
    num_gt = len(gt_rle_list)
    num_pred = len(pred_rle_list)

    if num_gt == 0:
        return {
            'instance_recall': None,
            'matched_iou': None,
            'unmatched_iou': None,
            'num_gt': 0,
            'num_detected': 0,
            'num_pred': num_pred,
            'per_instance_details': [],
        }

    if num_pred == 0:
        details = []
        for gi in range(num_gt):
            gt_area = _rle_area(gt_rle_list[gi])
            details.append({
                'gt_instance_id': gi + 1,
                'matched_pred_idx': None,
                'iou': 0.0,
                'detected': False,
                'gt_area': gt_area,
            })
        return {
            'instance_recall': 0.0,
            'matched_iou': 0.0,
            'unmatched_iou': 0.0,
            'num_gt': num_gt,
            'num_detected': 0,
            'num_pred': 0,
            'per_instance_details': details,
        }

    iou_matrix = _rle_iou_matrix(gt_rle_list, pred_rle_list)  # (N_gt, N_pred)
    cost = 1.0 - iou_matrix
    row_idx, col_idx = linear_sum_assignment(cost)

    assignment: Dict[int, tuple] = {}
    for r, c in zip(row_idx, col_idx):
        assignment[r] = (c, float(iou_matrix[r, c]))

    matched_ious: list[float] = []
    unmatched_ious: list[float] = []
    details: list[dict] = []
    detected = 0

    for gi in range(num_gt):
        gt_area = _rle_area(gt_rle_list[gi])
        if gi in assignment:
            pred_idx, iou_val = assignment[gi]
            is_match = iou_val >= iou_threshold
        else:
            best_iou = float(iou_matrix[gi].max()) if num_pred > 0 else 0.0
            pred_idx = None
            iou_val = best_iou
            is_match = False

        if is_match:
            detected += 1
            matched_ious.append(iou_val)
        else:
            best = float(iou_matrix[gi].max()) if num_pred > 0 else 0.0
            unmatched_ious.append(best)

        details.append({
            'gt_instance_id': gi + 1,
            'matched_pred_idx': int(pred_idx) if pred_idx is not None and is_match else None,
            'iou': float(iou_val),
            'detected': is_match,
            'gt_area': gt_area,
        })

    return {
        'instance_recall': detected / num_gt,
        'matched_iou': float(np.mean(matched_ious)) if matched_ious else 0.0,
        'unmatched_iou': float(np.mean(unmatched_ious)) if unmatched_ious else 0.0,
        'num_gt': num_gt,
        'num_detected': detected,
        'num_pred': num_pred,
        'per_instance_details': details,
    }


def rasterize_per_class_rles(
    rle_list: List[Dict],
    H: int,
    W: int,
) -> np.ndarray:
    """Pack per-class RLEs into an int32 last-wins raster.

    Used by the legacy diagnostic / pixel-metric paths in checkpoint_eval
    that expect a single int32 GT array per class. Within-class overlap
    pixels are owned by the last-painted instance — this is fine for
    taxonomy classification (a per-candidate label) but would lose RLE
    fidelity for instance recall, which is why we report RLE-aware
    metrics in a separate sub-block.
    """
    out = np.zeros((H, W), dtype=np.int32)
    for i, rle in enumerate(rle_list, start=1):
        binary = _rle_decode_binary(rle).astype(bool)
        if binary.shape != (H, W):
            raise ValueError(
                f"RLE shape {binary.shape} does not match expected {(H, W)}"
            )
        out[binary] = i
    return out


def _rle_decode_binary(rle: Dict) -> np.ndarray:
    """Decode a COCO RLE dict to a (H, W) bool array."""
    from src.utils.coco_utils import decode_rle as _decode
    return _decode(rle).astype(bool)


def _rle_area(rle: Dict) -> int:
    """Pixel area of an RLE without materializing the full raster."""
    return int(_rle_decode_binary(rle).sum())


def _rle_iou_matrix(gt_rles: List[Dict], pred_rles: List[Dict]) -> np.ndarray:
    """Pairwise IoU between GT and predicted RLE lists.

    Falls back to a numpy implementation when pycocotools is unavailable.
    """
    try:
        import pycocotools.mask as mask_utils
        # pycocotools expects byte counts, not str counts. Convert if needed.
        def _ensure_bytes(rle):
            r = dict(rle)
            counts = r.get("counts")
            if isinstance(counts, str):
                r["counts"] = counts.encode("ascii")
            return r

        gt_pc = [_ensure_bytes(r) for r in gt_rles]
        pred_pc = [_ensure_bytes(r) for r in pred_rles]
        iscrowd = [0] * len(gt_pc)
        ious = mask_utils.iou(pred_pc, gt_pc, iscrowd)
        # mask_utils.iou returns (N_pred, N_gt); we want (N_gt, N_pred).
        return np.asarray(ious).T.astype(np.float64)
    except (ImportError, ModuleNotFoundError):
        gt_bin = [_rle_decode_binary(r).ravel() for r in gt_rles]
        pred_bin = [_rle_decode_binary(r).ravel() for r in pred_rles]
        gt_arr = np.stack(gt_bin, axis=0).astype(np.float32) if gt_bin else np.zeros((0, 0), dtype=np.float32)
        pred_arr = np.stack(pred_bin, axis=0).astype(np.float32) if pred_bin else np.zeros((0, 0), dtype=np.float32)
        if gt_arr.size == 0 or pred_arr.size == 0:
            return np.zeros((len(gt_bin), len(pred_bin)), dtype=np.float64)
        intersection = gt_arr @ pred_arr.T
        gt_areas = gt_arr.sum(axis=1, keepdims=True)
        pred_areas = pred_arr.sum(axis=1, keepdims=True).T
        union = gt_areas + pred_areas - intersection
        return np.where(union > 0, intersection / union, 0.0).astype(np.float64)
