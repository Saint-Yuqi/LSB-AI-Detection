"""
Pred-centric candidate matching for silver labelling.

Unlike the GT-centric matchers in ``src.evaluation.metrics``, this module
iterates **predictions** and computes per-prediction signals: best-GT IoU,
area ratio, boundary F1, ambiguity, and multi-GT overlap.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
#  Data structures
# ---------------------------------------------------------------------------

@dataclass
class CandidateMatchResult:
    """Per-prediction matching signals used by the silver labeler."""
    pred_idx: int
    best_gt_id: int | None
    best_iou: float
    second_best_iou: float
    area_ratio: float | None
    boundary_f1: float | None
    overlaps_multiple_gt: bool
    confidence_score: float
    filter_status: str  # "kept" | "prior_rejected" | "core_rejected" | ...


@dataclass
class ImageExhaustivityResult:
    """Image-level coverage signals for satellite_ev / route_to_ev."""
    num_gt: int
    num_pred_matched: int
    num_gt_unmatched: int
    num_pred_unmatched: int
    mean_matched_iou: float
    exhaustivity_confident: bool
    total_gt_area: int = 0
    total_pred_area: int = 0
    unmatched_gt_area: int = 0
    unmatched_pred_area: int = 0
    recall: float = 1.0
    precision: float = 1.0


# ---------------------------------------------------------------------------
#  Boundary F1
# ---------------------------------------------------------------------------

_BOUNDARY_DILATION = 2  # pixels


def _boundary_pixels(mask: np.ndarray, dilation: int = _BOUNDARY_DILATION) -> np.ndarray:
    """Morphological boundary: dilate - erode."""
    import cv2
    kernel = np.ones((dilation * 2 + 1,) * 2, np.uint8)
    dilated = cv2.dilate(mask, kernel)
    eroded = cv2.erode(mask, kernel)
    return (dilated - eroded).astype(bool)


def _boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    bp = _boundary_pixels(pred_mask.astype(np.uint8))
    bg = _boundary_pixels(gt_mask.astype(np.uint8))
    tp = float(np.logical_and(bp, bg).sum())
    fp = float(np.logical_and(bp, ~bg).sum())
    fn = float(np.logical_and(~bp, bg).sum())
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


# ---------------------------------------------------------------------------
#  IoU matrix construction (vectorized, adapted from metrics.py)
# ---------------------------------------------------------------------------

def _build_iou_matrix(
    pred_masks_flat: np.ndarray,
    gt_masks_flat: np.ndarray,
) -> np.ndarray:
    """Build (N_pred, N_gt) IoU matrix from flattened binary masks.

    Parameters
    ----------
    pred_masks_flat : (N_pred, HW) float32
    gt_masks_flat   : (N_gt, HW) float32
    """
    intersection = pred_masks_flat @ gt_masks_flat.T  # (N_pred, N_gt)
    pred_areas = pred_masks_flat.sum(axis=1, keepdims=True)  # (N_pred, 1)
    gt_areas = gt_masks_flat.sum(axis=1, keepdims=True).T    # (1, N_gt)
    union = pred_areas + gt_areas - intersection
    return np.where(union > 0, intersection / union, 0.0)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def match_candidates_to_gt(
    pred_masks: list[dict[str, Any]],
    gt_instance_map: np.ndarray,
    *,
    min_overlap_thresh: float = 0.05,
    iou_matrix: np.ndarray | None = None,
) -> list[CandidateMatchResult]:
    """Pred-centric matching: for each prediction, compute all signals.

    Parameters
    ----------
    pred_masks
        List of mask dicts.  Must contain ``segmentation`` (H, W) bool/uint8,
        and optionally ``score`` / ``stability_score``,
        ``filter_status``.
    gt_instance_map
        (H, W) int array, 0 = background, >=1 = GT instance IDs.
    min_overlap_thresh
        Minimum IoU with a GT to be counted as "overlapping".
    iou_matrix
        Pre-computed (N_pred, N_gt) IoU matrix; if ``None`` it is built here.
    """
    gt_ids = np.unique(gt_instance_map)
    gt_ids = gt_ids[gt_ids > 0]
    num_gt = len(gt_ids)
    num_pred = len(pred_masks)

    if num_pred == 0:
        return []

    hw = gt_instance_map.size

    if iou_matrix is None:
        pred_flat = np.stack(
            [m["segmentation"].ravel() for m in pred_masks], axis=0,
        ).astype(np.float32)

        if num_gt > 0:
            gt_flat = np.stack(
                [(gt_instance_map == gid).ravel() for gid in gt_ids], axis=0,
            ).astype(np.float32)
            iou_matrix = _build_iou_matrix(pred_flat, gt_flat)
        else:
            iou_matrix = np.zeros((num_pred, 0), dtype=np.float32)

    results: list[CandidateMatchResult] = []

    for pi in range(num_pred):
        m = pred_masks[pi]
        pred_area = float(m["segmentation"].sum())
        confidence = float(
            m.get("score", m.get("stability_score", 0.0))
        )
        filt = m.get("filter_status", "kept")

        if num_gt == 0:
            results.append(CandidateMatchResult(
                pred_idx=pi,
                best_gt_id=None,
                best_iou=0.0,
                second_best_iou=0.0,
                area_ratio=None,
                boundary_f1=None,
                overlaps_multiple_gt=False,
                confidence_score=confidence,
                filter_status=filt,
            ))
            continue

        row = iou_matrix[pi]  # (N_gt,)
        sorted_indices = np.argsort(row)[::-1]
        best_idx = sorted_indices[0]
        best_iou = float(row[best_idx])
        second_best_iou = float(row[sorted_indices[1]]) if num_gt > 1 else 0.0

        overlaps = int((row >= min_overlap_thresh).sum())
        overlaps_multi = overlaps >= 2

        best_gt_id: int | None = None
        area_ratio: float | None = None
        bf1: float | None = None

        if best_iou > 0:
            best_gt_id = int(gt_ids[best_idx])
            gt_mask = (gt_instance_map == best_gt_id)
            gt_area = float(gt_mask.sum())
            area_ratio = pred_area / gt_area if gt_area > 0 else None
            bf1 = _boundary_f1(
                m["segmentation"].astype(np.uint8),
                gt_mask.astype(np.uint8),
            )

        results.append(CandidateMatchResult(
            pred_idx=pi,
            best_gt_id=best_gt_id,
            best_iou=best_iou,
            second_best_iou=second_best_iou,
            area_ratio=area_ratio,
            boundary_f1=bf1,
            overlaps_multiple_gt=overlaps_multi,
            confidence_score=confidence,
            filter_status=filt,
        ))

    return results


def compute_image_exhaustivity(
    pred_masks: list[dict[str, Any]],
    gt_instance_map: np.ndarray,
    candidate_results: list[CandidateMatchResult],
    *,
    match_iou_thresh: float = 0.3,
    confident_recall_thresh: float = 0.9,
    confident_precision_thresh: float = 0.9,
) -> ImageExhaustivityResult:
    """Image-level exhaustivity assessment for EV / route_to_ev routing."""
    gt_ids = np.unique(gt_instance_map)
    gt_ids = gt_ids[gt_ids > 0]
    num_gt = len(gt_ids)

    matched_gt: set[int] = set()
    matched_ious: list[float] = []
    num_pred_matched = 0
    unmatched_pred_union = np.zeros_like(gt_instance_map, dtype=bool)
    total_pred_union = np.zeros_like(gt_instance_map, dtype=bool)

    for pred, cr in zip(pred_masks, candidate_results):
        pred_mask = pred["segmentation"].astype(bool)
        total_pred_union |= pred_mask
        if cr.best_gt_id is not None and cr.best_iou >= match_iou_thresh:
            matched_gt.add(cr.best_gt_id)
            matched_ious.append(cr.best_iou)
            num_pred_matched += 1
        else:
            unmatched_pred_union |= pred_mask

    num_gt_unmatched = num_gt - len(matched_gt)
    num_pred_unmatched = len(candidate_results) - num_pred_matched
    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    unmatched_gt_ids = [int(gid) for gid in gt_ids if int(gid) not in matched_gt]
    unmatched_gt_union = (
        np.isin(gt_instance_map, unmatched_gt_ids)
        if unmatched_gt_ids
        else np.zeros_like(gt_instance_map, dtype=bool)
    )

    total_gt_area = int((gt_instance_map > 0).sum())
    total_pred_area = int(total_pred_union.sum())
    unmatched_gt_area = int(unmatched_gt_union.sum())
    unmatched_pred_area = int(unmatched_pred_union.sum())

    recall = len(matched_gt) / num_gt if num_gt > 0 else 1.0
    precision = (
        num_pred_matched / len(candidate_results)
        if candidate_results
        else 1.0
    )
    confident = (
        recall >= confident_recall_thresh
        and precision >= confident_precision_thresh
    )

    return ImageExhaustivityResult(
        num_gt=num_gt,
        num_pred_matched=num_pred_matched,
        num_gt_unmatched=num_gt_unmatched,
        num_pred_unmatched=num_pred_unmatched,
        mean_matched_iou=mean_iou,
        exhaustivity_confident=confident,
        total_gt_area=total_gt_area,
        total_pred_area=total_pred_area,
        unmatched_gt_area=unmatched_gt_area,
        unmatched_pred_area=unmatched_pred_area,
        recall=recall,
        precision=precision,
    )
