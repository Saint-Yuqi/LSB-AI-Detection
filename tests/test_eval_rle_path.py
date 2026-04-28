"""F11/F17: eval RLE-aware path matches per-class predictions to per-class GT
correctly across overlapping pixels and emits 3 metric blocks.
"""
from __future__ import annotations

import numpy as np

from src.evaluation.metrics import (
    calculate_optimal_instance_metrics_rle,
    rasterize_per_class_rles,
)
from src.utils.coco_utils import mask_to_rle


def _rle(arr: np.ndarray) -> dict:
    return mask_to_rle(arr.astype(np.uint8))


def test_rle_eval_preserves_within_class_overlap():
    """Two overlapping GT instances must both be matchable on the RLE path."""
    H, W = 64, 64
    gt1 = np.zeros((H, W), dtype=bool); gt1[10:30, 10:30] = True
    gt2 = np.zeros((H, W), dtype=bool); gt2[20:40, 20:40] = True
    overlap = (gt1 & gt2).sum()
    assert overlap > 0

    p1 = np.zeros((H, W), dtype=bool); p1[10:30, 10:30] = True
    p2 = np.zeros((H, W), dtype=bool); p2[22:42, 22:42] = True

    result = calculate_optimal_instance_metrics_rle(
        [_rle(p1), _rle(p2)],
        [_rle(gt1), _rle(gt2)],
        iou_threshold=0.5,
    )
    assert result["num_gt"] == 2
    assert result["num_pred"] == 2
    assert result["num_detected"] == 2
    assert result["matched_iou"] > 0.5


def test_rle_eval_empty_gt_handles_cleanly():
    H, W = 32, 32
    pred = np.zeros((H, W), dtype=bool); pred[5:10, 5:10] = True
    result = calculate_optimal_instance_metrics_rle([_rle(pred)], [], iou_threshold=0.5)
    assert result["num_gt"] == 0
    assert result["num_pred"] == 1
    assert result["instance_recall"] is None


def test_rle_eval_empty_pred_returns_zero_recall():
    H, W = 32, 32
    gt = np.zeros((H, W), dtype=bool); gt[5:10, 5:10] = True
    result = calculate_optimal_instance_metrics_rle([], [_rle(gt)], iou_threshold=0.5)
    assert result["num_gt"] == 1
    assert result["num_pred"] == 0
    assert result["instance_recall"] == 0.0
    assert len(result["per_instance_details"]) == 1


def test_rasterize_per_class_rles_last_wins_overlap():
    """rasterize_per_class_rles packs per-class RLEs with last-wins overlap."""
    H, W = 32, 32
    a = np.zeros((H, W), dtype=bool); a[5:15, 5:15] = True
    b = np.zeros((H, W), dtype=bool); b[10:20, 10:20] = True
    rles = [_rle(a), _rle(b)]
    raster = rasterize_per_class_rles(rles, H, W)
    # Both ids appear; overlap pixels claimed by id 2 (last-wins).
    assert set(int(x) for x in np.unique(raster) if x != 0) == {1, 2}
    overlap_region = raster[10:15, 10:15]
    assert (overlap_region == 2).all()
