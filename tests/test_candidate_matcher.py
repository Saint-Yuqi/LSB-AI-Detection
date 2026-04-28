"""
Tests for pred-centric candidate matching.

Verifies that ``candidate_matcher`` correctly computes per-prediction
signals against known synthetic masks, and does NOT call
``calculate_matched_metrics``.

Usage:
    pytest tests/test_candidate_matcher.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.candidate_matcher import (
    CandidateMatchResult,
    ImageExhaustivityResult,
    compute_image_exhaustivity,
    match_candidates_to_gt,
)


def _make_gt(H: int, W: int, regions: dict[int, tuple]) -> np.ndarray:
    """Create a synthetic GT instance map with box-shaped instances."""
    gt = np.zeros((H, W), dtype=np.uint8)
    for inst_id, (y0, x0, y1, x1) in regions.items():
        gt[y0:y1, x0:x1] = inst_id
    return gt


def _make_pred(H: int, W: int, y0: int, x0: int, y1: int, x1: int,
               **extra) -> dict:
    seg = np.zeros((H, W), dtype=bool)
    seg[y0:y1, x0:x1] = True
    d = {"segmentation": seg, **extra}
    return d


class TestMatchCandidatesToGt:
    def test_perfect_match(self):
        gt = _make_gt(100, 100, {1: (10, 10, 30, 30)})
        pred = [_make_pred(100, 100, 10, 10, 30, 30)]
        results = match_candidates_to_gt(pred, gt)
        assert len(results) == 1
        assert results[0].best_gt_id == 1
        assert results[0].best_iou == pytest.approx(1.0, abs=1e-6)
        assert results[0].area_ratio == pytest.approx(1.0, abs=1e-6)
        assert results[0].overlaps_multiple_gt is False

    def test_no_gt(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        pred = [_make_pred(50, 50, 5, 5, 15, 15)]
        results = match_candidates_to_gt(pred, gt)
        assert len(results) == 1
        assert results[0].best_gt_id is None
        assert results[0].best_iou == 0.0
        assert results[0].area_ratio is None
        assert results[0].boundary_f1 is None

    def test_no_preds(self):
        gt = _make_gt(50, 50, {1: (10, 10, 20, 20)})
        results = match_candidates_to_gt([], gt)
        assert results == []

    def test_partial_overlap(self):
        gt = _make_gt(100, 100, {1: (10, 10, 30, 30)})
        pred = [_make_pred(100, 100, 10, 10, 20, 20)]
        results = match_candidates_to_gt(pred, gt)
        assert 0 < results[0].best_iou < 1.0
        assert results[0].best_gt_id == 1

    def test_overlaps_multiple_gt(self):
        gt = _make_gt(100, 100, {
            1: (10, 10, 30, 30),
            2: (10, 25, 30, 50),
        })
        pred = [_make_pred(100, 100, 10, 10, 30, 50)]
        results = match_candidates_to_gt(pred, gt, min_overlap_thresh=0.05)
        assert results[0].overlaps_multiple_gt is True

    def test_ambiguity_signal(self):
        gt = _make_gt(100, 100, {
            1: (10, 10, 30, 30),
            2: (10, 28, 30, 48),
        })
        pred = [_make_pred(100, 100, 10, 19, 30, 39)]
        results = match_candidates_to_gt(pred, gt)
        assert results[0].second_best_iou > 0
        assert results[0].best_iou > results[0].second_best_iou

    def test_area_ratio(self):
        gt = _make_gt(100, 100, {1: (10, 10, 30, 30)})
        pred = [_make_pred(100, 100, 10, 10, 40, 40)]
        results = match_candidates_to_gt(pred, gt)
        assert results[0].area_ratio is not None
        assert results[0].area_ratio > 1.0

    def test_filter_status_preserved(self):
        gt = _make_gt(50, 50, {1: (10, 10, 20, 20)})
        pred = [_make_pred(50, 50, 10, 10, 20, 20,
                           filter_status="prior_rejected")]
        results = match_candidates_to_gt(pred, gt)
        assert results[0].filter_status == "prior_rejected"


class TestImageExhaustivity:
    def test_fully_matched(self):
        gt = _make_gt(100, 100, {
            1: (10, 10, 30, 30),
            2: (50, 50, 70, 70),
        })
        preds = [
            _make_pred(100, 100, 10, 10, 30, 30),
            _make_pred(100, 100, 50, 50, 70, 70),
        ]
        cr = match_candidates_to_gt(preds, gt)
        exh = compute_image_exhaustivity(preds, gt, cr, match_iou_thresh=0.3)
        assert exh.num_gt == 2
        assert exh.num_pred_matched == 2
        assert exh.num_gt_unmatched == 0
        assert exh.exhaustivity_confident is True

    def test_missing_gt(self):
        gt = _make_gt(100, 100, {
            1: (10, 10, 30, 30),
            2: (50, 50, 70, 70),
        })
        preds = [_make_pred(100, 100, 10, 10, 30, 30)]
        cr = match_candidates_to_gt(preds, gt)
        exh = compute_image_exhaustivity(preds, gt, cr)
        assert exh.num_gt_unmatched == 1
        assert exh.exhaustivity_confident is False

    def test_empty_image(self):
        gt = np.zeros((50, 50), dtype=np.uint8)
        cr = match_candidates_to_gt([], gt)
        exh = compute_image_exhaustivity([], gt, cr)
        assert exh.num_gt == 0
        assert exh.exhaustivity_confident is True

    def test_tracks_unmatched_areas(self):
        gt = _make_gt(100, 100, {
            1: (10, 10, 30, 30),
            2: (50, 50, 70, 70),
        })
        preds = [
            _make_pred(100, 100, 10, 10, 30, 30),
            _make_pred(100, 100, 75, 75, 90, 90),
        ]
        cr = match_candidates_to_gt(preds, gt)
        exh = compute_image_exhaustivity(preds, gt, cr)
        assert exh.num_gt_unmatched == 1
        assert exh.num_pred_unmatched == 1
        assert exh.unmatched_gt_area == 400
        assert exh.unmatched_pred_area == 225
        assert exh.total_gt_area == 800
        assert exh.total_pred_area == 625
