"""
Tests for silver labeler: no minor_fix/redraw in output, abstain on ambiguous,
multi-signal policy, uses candidate_matcher not calculate_matched_metrics.

Usage:
    pytest tests/test_silver_labeler.py -v
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
)
from src.review.silver_labeler import (
    SilverLabel,
    SilverPolicyConfig,
    _sort_instances_by_area,
    label_satellite_ev,
    label_satellite_ev_gt,
    label_satellite_mv_authoritative_candidates,
    label_satellite_mv_candidates,
    label_stream_ev,
    label_stream_ev_gt,
)


@pytest.fixture
def policy():
    return SilverPolicyConfig()


def _cr(
    best_iou: float = 0.0,
    second_best_iou: float = 0.0,
    area_ratio: float | None = None,
    boundary_f1: float | None = None,
    overlaps_multi: bool = False,
    confidence: float = 0.9,
    filter_status: str = "kept",
    pred_idx: int = 0,
) -> CandidateMatchResult:
    return CandidateMatchResult(
        pred_idx=pred_idx,
        best_gt_id=1 if best_iou > 0 else None,
        best_iou=best_iou,
        second_best_iou=second_best_iou,
        area_ratio=area_ratio,
        boundary_f1=boundary_f1,
        overlaps_multiple_gt=overlaps_multi,
        confidence_score=confidence,
        filter_status=filter_status,
    )


def _exh(confident: bool = True) -> ImageExhaustivityResult:
    return ImageExhaustivityResult(
        num_gt=2, num_pred_matched=2, num_gt_unmatched=0,
        num_pred_unmatched=0, mean_matched_iou=0.9,
        exhaustivity_confident=confident,
    )


class TestSatelliteMvSilver:
    def test_accept_high_quality(self, policy):
        cr = _cr(best_iou=0.85, area_ratio=1.0, boundary_f1=0.6, confidence=0.8)
        labels = label_satellite_mv_candidates("s", [cr], _exh(), policy)
        assert len(labels) == 1
        assert labels[0].decision_label == "accept"

    def test_reject_low_iou(self, policy):
        cr = _cr(best_iou=0.02)
        labels = label_satellite_mv_candidates("s", [cr], _exh(), policy)
        assert len(labels) == 1
        assert labels[0].decision_label == "reject"

    def test_reject_prior_filtered(self, policy):
        cr = _cr(best_iou=0.8, filter_status="prior_rejected",
                 area_ratio=1.0, boundary_f1=0.5)
        labels = label_satellite_mv_candidates("s", [cr], _exh(), policy)
        assert len(labels) == 1
        assert labels[0].decision_label == "reject"

    def test_reject_core_filtered(self, policy):
        cr = _cr(best_iou=0.8, filter_status="core_rejected",
                 area_ratio=1.0, boundary_f1=0.5)
        labels = label_satellite_mv_candidates("s", [cr], _exh(), policy)
        assert labels[0].decision_label == "reject"

    def test_no_minor_fix(self, policy):
        candidates = [
            _cr(best_iou=0.85, area_ratio=1.0, boundary_f1=0.6,
                confidence=0.8, pred_idx=i)
            for i in range(5)
        ]
        labels = label_satellite_mv_candidates("s", candidates, _exh(), policy)
        for lab in labels:
            assert lab.decision_label != "minor_fix"

    def test_no_redraw(self, policy):
        candidates = [_cr(best_iou=v) for v in (0.01, 0.05, 0.5, 0.9)]
        labels = label_satellite_mv_candidates("s", candidates, _exh(), policy)
        for lab in labels:
            assert lab.decision_label != "redraw"

    def test_abstain_on_ambiguous(self, policy):
        cr = _cr(best_iou=0.5, second_best_iou=0.45, area_ratio=1.0,
                 boundary_f1=0.4, overlaps_multi=True)
        labels = label_satellite_mv_candidates("s", [cr], _exh(), policy)
        assert len(labels) == 0

    def test_route_to_ev_when_uncertain(self, policy):
        cr = _cr(best_iou=0.4, area_ratio=1.0, boundary_f1=0.4, confidence=0.6)
        labels = label_satellite_mv_candidates(
            "s", [cr], _exh(confident=False), policy,
        )
        assert len(labels) == 1
        assert labels[0].decision_label == "route_to_ev"

    def test_authoritative_keys_use_gt_instance_ids(self, policy):
        cr = _cr(
            best_iou=0.85,
            area_ratio=1.0,
            boundary_f1=0.6,
            confidence=0.8,
        )
        cr.best_gt_id = 7

        labels = label_satellite_mv_authoritative_candidates(
            "s", [cr], _exh(), policy,
        )

        assert len(labels) == 1
        assert labels[0].candidate_key == "inst_007"
        assert labels[0].decision_label == "accept"

    def test_authoritative_prefers_accept_over_route_to_ev(self, policy):
        accept = _cr(
            best_iou=0.88,
            area_ratio=1.0,
            boundary_f1=0.6,
            confidence=0.8,
            pred_idx=0,
        )
        accept.best_gt_id = 3
        routed = _cr(
            best_iou=0.4,
            area_ratio=1.0,
            boundary_f1=0.4,
            confidence=0.6,
            pred_idx=1,
        )
        routed.best_gt_id = 3

        labels = label_satellite_mv_authoritative_candidates(
            "s", [accept, routed], _exh(confident=False), policy,
        )

        assert len(labels) == 1
        assert labels[0].candidate_key == "inst_003"
        assert labels[0].decision_label == "accept"

    def test_authoritative_skips_reject_only_matches(self, policy):
        cr = _cr(best_iou=0.01)
        cr.best_gt_id = 4

        labels = label_satellite_mv_authoritative_candidates(
            "s", [cr], _exh(), policy,
        )

        assert labels == []


@pytest.mark.skip(reason="deprecated GT migration")
class TestSatelliteEvSilver:
    def test_confirm_complete(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=3, num_pred_matched=3, num_gt_unmatched=0,
            num_pred_unmatched=0, mean_matched_iou=0.9,
            exhaustivity_confident=True,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "confirm_complete"

    def test_confirm_empty(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=0, num_pred_matched=0, num_gt_unmatched=0,
            num_pred_unmatched=0, mean_matched_iou=0.0,
            exhaustivity_confident=True,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "confirm_empty"

    def test_no_loose_confirm_complete(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=3, num_pred_matched=3, num_gt_unmatched=0,
            num_pred_unmatched=1, mean_matched_iou=0.98,
            exhaustivity_confident=True,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "remove_fp"

    def test_mixed_fp_dominant_resolves_remove_fp(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=5, num_pred_matched=4, num_gt_unmatched=1,
            num_pred_unmatched=1, mean_matched_iou=0.95,
            exhaustivity_confident=False,
            total_gt_area=1000,
            total_pred_area=1000,
            unmatched_gt_area=10,
            unmatched_pred_area=120,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "remove_fp"

    def test_mixed_fn_dominant_resolves_add_missing(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=5, num_pred_matched=4, num_gt_unmatched=1,
            num_pred_unmatched=1, mean_matched_iou=0.95,
            exhaustivity_confident=False,
            total_gt_area=1000,
            total_pred_area=1000,
            unmatched_gt_area=150,
            unmatched_pred_area=20,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "add_missing"

    def test_balanced_mixed_case_still_abstains(self, policy):
        exh = ImageExhaustivityResult(
            num_gt=5, num_pred_matched=4, num_gt_unmatched=1,
            num_pred_unmatched=1, mean_matched_iou=0.95,
            exhaustivity_confident=False,
            total_gt_area=1000,
            total_pred_area=1000,
            unmatched_gt_area=60,
            unmatched_pred_area=80,
        )
        lab = label_satellite_ev("s", exh, policy)
        assert lab is None

    def test_no_redraw(self, policy):
        for exh in [
            _exh(True), _exh(False),
            ImageExhaustivityResult(1, 0, 1, 0, 0.0, False),
        ]:
            lab = label_satellite_ev("s", exh, policy)
            if lab is not None:
                assert lab.decision_label != "redraw"


@pytest.mark.skip(reason="deprecated GT migration")
class TestStreamEvSilver:
    def test_no_redraw(self, policy):
        exh = _exh(True)
        lab = label_stream_ev("s", exh, policy)
        if lab is not None:
            assert lab.decision_label != "redraw"

    def test_confirm_empty(self, policy):
        exh = ImageExhaustivityResult(0, 0, 0, 0, 0.0, True)
        with pytest.warns(DeprecationWarning):
            lab = label_stream_ev("s", exh, policy)
        assert lab is not None
        assert lab.decision_label == "confirm_empty"


# ---------------------------------------------------------------------------
#  GT-driven satellite EV tests
# ---------------------------------------------------------------------------

class TestSortInstancesByArea:
    def test_sorted_descending_by_area(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[0:10, 0:10] = 1    # area = 100
        imap[0:20, 50:70] = 2   # area = 400
        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "satellites"},
        ]
        result = _sort_instances_by_area(imap, instances, "satellites")
        assert result[0]["id"] == 2  # larger first
        assert result[1]["id"] == 1

    def test_ties_broken_by_id(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[0:10, 0:10] = 3    # area = 100
        imap[50:60, 50:60] = 1  # area = 100
        instances = [
            {"id": 3, "type": "satellites"},
            {"id": 1, "type": "satellites"},
        ]
        result = _sort_instances_by_area(imap, instances, "satellites")
        # Same area → ascending id
        assert result[0]["id"] == 1
        assert result[1]["id"] == 3

    def test_filters_by_type(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[0:10, 0:10] = 1
        imap[50:60, 50:60] = 2
        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "streams"},
        ]
        result = _sort_instances_by_area(imap, instances, "satellites")
        assert len(result) == 1
        assert result[0]["id"] == 1


class TestSatelliteEvGtDriven:
    def test_empty_field(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        instances: list[dict] = []
        labels = label_satellite_ev_gt("s", imap, instances)
        assert len(labels) == 1
        assert labels[0].decision_label == "confirm_empty"
        assert labels[0].candidate_key == "image:gt_empty"

    def test_single_satellite_produces_two_variants(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        instances = [{"id": 1, "type": "satellites"}]
        labels = label_satellite_ev_gt("s", imap, instances)
        # gt_complete + drop_top1
        assert len(labels) == 2
        keys = {l.candidate_key for l in labels}
        assert "image:gt_complete" in keys
        assert "image:drop_top1" in keys

    def test_two_satellites_produces_two_variants(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1  # area 400
        imap[50:80, 50:80] = 2  # area 900
        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "satellites"},
        ]
        labels = label_satellite_ev_gt("s", imap, instances)
        assert len(labels) == 2  # gt_complete, drop_top2
        keys = {l.candidate_key for l in labels}
        assert "image:drop_top2" in keys

    def test_hides_only_top_three_satellites(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[0:10, 0:10] = 1       # area 100
        imap[0:20, 20:40] = 2      # area 400
        imap[40:70, 40:70] = 3     # area 900
        imap[80:85, 80:85] = 4     # area 25
        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "satellites"},
            {"id": 3, "type": "satellites"},
            {"id": 4, "type": "satellites"},
        ]
        labels = label_satellite_ev_gt("s", imap, instances)
        label_map = {l.candidate_key: l for l in labels}
        drop = label_map["image:drop_top3"]
        assert drop.signals["hidden_instance_ids"] == [3, 2, 1]
        assert drop.signals["visible_instance_ids"] == [4]

    def test_visible_hidden_disjoint(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        imap[50:80, 50:80] = 2
        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "satellites"},
        ]
        labels = label_satellite_ev_gt("s", imap, instances)
        for lab in labels:
            vis = set(lab.signals["visible_instance_ids"])
            hid = set(lab.signals["hidden_instance_ids"])
            assert vis & hid == set(), f"Overlap in {lab.candidate_key}"

    def test_decision_labels(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        instances = [{"id": 1, "type": "satellites"}]
        labels = label_satellite_ev_gt("s", imap, instances)
        label_map = {l.candidate_key: l.decision_label for l in labels}
        assert label_map["image:gt_complete"] == "confirm_complete"
        assert label_map["image:drop_top1"] == "add_missing"

    def test_streams_ignored(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1  # stream
        instances = [{"id": 1, "type": "streams"}]
        labels = label_satellite_ev_gt("s", imap, instances)
        assert len(labels) == 1
        assert labels[0].decision_label == "confirm_empty"


class TestStreamEvGtDriven:
    def test_empty_field(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        instances: list[dict] = []
        labels = label_stream_ev_gt("s", imap, instances)
        assert len(labels) == 1
        assert labels[0].decision_label == "confirm_empty"

    def test_single_stream_produces_two_variants(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        instances = [{"id": 1, "type": "streams"}]
        labels = label_stream_ev_gt("s", imap, instances)
        assert len(labels) == 2
        label_map = {l.candidate_key: l.decision_label for l in labels}
        assert label_map["image:gt_complete"] == "confirm_complete"
        assert label_map["image:drop_top1"] == "add_missing_fragment"

    def test_two_streams_still_produce_one_missing_variant(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1   # area 400
        imap[40:80, 40:80] = 2   # area 1600
        instances = [
            {"id": 1, "type": "streams"},
            {"id": 2, "type": "streams"},
        ]
        labels = label_stream_ev_gt("s", imap, instances)
        assert len(labels) == 2
        label_map = {l.candidate_key: l for l in labels}
        drop = label_map["image:drop_top1"]
        assert drop.signals["hidden_instance_ids"] == [2]
        assert drop.signals["visible_instance_ids"] == [1]

    def test_visible_hidden_disjoint(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        imap[50:80, 50:80] = 2
        instances = [
            {"id": 1, "type": "streams"},
            {"id": 2, "type": "streams"},
        ]
        labels = label_stream_ev_gt("s", imap, instances)
        for lab in labels:
            vis = set(lab.signals["visible_instance_ids"])
            hid = set(lab.signals["hidden_instance_ids"])
            assert vis & hid == set()

    def test_satellites_ignored(self):
        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[10:30, 10:30] = 1
        instances = [{"id": 1, "type": "satellites"}]
        labels = label_stream_ev_gt("s", imap, instances)
        assert len(labels) == 1
        assert labels[0].decision_label == "confirm_empty"
