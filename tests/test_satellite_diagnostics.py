"""Unit tests for the raw satellite diagnostic taxonomy.

Usage:
    pytest tests/test_satellite_diagnostics.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.satellite_diagnostics import (
    DiagnosticCfg,
    TaxonomyEntry,
    aggregate_diagnostics,
    build_candidate_table,
    classify,
    classify_candidates,
)


def test_one_to_one_complete_ratio_override_is_inclusive() -> None:
    cfg = DiagnosticCfg(
        complete_one_to_one_min_completeness=0.95,
        complete_one_to_one_max_seed_ratio=3.0,
    )

    label, reason = classify(
        matched_gt_id=8,
        purity=0.48,
        completeness=0.95,
        seed_gt_ratio=3.0,
        is_one_to_one=True,
        cfg=cfg,
    )

    assert label == "compact_complete"
    assert reason == "one_to_one_complete_seed_ratio_ok"


def test_ratio_override_rejects_when_seed_gt_ratio_exceeds_cap() -> None:
    cfg = DiagnosticCfg(complete_one_to_one_max_seed_ratio=3.0)

    label, reason = classify(
        matched_gt_id=8,
        purity=0.48,
        completeness=1.0,
        seed_gt_ratio=3.01,
        is_one_to_one=True,
        cfg=cfg,
    )

    assert label == "reject_low_purity"
    assert reason == "mixed_coverage"


def test_ratio_override_requires_one_to_one() -> None:
    cfg = DiagnosticCfg(complete_one_to_one_max_seed_ratio=3.0)

    label, reason = classify(
        matched_gt_id=12,
        purity=0.49,
        completeness=0.99,
        seed_gt_ratio=2.0,
        is_one_to_one=False,
        cfg=cfg,
    )

    assert label == "reject_low_purity"
    assert reason == "mixed_coverage"


def test_ratio_override_requires_near_complete_coverage() -> None:
    cfg = DiagnosticCfg(
        complete_one_to_one_min_completeness=0.95,
        complete_one_to_one_max_seed_ratio=3.0,
    )

    label, reason = classify(
        matched_gt_id=12,
        purity=0.49,
        completeness=0.94,
        seed_gt_ratio=2.0,
        is_one_to_one=True,
        cfg=cfg,
    )

    assert label == "reject_low_purity"
    assert reason == "mixed_coverage"


def test_pure_diffuse_case_still_falls_through() -> None:
    label, reason = classify(
        matched_gt_id=2,
        purity=1.0,
        completeness=0.30,
        seed_gt_ratio=0.30,
        is_one_to_one=True,
        cfg=DiagnosticCfg(),
    )

    assert label == "diffuse_core"
    assert reason == "pure_but_core_only"


def test_build_candidate_table_emits_post_counts() -> None:
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    signal = np.zeros((16, 16), dtype=np.float32)

    raw_hit = np.zeros((16, 16), dtype=bool)
    raw_hit[4:8, 4:8] = True
    raw_fp = np.zeros((16, 16), dtype=bool)
    raw_fp[10:12, 10:12] = True

    raw_sats = [
        {
            "raw_index": 0,
            "candidate_id": "sat_0000",
            "segmentation": raw_hit,
            "score": 0.9,
            "area_clean": int(raw_hit.sum()),
        },
        {
            "raw_index": 1,
            "candidate_id": "sat_0001",
            "segmentation": raw_fp,
            "score": 0.9,
            "area_clean": int(raw_fp.sum()),
        },
    ]

    report = build_candidate_table(
        raw_sats=raw_sats,
        post_sats=[raw_sats[0]],
        gt_sat_map=gt,
        render_signal=signal,
        H=16,
        W=16,
        cfg=DiagnosticCfg(),
        roi_bbox=(0, 0, 16, 16),
    )

    assert report["counts_by_label_roi"] == {
        "compact_complete": 1,
        "diffuse_core": 0,
        "reject_unmatched": 1,
        "reject_low_purity": 0,
    }
    assert report["counts_post_by_label_roi"] == {
        "compact_complete": 1,
        "diffuse_core": 0,
        "reject_unmatched": 0,
        "reject_low_purity": 0,
    }

    summary = aggregate_diagnostics(
        [report["per_candidate"]],
        [report["counts_post_by_label"]],
        [report["counts_post_by_label_roi"]],
    )
    assert summary["roi_matched_unmatched"] == {
        "matched_raw": 1,
        "unmatched_raw": 1,
        "matched_post": 1,
        "unmatched_post": 0,
    }


def test_roi_counts_use_mask_centroid_not_edge_touch() -> None:
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    signal = np.zeros((16, 16), dtype=np.float32)

    hit = np.zeros((16, 16), dtype=bool)
    hit[4:8, 4:8] = True

    edge_touching_fp = np.zeros((16, 16), dtype=bool)
    edge_touching_fp[9:13, 2:4] = True

    raw_sats = [
        {
            "raw_index": 0,
            "candidate_id": "sat_0000",
            "segmentation": hit,
            "score": 0.9,
            "area_clean": int(hit.sum()),
        },
        {
            "raw_index": 1,
            "candidate_id": "sat_0001",
            "segmentation": edge_touching_fp,
            "score": 0.9,
            "area_clean": int(edge_touching_fp.sum()),
        },
    ]

    report = build_candidate_table(
        raw_sats=raw_sats,
        post_sats=raw_sats,
        gt_sat_map=gt,
        render_signal=signal,
        H=16,
        W=16,
        cfg=DiagnosticCfg(),
        roi_bbox=(0, 0, 10, 16),
    )

    assert report["counts_by_label"] == {
        "compact_complete": 1,
        "diffuse_core": 0,
        "reject_unmatched": 1,
        "reject_low_purity": 0,
    }
    assert report["counts_by_label_roi"] == {
        "compact_complete": 1,
        "diffuse_core": 0,
        "reject_unmatched": 0,
        "reject_low_purity": 0,
    }
    assert report["counts_post_by_label_roi"] == report["counts_by_label_roi"]


def test_classify_candidates_returns_minimal_entries_without_render_signal() -> None:
    """The shared primitive doesn't need render_signal / score /
    area_clean — only ``segmentation`` on each mask. ``raw_index`` and
    ``candidate_id`` are forwarded when present.
    """
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1  # satellite GT instance 1

    hit = np.zeros((16, 16), dtype=bool)
    hit[4:8, 4:8] = True

    miss_outside_roi = np.zeros((16, 16), dtype=bool)
    miss_outside_roi[12:14, 12:14] = True

    sats = [
        {"raw_index": 0, "candidate_id": "sat_0000", "segmentation": hit},
        # Missing raw_index/candidate_id — still classifies fine (None-forwarded).
        {"segmentation": miss_outside_roi},
    ]

    entries = classify_candidates(
        sats=sats,
        gt_sat_map=gt,
        H=16,
        W=16,
        cfg=DiagnosticCfg(),
        roi_bbox=(0, 0, 10, 10),
    )

    assert len(entries) == 2
    assert all(isinstance(e, TaxonomyEntry) for e in entries)

    assert entries[0].raw_index == 0
    assert entries[0].candidate_id == "sat_0000"
    assert entries[0].matched_gt_id == 1
    assert entries[0].taxonomy_label == "compact_complete"
    assert entries[0].intersects_roi is True

    assert entries[1].raw_index is None
    assert entries[1].candidate_id is None
    assert entries[1].matched_gt_id is None
    assert entries[1].taxonomy_label == "reject_unmatched"
    assert entries[1].intersects_roi is False
