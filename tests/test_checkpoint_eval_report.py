from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.evaluation.checkpoint_eval import Sample, aggregate, compute_sample_report


_DEFAULT_CFG = {"matching": {"iou_threshold": 0.5}, "diagnostics": {"enabled": False}}


def _sat_mask(rows: slice, cols: slice, shape: tuple[int, int] = (16, 16)) -> dict:
    seg = np.zeros(shape, dtype=bool)
    seg[rows, cols] = True
    return {"type_label": "satellites", "segmentation": seg}


def _stream_mask(rows: slice, cols: slice, shape: tuple[int, int] = (16, 16)) -> dict:
    seg = np.zeros(shape, dtype=bool)
    seg[rows, cols] = True
    return {"type_label": "streams", "segmentation": seg}


def test_fbox_gold_report_is_satellite_only() -> None:
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    sample = Sample(
        base_key="00011_eo",
        galaxy_id=11,
        view="eo",
        benchmark_mode="fbox_gold_satellites",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites"},
        roi_bbox_1024=(0, 0, 16, 16),
    )
    sat = _sat_mask(slice(4, 8), slice(4, 8))
    report, diag = compute_sample_report(sample, [sat], ([], [sat]), None, _DEFAULT_CFG)
    summary = aggregate([report])

    assert diag is None
    assert "num_gt_satellites" in report
    assert "num_gt_streams" not in report
    assert set(report["layers"]["raw"]) == {"satellites"}
    assert set(report["layers"]["post_pred_only"]) == {"satellites"}
    assert set(summary["raw"]) == {"satellites"}
    assert set(summary["post_pred_only"]) == {"satellites"}


def test_fbox_satellites_block_uses_taxonomy_schema() -> None:
    """The official satellite block MUST carry taxonomy keys only — no
    detection fields like ``tp/fp/fn/detection/per_instance/matched_iou_mean``.
    """
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    sample = Sample(
        base_key="00011_eo",
        galaxy_id=11,
        view="eo",
        benchmark_mode="fbox_gold_satellites",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites"},
        roi_bbox_1024=(0, 0, 16, 16),
    )
    # One perfect hit on GT, one false-positive elsewhere.
    sat_tp = _sat_mask(slice(4, 8), slice(4, 8))
    sat_fp = _sat_mask(slice(12, 14), slice(12, 14))
    report, _ = compute_sample_report(
        sample, [sat_tp, sat_fp], ([], [sat_tp, sat_fp]), None, _DEFAULT_CFG,
    )

    roi_block = report["layers"]["raw"]["satellites"]["roi"]
    full_block = report["layers"]["raw"]["satellites"]["full_frame"]

    expected_keys = {
        "num_gt", "num_pred",
        "matched_candidates", "unmatched_candidates",
        "unique_gt_covered", "gt_uncovered",
        "precision", "recall", "f1",
        "counts_by_label", "pixel", "is_empty_trivial",
    }
    assert set(roi_block) == expected_keys
    assert set(full_block) == expected_keys
    # No legacy detection residue.
    for leftover in ("tp", "fp", "fn", "detection", "per_instance", "matched_iou_mean"):
        assert leftover not in roi_block
        assert leftover not in full_block

    assert roi_block["num_gt"] == 1
    assert roi_block["num_pred"] == 2
    assert roi_block["matched_candidates"] == 1
    assert roi_block["unmatched_candidates"] == 1
    assert roi_block["unique_gt_covered"] == 1
    assert roi_block["gt_uncovered"] == 0
    assert roi_block["precision"] == pytest.approx(0.5)
    assert roi_block["recall"] == pytest.approx(1.0)
    assert roi_block["f1"] == pytest.approx(2 * 0.5 * 1.0 / (0.5 + 1.0))


def test_fbox_num_gt_invariant_full_eq_roi_eq_total() -> None:
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    gt[10:12, 10:12] = 2
    sample = Sample(
        base_key="00011_eo",
        galaxy_id=11,
        view="eo",
        benchmark_mode="fbox_gold_satellites",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites", 2: "satellites"},
        roi_bbox_1024=(0, 0, 16, 16),
    )
    report, _ = compute_sample_report(sample, [], ([], []), None, _DEFAULT_CFG)
    sat = report["layers"]["raw"]["satellites"]
    assert sat["full_frame"]["num_gt"] == report["num_gt_satellites"] == 2
    assert sat["roi"]["num_gt"] == report["num_gt_satellites"] == 2


def test_fbox_invariant_raises_when_roi_misses_gt() -> None:
    """GT sitting outside the ROI must raise — fbox GT is curated inside ROI."""
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[0:2, 0:2] = 1
    sample = Sample(
        base_key="00011_eo",
        galaxy_id=11,
        view="eo",
        benchmark_mode="fbox_gold_satellites",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites"},
        roi_bbox_1024=(8, 8, 16, 16),  # ROI far from the satellite.
    )
    with pytest.raises(RuntimeError, match="fbox invariant broken"):
        compute_sample_report(sample, [], ([], []), None, _DEFAULT_CFG)


def test_gt_canonical_combined_is_composite_container() -> None:
    """``combined`` in gt_canonical is per-scope ``{satellites, streams}``
    — NOT a mixed detection block.
    """
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1   # satellite
    gt[10:14, 2:6] = 2  # stream
    sample = Sample(
        base_key="gcan_001",
        galaxy_id=1,
        view="eo",
        benchmark_mode="gt_canonical",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites", 2: "streams"},
        roi_bbox_1024=None,
    )
    sat = _sat_mask(slice(4, 8), slice(4, 8))
    stream = _stream_mask(slice(10, 14), slice(2, 6))
    report, _ = compute_sample_report(
        sample, [sat, stream], ([stream], [sat]), None, _DEFAULT_CFG,
    )

    layer = report["layers"]["raw"]
    assert set(layer) == {"satellites", "streams", "combined"}

    combined = layer["combined"]
    assert set(combined) == {"full_frame", "roi"}
    assert combined["roi"] is None  # gt_canonical has no ROI.

    full = combined["full_frame"]
    assert set(full) == {"satellites", "streams"}

    # Per-type blocks are the SAME objects as the per-type layer entries.
    assert full["satellites"] is layer["satellites"]["full_frame"]
    assert full["streams"] is layer["streams"]["full_frame"]

    # Satellites side uses taxonomy fields.
    sat_block = full["satellites"]
    assert "matched_candidates" in sat_block
    assert "counts_by_label" in sat_block
    for leftover in ("tp", "fp", "fn", "detection"):
        assert leftover not in sat_block

    # Streams side uses detection fields.
    str_block = full["streams"]
    for det_key in ("tp", "fp", "fn", "detection", "per_instance", "matched_iou_mean"):
        assert det_key in str_block
    assert "matched_candidates" not in str_block
    assert "counts_by_label" not in str_block


def test_gt_canonical_summary_has_no_combined_key() -> None:
    gt = np.zeros((16, 16), dtype=np.int32)
    gt[4:8, 4:8] = 1
    gt[10:14, 2:6] = 2
    sample = Sample(
        base_key="gcan_001",
        galaxy_id=1,
        view="eo",
        benchmark_mode="gt_canonical",
        render_1024_path=Path("unused.png"),
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites", 2: "streams"},
        roi_bbox_1024=None,
    )
    sat = _sat_mask(slice(4, 8), slice(4, 8))
    stream = _stream_mask(slice(10, 14), slice(2, 6))
    report, _ = compute_sample_report(
        sample, [sat, stream], ([stream], [sat]), None, _DEFAULT_CFG,
    )
    summary = aggregate([report])
    assert set(summary["raw"]) == {"satellites", "streams"}
    assert set(summary["post_pred_only"]) == {"satellites", "streams"}
    # Taxonomy micro shape on the satellites side.
    sat_micro = summary["raw"]["satellites"]["full_frame"]["micro"]
    assert {"matched_candidates", "unmatched_candidates", "num_pred",
            "unique_gt_covered", "num_gt", "precision", "recall", "f1"} == set(sat_micro)
    # Detection micro shape on the streams side.
    str_micro = summary["raw"]["streams"]["full_frame"]["micro"]
    assert {"tp", "fp", "fn", "precision", "recall", "f1"} == set(str_micro)
