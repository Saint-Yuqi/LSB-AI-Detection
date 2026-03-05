"""
Unit tests for type-aware SAM3 evaluation.

Tests cover: GT discovery priority matrix, per-type metric splits,
class-agnostic combined matching, save_results detail stripping,
dynamic aggregate, and legacy schema compat in visualization.

Usage:
    pytest tests/test_eval_type_aware.py -v
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.sam3_eval import (
    discover_pairs,
    _load_gt_by_type,
    _compute_slice_metrics,
    _strip_details,
    _aggregate_group,
    save_results,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_gt_dir(tmp: Path, base_key: str, *, full_mode: bool = True,
                 n_streams: int = 3, n_satellites: int = 2):
    """Create a GT directory with instance_map + instances.json."""
    gt_dir = tmp / base_key
    gt_dir.mkdir(parents=True)

    H, W = 64, 64
    instance_map = np.zeros((H, W), dtype=np.uint8)

    instances = []
    gid = 1

    # Streams: horizontal bands
    for i in range(n_streams):
        y_start = i * 8
        instance_map[y_start:y_start + 5, 10:50] = gid
        instances.append({"id": gid, "type": "streams"})
        gid += 1

    # Satellites: small squares
    for i in range(n_satellites):
        y_start = 40 + i * 8
        instance_map[y_start:y_start + 5, 10:20] = gid
        instances.append({"id": gid, "type": "satellites"})
        gid += 1

    if full_mode:
        Image.fromarray(instance_map).save(gt_dir / "instance_map_uint8.png")
        (gt_dir / "instances.json").write_text(json.dumps(instances))
    else:
        # Legacy mode: streams_instance_map.npy only
        streams_map = np.where(
            np.isin(instance_map, [i["id"] for i in instances if i["type"] == "streams"]),
            instance_map, 0
        )
        np.save(gt_dir / "streams_instance_map.npy", streams_map)


def _make_render_dir(tmp: Path, base_key: str):
    """Create a render directory with a dummy 0000.png."""
    render_dir = tmp / base_key
    render_dir.mkdir(parents=True)
    img = Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8))
    img.save(render_dir / "0000.png")


def _make_pred_mask(H: int, W: int, y0: int, y1: int, x0: int, x1: int,
                    type_label: str = "streams") -> dict:
    """Create a prediction mask dict."""
    seg = np.zeros((H, W), dtype=bool)
    seg[y0:y1, x0:x1] = True
    return {"segmentation": seg, "type_label": type_label, "confidence": 0.9}


# ---------------------------------------------------------------------------
# Discovery Tests
# ---------------------------------------------------------------------------

class TestDiscoverPairs:
    def test_full_mode(self, tmp_path):
        render_root = tmp_path / "renders"
        gt_root = tmp_path / "gt"
        _make_render_dir(render_root, "00011_eo")
        _make_gt_dir(gt_root, "00011_eo", full_mode=True)

        pairs = discover_pairs(render_root, gt_root)
        assert len(pairs) == 1
        assert pairs[0]["gt_mode"] == "full"
        assert pairs[0]["gt_instances_path"] is not None

    def test_streams_only_fallback(self, tmp_path):
        render_root = tmp_path / "renders"
        gt_root = tmp_path / "gt"
        _make_render_dir(render_root, "00011_eo")
        _make_gt_dir(gt_root, "00011_eo", full_mode=False)

        pairs = discover_pairs(render_root, gt_root)
        assert len(pairs) == 1
        assert pairs[0]["gt_mode"] == "streams_only"
        assert pairs[0]["gt_instances_path"] is None

    def test_corrupt_gt_skip(self, tmp_path):
        """instances.json present but instance_map_uint8.png missing → skip."""
        render_root = tmp_path / "renders"
        gt_root = tmp_path / "gt"
        _make_render_dir(render_root, "00011_eo")

        gt_dir = gt_root / "00011_eo"
        gt_dir.mkdir(parents=True)
        (gt_dir / "instances.json").write_text('[{"id":1,"type":"streams"}]')
        # No instance_map_uint8.png!

        pairs = discover_pairs(render_root, gt_root)
        assert len(pairs) == 0

    def test_map_without_json_skip(self, tmp_path):
        """instance_map_uint8.png present but instances.json missing → skip."""
        render_root = tmp_path / "renders"
        gt_root = tmp_path / "gt"
        _make_render_dir(render_root, "00011_eo")

        gt_dir = gt_root / "00011_eo"
        gt_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((64, 64), dtype=np.uint8)).save(
            gt_dir / "instance_map_uint8.png")
        # No instances.json!

        pairs = discover_pairs(render_root, gt_root)
        assert len(pairs) == 0


# ---------------------------------------------------------------------------
# GT Type Split Tests
# ---------------------------------------------------------------------------

class TestLoadGtByType:
    def test_full_mode_split(self, tmp_path):
        gt_root = tmp_path / "gt"
        _make_gt_dir(gt_root, "00011_eo", full_mode=True, n_streams=3, n_satellites=2)

        pair = {
            "gt_mode": "full",
            "gt_path": gt_root / "00011_eo" / "instance_map_uint8.png",
            "gt_instances_path": gt_root / "00011_eo" / "instances.json",
            "base_key": "00011_eo",
        }
        gt_streams, gt_satellites, gt_all = _load_gt_by_type(pair)

        # Stream IDs: 1,2,3 — satellite IDs: 4,5
        assert set(np.unique(gt_streams)) - {0} == {1, 2, 3}
        assert set(np.unique(gt_satellites)) - {0} == {4, 5}
        assert set(np.unique(gt_all)) - {0} == {1, 2, 3, 4, 5}

    def test_legacy_fallback(self, tmp_path):
        gt_root = tmp_path / "gt"
        _make_gt_dir(gt_root, "00011_eo", full_mode=False, n_streams=3, n_satellites=2)

        pair = {
            "gt_mode": "streams_only",
            "gt_path": gt_root / "00011_eo" / "streams_instance_map.npy",
            "gt_instances_path": None,
            "base_key": "00011_eo",
        }
        gt_streams, gt_satellites, gt_all = _load_gt_by_type(pair)

        assert len(set(np.unique(gt_streams)) - {0}) == 3
        assert np.all(gt_satellites == 0)  # No satellite GT in legacy
        np.testing.assert_array_equal(gt_all, gt_streams)


# ---------------------------------------------------------------------------
# Metric Slice Tests
# ---------------------------------------------------------------------------

class TestComputeSliceMetrics:
    def test_no_satellite_gt(self):
        """GT has 0 satellites → instance_recall = None (per metrics.py L391)."""
        H, W = 64, 64
        gt_empty = np.zeros((H, W), dtype=np.int32)
        preds = [_make_pred_mask(H, W, 0, 10, 0, 10, "satellites")]

        result = _compute_slice_metrics(preds, gt_empty, H, W, 0.5)
        assert result["num_gt"] == 0
        assert result["instance_recall"] is None

    def test_satellite_gt_no_preds(self):
        """GT has satellites, 0 preds → instance_recall = 0.0 (per L414)."""
        H, W = 64, 64
        gt = np.zeros((H, W), dtype=np.int32)
        gt[10:20, 10:20] = 1  # one GT instance

        result = _compute_slice_metrics([], gt, H, W, 0.5)
        assert result["num_gt"] == 1
        assert result["instance_recall"] == 0.0
        assert result["num_detected"] == 0

    def test_combined_includes_all_types(self):
        """Combined metrics include ALL pred masks regardless of type_label."""
        H, W = 64, 64
        gt = np.zeros((H, W), dtype=np.int32)
        gt[0:10, 0:10] = 1

        # Pred with unknown type_label
        debris_mask = _make_pred_mask(H, W, 0, 10, 0, 10, "debris")

        result = _compute_slice_metrics([debris_mask], gt, H, W, 0.5)
        assert result["num_pred"] == 1
        assert result["dice"] > 0.9  # Good overlap


# ---------------------------------------------------------------------------
# save_results / _strip_details Tests
# ---------------------------------------------------------------------------

class TestStripDetails:
    def test_recursive_dict(self):
        obj = {
            "streams": {
                "raw": {"dice": 0.8, "per_instance_details": [1, 2, 3]},
                "post": {"dice": 0.7, "per_instance_details": [4, 5]},
            }
        }
        result = _strip_details(obj)
        assert "per_instance_details" not in result["streams"]["raw"]
        assert "per_instance_details" not in result["streams"]["post"]
        assert result["streams"]["raw"]["dice"] == 0.8

    def test_recursive_list(self):
        obj = [
            {"per_instance_details": [1], "dice": 0.5},
            {"nested": {"per_instance_details": [2], "ok": True}},
        ]
        result = _strip_details(obj)
        assert "per_instance_details" not in result[0]
        assert "per_instance_details" not in result[1]["nested"]
        assert result[0]["dice"] == 0.5
        assert result[1]["nested"]["ok"] is True

    def test_arbitrary_depth(self):
        obj = {"a": {"b": {"c": {"per_instance_details": [], "val": 42}}}}
        result = _strip_details(obj)
        assert "per_instance_details" not in result["a"]["b"]["c"]
        assert result["a"]["b"]["c"]["val"] == 42


# ---------------------------------------------------------------------------
# Aggregation Tests
# ---------------------------------------------------------------------------

class TestAggregateGroup:
    def test_dynamic_type_inference(self):
        """Type keys are inferred dynamically, not hardcoded."""
        images = [
            {
                "streams": {"raw": {"dice": 0.8, "tp": 10, "fp": 2, "fn": 1},
                            "post": {"dice": 0.7, "tp": 9, "fp": 1, "fn": 2}},
                "custom_type": {"raw": {"dice": 0.5, "tp": 5, "fp": 3, "fn": 2},
                                "post": {"dice": 0.4, "tp": 4, "fp": 2, "fn": 3}},
                "galaxy_id": 11,
            }
        ]
        result = _aggregate_group(images)
        assert "streams" in result
        assert "custom_type" in result
        assert "raw" in result["streams"]
        assert "post" in result["custom_type"]

    def test_per_galaxy_structure(self):
        """Verify per-galaxy aggregation produces per-type metrics."""
        from src.evaluation.sam3_eval import aggregate_results

        images = [
            {
                "galaxy_id": 11,
                "streams": {"raw": {"dice": 0.8, "tp": 10, "fp": 2, "fn": 1},
                            "post": {"dice": 0.7, "tp": 9, "fp": 1, "fn": 2}},
                "satellites": {"raw": {"dice": 0.6, "tp": 6, "fp": 4, "fn": 3},
                               "post": {"dice": 0.6, "tp": 6, "fp": 4, "fn": 3}},
                "combined": {"raw": {"dice": 0.7, "tp": 16, "fp": 6, "fn": 4},
                             "post": {"dice": 0.65, "tp": 15, "fp": 5, "fn": 5}},
            }
        ]
        result = aggregate_results(images, group_by="galaxy")
        assert "11" in result
        assert "streams" in result["11"]
        assert "satellites" in result["11"]
        assert "combined" in result["11"]
        assert "raw" in result["11"]["streams"]
        assert "post" in result["11"]["satellites"]


# ---------------------------------------------------------------------------
# Visualization Legacy Compat Tests
# ---------------------------------------------------------------------------

class TestVisualizeLegacyCompat:
    def test_legacy_combined_empty(self):
        """Legacy JSON → combined = empty dict, not streams proxy."""
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

        # Inline the extraction function to test it
        from visualize_eval_metrics import _extract_all_metrics

        legacy_doc = {
            "summary": {
                "overall": {
                    "raw": {"macro_mean_dice": 0.8},
                    "post": {"macro_mean_dice": 0.7},
                    "n_images": 10,
                }
            }
        }
        result = _extract_all_metrics(legacy_doc)

        # Streams should have data
        assert result["streams"]["raw"]["macro_mean_dice"] == 0.8

        # Combined should be EMPTY (not a proxy for streams)
        assert result["combined"]["raw"] == {}
        assert result["combined"]["post"] == {}

        # Satellites should be EMPTY
        assert result["satellites"]["raw"] == {}
