"""
Behavior-locking tests for artifact serialization and export phase logic.

Pins: _save_predictions_json schema, export bbox correctness via
coco_utils.get_bbox_from_mask, COCO annotation count with skip conditions.

Usage:
    pytest tests/test_artifacts.py -v
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_unified_dataset import _save_predictions_json
from src.utils.coco_utils import get_bbox_from_mask


class TestSavePredictionsJsonSchema:
    def test_required_top_level_keys(self, tmp_path):
        masks = [
            {
                "segmentation": np.zeros((64, 64), dtype=bool),
                "type_label": "streams",
                "predicted_iou": 0.9,
                "area": 100,
                "bbox": [10, 10, 20, 20],
            }
        ]
        masks[0]["segmentation"][10:20, 10:20] = True

        out = tmp_path / "preds.json"
        _save_predictions_json(out, masks, H_work=64, W_work=64, engine="sam3", layer="raw")

        doc = json.loads(out.read_text())
        assert "schema_version" in doc
        assert doc["schema_version"] == 1
        assert "rle_convention" in doc
        assert doc["rle_convention"] == "coco_rle_fortran"
        assert "predictions" in doc
        assert "H_work" in doc
        assert "W_work" in doc
        assert "engine" in doc
        assert "layer" in doc
        assert "n_predictions" in doc
        assert "created_at" in doc

    def test_prediction_entry_keys(self, tmp_path):
        seg = np.zeros((64, 64), dtype=bool)
        seg[5:15, 5:15] = True
        masks = [
            {
                "segmentation": seg,
                "type_label": "satellites",
                "predicted_iou": 0.85,
                "area": 100,
                "bbox": [5, 5, 10, 10],
            }
        ]

        out = tmp_path / "preds.json"
        _save_predictions_json(out, masks, 64, 64)

        doc = json.loads(out.read_text())
        pred = doc["predictions"][0]
        assert "type_label" in pred
        assert "score" in pred
        assert "area" in pred
        assert "bbox_xywh" in pred
        assert "rle" in pred

    def test_empty_masks(self, tmp_path):
        out = tmp_path / "preds.json"
        _save_predictions_json(out, [], 64, 64)

        doc = json.loads(out.read_text())
        assert doc["n_predictions"] == 0
        assert doc["predictions"] == []

    def test_skip_none_segmentation(self, tmp_path):
        masks = [{"segmentation": None, "type_label": "streams"}]

        out = tmp_path / "preds.json"
        _save_predictions_json(out, masks, 64, 64)

        doc = json.loads(out.read_text())
        assert doc["n_predictions"] == 0


class TestExportBboxCorrectness:
    """Test bbox via coco_utils.get_bbox_from_mask (the target replacement)."""

    def test_simple_rect(self):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 1
        bbox = get_bbox_from_mask(mask)
        # [x, y, w, h]
        assert bbox == [30.0, 20.0, 30.0, 20.0]

    def test_single_pixel(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10, 20] = 1
        bbox = get_bbox_from_mask(mask)
        assert bbox == [20.0, 10.0, 1.0, 1.0]

    def test_empty_mask(self):
        mask = np.zeros((50, 50), dtype=np.uint8)
        bbox = get_bbox_from_mask(mask)
        assert bbox == [0.0, 0.0, 0.0, 0.0]


class TestCocoAnnotationCounts:
    """Verify export skip conditions: missing renders and zero-area masks."""

    def _build_export_fixture(self, tmp_path, *, skip_render=False, add_empty_mask=False):
        """Build a minimal filesystem fixture matching export phase expectations."""
        from PIL import Image
        from prepare_unified_dataset import BaseKey, VariantKey

        output_root = tmp_path / "output"
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        gt_dir.mkdir(parents=True)

        H, W = 64, 64
        instance_map = np.zeros((H, W), dtype=np.uint8)
        instance_map[5:15, 5:15] = 1  # stream
        instance_map[30:40, 30:40] = 2  # satellite
        if add_empty_mask:
            pass  # ID 3 listed but zero pixels

        Image.fromarray(instance_map).save(gt_dir / "instance_map_uint8.png")

        instances = [
            {"id": 1, "type": "streams"},
            {"id": 2, "type": "satellites"},
        ]
        if add_empty_mask:
            instances.append({"id": 3, "type": "satellites"})
        (gt_dir / "instances.json").write_text(json.dumps(instances))

        variants = [{"name": "asinh_stretch"}, {"name": "linear_magnitude"}]

        for v in variants:
            render_dir = (
                output_root / "renders" / "current" / v["name"] / str(key)
            )
            if skip_render and v["name"] == "linear_magnitude":
                continue  # simulate missing render
            render_dir.mkdir(parents=True)
            img = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
            img.save(render_dir / "0000.png")

        return output_root, [key], variants, H, W

    def test_full_export_counts(self, tmp_path):
        output_root, keys, variants, H, W = self._build_export_fixture(tmp_path)
        from prepare_unified_dataset import run_export_phase
        import logging

        config = {
            "paths": {"firebox_root": "/dummy", "output_root": str(output_root)},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [H, W]},
            "preprocessing_variants": variants,
            "data_selection": {"galaxy_ids": [11], "orientations": ["eo"]},
        }
        logger = logging.getLogger("test")
        run_export_phase(config, keys, logger)

        coco = json.loads(
            (output_root / "sam3_prepared" / "annotations.json").read_text()
        )
        assert len(coco["images"]) == 2  # 1 key x 2 variants
        assert len(coco["annotations"]) == 4  # 2 instances x 2 variants

    def test_missing_render_skipped(self, tmp_path):
        output_root, keys, variants, H, W = self._build_export_fixture(
            tmp_path, skip_render=True
        )
        from prepare_unified_dataset import run_export_phase
        import logging

        config = {
            "paths": {"firebox_root": "/dummy", "output_root": str(output_root)},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [H, W]},
            "preprocessing_variants": variants,
            "data_selection": {"galaxy_ids": [11], "orientations": ["eo"]},
        }
        logger = logging.getLogger("test")
        run_export_phase(config, keys, logger)

        coco = json.loads(
            (output_root / "sam3_prepared" / "annotations.json").read_text()
        )
        assert len(coco["images"]) == 1  # only asinh_stretch
        assert len(coco["annotations"]) == 2  # 2 instances x 1 variant

    def test_zero_area_mask_skipped(self, tmp_path):
        output_root, keys, variants, H, W = self._build_export_fixture(
            tmp_path, add_empty_mask=True
        )
        from prepare_unified_dataset import run_export_phase
        import logging

        config = {
            "paths": {"firebox_root": "/dummy", "output_root": str(output_root)},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [H, W]},
            "preprocessing_variants": variants,
            "data_selection": {"galaxy_ids": [11], "orientations": ["eo"]},
        }
        logger = logging.getLogger("test")
        run_export_phase(config, keys, logger)

        coco = json.loads(
            (output_root / "sam3_prepared" / "annotations.json").read_text()
        )
        # ID 3 has 0 pixels, should be skipped
        assert len(coco["annotations"]) == 4  # 2 real instances x 2 variants
