"""
Behavior-locking tests for GT resize and _merge_instances.

Pins: INTER_NEAREST preserves IDs, merge ID allocation starts at
max_stream_id + 1, overlap_policy="keep_streams", uint8 cap assertion.

Usage:
    pytest tests/test_gt_phase.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_unified_dataset import _merge_instances


class TestGtResizePreservesIds:
    """GT resize with INTER_NEAREST must not introduce new ID values."""

    def test_resize_preserves_ids(self):
        original = np.zeros((200, 200), dtype=np.int32)
        original[10:50, 10:50] = 1
        original[60:90, 60:90] = 2
        original[120:180, 120:180] = 5

        target_size = (128, 128)
        resized = cv2.resize(original, target_size, interpolation=cv2.INTER_NEAREST)

        original_ids = set(np.unique(original))
        resized_ids = set(np.unique(resized))
        assert resized_ids.issubset(original_ids), (
            f"Resize introduced new IDs: {resized_ids - original_ids}"
        )

    def test_resize_no_interpolation_artifacts(self):
        original = np.zeros((256, 256), dtype=np.int32)
        original[0:128, 0:128] = 3
        original[128:256, 128:256] = 7

        resized = cv2.resize(original, (64, 64), interpolation=cv2.INTER_NEAREST)
        assert set(np.unique(resized)) == {0, 3, 7}


class TestMergeInstances:
    def _make_streams(self, H=64, W=64):
        m = np.zeros((H, W), dtype=np.int32)
        m[5:15, 5:15] = 1
        m[20:30, 20:30] = 2
        return m

    def test_new_ids_start_at_max_stream_plus_one(self):
        streams = self._make_streams()
        max_stream_id = 2
        masks = [
            {"segmentation": np.ones((64, 64), dtype=bool) * False},
        ]
        masks[0]["segmentation"][40:50, 40:50] = True

        instance_map, instances, id_map, _ = _merge_instances(
            streams, masks, max_stream_id, "keep_streams"
        )
        new_ids = [inst["id"] for inst in instances if inst["type"] != "streams"]
        assert new_ids == [3]

    def test_multiple_inferred_masks_sequential_ids(self):
        streams = self._make_streams()
        max_stream_id = 2
        masks = []
        for i in range(3):
            seg = np.zeros((64, 64), dtype=bool)
            seg[40 + i * 5 : 45 + i * 5, 40:50] = True
            masks.append({"segmentation": seg, "type_label": "satellites"})

        instance_map, instances, id_map, _ = _merge_instances(
            streams, masks, max_stream_id, "keep_streams"
        )
        new_ids = sorted(
            inst["id"] for inst in instances if inst["type"] == "satellites"
        )
        assert new_ids == [3, 4, 5]

    def test_keep_streams_policy(self):
        streams = self._make_streams()
        max_stream_id = 2

        seg = np.zeros((64, 64), dtype=bool)
        seg[5:15, 5:15] = True  # overlaps stream ID=1
        masks = [{"segmentation": seg, "type_label": "satellites"}]

        instance_map, _, _, overlap_stats = _merge_instances(
            streams, masks, max_stream_id, "keep_streams"
        )
        assert overlap_stats["overlap_px"] > 0
        # Stream pixels should remain stream ID, not overwritten
        assert np.all(instance_map[5:15, 5:15] == 1)

    def test_overwrite_policy(self):
        streams = self._make_streams()
        max_stream_id = 2

        seg = np.zeros((64, 64), dtype=bool)
        seg[5:15, 5:15] = True  # overlaps stream ID=1
        masks = [{"segmentation": seg, "type_label": "satellites"}]

        instance_map, _, _, _ = _merge_instances(
            streams, masks, max_stream_id, "overwrite"
        )
        assert np.any(instance_map[5:15, 5:15] == 3)

    def test_uint8_cap_assertion(self):
        streams = np.zeros((64, 64), dtype=np.int32)
        max_stream_id = 250
        masks = []
        for i in range(10):
            seg = np.zeros((64, 64), dtype=bool)
            seg[i, 0] = True
            masks.append({"segmentation": seg})

        with pytest.raises(AssertionError, match="Too many instances"):
            _merge_instances(streams, masks, max_stream_id, "keep_streams")

    def test_output_dtype_is_uint8(self):
        streams = self._make_streams()
        instance_map, _, _, _ = _merge_instances(
            streams, [], 2, "keep_streams"
        )
        assert instance_map.dtype == np.uint8

    def test_empty_inferred_masks(self):
        streams = self._make_streams()
        instance_map, instances, id_map, overlap_stats = _merge_instances(
            streams, [], 2, "keep_streams"
        )
        assert len(instances) == 2  # only GT streams
        assert overlap_stats["overlap_px"] == 0
        assert overlap_stats["overlap_rate"] == 0.0

    def test_type_label_preserved(self):
        streams = self._make_streams()
        seg = np.zeros((64, 64), dtype=bool)
        seg[40:50, 40:50] = True
        masks = [{"segmentation": seg, "type_label": "satellites"}]

        _, instances, _, _ = _merge_instances(streams, masks, 2, "keep_streams")
        sat_instances = [i for i in instances if i["type"] == "satellites"]
        assert len(sat_instances) == 1
