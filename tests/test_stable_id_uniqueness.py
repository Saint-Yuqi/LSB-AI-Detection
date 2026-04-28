"""F19: assign_stable_ids stamps unique raw_index across combined mask lists.

PNbody runs both prompts in one runner.run, so the combined mask list is
already a single sequence — assign_stable_ids gives unique raw_index by
construction. This test pins that contract.
"""
from __future__ import annotations

import numpy as np

from src.pipelines.unified_dataset.artifacts import assign_stable_ids


def _seg(H=32, W=32):
    s = np.zeros((H, W), dtype=bool)
    s[5:10, 5:10] = True
    return s


def test_unique_raw_index_across_combined_list():
    masks = [
        {"segmentation": _seg(), "type_label": "tidal_features"},
        {"segmentation": _seg(), "type_label": "tidal_features"},
        {"segmentation": _seg(), "type_label": "satellites"},
        {"segmentation": _seg(), "type_label": "satellites"},
    ]
    assign_stable_ids(masks)
    raw_indexes = [m["raw_index"] for m in masks]
    assert len(set(raw_indexes)) == len(raw_indexes), \
        "raw_index collided across types — predictions JSON lookup would be ambiguous"
    assert raw_indexes == [0, 1, 2, 3]


def test_assign_stable_ids_is_idempotent():
    masks = [{"segmentation": _seg(), "type_label": "satellites"}]
    assign_stable_ids(masks)
    snapshot = dict(masks[0])
    assign_stable_ids(masks)
    assert masks[0]["raw_index"] == snapshot["raw_index"]
    assert masks[0]["candidate_id"] == snapshot["candidate_id"]
    assert masks[0]["candidate_rle_sha1"] == snapshot["candidate_rle_sha1"]


def test_candidate_rle_sha1_matches_serialized_rle_sha1():
    """F9: the upstream stamp computes the same sha1 that save_predictions_json emits."""
    from src.pipelines.unified_dataset.artifacts import _rle_sha1, save_predictions_json
    from src.utils.coco_utils import mask_to_rle
    import json
    import tempfile
    from pathlib import Path

    masks = [{"segmentation": _seg(), "type_label": "satellites", "score": 0.9}]
    assign_stable_ids(masks)
    upstream_sha1 = masks[0]["candidate_rle_sha1"]

    expected = _rle_sha1(mask_to_rle(masks[0]["segmentation"].astype(np.uint8)))
    assert upstream_sha1 == expected

    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "predictions.json"
        save_predictions_json(out, masks, 32, 32, engine="sam3", layer="raw")
        doc = json.loads(out.read_text())
        assert doc["predictions"][0]["candidate_rle_sha1"] == upstream_sha1
