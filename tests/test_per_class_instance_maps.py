"""Regression tests for the new (tidal_v1) per-class GT writer.

Covers:
    - F2: within-class overlap is preserved when SAM-derived rows are
      sourced from sam3_predictions_post.json (not the per-class map).
    - F6: the writer merges manifest fields written by gt.py rather than
      stomping them.
    - F13: each row carries id == global_id plus local_id, with
      class-encoded global ids that do not collide across classes.
    - F20: a row declaring source=='sam3_post' that is missing from the
      predictions JSON should be a hard error in downstream consumers
      (export, eval); here we validate the writer side only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.pipelines.unified_dataset.artifacts import (
    save_per_class_instance_maps,
    save_predictions_json,
)


def _seg(H: int, W: int, y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    s = np.zeros((H, W), dtype=bool)
    s[y0:y1, x0:x1] = True
    return s


def test_within_class_overlap_preserved_via_predictions_json(tmp_path: Path):
    """F2: two satellite masks with overlapping pixels stay distinct in the JSON."""
    H, W = 64, 64
    gt_dir = tmp_path

    s1 = _seg(H, W, 20, 35, 20, 35)
    s2 = _seg(H, W, 30, 45, 30, 45)   # overlaps s1 at [30:35, 30:35]
    overlap_pixels = int((s1 & s2).sum())
    assert overlap_pixels > 0

    sat_masks = [
        {
            "segmentation": s1,
            "type_label": "satellites",
            "candidate_id": "sat_0000",
            "raw_index": 0,
            "candidate_rle_sha1": "deadbeef00000000",
        },
        {
            "segmentation": s2,
            "type_label": "satellites",
            "candidate_id": "sat_0001",
            "raw_index": 1,
            "candidate_rle_sha1": "cafebabe00000000",
        },
    ]
    save_per_class_instance_maps(
        gt_dir,
        {"tidal_features": [], "satellites": sat_masks, "inner_galaxy": []},
        H, W,
    )
    save_predictions_json(
        gt_dir / "sam3_predictions_post.json",
        sat_masks, H, W, engine="sam3", layer="post",
    )

    # When export sources from predictions JSON, both RLEs cover the overlap.
    pred_doc = json.loads((gt_dir / "sam3_predictions_post.json").read_text())
    raw_to_rle = {p["raw_index"]: p["rle"] for p in pred_doc["predictions"]}
    from src.utils.coco_utils import decode_rle
    bin0 = decode_rle(raw_to_rle[0]).astype(bool)
    bin1 = decode_rle(raw_to_rle[1]).astype(bool)
    assert (bin0 & bin1).sum() == overlap_pixels, "within-class overlap collapsed!"


def test_manifest_merge_preserves_gt_phase_fields(tmp_path: Path):
    """F6: gt.py-written manifest fields survive the inference-phase writer."""
    H, W = 32, 32
    gt_dir = tmp_path

    pre_manifest = {
        "gt_path_version": "tidal_v1",
        "sb_threshold_used": 31.5,
        "source_mask": "/path/fits.gz",
        "source_mask_sha1": "abc123",
        "fits_phase_created_at": "2026-04-28T00:00:00",
    }
    (gt_dir / "manifest.json").write_text(json.dumps(pre_manifest))

    save_per_class_instance_maps(
        gt_dir,
        {"tidal_features": [], "satellites": [], "inner_galaxy": []},
        H, W,
    )
    merged = json.loads((gt_dir / "manifest.json").read_text())

    # FITS-phase keys must survive
    assert merged["source_mask"] == "/path/fits.gz"
    assert merged["source_mask_sha1"] == "abc123"
    assert merged["fits_phase_created_at"] == "2026-04-28T00:00:00"
    assert merged["sb_threshold_used"] == 31.5
    # Inference-phase keys must be added
    assert merged["instance_map_uint8_semantics"] == "qa_preview_last_wins"
    assert merged["authoritative_mask_source"] == "per_class_npy_plus_sam3_predictions_post"


def test_global_id_encoding_and_id_alias(tmp_path: Path):
    """F13: id == global_id, both encode class index; legacy readers stay valid."""
    H, W = 32, 32
    gt_dir = tmp_path

    masks_by_type = {
        "tidal_features": [{"segmentation": _seg(H, W, 0, 5, 0, 5), "source_instance_id": 9}],
        "satellites": [{
            "segmentation": _seg(H, W, 10, 15, 10, 15),
            "type_label": "satellites",
            "candidate_id": "sat_0000",
            "raw_index": 0,
            "candidate_rle_sha1": "0",
        }],
        "inner_galaxy": [{
            "segmentation": _seg(H, W, 20, 25, 20, 25),
            "type_label": "inner_galaxy",
            "candidate_id": "sat_0001",
            "raw_index": 1,
            "candidate_rle_sha1": "0",
        }],
    }
    save_per_class_instance_maps(gt_dir, masks_by_type, H, W)
    instances = json.loads((gt_dir / "instances.json").read_text())

    by_type = {r["type_label"]: r for r in instances}
    # id == global_id everywhere
    for r in instances:
        assert r["id"] == r["global_id"]
        assert "local_id" in r
        assert "map_file" in r

    # Class-encoded global ids do not collide across classes
    assert by_type["tidal_features"]["global_id"] == 100001
    assert by_type["satellites"]["global_id"] == 200001
    assert by_type["inner_galaxy"]["global_id"] == 300001


def test_three_per_class_npys_emitted(tmp_path: Path):
    """All three per-class npy files exist even when one bucket is empty."""
    H, W = 32, 32
    gt_dir = tmp_path

    save_per_class_instance_maps(
        gt_dir,
        {
            "tidal_features": [{"segmentation": _seg(H, W, 0, 5, 0, 5), "source_instance_id": 1}],
            "satellites": [],
            "inner_galaxy": [],
        },
        H, W,
    )
    assert (gt_dir / "tidal_features_instance_map.npy").exists()
    assert (gt_dir / "satellites_instance_map.npy").exists()
    assert (gt_dir / "inner_galaxy_instance_map.npy").exists()
    # Empty buckets produce all-zero arrays
    sats = np.load(gt_dir / "satellites_instance_map.npy")
    assert sats.shape == (H, W)
    assert sats.max() == 0
