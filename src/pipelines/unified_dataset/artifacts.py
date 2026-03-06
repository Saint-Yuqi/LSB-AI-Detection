"""
Artifact serialization helpers for the unified dataset pipeline.

Handles: prediction JSON, satellites NPZ cache, instance map merging.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.coco_utils import mask_to_rle


def save_predictions_json(
    path: Path,
    masks: list[dict[str, Any]],
    H_work: int,
    W_work: int,
    engine: str = "sam3",
    layer: str = "raw",
) -> None:
    """Save mask predictions to JSON with RLE encoding and schema header."""
    predictions = []
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        rle = mask_to_rle(seg.astype(np.uint8))
        predictions.append({
            "type_label": m.get("type_label", "unknown"),
            "score": round(m.get("predicted_iou", 0.0), 4),
            "area": m.get("area", int(seg.sum())),
            "bbox_xywh": m.get("bbox", [0, 0, 0, 0]),
            "rle": rle,
        })

    doc = {
        "schema_version": 1,
        "rle_convention": "coco_rle_fortran",
        "H_work": H_work,
        "W_work": W_work,
        "engine": engine,
        "layer": layer,
        "n_predictions": len(predictions),
        "created_at": datetime.now().isoformat(),
        "predictions": predictions,
    }
    path.write_text(json.dumps(doc, indent=2))


def save_satellites_cache(
    path: Path,
    kept: list[dict],
    ambiguous: list[dict],
    dup_rejected: list[dict],
    prior_rejected: list[dict],
    core_rejected: list[dict],
    input_image_sha1: str,
    H: int, W: int,
) -> None:
    """Save cache with RLE-encoded masks and metadata."""
    def encode_bucket(masks):
        encoded = []
        for m in masks:
            seg = m.get("segmentation", np.zeros((H, W), dtype=bool))
            rle = mask_to_rle(seg.astype(np.uint8))
            encoded.append({
                "rle": rle,
                "area": m.get("area", 0),
                "centroid_x": m.get("centroid_x", 0),
                "centroid_y": m.get("centroid_y", 0),
                "stability_score": m.get("stability_score", 0),
                "predicted_iou": m.get("predicted_iou", 0),
            })
        return encoded

    cache = {
        "kept": encode_bucket(kept),
        "ambiguous": encode_bucket(ambiguous),
        "dup_rejected": encode_bucket(dup_rejected),
        "prior_rejected": encode_bucket(prior_rejected),
        "core_rejected": encode_bucket(core_rejected),
        "metadata": {
            "input_image_sha1": input_image_sha1,
            "H": H, "W": W,
            "created_at": datetime.now().isoformat(),
        }
    }
    np.savez_compressed(path, cache=cache)


def merge_instances(
    streams_map: np.ndarray,
    inferred_masks: list[dict],
    max_stream_id: int,
    overlap_policy: str,
) -> tuple[np.ndarray, list[dict], dict, dict]:
    """Merge streams GT + inferred masks into final instance map.
    Uses type_label from each mask dict for instances_list type field."""
    instance_map = streams_map.copy().astype(np.int32)

    stream_ids = sorted([int(x) for x in np.unique(streams_map) if x > 0])
    instances_list = [{"id": sid, "type": "streams"} for sid in stream_ids]

    id_map: dict[str, dict] = {
        "streams": {str(sid): sid for sid in stream_ids},
        "satellites": {},
    }

    overlap_px = 0
    total_inferred_px = 0

    for i, m in enumerate(inferred_masks):
        new_id = max_stream_id + i + 1
        seg = m.get("segmentation", np.zeros_like(streams_map, dtype=bool))
        type_label = m.get("type_label", "satellites")

        if overlap_policy == "keep_streams":
            mask = seg & (instance_map == 0)
        else:
            mask = seg

        overlap_px += int((seg & (streams_map > 0)).sum())
        total_inferred_px += int(seg.sum())

        instance_map[mask] = new_id
        instances_list.append({"id": new_id, "type": type_label})
        id_map.setdefault(type_label, {})[str(i)] = new_id

    overlap_rate = overlap_px / total_inferred_px if total_inferred_px > 0 else 0.0

    assert instance_map.max() <= 255, f"Too many instances: {instance_map.max()}"
    instance_map = instance_map.astype(np.uint8)

    return instance_map, instances_list, id_map, {"overlap_px": overlap_px, "overlap_rate": overlap_rate}
