"""
Artifact serialization helpers for the unified dataset pipeline.

Handles: prediction JSON, satellites NPZ cache, instance map merging.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.coco_utils import decode_rle, mask_to_rle

_SERIALIZED_MASK_EXTRA_KEYS = (
    "area_clean",
    "area_raw",
    "bbox_w",
    "bbox_h",
    "centroid_x",
    "centroid_y",
    "dist_to_center",
    "aspect_sym_moment",
    "aspect_sym_boundary",
    "aspect_sym",
    "curvature_ratio",
    "solidity",
    "reject_reason",
)

_SERIALIZED_IDENTITY_KEYS = (
    "candidate_id",
    "raw_index",
    "candidate_rle_sha1",
)


def _json_ready(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _rle_sha1(rle: dict[str, Any]) -> str:
    counts = rle.get("counts")
    if isinstance(counts, str):
        blob = counts.encode("ascii")
    else:
        blob = json.dumps(rle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _candidate_id(type_label: str, type_index: int) -> str:
    prefix_map = {"satellites": "sat", "streams": "stream"}
    prefix = prefix_map.get(type_label, type_label.replace(" ", "_")[:8] or "cand")
    return f"{prefix}_{type_index:04d}"


def assign_stable_ids(masks: list[dict[str, Any]]) -> None:
    """Stamp raw_index + candidate_id + candidate_rle_sha1 in place.

    Contract:
        - ``raw_index`` = position in the provided list. All masks share the
          same index space (global ordinal). This is the source-identity key
          that survives later filtering layers.
        - ``candidate_id`` = ``f"{prefix}_{type_index:04d}"`` where prefix
          depends on ``type_label`` and ``type_index`` is a per-type ordinal
          over this same list.
        - ``candidate_rle_sha1`` = sha1 of the COCO-RLE encoding of
          ``segmentation``. Stamped using ``mask_to_rle`` then ``_rle_sha1``
          so that the value matches what ``save_predictions_json`` later
          writes for the same mask.
        - Idempotent: a mask that already carries a key is left untouched,
          so callers may assign IDs once upstream and safely call this again
          (e.g. inside the post pipeline's raw-retrieval stage).

    This helper is the single place that defines the cross-layer semantics
    of ``raw_index`` / ``candidate_rle_sha1``. ``save_predictions_json``
    reuses these IDs when present; downstream diagnostics / notebooks can
    trust them to match ``predictions_raw.json``.
    """
    type_counts: dict[str, int] = {}
    for i, m in enumerate(masks):
        type_label = m.get("type_label", "unknown")
        type_index = type_counts.get(type_label, 0)
        if "raw_index" not in m:
            m["raw_index"] = i
        if "candidate_id" not in m:
            m["candidate_id"] = _candidate_id(type_label, type_index)
        if "candidate_rle_sha1" not in m:
            seg = m.get("segmentation")
            if seg is not None:
                rle = mask_to_rle(np.asarray(seg).astype(np.uint8))
                m["candidate_rle_sha1"] = _rle_sha1(rle)
        type_counts[type_label] = type_index + 1


def save_predictions_json(
    path: Path,
    masks: list[dict[str, Any]],
    H_work: int,
    W_work: int,
    engine: str = "sam3",
    layer: str = "raw",
) -> None:
    """Save mask predictions to JSON with RLE encoding and schema header.

    ``raw_index`` / ``candidate_id`` semantics:
        - If a mask already carries these keys (typically stamped by
          ``assign_stable_ids`` on the full raw list upstream), they are
          written through verbatim. In that case ``raw_index`` is the
          *source-raw ordinal* — same value across raw / post_pred_only /
          post_gt_aware JSONs, and NOT the row position inside this file.
          Callers that want "row in this file" should use the array index
          of the ``predictions`` list directly.
        - If they are missing, this function falls back to the legacy local
          behaviour: ``raw_index`` = row position inside this file,
          ``candidate_id`` = ``{prefix}_{type_index:04d}`` within this list.
    """
    predictions = []
    type_counts: dict[str, int] = {}
    for m in masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        rle = mask_to_rle(seg.astype(np.uint8))
        type_label = m.get("type_label", "unknown")
        type_index = type_counts.get(type_label, 0)
        raw_index = int(m.get("raw_index", len(predictions)))
        candidate_id = m.get("candidate_id", _candidate_id(type_label, type_index))
        predictions.append({
            "type_label": type_label,
            "score": round(m.get("score", 0.0), 4),
            "area": m.get("area", int(seg.sum())),
            "bbox_xywh": m.get("bbox", [0, 0, 0, 0]),
            "rle": rle,
            "candidate_id": candidate_id,
            "raw_index": raw_index,
            "candidate_rle_sha1": _rle_sha1(rle),
            **{
                key: _json_ready(m[key])
                for key in _SERIALIZED_MASK_EXTRA_KEYS
                if key in m and m[key] is not None
            },
        })
        type_counts[type_label] = type_index + 1

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


def load_predictions_json(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Load prediction JSON and decode masks back into mask dicts."""
    doc = json.loads(path.read_text())
    masks: list[dict[str, Any]] = []
    for pred in doc.get("predictions", []):
        mask = {
            "segmentation": decode_rle(pred["rle"]).astype(bool),
            "type_label": pred.get("type_label", "unknown"),
            "score": pred.get("score", 0.0),
            "area": pred.get("area", 0),
            "bbox": pred.get("bbox_xywh", [0, 0, 0, 0]),
        }
        for key in _SERIALIZED_IDENTITY_KEYS:
            if key in pred:
                mask[key] = pred[key]
        for key in _SERIALIZED_MASK_EXTRA_KEYS:
            if key in pred:
                mask[key] = pred[key]
        masks.append(mask)
    return doc, masks


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
                "score": m.get("score", 0),
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


def rasterize_pseudo_gt(
    masks: list[dict[str, Any]],
    H: int,
    W: int,
    overlap_policy: str = "last_wins",
) -> tuple[np.ndarray, list[dict]]:
    """Convert post-filtered masks to an instance map + instances list.

    Assigns instance IDs 1..N. Last-writer-wins for pixel overlap.
    Returns (instance_map_uint8, instances_list).

    Raises:
        AssertionError: if N > 255 (uint8 overflow).
    """
    instance_map, instances_list, _ = build_pseudo_gt_artifacts(
        masks,
        H,
        W,
        overlap_policy=overlap_policy,
    )
    return instance_map, instances_list


def build_pseudo_gt_artifacts(
    masks: list[dict[str, Any]],
    H: int,
    W: int,
    overlap_policy: str = "last_wins",
) -> tuple[np.ndarray, list[dict], dict[str, dict[str, int]]]:
    """Build a pseudo GT instance map, instances list, and type-aware id map."""
    assert len(masks) <= 255, f"Too many masks for uint8: {len(masks)}"

    instance_map = np.zeros((H, W), dtype=np.int32)
    instances_list: list[dict] = []
    id_map: dict[str, dict[str, int]] = {}

    if overlap_policy == "stream_first":
        ordered_masks = [
            *[m for m in masks if m.get("type_label") == "streams"],
            *[m for m in masks if m.get("type_label") != "streams"],
        ]
    else:
        ordered_masks = list(masks)

    type_counts: dict[str, int] = {}
    next_id = 1

    for m in ordered_masks:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        seg_bool = seg.astype(bool)
        type_label = m.get("type_label", "unknown")

        if overlap_policy == "stream_first" and type_label != "streams":
            seg_bool = seg_bool & (instance_map == 0)
            if seg_bool.sum() == 0:
                continue

        instance_map[seg_bool] = next_id
        instances_list.append({"id": next_id, "type": type_label})
        type_index = type_counts.get(type_label, 0)
        id_map.setdefault(type_label, {})[str(type_index)] = next_id
        type_counts[type_label] = type_index + 1
        next_id += 1

    return instance_map.astype(np.uint8), instances_list, id_map


def save_pseudo_gt(
    gt_dir: Path,
    masks: list[dict[str, Any]],
    H: int,
    W: int,
    overlap_policy: str = "last_wins",
    write_id_map: bool = False,
) -> tuple[np.ndarray, list[dict]]:
    """Rasterize post-filtered masks and write instance_map_uint8.png + instances.json."""
    from PIL import Image as _PILImage

    instance_map, instances_list, id_map = build_pseudo_gt_artifacts(
        masks,
        H,
        W,
        overlap_policy=overlap_policy,
    )

    gt_dir.mkdir(parents=True, exist_ok=True)
    _PILImage.fromarray(instance_map).save(gt_dir / "instance_map_uint8.png")
    (gt_dir / "instances.json").write_text(json.dumps(instances_list, indent=2))
    if write_id_map:
        (gt_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))

    return instance_map, instances_list


_PER_CLASS_MAP_FILES: dict[str, str] = {
    "tidal_features": "tidal_features_instance_map.npy",
    "satellites": "satellites_instance_map.npy",
    "inner_galaxy": "inner_galaxy_instance_map.npy",
}

_TIDAL_V1_CLASSES: tuple[str, ...] = ("tidal_features", "satellites", "inner_galaxy")

# Class index for the global_id encoding rule (F13).
_CLASS_INDEX: dict[str, int] = {"tidal_features": 1, "satellites": 2, "inner_galaxy": 3}
_GLOBAL_ID_STRIDE = 100_000


def _global_id(type_label: str, local_id: int) -> int:
    return _CLASS_INDEX[type_label] * _GLOBAL_ID_STRIDE + int(local_id)


def save_per_class_instance_maps(
    gt_dir: Path,
    masks_by_type: dict[str, list[dict[str, Any]]],
    H: int,
    W: int,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Write per-class instance maps + per-row metadata for the new GT path.

    Output layout (under ``gt_dir``):
        tidal_features_instance_map.npy   int32, 0=bg, 1..N local ids
        satellites_instance_map.npy        int32, 0=bg, 1..N local ids
        inner_galaxy_instance_map.npy      int32, 0=bg, 1..N local ids
        instances.json                     list[dict], two row shapes (SAM- vs FITS-derived)
        id_map.json                        {class_name: {str(local_id): global_id}}
        instance_map_uint8.png             QA preview (lossy, last-wins)
        manifest.json                      MERGED with whatever gt.py wrote earlier

    Row schema (instances.json):
        - SAM-derived rows carry: ``candidate_id``, ``raw_index``,
          ``candidate_rle_sha1``, ``source: "sam3_post"``.
        - FITS-derived rows carry: ``source_instance_id``, ``raw_index: null``,
          ``source: "firebox_sb31.5_fits"``.
        - Both shapes carry: ``id``, ``global_id`` (== ``id``), ``local_id``,
          ``type`` and ``type_label`` (identical alias for one cycle),
          ``map_file``.

    The QA PNG is a single uint8 raster with last-wins overlap. When the
    union exceeds 255 distinct ids we log a warning and clip to uint8
    rather than asserting (F: artifacts.py overflow tolerance).
    """
    from PIL import Image as _PILImage

    gt_dir.mkdir(parents=True, exist_ok=True)

    per_class_maps: dict[str, np.ndarray] = {}
    instances_list: list[dict[str, Any]] = []
    id_map: dict[str, dict[str, int]] = {cls: {} for cls in _TIDAL_V1_CLASSES}

    for cls in _TIDAL_V1_CLASSES:
        masks = masks_by_type.get(cls, []) or []
        m_map = np.zeros((H, W), dtype=np.int32)
        for local_idx, m in enumerate(masks, start=1):
            seg = m.get("segmentation")
            if seg is None:
                continue
            seg_bool = np.asarray(seg).astype(bool)
            if seg_bool.sum() == 0:
                continue
            m_map[seg_bool] = local_idx

            global_id = _global_id(cls, local_idx)
            id_map[cls][str(local_idx)] = global_id

            row: dict[str, Any] = {
                "id": global_id,
                "global_id": global_id,
                "local_id": local_idx,
                "type": cls,
                "type_label": cls,
                "map_file": _PER_CLASS_MAP_FILES[cls],
            }
            if "candidate_id" in m or "raw_index" in m or "candidate_rle_sha1" in m:
                row["source"] = "sam3_post"
                if "candidate_id" in m:
                    row["candidate_id"] = m["candidate_id"]
                if "raw_index" in m:
                    row["raw_index"] = int(m["raw_index"])
                if "candidate_rle_sha1" in m:
                    row["candidate_rle_sha1"] = m["candidate_rle_sha1"]
            elif "source_instance_id" in m:
                row["source"] = m.get("source", "firebox_sb31.5_fits")
                row["source_instance_id"] = int(m["source_instance_id"])
                row["raw_index"] = None
            else:
                row["source"] = m.get("source", "unknown")
                row["raw_index"] = None
            instances_list.append(row)

        np.save(gt_dir / _PER_CLASS_MAP_FILES[cls], m_map)
        per_class_maps[cls] = m_map

    (gt_dir / "instances.json").write_text(json.dumps(instances_list, indent=2))
    (gt_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))

    # QA preview: last-wins union of all three classes, packed into uint8.
    qa = np.zeros((H, W), dtype=np.int32)
    next_id = 1
    for cls in _TIDAL_V1_CLASSES:
        m_map = per_class_maps[cls]
        for local_id in sorted(int(x) for x in np.unique(m_map) if x != 0):
            qa[m_map == local_id] = next_id
            next_id += 1
    if next_id - 1 > 255:
        import logging

        logging.getLogger(__name__).warning(
            "QA preview overflow: %d total instances exceed uint8; clipping. "
            "instance_map_uint8.png is a lossy preview only.",
            next_id - 1,
        )
    qa_uint8 = np.clip(qa, 0, 255).astype(np.uint8)
    _PILImage.fromarray(qa_uint8).save(gt_dir / "instance_map_uint8.png")

    # Manifest merge (F6): preserve fields gt.py already wrote (fits_source,
    # source_mask, etc.); add tidal_v1 semantic stamps without stomping.
    manifest_path = gt_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest.update({
        "gt_path_version": "tidal_v1",
        "instance_map_uint8_semantics": "qa_preview_last_wins",
        "authoritative_mask_source": "per_class_npy_plus_sam3_predictions_post",
        "n_tidal_features": int((per_class_maps["tidal_features"] > 0).any()) and int(per_class_maps["tidal_features"].max()),
        "n_satellites": int(per_class_maps["satellites"].max()),
        "n_inner_galaxy": int(per_class_maps["inner_galaxy"].max()),
        "categories": list(_TIDAL_V1_CLASSES),
        "instances_updated_at": datetime.now().isoformat(),
    })
    if extra_manifest:
        manifest.update(extra_manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    return per_class_maps, instances_list


def save_per_class_pseudo_gt(
    gt_dir: Path,
    masks_by_type: dict[str, list[dict[str, Any]]],
    H: int,
    W: int,
    *,
    extra_manifest: dict[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    """Pseudo-GT writer for PNbody. Semantically identical to GT writer."""
    return save_per_class_instance_maps(
        gt_dir, masks_by_type, H, W, extra_manifest=extra_manifest
    )


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
