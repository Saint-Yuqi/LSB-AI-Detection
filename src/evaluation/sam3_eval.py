"""
SAM3 evaluation core — type-aware discovery, inference, aggregation, serialization.

Usage:
    from src.evaluation.sam3_eval import discover_pairs, run_and_evaluate, aggregate_results
    pairs = discover_pairs(render_dir, gt_dir)
    results = [run_and_evaluate(runner, pair, ...) for pair in pairs]
    summary = aggregate_results(results)

Env:
    Requires: numpy, scipy, PIL. SAM3 runner imported lazily.

Per-image result schema (type-aware):
    result["streams"]["raw"]     = {dice, precision, ...}
    result["streams"]["post"]    = {dice, precision, ...}
    result["satellites"]["raw"]  = {dice, precision, ...}
    result["satellites"]["post"] = {dice, precision, ...}   # = raw (no filter)
    result["combined"]["raw"]    = {dice, precision, ...}   # class-agnostic
    result["combined"]["post"]   = {dice, precision, ...}
"""
from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from src.evaluation.metrics import (
    calculate_pixel_metrics,
    calculate_optimal_instance_metrics,
)

logger = logging.getLogger(__name__)


# =========================================================================== #
#  Discovery
# =========================================================================== #


def discover_pairs(
    render_dir: Path,
    gt_dir: Path,
    max_samples: Optional[int] = None,
) -> list[dict[str, Any]]:
    """
    Walk render_dir/{base_key}/0000.png ↔ gt_dir/{base_key}/ GT files.

    Priority matrix for GT files:
        instances.json + instance_map_uint8.png → full mode (streams + satellites)
        instances.json + NO instance_map        → skip (corrupt GT)
        NO instances.json + instance_map_uint8   → skip (can't split types)
        NO instances.json + streams_instance_map → legacy fallback (streams only)
        nothing                                  → skip

    Returns list of dicts with gt_mode="full"|"streams_only".
    """
    render_dir = Path(render_dir)
    gt_dir = Path(gt_dir)

    pattern = re.compile(r"^(\d+)_([^_]+)$")
    pairs: list[dict[str, Any]] = []

    for subdir in sorted(render_dir.iterdir()):
        if not subdir.is_dir():
            continue
        m = pattern.match(subdir.name)
        if m is None:
            continue

        render_path = subdir / "0000.png"
        if not render_path.exists():
            continue

        gt_subdir = gt_dir / subdir.name
        if not gt_subdir.is_dir():
            continue

        instances_json_path = gt_subdir / "instances.json"
        full_map_path = gt_subdir / "instance_map_uint8.png"
        streams_npy_path = gt_subdir / "streams_instance_map.npy"

        if instances_json_path.exists():
            if not full_map_path.exists():
                logger.warning(
                    f"SKIP {subdir.name}: instances.json present but "
                    f"instance_map_uint8.png missing (corrupt GT)"
                )
                continue
            # Full mode — validate ID consistency
            gt_mode = "full"
            gt_path = full_map_path
            gt_instances_path = instances_json_path
        elif full_map_path.exists() and not instances_json_path.exists():
            logger.warning(
                f"SKIP {subdir.name}: instance_map_uint8.png present but "
                f"instances.json missing (can't split types)"
            )
            continue
        elif streams_npy_path.exists():
            gt_mode = "streams_only"
            gt_path = streams_npy_path
            gt_instances_path = None
        else:
            continue

        pairs.append({
            "base_key": subdir.name,
            "galaxy_id": int(m.group(1)),
            "view_id": m.group(2),
            "orientation": m.group(2),
            "render_path": render_path,
            "gt_path": gt_path,
            "gt_instances_path": gt_instances_path,
            "gt_mode": gt_mode,
        })

    if max_samples is not None:
        pairs = pairs[:max_samples]

    return pairs


def _load_gt_by_type(
    pair: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load and split GT into per-type instance maps.

    Returns (gt_streams_map, gt_satellites_map, gt_all_map)
    where each is (H, W) int32 with instance IDs > 0.
    """
    if pair["gt_mode"] == "full":
        # Full mode: instance_map_uint8.png + instances.json
        full_map = np.array(Image.open(pair["gt_path"])).astype(np.int32)
        with open(pair["gt_instances_path"]) as f:
            instances = json.load(f)

        # ID consistency check
        map_ids = set(np.unique(full_map).tolist()) - {0}
        json_ids = {inst["id"] for inst in instances}
        if map_ids != json_ids:
            diff = map_ids.symmetric_difference(json_ids)
            logger.warning(
                f"{pair['base_key']}: GT ID mismatch — "
                f"map_only={map_ids - json_ids}, json_only={json_ids - map_ids}, "
                f"evaluating with intersection ({len(map_ids & json_ids)} IDs)"
            )

        # Build type → ID sets
        stream_ids = {inst["id"] for inst in instances if inst["type"] == "streams"}
        satellite_ids = {inst["id"] for inst in instances if inst["type"] == "satellites"}

        gt_streams = np.where(np.isin(full_map, list(stream_ids)), full_map, 0)
        gt_satellites = np.where(np.isin(full_map, list(satellite_ids)), full_map, 0)
        gt_all = full_map

    else:
        # Legacy fallback: streams only
        gt_streams = np.load(pair["gt_path"]).astype(np.int32)
        H, W = gt_streams.shape
        gt_satellites = np.zeros((H, W), dtype=np.int32)
        gt_all = gt_streams.copy()

    return gt_streams, gt_satellites, gt_all


# =========================================================================== #
#  Inference + Evaluation (single image)
# =========================================================================== #


def _combine_pred_masks(masks: list[dict]) -> np.ndarray:
    """Union of all pred segmentation masks → binary (H, W)."""
    if not masks:
        return None
    combined = np.zeros_like(masks[0]["segmentation"], dtype=bool)
    for m in masks:
        combined |= m["segmentation"].astype(bool)
    return combined


def _compute_slice_metrics(
    pred_masks: list[dict],
    gt_instance_map: np.ndarray,
    H_work: int,
    W_work: int,
    match_iou_thresh: float,
) -> dict[str, Any]:
    """Compute pixel + instance metrics for one type slice."""
    gt_binary = (gt_instance_map > 0).astype(bool)
    combined = _combine_pred_masks(pred_masks)
    if combined is None:
        combined = np.zeros((H_work, W_work), dtype=bool)

    pixel = calculate_pixel_metrics(combined, gt_binary)
    instance = calculate_optimal_instance_metrics(
        pred_masks, gt_instance_map, match_iou_thresh,
    )

    return {
        "dice": pixel["dice"],
        "precision": pixel["precision"],
        "recall_pixel": pixel["recall"],
        "capped_hausdorff95": pixel["capped_hausdorff95"],
        "binary_iou": _binary_iou(combined, gt_binary),
        "tp": pixel["tp"],
        "fp": pixel["fp"],
        "fn": pixel["fn"],
        "instance_recall": instance["instance_recall"],
        "matched_iou": instance["matched_iou"],
        "unmatched_iou": instance["unmatched_iou"],
        "num_gt": instance["num_gt"],
        "num_detected": instance["num_detected"],
        "num_pred": instance["num_pred"],
        "per_instance_details": instance["per_instance_details"],
    }


def run_and_evaluate(
    runner: Any,
    pair: dict[str, Any],
    prompts: list[dict[str, str]],
    H_work: int,
    W_work: int,
    match_iou_thresh: float = 0.5,
    streams_filter: Any = None,
) -> dict[str, Any]:
    """
    Run SAM3 inference on one image and compute per-type metrics.

    Computes metrics for 3 type slices × 2 layers:
        streams.{raw,post}     — StreamsSanityFilter applied for post
        satellites.{raw,post}  — post = raw (no filter)
        combined.{raw,post}    — class-agnostic: ALL masks vs ALL GT

    Args:
        runner:           SAM3PromptRunner instance.
        pair:             dict from discover_pairs (includes gt_mode).
        prompts:          text prompts for SAM3.
        H_work, W_work:  working grid resolution.
        match_iou_thresh: IoU threshold for instance matching.
        streams_filter:   StreamsSanityFilter or None (for post layer).
    """
    image_pil = Image.open(pair["render_path"]).convert("RGB")

    # Load type-split GT
    gt_streams, gt_satellites, gt_all = _load_gt_by_type(pair)

    # --- SAM3 inference ---
    masks, time_ms = runner.run(image_pil, prompts)

    # --- Split predictions by type ---
    stream_masks_raw = [m for m in masks if m.get("type_label") == "streams"]
    satellite_masks_raw = [m for m in masks if m.get("type_label") == "satellites"]

    # --- Post layers ---
    # Streams: apply StreamsSanityFilter
    if streams_filter is not None and stream_masks_raw:
        stream_masks_post, _ = streams_filter.filter(stream_masks_raw, H_work, W_work)
    else:
        stream_masks_post = list(stream_masks_raw)

    # Satellites: no filter, post = raw
    satellite_masks_post = list(satellite_masks_raw)

    # Combined: ALL masks regardless of type_label (future-proof)
    all_masks_raw = list(masks)
    all_masks_post = stream_masks_post + satellite_masks_post
    # Include any masks with unknown type_labels in post too
    known_types = {"streams", "satellites"}
    other_masks = [m for m in masks if m.get("type_label") not in known_types]
    all_masks_post = all_masks_post + other_masks

    # --- Metadata ---
    result: dict[str, Any] = {
        "base_key": pair["base_key"],
        "galaxy_id": pair["galaxy_id"],
        "orientation": pair["orientation"],
        "gt_mode": pair["gt_mode"],
        "n_raw_total": len(masks),
        "n_raw_streams": len(stream_masks_raw),
        "n_raw_satellites": len(satellite_masks_raw),
        "inference_time_ms": round(time_ms, 2),
    }

    # --- Compute per-type metrics ---
    # Streams
    result["streams"] = {
        "raw": _compute_slice_metrics(
            stream_masks_raw, gt_streams, H_work, W_work, match_iou_thresh),
        "post": _compute_slice_metrics(
            stream_masks_post, gt_streams, H_work, W_work, match_iou_thresh),
    }

    # Satellites
    sat_raw = _compute_slice_metrics(
        satellite_masks_raw, gt_satellites, H_work, W_work, match_iou_thresh)
    result["satellites"] = {"raw": sat_raw, "post": dict(sat_raw)}

    # Combined (class-agnostic)
    result["combined"] = {
        "raw": _compute_slice_metrics(
            all_masks_raw, gt_all, H_work, W_work, match_iou_thresh),
        "post": _compute_slice_metrics(
            all_masks_post, gt_all, H_work, W_work, match_iou_thresh),
    }

    # Keep raw masks for overlay (stripped from JSON by save_results)
    result["_stream_masks_raw"] = stream_masks_raw
    result["_satellite_masks_raw"] = satellite_masks_raw

    return result


def _binary_iou(pred: np.ndarray, gt: np.ndarray) -> Optional[float]:
    """Simple binary IoU."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = int((pred_b & gt_b).sum())
    union = int((pred_b | gt_b).sum())
    if union == 0:
        return None
    return inter / union


# =========================================================================== #
#  Aggregation
# =========================================================================== #

_METRIC_KEYS = [
    "dice", "precision", "recall_pixel", "capped_hausdorff95",
    "binary_iou", "instance_recall", "matched_iou", "unmatched_iou",
]


def aggregate_results(
    per_image: list[dict[str, Any]],
    group_by: str = "overall",
) -> dict[str, Any]:
    """
    Aggregate per-image results into macro/micro stats.

    group_by: "overall" or "galaxy" (merge eo+fo by galaxy_id).

    Returns dict with per-type (streams/satellites/combined) metrics
    for both raw & post layers.
    """
    if group_by == "galaxy":
        groups: dict[Any, list] = defaultdict(list)
        for r in per_image:
            groups[r["galaxy_id"]].append(r)
        return {
            str(gid): _aggregate_group(images)
            for gid, images in sorted(groups.items())
        }

    return _aggregate_group(per_image)


def _aggregate_group(images: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image results — dynamically infers type keys."""
    result: dict[str, Any] = {}

    if not images:
        return result

    # Dynamic type inference: find keys that contain {raw, post} sub-dicts
    type_keys = [
        k for k in images[0]
        if isinstance(images[0].get(k), dict) and "raw" in images[0][k]
    ]

    for type_key in type_keys:
        result[type_key] = {}
        for layer in ("raw", "post"):
            result[type_key][layer] = _aggregate_layer(
                images, layer, type_key=type_key
            )

    result["n_images"] = len(images)
    return result


def _aggregate_layer(
    images: list[dict[str, Any]],
    layer: str,
    type_key: Optional[str] = None,
) -> dict[str, Any]:
    """Null-aware macro + micro aggregation for one layer of one type."""
    # Collect metric values (skip None)
    metric_vals: dict[str, list[float]] = {k: [] for k in _METRIC_KEYS}
    total_tp, total_fp, total_fn = 0, 0, 0
    n_empty_pred, n_empty_gt, n_both_empty = 0, 0, 0

    for img in images:
        # Navigate to the correct nesting level
        if type_key is not None:
            layer_data = img.get(type_key, {}).get(layer, {})
        else:
            layer_data = img.get(layer, {})

        tp = layer_data.get("tp", 0)
        fp = layer_data.get("fp", 0)
        fn = layer_data.get("fn", 0)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        pred_empty = (tp + fp) == 0
        gt_empty = (tp + fn) == 0
        if pred_empty and gt_empty:
            n_both_empty += 1
        elif pred_empty:
            n_empty_pred += 1
        elif gt_empty:
            n_empty_gt += 1

        for k in _METRIC_KEYS:
            v = layer_data.get(k)
            if v is not None:
                metric_vals[k].append(v)

    agg: dict[str, Any] = {}

    # Macro-average
    for k in _METRIC_KEYS:
        vals = metric_vals[k]
        agg[f"n_defined_{k}"] = len(vals)
        if vals:
            agg[f"macro_mean_{k}"] = float(np.mean(vals))
            agg[f"macro_std_{k}"] = float(np.std(vals))
        else:
            agg[f"macro_mean_{k}"] = None
            agg[f"macro_std_{k}"] = None

    # Micro-average (pooled pixels)
    agg["micro_tp"] = total_tp
    agg["micro_fp"] = total_fp
    agg["micro_fn"] = total_fn
    micro_denom_dice = 2 * total_tp + total_fp + total_fn
    agg["micro_dice"] = (2.0 * total_tp / micro_denom_dice) if micro_denom_dice > 0 else None
    agg["micro_precision"] = (total_tp / (total_tp + total_fp)) if (total_tp + total_fp) > 0 else None
    agg["micro_recall"] = (total_tp / (total_tp + total_fn)) if (total_tp + total_fn) > 0 else None

    # Sample counts
    agg["n_empty_pred"] = n_empty_pred
    agg["n_empty_gt"] = n_empty_gt
    agg["n_both_empty"] = n_both_empty

    return agg


# =========================================================================== #
#  Serialization
# =========================================================================== #


def _strip_details(obj: Any) -> Any:
    """Recursively strip per_instance_details from nested dicts/lists."""
    if isinstance(obj, dict):
        return {
            k: _strip_details(v) for k, v in obj.items()
            if k != "per_instance_details"
        }
    if isinstance(obj, (list, tuple)):
        return [_strip_details(item) for item in obj]
    return obj


def save_results(
    output_dir: Path,
    config: dict[str, Any],
    summary: dict[str, Any],
    per_image: list[dict[str, Any]],
) -> Path:
    """Save evaluation results to JSON. Returns path to saved file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"eval_results_{timestamp}.json"

    # Recursively strip per_instance_details + private keys
    per_image_slim = []
    for img in per_image:
        clean = {k: v for k, v in img.items() if not k.startswith("_")}
        per_image_slim.append(_strip_details(clean))

    doc = {
        "config": config,
        "summary": summary,
        "per_image": per_image_slim,
        "created_at": datetime.now().isoformat(),
    }

    out_path.write_text(json.dumps(doc, indent=2, default=_json_default))
    return out_path


def _json_default(obj: Any) -> Any:
    """Handle numpy types in JSON serialization."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# =========================================================================== #
#  Overlay
# =========================================================================== #

# Color palettes for type-aware overlay
_STREAM_COLORS = [(100, 149, 237), (65, 105, 225), (30, 144, 255), (0, 191, 255)]
_SATELLITE_COLORS = [(60, 179, 113), (46, 139, 87), (0, 200, 83), (102, 205, 170)]


def save_eval_overlay(
    path: Path,
    render_path: Path,
    gt_streams_map: np.ndarray,
    gt_satellites_map: np.ndarray,
    stream_masks: list[dict],
    satellite_masks: list[dict],
) -> None:
    """
    Type-aware QA overlay.

    GT contours: white = streams, yellow = satellites
    Pred fills:  blue palette = streams, green palette = satellites
    """
    image = np.array(Image.open(render_path).convert("RGB"))
    overlay = image.copy()

    # GT stream contours (white)
    for gid in np.unique(gt_streams_map):
        if gid == 0:
            continue
        gt_bin = (gt_streams_map == gid).astype(np.uint8)
        contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    # GT satellite contours (yellow)
    for gid in np.unique(gt_satellites_map):
        if gid == 0:
            continue
        gt_bin = (gt_satellites_map == gid).astype(np.uint8)
        contours, _ = cv2.findContours(gt_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 0), 2)

    # Pred stream fills (blue palette)
    _draw_pred_fills(overlay, stream_masks, _STREAM_COLORS)

    # Pred satellite fills (green palette)
    _draw_pred_fills(overlay, satellite_masks, _SATELLITE_COLORS)

    # Legend
    cv2.putText(overlay, "GT streams (white)", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.putText(overlay, "GT satellites (yellow)", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    cv2.putText(overlay, "Pred streams (blue)", (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 149, 237), 1)
    cv2.putText(overlay, "Pred satellites (green)", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (60, 179, 113), 1)

    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def _draw_pred_fills(
    overlay: np.ndarray,
    masks: list[dict],
    colors: list[tuple[int, int, int]],
    alpha: float = 0.45,
) -> None:
    """Draw semi-transparent prediction fills on overlay."""
    for i, m in enumerate(masks):
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        mask_bool = seg.astype(bool)
        color = np.array(colors[i % len(colors)], dtype=np.uint8)
        overlay[mask_bool] = (
            overlay[mask_bool].astype(np.float32) * (1.0 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)
