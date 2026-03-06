"""
Overlay visualisation – draw coloured contours on 8-bit image.

Usage:
    from src.visualization.overlay import save_overlay
    save_overlay(image, kept, core_rejected, prior_rejected, duplicates, ambiguous, out_path)

Layers (draw order: background → foreground):
    Gray   (128,128,128) – prior_rejected (thickness=1)
    Cyan   (0,255,255)   – duplicate_rejected (thickness=1)
    Yellow (255,255,0)   – ambiguous (thickness=2)
    Red    (255,0,0)     – core_rejected (thickness=2)
    Green  (0,255,0)     – kept masks (thickness=2)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# RGB order (canvas is RGB, converted to BGR at save)
_COLOUR_KEPT = (0, 255, 0)           # Green
_COLOUR_CORE = (255, 0, 0)           # Red
_COLOUR_PRIOR = (128, 128, 128)      # Gray
_COLOUR_DUPLICATE = (0, 255, 255)    # Cyan
_COLOUR_AMBIGUOUS = (255, 255, 0)    # Yellow


def _draw_contours(
    canvas: np.ndarray,
    masks: list[dict[str, Any]],
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw contours for a list of masks on canvas (in-place)."""
    for m in masks:
        seg = m["segmentation"].astype(np.uint8)
        cnts, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, colour, thickness)


def save_overlay(
    image: np.ndarray,
    kept: list[dict[str, Any]],
    core_rejected: list[dict[str, Any]] | None = None,
    prior_rejected: list[dict[str, Any]] | None = None,
    duplicate_rejected: list[dict[str, Any]] | None = None,
    ambiguous: list[dict[str, Any]] | None = None,
    out_path: str | Path = "overlay.png",
    draw_prior: bool = False,
    draw_duplicate: bool = True,
    draw_ambiguous: bool = True,
) -> None:
    """
    Save overlay PNG with multi-layer contours.

    Args:
        image: (H, W, 3) uint8 RGB.
        kept: masks to show in green.
        core_rejected: masks to show in red.
        prior_rejected: masks optionally shown in gray (if draw_prior=True).
        duplicate_rejected: masks shown in cyan (same-target duplicates).
        ambiguous: masks shown in yellow (borderline cases).
        out_path: output file path.
        draw_prior: if True, draw prior_rejected in gray.
        draw_duplicate: if True, draw duplicate_rejected in cyan.
        draw_ambiguous: if True, draw ambiguous in yellow.
    """
    canvas = image.copy()

    # Draw layers in order: background → foreground
    if draw_prior and prior_rejected:
        _draw_contours(canvas, prior_rejected, _COLOUR_PRIOR, thickness=1)

    if draw_duplicate and duplicate_rejected:
        _draw_contours(canvas, duplicate_rejected, _COLOUR_DUPLICATE, thickness=1)

    if draw_ambiguous and ambiguous:
        _draw_contours(canvas, ambiguous, _COLOUR_AMBIGUOUS, thickness=2)

    if core_rejected:
        _draw_contours(canvas, core_rejected, _COLOUR_CORE, thickness=2)

    _draw_contours(canvas, kept, _COLOUR_KEPT, thickness=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGB → BGR for cv2.imwrite
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def save_evaluation_overlay(
    path: Path,
    image: np.ndarray,
    streams_map: np.ndarray,
    predictions: list[dict[str, Any]],
) -> None:
    """QA overlay: GT streams as solid white contours, predictions as semi-transparent fills."""
    overlay = image.copy()

    gt_ids = np.unique(streams_map)
    gt_ids = gt_ids[gt_ids > 0]
    for gid in gt_ids:
        gt_binary = (streams_map == gid).astype(np.uint8)
        contours, _ = cv2.findContours(gt_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (255, 255, 255), 2)

    stream_colors = [(100, 149, 237), (65, 105, 225), (30, 144, 255)]
    sat_colors = [(255, 165, 0), (255, 140, 0), (255, 127, 80)]
    stream_idx, sat_idx = 0, 0

    for m in predictions:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        mask_bool = seg.astype(bool)
        tl = m.get("type_label", "")

        if tl == "streams":
            color = np.array(stream_colors[stream_idx % len(stream_colors)], dtype=np.uint8)
            stream_idx += 1
        else:
            color = np.array(sat_colors[sat_idx % len(sat_colors)], dtype=np.uint8)
            sat_idx += 1

        alpha = 0.45
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * (1 - alpha)
                              + color.astype(np.float32) * alpha).astype(np.uint8)

        seg_u8 = seg.astype(np.uint8)
        contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1)

        score = m.get("predicted_iou", 0.0)
        bbox = m.get("bbox", None)
        if bbox and len(bbox) == 4:
            x, y = int(bbox[0]), max(int(bbox[1]) - 5, 12)
            cv2.putText(overlay, f"{tl[0]}:{score:.2f}", (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        color.tolist(), 1, cv2.LINE_AA)

    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_instance_overlay(path: Path, image: np.ndarray, instance_map: np.ndarray) -> None:
    """Generate QA overlay with colored instances from a merged instance_map."""
    overlay = image.copy()

    unique_ids = np.unique(instance_map)
    unique_ids = unique_ids[unique_ids > 0]

    np.random.seed(42)
    colors = np.random.randint(50, 255, size=(256, 3))

    for inst_id in unique_ids:
        mask = instance_map == inst_id
        color = colors[int(inst_id) % 256]
        overlay[mask] = (overlay[mask] * 0.5 + color * 0.5).astype(np.uint8)

    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
