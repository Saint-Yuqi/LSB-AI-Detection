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
