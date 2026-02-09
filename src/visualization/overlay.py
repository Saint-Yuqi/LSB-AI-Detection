"""
Overlay visualisation – draw coloured contours on 8-bit image.

Usage:
    from src.visualization.overlay import save_overlay
    save_overlay(image, kept, core_rejected, prior_rejected, out_path)

Layers:
    Green  – kept masks
    Red    – core-rejected masks
    Gray   – prior-rejected masks (optional)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


# RGB order (canvas is RGB, converted to BGR at save)
_COLOUR_KEPT = (0, 255, 0)          # Green
_COLOUR_CORE = (255, 0, 0)          # Red (was BGR, now RGB)
_COLOUR_PRIOR = (128, 128, 128)     # Gray


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
    out_path: str | Path = "overlay.png",
    draw_prior: bool = False,
) -> None:
    """
    Save overlay PNG with multi-layer contours.

    Args:
        image: (H, W, 3) uint8 RGB.
        kept: masks to show in green.
        core_rejected: masks to show in red.
        prior_rejected: masks optionally shown in gray.
        out_path: output file path.
        draw_prior: if True, also draw prior_rejected in gray.
    """
    canvas = image.copy()
    # Draw layers in order: prior (background) → core → kept (foreground)
    if draw_prior and prior_rejected:
        _draw_contours(canvas, prior_rejected, _COLOUR_PRIOR, thickness=1)
    if core_rejected:
        _draw_contours(canvas, core_rejected, _COLOUR_CORE, thickness=2)
    _draw_contours(canvas, kept, _COLOUR_KEPT, thickness=2)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Convert RGB → BGR for cv2.imwrite
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
