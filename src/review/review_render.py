"""
Review-asset rendering: candidate crops, bare context, stamped context,
and full-image EV views.

Rendering conventions follow ``src/visualization/overlay.py``
(RGB canvas, cv2 contours).  All outputs strip PNG metadata timestamps
for byte-determinism.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from src.review.render_spec import RenderSpec
from src.review.schemas import CropSpec
from src.utils.coco_utils import decode_rle


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _save_deterministic_png(img_rgb: np.ndarray, path: Path) -> None:
    """Save *img_rgb* (H, W, 3 uint8 RGB) as PNG with no metadata chunks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    pil = Image.fromarray(img_rgb)
    pil.save(str(path), format="PNG", optimize=False)


def _crop_around(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    size: int,
    pad_value: int = 0,
) -> np.ndarray:
    """Centre-crop ``size x size`` from *image*, zero-padding at edges."""
    h, w = image.shape[:2]
    half = size // 2
    x0 = center_x - half
    y0 = center_y - half

    pad_top = max(0, -y0)
    pad_left = max(0, -x0)
    pad_bottom = max(0, (y0 + size) - h)
    pad_right = max(0, (x0 + size) - w)

    y0c = max(0, y0)
    x0c = max(0, x0)
    y1c = min(h, y0 + size)
    x1c = min(w, x0 + size)

    crop = image[y0c:y1c, x0c:x1c].copy()

    if pad_top or pad_bottom or pad_left or pad_right:
        crop = np.pad(
            crop,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0))
            if crop.ndim == 3
            else ((pad_top, pad_bottom), (pad_left, pad_right)),
            mode="constant",
            constant_values=pad_value,
        )
    return crop


# ---------------------------------------------------------------------------
#  Candidate crop (satellite_mv)
# ---------------------------------------------------------------------------

def render_candidate_crop(
    full_image: np.ndarray,
    candidate_rle: dict[str, Any],
    crop_spec: CropSpec,
    spec: RenderSpec,
) -> np.ndarray:
    """Render a centre-cropped view with semi-transparent contour overlay.

    Returns an RGB uint8 array of shape ``(crop_size, crop_size, 3)``.
    """
    mask = decode_rle(candidate_rle)

    crop = _crop_around(
        full_image,
        crop_spec.center_x,
        crop_spec.center_y,
        crop_spec.size,
    )

    mask_crop = _crop_around(
        mask,
        crop_spec.center_x,
        crop_spec.center_y,
        crop_spec.size,
    )

    contours, _ = cv2.findContours(
        mask_crop.astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    overlay = crop.copy()
    cv2.drawContours(
        overlay, contours, -1,
        spec.contour_color,
        spec.contour_thickness,
    )
    alpha = spec.contour_alpha
    result = cv2.addWeighted(overlay, alpha, crop, 1 - alpha, 0)
    return result.astype(np.uint8)


# ---------------------------------------------------------------------------
#  Bare context (shared, unannotated)
# ---------------------------------------------------------------------------

def render_bare_context(full_image: np.ndarray) -> np.ndarray:
    """Return the full image without any annotations (identity copy)."""
    return full_image.copy()


# ---------------------------------------------------------------------------
#  Stamped context (per-candidate, produced at ETL time)
# ---------------------------------------------------------------------------

def stamp_context(
    bare_context: np.ndarray,
    candidate_bbox_xywh: tuple[int, int, int, int],
    candidate_id: str,
    spec: RenderSpec,
) -> np.ndarray:
    """Draw a thin coloured bbox + candidate_id label onto a bare context copy.

    This is called at ETL time, NOT at asset-build time, to preserve
    the dedup invariant (one shared bare context per sample).
    """
    canvas = bare_context.copy()
    x, y, w, h = candidate_bbox_xywh
    cv2.rectangle(
        canvas, (x, y), (x + w, y + h),
        spec.bbox_color, spec.bbox_thickness,
    )
    cv2.putText(
        canvas, candidate_id,
        (x, max(y - 4, 10)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        spec.bbox_color, 1, cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
#  Full-image EV view
# ---------------------------------------------------------------------------

def render_ev_full_image(
    full_image: np.ndarray,
    fragment_hints: list[dict[str, Any]] | None,
    spec: RenderSpec,
) -> np.ndarray:
    """Return full-image review view, with optional stream fragment hints."""
    canvas = full_image.copy()
    if spec.fragment_hints_enabled and fragment_hints:
        for hint in fragment_hints:
            rle = hint.get("rle")
            if rle is None:
                continue
            mask = decode_rle(rle).astype(np.uint8)
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            overlay = canvas.copy()
            cv2.drawContours(
                overlay, contours, -1,
                spec.contour_color, spec.contour_thickness,
            )
            canvas = cv2.addWeighted(
                overlay, spec.contour_alpha,
                canvas, 1 - spec.contour_alpha, 0,
            )
    return canvas


# ---------------------------------------------------------------------------
#  Public save wrappers
# ---------------------------------------------------------------------------

def save_crop(
    full_image: np.ndarray,
    candidate_rle: dict[str, Any],
    crop_spec: CropSpec,
    spec: RenderSpec,
    out_path: Path,
) -> None:
    """Render and save a candidate crop PNG."""
    img = render_candidate_crop(full_image, candidate_rle, crop_spec, spec)
    _save_deterministic_png(img, out_path)


def save_bare_context(full_image: np.ndarray, out_path: Path) -> None:
    """Save a bare (unannotated) context PNG."""
    _save_deterministic_png(render_bare_context(full_image), out_path)


def save_stamped_context(
    bare_context: np.ndarray,
    candidate_bbox_xywh: tuple[int, int, int, int],
    candidate_id: str,
    spec: RenderSpec,
    out_path: Path,
) -> None:
    """Stamp and save a per-candidate context PNG (ETL time)."""
    img = stamp_context(bare_context, candidate_bbox_xywh, candidate_id, spec)
    _save_deterministic_png(img, out_path)


def save_ev_full_image(
    full_image: np.ndarray,
    fragment_hints: list[dict[str, Any]] | None,
    spec: RenderSpec,
    out_path: Path,
) -> None:
    """Render and save a full-image EV review PNG."""
    img = render_ev_full_image(full_image, fragment_hints, spec)
    _save_deterministic_png(img, out_path)
