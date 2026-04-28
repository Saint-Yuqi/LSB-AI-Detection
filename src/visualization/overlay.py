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

import logging
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)


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


_COLOUR_GT_STREAM = (255, 255, 255)
_COLOUR_GT_SATELLITE = (255, 255, 0)
_COLOUR_STREAM_CONTOUR = (100, 149, 237)
_COLOUR_SATELLITE_CONTOUR = (255, 140, 0)
_COLOUR_INNER_GALAXY_CONTOUR = (200, 100, 255)   # purple — tidal_v1 distinct class


def _is_tidal_feature_label(tl: str) -> bool:
    """tidal_features (new) and streams (legacy) share visual semantics."""
    return tl in ("streams", "stellar stream", "tidal_features")

_GTMaskInput = np.ndarray | list[dict[str, Any]]


def _draw_mask_contour_with_score(
    canvas: np.ndarray,
    seg: np.ndarray,
    colour: tuple[int, int, int],
    label: str,
    score: float,
    thickness: int = 2,
) -> None:
    """Draw only the contour of `seg` plus a compact score label, no alpha fill."""
    seg_u8 = seg.astype(np.uint8)
    contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return
    cv2.drawContours(canvas, contours, -1, colour, thickness)

    rows, cols = np.where(seg_u8)
    if len(rows) == 0:
        return
    x = int(cols.min())
    y = max(int(rows.min()) - 5, 12)
    cv2.putText(
        canvas,
        f"{label}:{score:.2f}",
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        colour,
        1,
        cv2.LINE_AA,
    )


def _normalize_gt_masks(gt: _GTMaskInput | None) -> list[np.ndarray]:
    """Convert a GT instance map or GT mask dict list into binary masks."""
    if gt is None:
        return []

    if isinstance(gt, np.ndarray):
        if gt.ndim != 2:
            raise ValueError(f"GT instance map must be 2D, got shape={gt.shape}")
        gt_ids = np.unique(gt)
        return [(gt == gid).astype(np.uint8) for gid in gt_ids if gid > 0]

    masks: list[np.ndarray] = []
    for mask in gt:
        seg = mask.get("segmentation")
        if seg is None:
            raise KeyError("GT mask dict missing 'segmentation'")
        masks.append(seg.astype(np.uint8))
    return masks


def _draw_binary_contours(
    canvas: np.ndarray,
    masks: list[np.ndarray],
    colour: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    """Draw contours for a list of binary masks."""
    for seg in masks:
        if seg.sum() == 0:
            continue
        contours, _ = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, contours, -1, colour, thickness)


def _draw_legend_with_backdrop(
    canvas: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
) -> None:
    """Draw a readable legend with a semi-transparent dark backing box."""
    if not lines:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    line_gap = 8
    padding = 8
    x0 = 10
    y0 = 18

    text_sizes = [
        cv2.getTextSize(text, font, font_scale, text_thickness)[0]
        for text, _ in lines
    ]
    max_width = max(width for width, _ in text_sizes)
    total_height = sum(height for _, height in text_sizes) + line_gap * (len(lines) - 1)
    box_w = max_width + padding * 2
    box_h = total_height + padding * 2

    overlay = canvas.copy()
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x0 + box_w, y0 + box_h),
        (0, 0, 0),
        thickness=-1,
    )
    cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, dst=canvas)

    y = y0 + padding
    for (text, colour), (_, height) in zip(lines, text_sizes, strict=False):
        y += height
        cv2.putText(
            canvas,
            text,
            (x0 + padding, y),
            font,
            font_scale,
            colour,
            text_thickness,
            cv2.LINE_AA,
        )
        y += line_gap


def save_evaluation_overlay(
    path: Path,
    image: np.ndarray,
    gt_streams: _GTMaskInput,
    predictions: list[dict[str, Any]],
    gt_satellites: _GTMaskInput | None = None,
) -> None:
    """QA overlay (contour-only):

    - GT stream contours in white (thickness=2).
    - Optional GT satellite contours in yellow (thickness=2).
    - Post stream contours in blue (thickness=2) with 's:{score}' label.
    - Post satellite contours in orange (thickness=2) with 's:{score}' label.

    ``gt_streams`` / ``gt_satellites`` may be either an integer GT instance map
    (0 = background) or a list of GT mask dicts containing ``segmentation``.
    """
    overlay = image.copy()

    _draw_binary_contours(
        overlay,
        _normalize_gt_masks(gt_streams),
        _COLOUR_GT_STREAM,
        thickness=2,
    )
    gt_satellite_masks = _normalize_gt_masks(gt_satellites)
    if gt_satellite_masks:
        _draw_binary_contours(
            overlay,
            gt_satellite_masks,
            _COLOUR_GT_SATELLITE,
            thickness=2,
        )

    for m in predictions:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        tl = m.get("type_label", "")
        score = float(m.get("score", 0.0))
        if _is_tidal_feature_label(tl):
            _draw_mask_contour_with_score(overlay, seg, _COLOUR_STREAM_CONTOUR, "s", score, thickness=2)
        elif tl == "inner_galaxy":
            _draw_mask_contour_with_score(overlay, seg, _COLOUR_INNER_GALAXY_CONTOUR, "s", score, thickness=2)
        else:
            _draw_mask_contour_with_score(overlay, seg, _COLOUR_SATELLITE_CONTOUR, "s", score, thickness=2)

    legend_lines: list[tuple[str, tuple[int, int, int]]] = [
        ("GT streams (white)", _COLOUR_GT_STREAM),
    ]
    if gt_satellite_masks:
        legend_lines.append(("GT satellites (yellow)", _COLOUR_GT_SATELLITE))
    legend_lines.extend([
        ("Post streams (blue)", _COLOUR_STREAM_CONTOUR),
        ("Post satellites (orange)", _COLOUR_SATELLITE_CONTOUR),
    ])
    _draw_legend_with_backdrop(overlay, legend_lines)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_raw_overlay(
    path: Path,
    image: np.ndarray,
    raw_streams: list[dict[str, Any]],
    raw_satellites: list[dict[str, Any]],
) -> None:
    """Contour-only overlay for raw SAM3 output, pre any post-processing.

    - Raw stream contours in blue with 's:{score}' label.
    - Raw satellite contours in orange with 's:{score}' label.
    """
    overlay = image.copy()

    for m in raw_streams:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        score = float(m.get("score", 0.0))
        _draw_mask_contour_with_score(overlay, seg, _COLOUR_STREAM_CONTOUR, "s", score, thickness=2)

    for m in raw_satellites:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        score = float(m.get("score", 0.0))
        _draw_mask_contour_with_score(overlay, seg, _COLOUR_SATELLITE_CONTOUR, "s", score, thickness=2)

    cv2.imwrite(str(path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_pseudo_label_overlay(
    path: Path,
    image: np.ndarray,
    predictions: list[dict[str, Any]],
) -> None:
    """QA overlay for pseudo-labels: prediction fills + contours + scores (no GT)."""
    overlay = image.copy()

    stream_colors = [(100, 149, 237), (65, 105, 225), (30, 144, 255)]
    sat_colors = [(255, 165, 0), (255, 140, 0), (255, 127, 80)]
    inner_colors = [(200, 100, 255), (170, 80, 230), (220, 120, 255)]
    stream_idx, sat_idx, inner_idx = 0, 0, 0

    for m in predictions:
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        mask_bool = seg.astype(bool)
        tl = m.get("type_label", "")

        if _is_tidal_feature_label(tl):
            color = np.array(stream_colors[stream_idx % len(stream_colors)], dtype=np.uint8)
            stream_idx += 1
        elif tl == "inner_galaxy":
            color = np.array(inner_colors[inner_idx % len(inner_colors)], dtype=np.uint8)
            inner_idx += 1
        else:
            color = np.array(sat_colors[sat_idx % len(sat_colors)], dtype=np.uint8)
            sat_idx += 1

        alpha = 0.45
        overlay[mask_bool] = (overlay[mask_bool].astype(np.float32) * (1 - alpha)
                              + color.astype(np.float32) * alpha).astype(np.uint8)

        seg_u8 = seg.astype(np.uint8)
        contours, _ = cv2.findContours(seg_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, color.tolist(), 1)

        score = m.get("score", 0.0)
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


def _draw_gt_type_aware_contours(
    canvas: np.ndarray,
    gt_instance_map: np.ndarray,
    gt_type_of_id: dict[int, str] | None,
    thickness: int = 2,
) -> tuple[bool, bool]:
    """Draw per-instance GT contours colored by type.

    Streams -> white; satellites -> yellow. Missing type ids fall back to
    yellow (warn-once style via logger.warning).

    Returns:
        (has_streams, has_satellites) flags for caller legend composition.
    """
    has_streams = False
    has_satellites = False
    type_map = gt_type_of_id or {}

    for inst_id in np.unique(gt_instance_map):
        if inst_id == 0:
            continue
        inst_type = type_map.get(int(inst_id))
        if inst_type == "streams":
            colour = _COLOUR_GT_STREAM
            has_streams = True
        elif inst_type == "satellites":
            colour = _COLOUR_GT_SATELLITE
            has_satellites = True
        else:
            logger.warning(
                "GT instance id=%s has unknown type=%r; defaulting to yellow.",
                int(inst_id), inst_type,
            )
            colour = _COLOUR_GT_SATELLITE
            has_satellites = True

        binary = (gt_instance_map == inst_id).astype(np.uint8)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(canvas, contours, -1, colour, thickness)

    return has_streams, has_satellites


def _draw_roi_box(
    canvas: np.ndarray,
    roi_bbox: tuple[int, int, int, int],
    colour: tuple[int, int, int] = _COLOUR_GT_SATELLITE,
    thickness: int = 2,
) -> None:
    """Draw a rectangle for a ``(y0, x0, y1, x1)`` ROI on the canvas.

    The contract matches ``src.evaluation.checkpoint_eval.Sample.roi_bbox_1024``
    (and ``FBOX_ROI_1024``). The (y, x) -> (x, y) swap required by cv2
    happens in this one place only.
    """
    y0, x0, y1, x1 = roi_bbox
    cv2.rectangle(canvas, (x0, y0), (x1, y1), colour, thickness)


def save_gt_contour_only_overlay(
    path: str | Path,
    render_rgb: np.ndarray,
    gt_instance_map: np.ndarray,
    sample_label: str,
    gt_type_of_id: dict[int, str] | None = None,
    roi_bbox: tuple[int, int, int, int] | None = None,
) -> None:
    """Evaluation overlay: type-aware GT contours + optional ROI box + label.

    Args:
        path:             Output PNG path.
        render_rgb:       (H, W, 3) uint8 RGB image (already at working grid).
        gt_instance_map:  (H, W) int array, 0 = background, positive = instance IDs.
        sample_label:     Text shown in the top-left label box
                          (e.g. "{base_key} | GT=N | SAT=N | STR=N").
                          Must be ASCII-only (OpenCV Hershey font limitation).
        gt_type_of_id:    Mapping ``{instance_id: "streams"|"satellites"}``.
                          Streams are drawn in white, satellites in yellow.
                          Ids missing from this mapping fall back to yellow
                          (logged as a warning).
        roi_bbox:         Optional ROI rectangle in ``(y0, x0, y1, x1)`` order
                          matching ``Sample.roi_bbox_1024``. Drawn in yellow.

    Intentionally does NOT draw prediction fills. Sole purpose: GT visual
    reference for eval QA with per-type colour differentiation.
    """
    canvas = render_rgb.copy()

    has_streams, has_satellites = _draw_gt_type_aware_contours(
        canvas, gt_instance_map, gt_type_of_id, thickness=2,
    )

    if roi_bbox is not None:
        _draw_roi_box(canvas, roi_bbox, _COLOUR_GT_SATELLITE, thickness=2)

    legend_lines: list[tuple[str, tuple[int, int, int]]] = [
        (sample_label, _COLOUR_GT_STREAM),
    ]
    if has_streams:
        legend_lines.append(("GT streams (white)", _COLOUR_GT_STREAM))
    if has_satellites:
        legend_lines.append(("GT satellites (yellow)", _COLOUR_GT_SATELLITE))
    if roi_bbox is not None:
        legend_lines.append(("ROI (yellow box)", _COLOUR_GT_SATELLITE))
    _draw_legend_with_backdrop(canvas, legend_lines)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


# --------------------------------------------------------------------------- #
#  Evaluation prediction overlay (GT + ROI + per-prediction contours + scores)
# --------------------------------------------------------------------------- #

_PRED_PALETTE_SEED = 42


def _build_pred_palettes(
    n_streams: int, n_satellites: int
) -> tuple[list[tuple[int, int, int]], list[tuple[int, int, int]]]:
    """Build deterministic per-type random RGB palettes.

    Streams use cool ranges (B dominant); satellites use warm ranges
    (R dominant). Seeding is fixed so repeated calls for the same counts
    produce byte-identical PNGs.
    """
    rng = np.random.default_rng(_PRED_PALETTE_SEED)
    stream_palette: list[tuple[int, int, int]] = []
    for _ in range(max(n_streams, 1)):
        r = int(rng.integers(40, 141))
        g = int(rng.integers(100, 201))
        b = int(rng.integers(150, 256))
        stream_palette.append((r, g, b))
    sat_palette: list[tuple[int, int, int]] = []
    for _ in range(max(n_satellites, 1)):
        r = int(rng.integers(200, 256))
        g = int(rng.integers(80, 201))
        b = int(rng.integers(30, 121))
        sat_palette.append((r, g, b))
    return stream_palette, sat_palette


def save_eval_prediction_overlay(
    path: str | Path,
    render_rgb: np.ndarray,
    gt_instance_map: np.ndarray,
    gt_type_of_id: dict[int, str] | None,
    predictions: list[dict[str, Any]],
    roi_bbox: tuple[int, int, int, int] | None,
    layer_label: str,
    sample_label: str,
) -> None:
    """QA overlay: type-aware GT + optional ROI + per-prediction contours.

    - GT streams drawn in white, GT satellites in yellow (shared with
      ``save_gt_contour_only_overlay``).
    - Optional yellow ROI rectangle, ``(y0, x0, y1, x1)`` order.
    - Each prediction gets its own colour from a per-type deterministic
      palette (seed=42): streams cool, satellites warm. Contour + compact
      ``{t}:{score:.2f}`` label (``t`` = ``s`` for streams, ``c`` for
      satellites). Note: the legacy helpers ``save_evaluation_overlay`` /
      ``save_raw_overlay`` mis-label satellites as ``s`` — that's a
      separate bug left alone here to keep the fix scoped.

    Args:
        path:             Output PNG path.
        render_rgb:       (H, W, 3) uint8 RGB image.
        gt_instance_map:  (H, W) int array, 0 = background, +ve = instance IDs.
        gt_type_of_id:    ``{id: "streams"|"satellites"}`` mapping.
        predictions:      List of pred dicts (each needs ``segmentation``,
                          ``type_label``, ``score``).
        roi_bbox:         Optional ``(y0, x0, y1, x1)`` ROI rectangle.
        layer_label:      One of ``"raw" | "post_pred_only" | "post_gt_aware"``.
                          Shown in the legend for debugging.
        sample_label:     ASCII top-line label (same format as the GT-only
                          overlay).
    """
    canvas = render_rgb.copy()

    has_gt_streams, has_gt_satellites = _draw_gt_type_aware_contours(
        canvas, gt_instance_map, gt_type_of_id, thickness=2,
    )
    if roi_bbox is not None:
        _draw_roi_box(canvas, roi_bbox, _COLOUR_GT_SATELLITE, thickness=2)

    stream_preds = [m for m in predictions if m.get("type_label") == "streams"]
    sat_preds = [m for m in predictions if m.get("type_label") == "satellites"]
    stream_palette, sat_palette = _build_pred_palettes(
        len(stream_preds), len(sat_preds),
    )

    has_pred_streams = False
    has_pred_satellites = False

    for idx, m in enumerate(stream_preds):
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        colour = stream_palette[idx % len(stream_palette)]
        score = float(m.get("score", 0.0))
        _draw_mask_contour_with_score(canvas, seg, colour, "s", score, thickness=2)
        has_pred_streams = True

    for idx, m in enumerate(sat_preds):
        seg = m.get("segmentation")
        if seg is None or seg.sum() == 0:
            continue
        colour = sat_palette[idx % len(sat_palette)]
        score = float(m.get("score", 0.0))
        _draw_mask_contour_with_score(canvas, seg, colour, "c", score, thickness=2)
        has_pred_satellites = True

    legend_lines: list[tuple[str, tuple[int, int, int]]] = [
        (sample_label, _COLOUR_GT_STREAM),
        (f"layer: {layer_label}", _COLOUR_GT_STREAM),
    ]
    if has_gt_streams:
        legend_lines.append(("GT streams (white)", _COLOUR_GT_STREAM))
    if has_gt_satellites:
        legend_lines.append(("GT satellites (yellow)", _COLOUR_GT_SATELLITE))
    if has_pred_streams:
        legend_lines.append(
            (f"Pred streams (cool, n={len(stream_preds)})", stream_palette[0]),
        )
    if has_pred_satellites:
        legend_lines.append(
            (f"Pred satellites (warm, n={len(sat_preds)})", sat_palette[0]),
        )
    if roi_bbox is not None:
        legend_lines.append(("ROI (yellow box)", _COLOUR_GT_SATELLITE))
    _draw_legend_with_backdrop(canvas, legend_lines)

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
