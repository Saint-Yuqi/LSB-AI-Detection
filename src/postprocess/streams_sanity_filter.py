"""
Streams sanity filter – lightweight guard against obvious false positives.

Usage:
    from src.postprocess.streams_sanity_filter import StreamsSanityFilter, load_streams_cfg
    cfg = load_streams_cfg()  # from mask_stats_summary.json
    flt = StreamsSanityFilter(min_area=cfg["min_area"], max_area_px=cfg["max_area_px"],
                              edge_touch_frac=cfg["edge_touch_frac"])
    kept, rejected = flt.filter(masks, H, W)

Args:
    masks:          list of mask dicts with 'segmentation' (H, W) bool numpy.
    min_area:       reject masks with area < min_area pixels.
    max_area_px:    reject masks with area > max_area_px (absolute, from GT stats).
    max_area_frac:  reject masks covering > max_area_frac of total image (fallback when max_area_px is None).
    edge_touch_frac: reject if > this fraction of mask pixels lie on image border.

Reject reasons:
    - sanity_area_low:  area < min_area
    - sanity_area_high: area > max_area_px (or max_area_frac * H * W)
    - sanity_edge:      edge_touch_ratio > edge_touch_frac

Contract:
    - Does NOT reject based on shape (aspect ratio, solidity) — streams are elongated.
    - Lightweight by design: no convex hull, no skeleton, no eigenvalue decomposition.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

_DEFAULT_STATS_JSON = Path(__file__).parents[2] / "outputs" / "mask_stats" / "mask_stats_summary.json"


def load_streams_cfg(
    stats_json: Path | str = _DEFAULT_STATS_JSON,
) -> dict[str, Any]:
    """Load streams filter thresholds from mask_stats_summary.json.

    Returns dict with keys: min_area (int), max_area_px (int|None), edge_touch_frac (float).
    Uses math.ceil for area thresholds to avoid over-filtering at boundary.
    """
    defaults: dict[str, Any] = {"min_area": 50, "max_area_px": None, "edge_touch_frac": 0.8}
    stats_json = Path(stats_json)

    if not stats_json.exists():
        warnings.warn(
            f"Stats not found: {stats_json}; using defaults. "
            "Run: python scripts/analyze_mask_stats.py",
            stacklevel=2,
        )
        return defaults

    try:
        with open(stats_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Failed to parse {stats_json}: {e}; using defaults", stacklevel=2)
        return defaults

    rec = data.get("filter_recommendations", {}).get("streams")
    if rec is None:
        warnings.warn(
            f"Key 'filter_recommendations.streams' missing in {stats_json}; using defaults",
            stacklevel=2,
        )
        return defaults

    min_area_val = rec.get("min_area")
    max_area_val = rec.get("max_area")
    if min_area_val is None:
        warnings.warn(f"Missing 'min_area' in streams recommendations; using {defaults['min_area']}", stacklevel=2)
    if max_area_val is None:
        warnings.warn(f"Missing 'max_area' in streams recommendations; max_area_px disabled", stacklevel=2)

    return {
        "min_area": int(math.ceil(min_area_val)) if min_area_val is not None else defaults["min_area"],
        "max_area_px": int(math.ceil(max_area_val)) if max_area_val is not None else None,
        "edge_touch_frac": defaults["edge_touch_frac"],
    }


class StreamsSanityFilter:
    """Lightweight false-positive guard for streams masks."""

    def __init__(
        self,
        min_area: int = 50,
        max_area_frac: float = 0.5,
        edge_touch_frac: float = 0.8,
        max_area_px: int | None = None,
    ):
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.edge_touch_frac = edge_touch_frac
        self.max_area_px = max_area_px

    def filter(
        self,
        masks: list[dict[str, Any]],
        H: int,
        W: int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Returns:
            (kept, rejected). Rejected masks get 'reject_reason' field.
        """
        if self.max_area_px is not None:
            max_area = self.max_area_px
        else:
            max_area = int(self.max_area_frac * H * W)
        kept, rejected = [], []

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)
            area = m.get("area", int(seg.sum()))

            # --- min area ---
            if area < self.min_area:
                m["reject_reason"] = "sanity_area_low"
                rejected.append(m)
                continue

            # --- max area ---
            if area > max_area:
                m["reject_reason"] = "sanity_area_high"
                rejected.append(m)
                continue

            # --- edge touch ---
            if self.edge_touch_frac < 1.0:
                edge_ratio = _edge_touch_ratio(seg, H, W)
                if edge_ratio > self.edge_touch_frac:
                    m["reject_reason"] = "sanity_edge"
                    rejected.append(m)
                    continue

            kept.append(m)

        return kept, rejected


def _edge_touch_ratio(seg: np.ndarray, H: int, W: int) -> float:
    """Fraction of mask pixels that lie on image border (border_px / total_mask_px)."""
    border_mask = np.zeros_like(seg, dtype=bool)
    border_mask[0, :] = True
    border_mask[-1, :] = True
    border_mask[:, 0] = True
    border_mask[:, -1] = True

    seg_bool = seg.astype(bool)
    border_px = int((seg_bool & border_mask).sum())
    total_px = int(seg_bool.sum())

    if total_px == 0:
        return 0.0
    return border_px / total_px
