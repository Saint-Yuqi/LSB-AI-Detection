"""
Streams sanity filter – lightweight guard against obvious false positives.

Usage:
    from src.postprocess.streams_sanity_filter import StreamsSanityFilter
    flt = StreamsSanityFilter(min_area=50, max_area_frac=0.5, edge_touch_frac=0.8)
    kept, rejected = flt.filter(masks, H, W)

Args:
    masks:          list of mask dicts with 'segmentation' (H, W) bool numpy.
    min_area:       reject masks with area < min_area pixels.
    max_area_frac:  reject masks covering > max_area_frac of total image.
    edge_touch_frac: reject if > this fraction of perimeter touches image border.

Reject reasons:
    - sanity_area_low:  area < min_area
    - sanity_area_high: area > max_area_frac * H * W
    - sanity_edge:      edge_touch_ratio > edge_touch_frac

Contract:
    - Does NOT reject based on shape (aspect ratio, solidity) — streams are elongated.
    - Lightweight by design: no convex hull, no skeleton, no eigenvalue decomposition.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class StreamsSanityFilter:
    """Lightweight false-positive guard for streams masks."""

    def __init__(
        self,
        min_area: int = 50,
        max_area_frac: float = 0.5,
        edge_touch_frac: float = 0.8,
    ):
        self.min_area = min_area
        self.max_area_frac = max_area_frac
        self.edge_touch_frac = edge_touch_frac

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
    """Fraction of mask perimeter pixels that touch image border."""
    # Border pixels: first/last row, first/last col
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
