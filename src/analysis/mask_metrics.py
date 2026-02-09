"""
Mask metrics – vectorised per-mask geometry computation.

Usage:
    from src.analysis.mask_metrics import compute_mask_metrics
    metrics = compute_mask_metrics(seg, H, W)

Keys:
    area, bbox_w, bbox_h, aspect_sym, solidity*, centroid_x, centroid_y, dist_to_center

* solidity is computed lazily to avoid slow convex hull for pruned masks (set compute_hull=False).
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import ConvexHull


def compute_mask_metrics(
    seg: np.ndarray,
    H: int,
    W: int,
    compute_hull: bool = True,
) -> dict[str, Any]:
    """
    Vectorised geometry of a binary mask.

    Args:
        seg: (H, W) uint8/bool binary mask.
        H, W: image size for dist_to_center.
        compute_hull: if False, skip convex hull (faster).

    Returns:
        dict of scalar metrics.
    """
    binary = seg.astype(np.uint8)
    area = int(np.sum(binary))
    if area == 0:
        return {"area": 0}

    rows, cols = np.where(binary)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    aspect_sym = max(bbox_w / bbox_h, bbox_h / bbox_w) if min(bbox_w, bbox_h) > 0 else 1.0

    # Centroid -----------------------------------------------------------------
    cen_y, cen_x = float(rows.mean()), float(cols.mean())
    cx, cy = W / 2.0, H / 2.0
    dist_to_center = float(np.hypot(cen_x - cx, cen_y - cy))

    # Solidity (lazy) ----------------------------------------------------------
    solidity = None
    if compute_hull:
        coords = np.column_stack((rows, cols))
        if len(coords) >= 3:
            try:
                hull = ConvexHull(coords)
                convex_area = float(hull.volume)
                solidity = area / convex_area if convex_area > 0 else 1.0
            except Exception:
                solidity = 1.0
        else:
            solidity = 1.0

    return {
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "aspect_sym": round(aspect_sym, 4),
        "solidity": round(solidity, 4) if solidity is not None else None,
        "centroid_x": round(cen_x, 2),
        "centroid_y": round(cen_y, 2),
        "dist_to_center": round(dist_to_center, 2),
    }


def append_metrics_to_masks(
    masks: list[dict[str, Any]],
    H: int,
    W: int,
    compute_hull: bool = True,
) -> None:
    """In-place add geometry metrics to each mask dict."""
    for m in masks:
        seg = m["segmentation"]
        metrics = compute_mask_metrics(seg, H, W, compute_hull)
        m.update(metrics)
