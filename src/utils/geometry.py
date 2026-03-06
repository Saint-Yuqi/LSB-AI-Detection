"""
Shared geometry helpers for mask analysis and post-processing.

Usage:
    from src.utils.geometry import discrete_convex_area
    area = discrete_convex_area(coords)

Why pixel-corner expansion:
    scipy.spatial.ConvexHull on pixel *centers* underestimates convex area for
    compact shapes (solidity > 1.0). Expanding each center to its 4 corners
    before hull computation gives the correct discrete convex area.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull

_PIXEL_CORNER_OFFSETS = np.array(
    [[-0.5, -0.5], [-0.5, 0.5], [0.5, -0.5], [0.5, 0.5]]
)


def discrete_convex_area(coords: np.ndarray) -> float:
    """Convex hull area of pixel coordinates, accounting for pixel extent.

    Args:
        coords: (N, 2) array of pixel coordinates (row, col). int or float.

    Returns:
        Convex hull area in px^2. Falls back to len(coords) if hull
        computation fails (degenerate geometry).
    """
    if len(coords) < 3:
        return float(len(coords))
    corners = (
        coords.astype(np.float64)[:, None, :] + _PIXEL_CORNER_OFFSETS[None, :, :]
    ).reshape(-1, 2)
    try:
        return float(ConvexHull(corners).volume)
    except Exception:
        return float(len(coords))
