"""
Mask metrics – vectorised per-mask geometry computation.

Usage:
    from src.analysis.mask_metrics import compute_mask_metrics, append_metrics_to_masks
    metrics = compute_mask_metrics(seg, H, W)
    append_metrics_to_masks(masks, H, W, compute_hull=True)

Contract:
    - Input `seg` must be decoded binary numpy mask (0/1 uint8), NOT RLE.
    - If AutoMaskRunner returns RLE, caller must decode before calling metrics.

Keys:
    area_raw, area_clean, bbox_w, bbox_h, centroid_xy, dist_to_center
    aspect_sym_moment:   major/minor from covariance eigenvalues (rotation-invariant)
    aspect_sym_boundary: major/minor from cv2.fitEllipse on contour (shape-boundary-aware)
    curvature_ratio:     skeleton_length / ellipse_major (>1 = curved/bent structure)
    aspect_sym:          alias for aspect_sym_moment (backward compat)
    solidity* (computed only if compute_hull=True AND area_clean >= min_area_for_hull)

Clean mask definition:
    1. Keep largest connected component (8-connectivity)
    2. Fill holes (binary_fill_holes)
    3. area_clean = sum(clean_mask)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.ndimage import label, binary_fill_holes
from scipy.spatial import ConvexHull

# Gate for hull computation: skip convex hull for tiny masks (unstable + slow)
MIN_AREA_FOR_HULL = 10
# fitEllipse needs >= 5 contour points; skeleton needs reasonable area
MIN_AREA_FOR_ELLIPSE = 5
MIN_AREA_FOR_SKELETON = 20


def _clean_mask(seg: np.ndarray) -> np.ndarray:
    """
    Keep largest connected component (8-connectivity) + fill holes.
    Returns cleaned binary mask (uint8 0/1).
    """
    binary = seg.astype(np.uint8)
    # 8-connectivity structure
    struct = np.ones((3, 3), dtype=np.uint8)
    labeled, n = label(binary, structure=struct)
    if n == 0:
        return binary
    # Find largest component (skip background label 0)
    sizes = np.bincount(labeled.ravel())
    sizes[0] = 0  # ignore background
    largest = np.argmax(sizes)
    clean = (labeled == largest).astype(np.uint8)
    # Fill holes
    return binary_fill_holes(clean).astype(np.uint8)


def compute_mask_metrics(
    seg: np.ndarray,
    H: int,
    W: int,
    compute_hull: bool = True,
) -> dict[str, Any]:
    """
    Vectorised geometry of a binary mask.

    Args:
        seg: (H, W) uint8/bool binary mask (MUST be decoded, not RLE).
        H, W: image size for dist_to_center.
        compute_hull: if False, skip convex hull (faster).

    Returns:
        dict of scalar metrics including area_raw, area_clean, centroid_xy, etc.
    """
    binary = seg.astype(np.uint8)
    area_raw = int(np.sum(binary))
    if area_raw == 0:
        return {"area_raw": 0, "area_clean": 0}

    # Clean mask for stable metrics
    clean = _clean_mask(binary)
    area_clean = int(np.sum(clean))

    # Compute centroid from CLEAN mask (reduces noise influence on grouping)
    rows, cols = np.where(clean) if area_clean > 0 else np.where(binary)
    if len(rows) == 0:
        return {"area_raw": area_raw, "area_clean": 0}

    # Bounding box (kept for reference)
    y_min, y_max = int(rows.min()), int(rows.max())
    x_min, x_max = int(cols.min()), int(cols.max())
    bbox_h = y_max - y_min + 1
    bbox_w = x_max - x_min + 1

    # --- Aspect ratios (dual method) ---
    aspect_sym_moment = _aspect_sym_moment(rows, cols)
    aspect_sym_boundary = _aspect_sym_boundary(clean, area_clean)
    curvature_ratio = _curvature_ratio(clean, area_clean, rows, cols)

    # Centroid (from clean mask)
    cen_y, cen_x = float(rows.mean()), float(cols.mean())
    centroid_xy = (round(cen_x, 2), round(cen_y, 2))

    # Distance to image center
    cx, cy = W / 2.0, H / 2.0
    dist_to_center = float(np.hypot(cen_x - cx, cen_y - cy))

    # Solidity (lazy, gated by area)
    solidity = None
    if compute_hull and area_clean >= MIN_AREA_FOR_HULL:
        hull_coords = np.column_stack((rows, cols))
        if len(hull_coords) >= 3:
            try:
                hull = ConvexHull(hull_coords)
                convex_area = float(hull.volume)  # 2D: volume = area
                solidity = area_clean / convex_area if convex_area > 0 else 1.0
            except Exception:
                solidity = 1.0
        else:
            solidity = 1.0

    return {
        "area_raw": area_raw,
        "area_clean": area_clean,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "aspect_sym_moment": round(aspect_sym_moment, 4),
        "aspect_sym_boundary": round(aspect_sym_boundary, 4) if aspect_sym_boundary is not None else None,
        "curvature_ratio": round(curvature_ratio, 4) if curvature_ratio is not None else None,
        "aspect_sym": round(aspect_sym_moment, 4),  # backward compat alias
        "solidity": round(solidity, 4) if solidity is not None else None,
        "centroid_xy": centroid_xy,
        "centroid_x": centroid_xy[0],  # legacy compat
        "centroid_y": centroid_xy[1],  # legacy compat
        "dist_to_center": round(dist_to_center, 2),
    }


# ---------------------------------------------------------------------------
#  Helpers: dual aspect ratio + curvature
# ---------------------------------------------------------------------------

def _aspect_sym_moment(rows: np.ndarray, cols: np.ndarray) -> float:
    """Major/minor from covariance eigenvalues (rotation-invariant, global shape)."""
    if len(rows) < 3:
        return 1.0
    coords = np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    return float(np.sqrt(eigvals[1] / eigvals[0]))  # major/minor ∈ [1, ∞)


def _aspect_sym_boundary(clean: np.ndarray, area_clean: int) -> float | None:
    """Major/minor from cv2.fitEllipse on largest contour (boundary-aware)."""
    if area_clean < MIN_AREA_FOR_ELLIPSE:
        return None
    try:
        import cv2
    except ImportError:
        return None
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=len)  # largest contour by point count
    if len(cnt) < 5:  # fitEllipse needs >= 5 points
        return None
    (_, (w_ell, h_ell), _) = cv2.fitEllipse(cnt)  # (center, (w,h), angle)
    if min(w_ell, h_ell) < 1e-6:
        return None
    return float(max(w_ell, h_ell) / min(w_ell, h_ell))


def _curvature_ratio(clean: np.ndarray, area_clean: int,
                     rows: np.ndarray, cols: np.ndarray) -> float | None:
    """skeleton_length / ellipse_major_axis. >1 means curved/bent structure."""
    if area_clean < MIN_AREA_FOR_SKELETON:
        return None
    try:
        from skimage.morphology import skeletonize
    except ImportError:
        return None
    skel = skeletonize(clean.astype(bool))
    skel_len = float(np.sum(skel))
    if skel_len < 1:
        return None
    # Major axis from covariance eigenvalues (consistent with moment method)
    coords = np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))
    if len(coords) < 3:
        return None
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    # semi-major ≈ 2 * sqrt(eigval_max)  (covers ~95% of a Gaussian)
    major_axis = 2.0 * np.sqrt(max(eigvals.max(), 1e-12))
    if major_axis < 1:
        return None
    return float(skel_len / major_axis)


def append_metrics_to_masks(
    masks: list[dict[str, Any]],
    H: int,
    W: int,
    compute_hull: bool = True,
) -> None:
    """
    In-place add geometry metrics to each mask dict.

    Contract: masks[i]["segmentation"] must be decoded binary numpy (0/1), NOT RLE.
    """
    for m in masks:
        seg = m["segmentation"]
        metrics = compute_mask_metrics(seg, H, W, compute_hull)
        m.update(metrics)
