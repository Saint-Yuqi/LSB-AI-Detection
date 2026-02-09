"""
Core exclusion filter – reject masks whose centroid falls inside central galaxy radius.

Usage:
    from src.postprocess.core_exclusion_filter import CoreExclusionFilter
    flt = CoreExclusionFilter(radius_frac=0.08)
    kept, core_hits, diag = flt.filter(masks, H, W)

Returns:
    kept: masks with centroid outside R_exclude
    core_hits: masks rejected by core filter
    diag: dict with keys {dist_p05, dist_p50, dist_p95, core_areas, core_solidities}
"""
from __future__ import annotations

from typing import Any

import numpy as np


class CoreExclusionFilter:
    """Exclude masks whose centroid is within radius_frac * min(H, W) of image centre."""

    def __init__(self, radius_frac: float = 0.08):
        self.radius_frac = radius_frac

    def filter(
        self,
        masks: list[dict[str, Any]],
        H: int,
        W: int,
    ) -> tuple[list[dict], list[dict], dict[str, Any]]:
        """
        Args:
            masks: list of SAM2 mask dicts with 'segmentation'.
            H, W: image dimensions.

        Returns:
            (kept, core_hits, diagnostics)
        """
        R_exclude = self.radius_frac * min(H, W)
        cx, cy = W / 2.0, H / 2.0

        kept, core_hits = [], []
        all_dists: list[float] = []

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)
            rows, cols = np.where(seg)
            if len(rows) == 0:
                core_hits.append(m)
                continue
            # Centroid
            cen_y, cen_x = float(rows.mean()), float(cols.mean())
            dist = np.hypot(cen_x - cx, cen_y - cy)
            m["centroid_x"] = cen_x
            m["centroid_y"] = cen_y
            m["dist_to_center"] = dist
            all_dists.append(dist)

            if dist < R_exclude:
                m["reject_reason"] = "core"
                core_hits.append(m)
            else:
                kept.append(m)

        # Diagnostics --------------------------------------------------------
        dists_arr = np.array(all_dists) if all_dists else np.array([0.0])
        diag: dict[str, Any] = {
            "R_exclude": R_exclude,
            "dist_p05": float(np.percentile(dists_arr, 5)),
            "dist_p50": float(np.percentile(dists_arr, 50)),
            "dist_p95": float(np.percentile(dists_arr, 95)),
        }
        # Rejected core hit distributions
        if core_hits:
            core_areas = [int(np.sum(m["segmentation"])) for m in core_hits]
            core_solidities = [m.get("solidity", 0.0) for m in core_hits]
            diag["core_area_min"] = min(core_areas)
            diag["core_area_max"] = max(core_areas)
            diag["core_area_mean"] = float(np.mean(core_areas))
            diag["core_solidity_mean"] = float(np.mean(core_solidities)) if any(core_solidities) else None
        else:
            diag["core_area_min"] = None
            diag["core_area_max"] = None
            diag["core_area_mean"] = None
            diag["core_solidity_mean"] = None

        return kept, core_hits, diag
