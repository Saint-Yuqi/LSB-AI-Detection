"""
Satellite prior filter – area / solidity / aspect_sym rules.

Usage:
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    cfg = load_filter_cfg()          # from mask_stats_summary.json
    flt = SatellitePriorFilter(cfg)
    kept, rejected = flt.filter(masks)

Args:
    masks: list of dicts with 'segmentation', 'area', etc.
    cfg (optional): dict with area_min, area_max, solidity_min, aspect_sym_max.
           Defaults loaded from outputs/mask_stats/mask_stats_summary.json → satellites.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

# Default recommendations source ------------------------------------
_DEFAULT_STATS_JSON = Path(__file__).parents[2] / "outputs" / "mask_stats" / "mask_stats_summary.json"


def load_filter_cfg(
    stats_json: Path | str = _DEFAULT_STATS_JSON,
    feature_type: str = "satellites",
) -> dict[str, Any]:
    """Load filter thresholds from mask_stats_summary.json. Falls back to hard-coded safe defaults."""
    defaults = {
        "area_min": 30,
        "area_max": 2000,
        "solidity_min": 0.80,
        "aspect_sym_max": 2.0,
    }
    stats_json = Path(stats_json)
    if not stats_json.exists():
        return defaults

    with open(stats_json) as f:
        data = json.load(f)
    rec = data.get("filter_recommendations", {}).get(feature_type, {})
    return {
        "area_min": rec.get("min_area", defaults["area_min"]),
        "area_max": rec.get("max_area", defaults["area_max"]),
        "solidity_min": rec.get("min_solidity", defaults["solidity_min"]),
        "aspect_sym_max": rec.get("aspect_sym_max", defaults["aspect_sym_max"]),
    }


def _compute_solidity(binary: np.ndarray) -> float:
    """Convex-hull-based solidity (area / convex_area)."""
    coords = np.argwhere(binary)
    if len(coords) < 3:
        return 1.0
    try:
        hull = ConvexHull(coords)
        convex_area = float(hull.volume)  # 2D: volume = area
    except Exception:
        convex_area = float(np.sum(binary))
    area = float(np.sum(binary))
    return area / convex_area if convex_area > 0 else 1.0


def _compute_aspect_sym(binary: np.ndarray) -> float:
    """max(w/h, h/w) bounding-box aspect symmetry."""
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return 1.0
    h = rows.max() - rows.min() + 1
    w = cols.max() - cols.min() + 1
    if min(w, h) == 0:
        return 1.0
    return max(w / h, h / w)


class SatellitePriorFilter:
    """Rule-based filter on area, solidity, aspect_sym."""

    def __init__(self, cfg: dict[str, Any] | None = None):
        self.cfg = cfg if cfg else load_filter_cfg()

    def filter(self, masks: list[dict[str, Any]]) -> tuple[list[dict], list[dict]]:
        """Return (kept, rejected) lists."""
        kept, rejected = [], []
        area_min = self.cfg["area_min"]
        area_max = self.cfg["area_max"]
        solidity_min = self.cfg["solidity_min"]
        aspect_sym_max = self.cfg["aspect_sym_max"]

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)
            area = int(np.sum(seg))
            if not (area_min <= area <= area_max):
                m["reject_reason"] = "area"
                rejected.append(m)
                continue

            solidity = _compute_solidity(seg)
            if solidity < solidity_min:
                m["reject_reason"] = "solidity"
                rejected.append(m)
                continue

            aspect_sym = _compute_aspect_sym(seg)
            if aspect_sym > aspect_sym_max:
                m["reject_reason"] = "aspect_sym"
                rejected.append(m)
                continue

            # Pass all checks
            m["solidity"] = solidity
            m["aspect_sym"] = aspect_sym
            kept.append(m)

        return kept, rejected
