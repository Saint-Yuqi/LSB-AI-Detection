"""
Satellite prior filter – area / solidity / aspect_sym rules with ambiguous zone.

Usage:
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    cfg = load_filter_cfg()          # from mask_stats_summary.json
    flt = SatellitePriorFilter(cfg)
    kept, rejected, ambiguous = flt.filter(masks)

Args:
    masks: list of dicts with 'segmentation', 'area_clean', 'solidity', 'aspect_sym_moment', 'dist_to_center'.
    cfg: dict with area_min, area_max, solidity_min, aspect_sym_max, ambiguous_factor, core_radius_frac.

Reject reasons (standardized):
    - prior_area_low: area_clean < area_min
    - prior_area_high: area_clean > ambiguous_max AND fails shape criteria
    - prior_solidity: solidity < solidity_min
    - prior_aspect: aspect_sym_moment > aspect_sym_max
    - ambiguous_area: area in (area_max, ambiguous_max] AND passes shape criteria

Ambiguous criteria (must pass ALL to be ambiguous instead of rejected):
    1. area in (area_max, area_max * (1 + ambiguous_factor)]
    2. solidity >= solidity_min
    3. aspect_sym_moment <= aspect_sym_max
    4. dist_to_center >= core_dist_threshold (from metrics, not H/W)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull

# Default recommendations source
_DEFAULT_STATS_JSON = Path(__file__).parents[2] / "outputs" / "mask_stats" / "mask_stats_summary.json"


def load_filter_cfg(
    stats_json: Path | str = _DEFAULT_STATS_JSON,
    feature_type: str = "satellites",
) -> dict[str, Any]:
    """Load filter thresholds from mask_stats_summary.json. Falls back to safe defaults."""
    defaults = {
        "area_min": 30,
        "area_max": 1842,
        "solidity_min": 0.83,
        "aspect_sym_max": 1.75,
        "ambiguous_factor": 0.25,
        "core_radius_frac": 0.08,
    }
    stats_json = Path(stats_json)
    if not stats_json.exists():
        return defaults

    try:
        with open(stats_json) as f:
            data = json.load(f)
        rec = data.get("filter_recommendations", {}).get(feature_type, {})
        return {
            "area_min": rec.get("min_area", defaults["area_min"]),
            "area_max": rec.get("max_area", defaults["area_max"]),
            "solidity_min": rec.get("min_solidity", defaults["solidity_min"]),
            "aspect_sym_max": rec.get("aspect_sym_moment_max", rec.get("aspect_sym_max", defaults["aspect_sym_max"])),
            "ambiguous_factor": defaults["ambiguous_factor"],
            "core_radius_frac": defaults["core_radius_frac"],
        }
    except Exception:
        return defaults


def _compute_solidity(binary: np.ndarray) -> float:
    """Convex-hull-based solidity (area / convex_area)."""
    coords = np.argwhere(binary)
    if len(coords) < 3:
        return 1.0
    try:
        hull = ConvexHull(coords)
        convex_area = float(hull.volume)
    except Exception:
        convex_area = float(np.sum(binary))
    area = float(np.sum(binary))
    return area / convex_area if convex_area > 0 else 1.0


def _compute_aspect_sym(binary: np.ndarray) -> float:
    """Ellipse axis ratio (major/minor) from covariance eigenvalues. Rotation-invariant."""
    rows, cols = np.where(binary)
    if len(rows) < 3:
        return 1.0
    coords = np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))
    cov = np.cov(coords, rowvar=False)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.clip(eigvals, 1e-12, None)
    return float(np.sqrt(eigvals[1] / eigvals[0]))  # major/minor ∈ [1, ∞)


class SatellitePriorFilter:
    """Rule-based filter on area_clean, solidity, aspect_sym with ambiguous zone."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        image_size: int = 1072,  # for core_dist_threshold fallback
    ):
        self.cfg = cfg if cfg else load_filter_cfg()
        # Core distance threshold (used for ambiguous check)
        # If mask's dist_to_center < this, it's likely core, not ambiguous satellite
        self.core_dist_threshold = self.cfg.get("core_radius_frac", 0.08) * image_size

    def filter(
        self,
        masks: list[dict[str, Any]],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Return (kept, rejected, ambiguous) lists.

        Each rejected/ambiguous mask gets 'reject_reason' field.
        Uses pre-computed metrics: area_clean, solidity, aspect_sym, dist_to_center.
        """
        kept, rejected, ambiguous = [], [], []

        area_min = self.cfg["area_min"]
        area_max = self.cfg["area_max"]
        solidity_min = self.cfg["solidity_min"]
        aspect_sym_max = self.cfg["aspect_sym_max"]
        ambiguous_factor = self.cfg.get("ambiguous_factor", 0.25)
        ambiguous_max = area_max * (1 + ambiguous_factor)

        for m in masks:
            seg = m["segmentation"].astype(np.uint8)

            # Use pre-computed area_clean, fall back to raw sum
            area = m.get("area_clean", int(np.sum(seg)))

            # --- Area check ---
            if area < area_min:
                m["reject_reason"] = "prior_area_low"
                rejected.append(m)
                continue

            # --- Get or compute shape metrics ---
            solidity = m.get("solidity")
            if solidity is None:
                solidity = _compute_solidity(seg)
                m["solidity"] = solidity

            # Prefer aspect_sym_moment (new), fall back to aspect_sym (legacy)
            aspect_sym = m.get("aspect_sym_moment") or m.get("aspect_sym")
            if aspect_sym is None:
                aspect_sym = _compute_aspect_sym(seg)
            m["aspect_sym_moment"] = aspect_sym
            m["aspect_sym"] = aspect_sym  # keep alias in sync

            dist_to_center = m.get("dist_to_center", float("inf"))

            # --- Ambiguous zone check (area > area_max) ---
            if area > area_max:
                # Check if qualifies as ambiguous (passes shape criteria)
                passes_shape = (
                    solidity >= solidity_min
                    and aspect_sym <= aspect_sym_max
                    and dist_to_center >= self.core_dist_threshold
                )

                if area <= ambiguous_max and passes_shape:
                    m["reject_reason"] = "ambiguous_area"
                    ambiguous.append(m)
                else:
                    m["reject_reason"] = "prior_area_high"
                    rejected.append(m)
                continue

            # --- Solidity check ---
            if solidity < solidity_min:
                m["reject_reason"] = "prior_solidity"
                rejected.append(m)
                continue

            # --- Aspect symmetry check ---
            if aspect_sym > aspect_sym_max:
                m["reject_reason"] = "prior_aspect"
                rejected.append(m)
                continue

            # Pass all checks
            kept.append(m)

        return kept, rejected, ambiguous
