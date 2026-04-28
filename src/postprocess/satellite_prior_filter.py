"""
Satellite prior filter – slim version: optional hard-center reject plus
area-min, solidity, aspect_sym.

Usage:
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    cfg = load_filter_cfg()          # from mask_stats_summary.json
    flt = SatellitePriorFilter(cfg)
    kept, rejected, ambiguous = flt.filter(masks)  # ambiguous is always []

Args:
    masks: list of dicts with 'segmentation', 'area_clean', 'solidity',
        'aspect_sym_moment', and optionally 'dist_to_center'.
    cfg: dict with area_min, solidity_min, aspect_sym_max, and optional
        hard_center_radius_frac.

Reject reasons:
    - prior_hard_center: dist_to_center_frac < hard_center_radius_frac
    - prior_area_low: area_clean < area_min
    - prior_solidity: solidity < solidity_min
    - prior_aspect:   aspect_sym_moment > aspect_sym_max

DR1 v4 change: removed `area_max` / `ambiguous_factor` / `ambiguous_max`.
    The ambiguous return slot is kept for backward-compatible call sites
    (pnbody) and is always an empty list. Callers may still pass the old
    `ambiguous_factor` / `core_radius_frac` keys in `cfg`; they are
    accepted and ignored so the old pnbody flow keeps working.
"""
from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.utils.geometry import discrete_convex_area

_DEFAULT_STATS_JSON = Path(__file__).parents[2] / "outputs" / "mask_stats" / "mask_stats_summary.json"

_FILTER_DEFAULTS: dict[str, Any] = {
    "area_min": 30,
    "solidity_min": 0.83,
    "aspect_sym_max": 1.75,
    "hard_center_radius_frac": 0.03,
    "hard_center_action": "drop",
}

_HARD_CENTER_ACTIONS = ("drop", "relabel_inner_galaxy")

_LEGACY_THRESHOLD_VERSION = "prior_v2_slim"
_CENTER_AWARE_THRESHOLD_VERSION = "prior_v3_center_aware"

_JSON_TO_CFG = [
    ("area_min", "min_area"),
    ("solidity_min", "min_solidity"),
    ("aspect_sym_max", "aspect_sym_moment_max"),
]


def load_filter_cfg(
    stats_json: Path | str = _DEFAULT_STATS_JSON,
    feature_type: str = "satellites",
) -> dict[str, Any]:
    """Load slim filter thresholds from mask_stats_summary.json with 3-tier guards.

    Tier 1: file existence. Tier 2: JSON parse. Tier 3: key/field presence.
    Each tier emits warnings.warn and falls back to defaults.
    Only `min_area`, `min_solidity`, `aspect_sym_moment_max` are consulted
    from stats. The optional hard-center threshold is static/config-driven.
    """
    defaults = dict(_FILTER_DEFAULTS)
    stats_json = Path(stats_json)

    if not stats_json.exists():
        warnings.warn(
            f"Stats not found: {stats_json}; using defaults. "
            "Run: python scripts/analysis/analyze_mask_stats.py",
            stacklevel=2,
        )
        return defaults

    try:
        with open(stats_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Failed to parse {stats_json}: {e}; using defaults", stacklevel=2)
        return defaults

    rec = data.get("filter_recommendations", {}).get(feature_type)
    if rec is None:
        warnings.warn(
            f"Key 'filter_recommendations.{feature_type}' missing in "
            f"{stats_json}; using defaults",
            stacklevel=2,
        )
        return defaults

    cfg: dict[str, Any] = dict(defaults)
    for cfg_key, json_key in _JSON_TO_CFG:
        val = rec.get(json_key)
        if json_key == "aspect_sym_moment_max" and val is None:
            val = rec.get("aspect_sym_max")
        if val is None:
            warnings.warn(
                f"Missing '{json_key}' in {feature_type} recommendations; "
                f"using default {defaults[cfg_key]}",
                stacklevel=2,
            )
            cfg[cfg_key] = defaults[cfg_key]
        elif cfg_key == "area_min":
            cfg[cfg_key] = int(math.ceil(val))
        else:
            cfg[cfg_key] = val

    return cfg


def _compute_solidity(binary: np.ndarray) -> float:
    """Convex-hull-based solidity (area / convex_area), pixel-corner-aware."""
    coords = np.argwhere(binary)
    if len(coords) < 3:
        return 1.0
    convex_area = discrete_convex_area(coords)
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


def _compute_dist_to_center(binary: np.ndarray) -> float:
    """Distance from mask centroid to image center in pixels."""
    rows, cols = np.where(binary)
    if len(rows) == 0:
        return float("inf")
    H, W = binary.shape
    cen_y = float(rows.mean())
    cen_x = float(cols.mean())
    return float(np.hypot(cen_x - W / 2.0, cen_y - H / 2.0))


class SatellitePriorFilter:
    """Slim rule-based filter: optional hard-center, area_min, solidity, aspect_sym."""

    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        image_size: int = 1072,  # kept for backward signature compat; unused
    ):
        base_cfg = load_filter_cfg() if cfg is None else {**_FILTER_DEFAULTS, **cfg}
        action = base_cfg.get("hard_center_action", "drop")
        if action not in _HARD_CENTER_ACTIONS:
            raise ValueError(
                f"Invalid hard_center_action {action!r}; expected one of {_HARD_CENTER_ACTIONS}"
            )
        self.cfg = base_cfg
        # image_size retained for signature compat with older callers.
        self._image_size = image_size

    @property
    def threshold_version(self) -> str:
        if self.cfg.get("hard_center_radius_frac") is None:
            return _LEGACY_THRESHOLD_VERSION
        return _CENTER_AWARE_THRESHOLD_VERSION

    def threshold_values(self) -> dict[str, Any]:
        return {
            "area_min": self.cfg["area_min"],
            "solidity_min": self.cfg["solidity_min"],
            "aspect_sym_max": self.cfg["aspect_sym_max"],
            "hard_center_radius_frac": self.cfg.get("hard_center_radius_frac"),
        }

    def decide(self, mask: dict[str, Any]) -> tuple[str, str]:
        """Return (decision, reason) for one candidate.

        decision is one of: 'pass', 'drop'.

        This 2-tuple shape is preserved for legacy callers regardless of the
        configured ``hard_center_action``. Use :meth:`decide_with_target` to
        observe the relabel target on the new path.
        """
        decision, reason, _ = self.decide_with_target(mask)
        # Map the new "relabel" decision back onto legacy semantics: from the
        # perspective of an existing 2-tuple caller, a relabel still means the
        # candidate did not pass the satellite branch — surface it as a drop
        # with the same reason.
        if decision == "relabel":
            return "drop", reason
        return decision, reason

    def decide_with_target(
        self, mask: dict[str, Any]
    ) -> tuple[str, str, str | None]:
        """Return (decision, reason, target_type) for one candidate.

        decision is one of: ``'pass'``, ``'drop'``, ``'relabel'``.
        target_type is set only on ``'relabel'`` (e.g. ``'inner_galaxy'``);
        ``None`` for the other two outcomes.

        When ``cfg["hard_center_action"] == "relabel_inner_galaxy"`` and the
        mask centroid lies within ``hard_center_radius_frac``, returns
        ``("relabel", "prior_hard_center", "inner_galaxy")``. Otherwise the
        legacy decision pipeline runs and the third slot is ``None``.
        """
        seg = mask["segmentation"].astype(np.uint8)
        hard_center_radius_frac = self.cfg.get("hard_center_radius_frac")
        hard_center_action = self.cfg.get("hard_center_action", "drop")

        if hard_center_radius_frac is not None:
            dist_to_center = mask.get("dist_to_center")
            if dist_to_center is None:
                dist_to_center = _compute_dist_to_center(seg)
                mask["dist_to_center"] = dist_to_center
            dist_frac = float(dist_to_center) / float(min(seg.shape))
            if dist_frac < float(hard_center_radius_frac):
                if hard_center_action == "relabel_inner_galaxy":
                    return "relabel", "prior_hard_center", "inner_galaxy"
                return "drop", "prior_hard_center", None

        area = mask.get("area_clean", int(np.sum(seg)))
        if area < self.cfg["area_min"]:
            return "drop", "prior_area_low", None

        solidity = mask.get("solidity")
        if solidity is None:
            solidity = _compute_solidity(seg)
            mask["solidity"] = solidity
        if float(solidity) < float(self.cfg["solidity_min"]):
            return "drop", "prior_solidity", None

        aspect_sym = mask.get("aspect_sym_moment") or mask.get("aspect_sym")
        if aspect_sym is None:
            aspect_sym = _compute_aspect_sym(seg)
        mask["aspect_sym_moment"] = aspect_sym
        mask["aspect_sym"] = aspect_sym
        if float(aspect_sym) > float(self.cfg["aspect_sym_max"]):
            return "drop", "prior_aspect", None

        return "pass", "pass_prior_filter", None

    def filter(
        self,
        masks: list[dict[str, Any]],
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """
        Return (kept, rejected, relabeled) lists.

        - ``kept`` are masks that passed the satellite branch unchanged.
        - ``rejected`` are masks dropped by any prior rule (including
          ``hard_center`` when ``hard_center_action == "drop"``).
        - ``relabeled`` is non-empty only when
          ``hard_center_action == "relabel_inner_galaxy"`` and at least one
          mask was relabeled. Each relabeled mask carries
          ``mask["type_label"] = "inner_galaxy"``,
          ``mask["reject_reason"] = "prior_hard_center"``,
          ``mask["relabel_target"] = "inner_galaxy"``.

        The 3-tuple shape preserves the legacy pnbody caller signature
        (third slot was historically ``ambiguous`` and always empty).
        """
        kept, rejected, relabeled = [], [], []

        for m in masks:
            decision, reason, target = self.decide_with_target(m)
            if decision == "drop":
                m["reject_reason"] = reason
                rejected.append(m)
                continue
            if decision == "relabel":
                m["reject_reason"] = reason
                m["relabel_target"] = target
                if target is not None:
                    m["type_label"] = target
                relabeled.append(m)
                continue
            kept.append(m)

        return kept, rejected, relabeled
