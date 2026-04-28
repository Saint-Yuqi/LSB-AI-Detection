"""
Core-region policy for DR1 satellite candidates.

DEPRECATED on the new (tidal_v1) GT path. The 3-class refactor handles
hard-center satellites via ``SatellitePriorFilter.decide_with_target`` —
they are relabeled as ``inner_galaxy`` rather than dropped, and the
runner skips this stage when constructed with
``enable_core_policy=False``. This module is retained for one transitional
cycle so legacy configs (default ``paths.gt_subdir``) and existing
imports / tests keep working unchanged.

Design (legacy path):
    - Hard core: `dist_to_center_frac < hard_core_radius_frac` -> unconditional reject.
    - Soft core: `hard <= dist_to_center_frac < soft_core_radius_frac`
        -> only survive if the strict soft-core rescue rules all hold.
    - Outside core: `dist_to_center_frac >= soft_core_radius_frac` -> pass.

Soft-core rescue (strict v1):
    area_clean_px     >= rescue_area_min_px
    score             >= rescue_score_min
    solidity          >= rescue_solidity_min
    aspect_sym_moment <= rescue_aspect_max

Reasons:
    drop_hard_core
    rescue_soft_core
    drop_soft_core
    pass_outside_core
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

THRESHOLD_VERSION = "core_policy_v1_strict"


@dataclass(frozen=True)
class SatelliteCorePolicy:
    hard_core_radius_frac: float = 0.03
    soft_core_radius_frac: float = 0.08
    rescue_area_min_px: int = 600
    rescue_solidity_min: float = 0.90
    rescue_aspect_max: float = 1.80
    rescue_score_min: float = 0.18

    @property
    def threshold_version(self) -> str:
        return THRESHOLD_VERSION

    def threshold_values(self) -> dict[str, Any]:
        return {
            "hard_core_radius_frac": self.hard_core_radius_frac,
            "soft_core_radius_frac": self.soft_core_radius_frac,
            "rescue_area_min_px": self.rescue_area_min_px,
            "rescue_solidity_min": self.rescue_solidity_min,
            "rescue_aspect_max": self.rescue_aspect_max,
            "rescue_score_min": self.rescue_score_min,
        }

    def decide(
        self,
        dist_to_center_frac: float,
        area_clean_px: int,
        score: float,
        solidity: float,
        aspect_sym_moment: float,
    ) -> tuple[str, str]:
        """Return (decision, reason).

        decision is one of: 'pass', 'rescue', 'drop'.
        """
        if dist_to_center_frac < self.hard_core_radius_frac:
            return "drop", "drop_hard_core"
        if dist_to_center_frac < self.soft_core_radius_frac:
            if (
                area_clean_px >= self.rescue_area_min_px
                and score >= self.rescue_score_min
                and solidity >= self.rescue_solidity_min
                and aspect_sym_moment <= self.rescue_aspect_max
            ):
                return "rescue", "rescue_soft_core"
            return "drop", "drop_soft_core"
        return "pass", "pass_outside_core"
