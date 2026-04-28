"""
Static three-tier size-aware score gate for DR1 satellite candidates.

Design:
    - No GT-derived thresholds. All values are static manual constants
      with explicit unit suffixes (*_px, *_score).
    - Reads `area_clean_px` and `score` from the candidate's
      `metrics_snapshot_thin`.

Rules:
    area_clean_px  <  small_area_max_px       -> score >= small_min_score
    area_clean_px  in [small_area_max_px, medium_area_max_px)
                                              -> score >= medium_min_score
    area_clean_px  >= medium_area_max_px      -> score >= large_min_score

Reasons:
    pass_small / drop_small_score
    pass_medium / drop_medium_score
    pass_large / drop_large_score
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

THRESHOLD_VERSION = "score_gate_v1_static"


@dataclass(frozen=True)
class SatelliteScoreGate:
    small_area_max_px: int = 200
    medium_area_max_px: int = 600
    small_min_score: float = 0.60
    medium_min_score: float = 0.20
    large_min_score: float = 0.18

    @property
    def threshold_version(self) -> str:
        return THRESHOLD_VERSION

    def threshold_values(self) -> dict[str, Any]:
        return {
            "small_area_max_px": self.small_area_max_px,
            "medium_area_max_px": self.medium_area_max_px,
            "small_min_score": self.small_min_score,
            "medium_min_score": self.medium_min_score,
            "large_min_score": self.large_min_score,
        }

    def decide(
        self,
        area_clean_px: int,
        score: float,
    ) -> tuple[str, str]:
        """Return (decision, reason).

        decision is 'pass' or 'drop'.
        reason is one of the six tokens listed at module top.
        """
        if area_clean_px < self.small_area_max_px:
            if score >= self.small_min_score:
                return "pass", "pass_small"
            return "drop", "drop_small_score"
        if area_clean_px < self.medium_area_max_px:
            if score >= self.medium_min_score:
                return "pass", "pass_medium"
            return "drop", "drop_medium_score"
        if score >= self.large_min_score:
            return "pass", "pass_large"
        return "drop", "drop_large_score"
