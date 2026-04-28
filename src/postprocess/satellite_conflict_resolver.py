"""
DR1 stream-vs-satellite conflict resolver.

Design:
    - Inputs are the GT `streams_instance_map.npy` (positive int IDs are
      real stream instances, 0 is background) and the list of alive
      satellite candidates that already survived prior + core policy.
    - For each satellite candidate, compute per-stream overlap with
      every positive stream ID. Pick the stream with the largest
      overlap as `matched_stream_id` (keeps real instance ID, NOT list
      index).
    - A satellite only wins over the matched stream when it is
      overwhelmingly compact-and-aligned; otherwise it is dropped and
      the decision event carries `matched_stream_id` plus a specific
      reason token.

Reasons:
    pass_no_stream_conflict
    satellite_wins
    drop_satellite (with fine-grained sub-reason):
        area_under_600_swallowed_by_stream
        not_compact_enough_to_win
        lost_to_stream_area_ge_600
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

THRESHOLD_VERSION = "conflict_policy_v1_dr1"


@dataclass(frozen=True)
class SatelliteConflictResolver:
    rescue_overlap_ratio_satellite: float = 0.90
    rescue_area_min_px: int = 600
    rescue_solidity_min: float = 0.90
    rescue_aspect_max: float = 1.80

    @property
    def threshold_version(self) -> str:
        return THRESHOLD_VERSION

    def threshold_values(self) -> dict[str, Any]:
        return {
            "rescue_overlap_ratio_satellite": self.rescue_overlap_ratio_satellite,
            "rescue_area_min_px": self.rescue_area_min_px,
            "rescue_solidity_min": self.rescue_solidity_min,
            "rescue_aspect_max": self.rescue_aspect_max,
        }

    def match_stream(
        self,
        sat_seg: np.ndarray,
        streams_instance_map: np.ndarray,
    ) -> tuple[int | None, int, float, float]:
        """Return (matched_stream_id, overlap_px, overlap_ratio_satellite, overlap_ratio_stream).

        matched_stream_id is None if the satellite does not overlap any
        positive stream instance. overlap ratios default to 0.0 in that
        case.
        """
        sat_bool = sat_seg.astype(bool)
        sat_area = int(sat_bool.sum())
        if sat_area == 0:
            return None, 0, 0.0, 0.0

        # Consider only pixels inside the satellite to avoid scanning
        # the whole image for IDs that cannot match.
        overlap_ids = streams_instance_map[sat_bool]
        overlap_ids = overlap_ids[overlap_ids > 0]
        if overlap_ids.size == 0:
            return None, 0, 0.0, 0.0

        ids, counts = np.unique(overlap_ids, return_counts=True)
        best_idx = int(np.argmax(counts))
        matched_id = int(ids[best_idx])
        overlap_px = int(counts[best_idx])

        stream_area = int((streams_instance_map == matched_id).sum())
        overlap_ratio_sat = overlap_px / sat_area if sat_area > 0 else 0.0
        overlap_ratio_stream = overlap_px / stream_area if stream_area > 0 else 0.0
        return matched_id, overlap_px, overlap_ratio_sat, overlap_ratio_stream

    def decide(
        self,
        matched_stream_id: int | None,
        overlap_ratio_satellite: float,
        area_clean_px: int,
        solidity: float,
        aspect_sym_moment: float,
    ) -> tuple[str, str, dict[str, Any]]:
        """Return (decision, reason, extras).

        decision is one of: 'pass', 'win', 'drop'.
        extras carries `matched_stream_id` (None when no overlap).
        """
        extras: dict[str, Any] = {"matched_stream_id": matched_stream_id}

        if matched_stream_id is None:
            return "pass", "pass_no_stream_conflict", extras

        compact = (
            solidity >= self.rescue_solidity_min
            and aspect_sym_moment <= self.rescue_aspect_max
        )
        overlap_ok = overlap_ratio_satellite >= self.rescue_overlap_ratio_satellite
        large_enough = area_clean_px >= self.rescue_area_min_px

        if overlap_ok and large_enough and compact:
            return "win", "satellite_wins", extras

        if area_clean_px < self.rescue_area_min_px:
            return "drop", "area_under_600_swallowed_by_stream", extras
        if not compact:
            return "drop", "not_compact_enough_to_win", extras
        return "drop", "lost_to_stream_area_ge_600", extras
