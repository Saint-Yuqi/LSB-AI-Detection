"""Unit tests for the DR1 stream-vs-satellite conflict resolver.

Verifies `matched_stream_id` is the real stream instance ID from
`streams_instance_map` (NOT a list index), and covers the three main
outcomes: no-conflict, satellite wins, dropped-and-swallowed.

Usage:
    pytest tests/test_satellite_conflict_resolver.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.satellite_conflict_resolver import SatelliteConflictResolver


@pytest.fixture()
def resolver() -> SatelliteConflictResolver:
    return SatelliteConflictResolver()


def _empty_streams_map(H: int = 64, W: int = 64) -> np.ndarray:
    return np.zeros((H, W), dtype=np.int32)


def _streams_map_with_two_instances(H: int = 64, W: int = 64) -> np.ndarray:
    """Create a streams GT map with two separate instances: IDs 7 and 42.

    Index into the map: (rows, cols). Instance 7 lives top-left,
    instance 42 lives bottom-right. Neither maps to its list index.
    """
    m = np.zeros((H, W), dtype=np.int32)
    m[2:10, 2:10] = 7
    m[40:60, 40:60] = 42
    return m


def test_no_overlap_returns_pass_without_matched_id(
    resolver: SatelliteConflictResolver,
) -> None:
    streams_map = _streams_map_with_two_instances()
    sat_seg = np.zeros_like(streams_map, dtype=bool)
    sat_seg[20:28, 20:28] = True  # completely outside both streams

    matched, _, _, _ = resolver.match_stream(sat_seg, streams_map)
    decision, reason, extras = resolver.decide(
        matched_stream_id=matched,
        overlap_ratio_satellite=0.0,
        area_clean_px=int(sat_seg.sum()),
        solidity=0.95,
        aspect_sym_moment=1.1,
    )
    assert decision == "pass"
    assert reason == "pass_no_stream_conflict"
    assert extras["matched_stream_id"] is None


def test_matched_stream_id_is_real_instance_id_not_list_index(
    resolver: SatelliteConflictResolver,
) -> None:
    streams_map = _streams_map_with_two_instances()
    sat_seg = np.zeros_like(streams_map, dtype=bool)
    sat_seg[3:9, 3:9] = True  # inside stream 7

    matched, overlap_px, ratio_sat, ratio_stream = resolver.match_stream(
        sat_seg, streams_map
    )
    assert matched == 7, "matched_stream_id must be the real stream ID (7), not its list index"
    assert overlap_px > 0
    assert 0.0 < ratio_sat <= 1.0
    assert 0.0 < ratio_stream <= 1.0


def test_matched_picks_max_overlap_id(resolver: SatelliteConflictResolver) -> None:
    streams_map = _streams_map_with_two_instances()
    sat_seg = np.zeros_like(streams_map, dtype=bool)
    # Heavily overlap stream 42 and lightly touch stream 7.
    sat_seg[40:60, 40:60] = True
    sat_seg[2:4, 2:4] = True
    matched, _, _, _ = resolver.match_stream(sat_seg, streams_map)
    assert matched == 42


def test_small_satellite_swallowed_by_stream_has_reason_and_matched_id(
    resolver: SatelliteConflictResolver,
) -> None:
    streams_map = _streams_map_with_two_instances()
    decision, reason, extras = resolver.decide(
        matched_stream_id=42,
        overlap_ratio_satellite=0.95,
        area_clean_px=180,  # < 600 tier
        solidity=0.95,
        aspect_sym_moment=1.2,
    )
    assert decision == "drop"
    assert reason == "area_under_600_swallowed_by_stream"
    assert extras["matched_stream_id"] == 42


def test_large_compact_satellite_wins_over_stream(
    resolver: SatelliteConflictResolver,
) -> None:
    decision, reason, extras = resolver.decide(
        matched_stream_id=7,
        overlap_ratio_satellite=0.95,
        area_clean_px=900,
        solidity=0.95,
        aspect_sym_moment=1.2,
    )
    assert decision == "win"
    assert reason == "satellite_wins"
    assert extras["matched_stream_id"] == 7


def test_non_compact_large_satellite_loses(resolver: SatelliteConflictResolver) -> None:
    decision, reason, extras = resolver.decide(
        matched_stream_id=7,
        overlap_ratio_satellite=0.95,
        area_clean_px=900,
        solidity=0.70,  # below rescue_solidity_min
        aspect_sym_moment=1.2,
    )
    assert decision == "drop"
    assert reason == "not_compact_enough_to_win"
    assert extras["matched_stream_id"] == 7


def test_large_compact_but_low_overlap_loses(
    resolver: SatelliteConflictResolver,
) -> None:
    decision, reason, extras = resolver.decide(
        matched_stream_id=7,
        overlap_ratio_satellite=0.50,  # below rescue_overlap_ratio_satellite
        area_clean_px=900,
        solidity=0.95,
        aspect_sym_moment=1.2,
    )
    assert decision == "drop"
    assert reason == "lost_to_stream_area_ge_600"
    assert extras["matched_stream_id"] == 7


def test_threshold_version_is_dr1_v1(resolver: SatelliteConflictResolver) -> None:
    assert resolver.threshold_version == "conflict_policy_v1_dr1"


def test_empty_streams_map_gives_pass_no_conflict(
    resolver: SatelliteConflictResolver,
) -> None:
    empty = _empty_streams_map()
    sat_seg = np.zeros_like(empty, dtype=bool)
    sat_seg[10:20, 10:20] = True
    matched, _, _, _ = resolver.match_stream(sat_seg, empty)
    assert matched is None
