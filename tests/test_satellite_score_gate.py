"""Unit tests for the DR1 static three-tier satellite score gate.

Usage:
    pytest tests/test_satellite_score_gate.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.satellite_score_gate import SatelliteScoreGate


@pytest.fixture()
def gate() -> SatelliteScoreGate:
    return SatelliteScoreGate()


@pytest.mark.parametrize(
    "area,score,expected_decision,expected_reason",
    [
        # Small tier (area < 200), threshold 0.60
        (1, 0.60, "pass", "pass_small"),
        (199, 0.60, "pass", "pass_small"),
        (199, 0.59, "drop", "drop_small_score"),
        (150, 0.80, "pass", "pass_small"),
        # Medium tier [200, 600), threshold 0.20
        (200, 0.20, "pass", "pass_medium"),
        (599, 0.20, "pass", "pass_medium"),
        (599, 0.19, "drop", "drop_medium_score"),
        (500, 0.20, "pass", "pass_medium"),
        # Large tier [600, inf), threshold 0.18
        (600, 0.18, "pass", "pass_large"),
        (10_000, 0.18, "pass", "pass_large"),
        (600, 0.17, "drop", "drop_large_score"),
        (3_500, 0.10, "drop", "drop_large_score"),
    ],
)
def test_three_tier_boundaries(
    gate: SatelliteScoreGate,
    area: int,
    score: float,
    expected_decision: str,
    expected_reason: str,
) -> None:
    decision, reason = gate.decide(area, score)
    assert decision == expected_decision
    assert reason == expected_reason


def test_threshold_version_is_static_v1(gate: SatelliteScoreGate) -> None:
    assert gate.threshold_version == "score_gate_v1_static"


def test_threshold_values_has_unit_safe_keys(gate: SatelliteScoreGate) -> None:
    values = gate.threshold_values()
    expected_keys = {
        "small_area_max_px",
        "medium_area_max_px",
        "small_min_score",
        "medium_min_score",
        "large_min_score",
    }
    assert set(values.keys()) == expected_keys


def test_custom_thresholds_override_defaults() -> None:
    custom = SatelliteScoreGate(
        small_area_max_px=100,
        medium_area_max_px=500,
        small_min_score=0.7,
        medium_min_score=0.4,
        large_min_score=0.2,
    )
    assert custom.decide(99, 0.7) == ("pass", "pass_small")
    assert custom.decide(99, 0.69) == ("drop", "drop_small_score")
    assert custom.decide(100, 0.4) == ("pass", "pass_medium")
    assert custom.decide(500, 0.2) == ("pass", "pass_large")
