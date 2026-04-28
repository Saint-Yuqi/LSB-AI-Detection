"""Unit tests for the DR1 satellite core-region policy.

Usage:
    pytest tests/test_satellite_core_policy.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.satellite_core_policy import SatelliteCorePolicy


@pytest.fixture()
def policy() -> SatelliteCorePolicy:
    return SatelliteCorePolicy()


def _rescue_ok_kwargs() -> dict:
    return {
        "area_clean_px": 700,
        "score": 0.50,
        "solidity": 0.95,
        "aspect_sym_moment": 1.20,
    }


def test_hard_core_always_dropped(policy: SatelliteCorePolicy) -> None:
    decision, reason = policy.decide(
        dist_to_center_frac=0.02,
        **_rescue_ok_kwargs(),
    )
    assert decision == "drop"
    assert reason == "drop_hard_core"


def test_hard_core_boundary_exact_hard_radius(policy: SatelliteCorePolicy) -> None:
    # dist == hard_core_radius_frac should enter soft region, not hard.
    decision, _ = policy.decide(
        dist_to_center_frac=policy.hard_core_radius_frac,
        **_rescue_ok_kwargs(),
    )
    assert decision == "rescue"


def test_outside_core_pass(policy: SatelliteCorePolicy) -> None:
    decision, reason = policy.decide(
        dist_to_center_frac=0.12,
        area_clean_px=100,
        score=0.20,
        solidity=0.5,
        aspect_sym_moment=3.0,
    )
    assert decision == "pass"
    assert reason == "pass_outside_core"


def test_outside_core_boundary_at_soft_radius(policy: SatelliteCorePolicy) -> None:
    decision, reason = policy.decide(
        dist_to_center_frac=policy.soft_core_radius_frac,
        area_clean_px=100,
        score=0.0,
        solidity=0.0,
        aspect_sym_moment=5.0,
    )
    assert decision == "pass"
    assert reason == "pass_outside_core"


def test_soft_core_full_rescue(policy: SatelliteCorePolicy) -> None:
    decision, reason = policy.decide(
        dist_to_center_frac=0.05,
        **_rescue_ok_kwargs(),
    )
    assert decision == "rescue"
    assert reason == "rescue_soft_core"


@pytest.mark.parametrize(
    "override",
    [
        {"area_clean_px": 599},  # area below rescue_area_min_px
        {"score": 0.17},  # below rescue_score_min
        {"solidity": 0.89},  # below rescue_solidity_min
        {"aspect_sym_moment": 1.81},  # above rescue_aspect_max
    ],
)
def test_soft_core_rescue_fails_when_any_gate_fails(
    policy: SatelliteCorePolicy,
    override: dict,
) -> None:
    kwargs = _rescue_ok_kwargs()
    kwargs.update(override)
    decision, reason = policy.decide(dist_to_center_frac=0.05, **kwargs)
    assert decision == "drop"
    assert reason == "drop_soft_core"


def test_threshold_version_strict_v1(policy: SatelliteCorePolicy) -> None:
    assert policy.threshold_version == "core_policy_v1_strict"


def test_threshold_values_has_expected_keys(policy: SatelliteCorePolicy) -> None:
    values = policy.threshold_values()
    assert set(values.keys()) == {
        "hard_core_radius_frac",
        "soft_core_radius_frac",
        "rescue_area_min_px",
        "rescue_solidity_min",
        "rescue_aspect_max",
        "rescue_score_min",
    }
