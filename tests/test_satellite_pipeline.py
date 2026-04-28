"""Unit tests for the DR1 SatellitePipelineRunner.

Verifies: fixed stage ordering, metrics_snapshot_thin whitelist,
soft-core rescue survival, and dropped-by-stream matched_stream_id.

Usage:
    pytest tests/test_satellite_pipeline.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.satellite_conflict_resolver import SatelliteConflictResolver
from src.postprocess.satellite_core_policy import SatelliteCorePolicy
from src.postprocess.satellite_pipeline import (
    STAGE_ORDER,
    SatellitePipelineRunner,
    _THIN_KEYS,
)
from src.postprocess.satellite_prior_filter import SatellitePriorFilter
from src.postprocess.satellite_score_gate import SatelliteScoreGate


H = 128
W = 128


def _build_runner() -> SatellitePipelineRunner:
    prior = SatellitePriorFilter(
        cfg={"area_min": 30, "solidity_min": 0.83, "aspect_sym_max": 1.75}
    )
    return SatellitePipelineRunner(
        score_gate=SatelliteScoreGate(),
        prior_filter=prior,
        core_policy=SatelliteCorePolicy(),
        conflict_resolver=SatelliteConflictResolver(),
    )


def _make_square_mask(cx: int, cy: int, size: int, score: float) -> dict:
    seg = np.zeros((H, W), dtype=bool)
    half = size // 2
    y0, y1 = max(cy - half, 0), min(cy + half, H)
    x0, x1 = max(cx - half, 0), min(cx + half, W)
    seg[y0:y1, x0:x1] = True
    return {
        "type_label": "satellites",
        "segmentation": seg,
        "score": score,
    }


def _empty_streams_map() -> np.ndarray:
    return np.zeros((H, W), dtype=np.int32)


def _streams_map_with_instance(stream_id: int, cx: int, cy: int, size: int) -> np.ndarray:
    m = _empty_streams_map()
    half = size // 2
    m[cy - half : cy + half, cx - half : cx + half] = stream_id
    return m


@pytest.fixture()
def runner() -> SatellitePipelineRunner:
    return _build_runner()


def test_stage_order_prefix_for_every_candidate(
    runner: SatellitePipelineRunner,
) -> None:
    # mask far from center, large, high score, no stream conflict -> passes everything
    sats = [_make_square_mask(cx=110, cy=110, size=40, score=0.70)]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-1")
    stages_seen = [ev["stage"] for ev in result.candidates[0]["history"]]
    # Prefix must equal STAGE_ORDER up to (but not including) diagnostics_emit,
    # which is image-level and not per-candidate.
    expected = [s for s in STAGE_ORDER if s != "diagnostics_emit"]
    assert stages_seen == expected


def test_every_history_stage_is_in_canonical_order(
    runner: SatellitePipelineRunner,
) -> None:
    sats = [
        _make_square_mask(cx=110, cy=110, size=40, score=0.70),  # passes all
        _make_square_mask(cx=110, cy=110, size=10, score=0.10),  # drops at score_gate
    ]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-2")
    for rec in result.candidates:
        stages_seen = [ev["stage"] for ev in rec["history"]]
        ordered_stages = [s for s in STAGE_ORDER if s in stages_seen]
        assert stages_seen == ordered_stages, (
            "history stages must appear in canonical STAGE_ORDER prefix"
        )


def test_metrics_snapshot_thin_whitelist_enforced(
    runner: SatellitePipelineRunner,
) -> None:
    sats = [_make_square_mask(cx=110, cy=110, size=40, score=0.70)]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-3")
    forbidden = {"bbox", "segmentation", "rle", "contours", "area", "area_raw"}
    for rec in result.candidates:
        for ev in rec["history"]:
            snap_keys = set(ev["metrics_snapshot_thin"].keys())
            assert snap_keys.issubset(_THIN_KEYS), (
                f"metrics_snapshot_thin contains non-whitelisted keys: "
                f"{snap_keys - _THIN_KEYS}"
            )
            assert not snap_keys.intersection(forbidden), (
                f"metrics_snapshot_thin leaked forbidden keys: "
                f"{snap_keys.intersection(forbidden)}"
            )


def test_small_low_score_dropped_at_score_gate(
    runner: SatellitePipelineRunner,
) -> None:
    sats = [_make_square_mask(cx=110, cy=110, size=10, score=0.10)]  # area 100
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-score")
    rec = result.candidates[0]
    assert rec["final_status"] == "dropped"
    drop_events = [ev for ev in rec["history"] if ev["decision"] == "drop"]
    assert drop_events
    assert drop_events[0]["stage"] == "size_aware_score_gate"
    assert drop_events[0]["reason"] in {"drop_small_score"}


def test_hard_core_candidate_dropped(runner: SatellitePipelineRunner) -> None:
    # Center the mask at image centre => dist_to_center ~ 0 => hard core
    sats = [_make_square_mask(cx=W // 2, cy=H // 2, size=40, score=0.70)]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-core")
    rec = result.candidates[0]
    prior_events = [
        ev for ev in rec["history"]
        if ev["stage"] == "satellite_prior_filter"
    ]
    assert prior_events
    assert prior_events[0]["decision"] == "drop"
    assert prior_events[0]["reason"] == "prior_hard_center"

    core_events = [
        ev for ev in rec["history"]
        if ev["stage"] == "core_exclusion_or_soft_core_rescue"
    ]
    assert core_events == []


def test_soft_core_rescue_survives_and_is_final_kept(
    runner: SatellitePipelineRunner,
) -> None:
    # Position the mask inside the soft ring (between 3% and 8% of min(H,W))
    min_side = min(H, W)
    r_soft = 0.05 * min_side  # inside soft ring
    cx = int(W // 2 + r_soft + 1)
    cy = H // 2
    # Large, high-solidity, circular-ish -> satisfies rescue criteria
    seg = np.zeros((H, W), dtype=bool)
    yy, xx = np.ogrid[:H, :W]
    seg[(yy - cy) ** 2 + (xx - cx) ** 2 <= 16 ** 2] = True  # ~804 px disk
    sat = {
        "type_label": "satellites",
        "segmentation": seg,
        "score": 0.70,
    }
    result = runner.run([sat], _empty_streams_map(), H, W, base_key="test-rescue")
    rec = result.candidates[0]
    assert rec["final_status"] == "kept"
    decisions = [(ev["stage"], ev["decision"], ev["reason"]) for ev in rec["history"]]
    assert (
        "core_exclusion_or_soft_core_rescue",
        "rescue",
        "rescue_soft_core",
    ) in decisions
    assert len(result.final_sats) == 1


def test_satellite_swallowed_by_stream_records_matched_id(
    runner: SatellitePipelineRunner,
) -> None:
    # Small satellite sitting inside a stream instance (ID 42).
    streams_map = _streams_map_with_instance(stream_id=42, cx=90, cy=90, size=40)
    sats = [_make_square_mask(cx=90, cy=90, size=12, score=0.70)]  # area 144 (small tier)
    result = runner.run(sats, streams_map, H, W, base_key="test-swallow")
    rec = result.candidates[0]
    assert rec["final_status"] == "dropped"
    assert rec["matched_stream_id"] == 42
    conflict_events = [
        ev for ev in rec["history"] if ev["stage"] == "stream_conflict_resolution"
    ]
    assert conflict_events, "swallowed candidate must have a conflict stage event"
    ev = conflict_events[0]
    assert ev["decision"] == "drop"
    assert ev["reason"] == "area_under_600_swallowed_by_stream"
    assert ev["threshold_values"]["matched_stream_id"] == 42


def test_image_summary_counts(runner: SatellitePipelineRunner) -> None:
    sats = [
        _make_square_mask(cx=110, cy=110, size=40, score=0.70),  # keep
        _make_square_mask(cx=110, cy=110, size=10, score=0.10),  # drop
    ]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-sum")
    summary = result.image_summary
    assert summary["base_key"] == "test-sum"
    assert summary["n_raw_satellites"] == 2
    assert summary["n_final_satellites"] == 1
    assert "thresholds_version" in summary
    assert summary["thresholds_version"]["score_gate"] == "score_gate_v1_static"
    assert summary["thresholds_version"]["core_policy"] == "core_policy_v1_strict"
    assert summary["thresholds_version"]["conflict_policy"] == "conflict_policy_v1_dr1"


def test_pipeline_has_no_manual_override_stage() -> None:
    """Reviewed exceptions are migrated into explicit GT edits, not injected here."""
    assert "manual_override" not in STAGE_ORDER


def test_runner_rejects_manual_overrides_kwarg() -> None:
    """Constructor no longer accepts a manual_overrides parameter."""
    with pytest.raises(TypeError):
        SatellitePipelineRunner(  # type: ignore[call-arg]
            score_gate=SatelliteScoreGate(),
            prior_filter=SatellitePriorFilter(
                cfg={"area_min": 30, "solidity_min": 0.83, "aspect_sym_max": 1.75}
            ),
            core_policy=SatelliteCorePolicy(),
            conflict_resolver=SatelliteConflictResolver(),
            manual_overrides={"ignored": {}},
        )


def test_candidate_id_and_sha1_are_populated(
    runner: SatellitePipelineRunner,
) -> None:
    sats = [
        _make_square_mask(cx=110, cy=110, size=40, score=0.70),
        _make_square_mask(cx=20, cy=20, size=40, score=0.70),
    ]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-ids")
    ids = {rec["candidate_id"] for rec in result.candidates}
    sha1s = {rec["candidate_rle_sha1"] for rec in result.candidates}
    assert ids == {"sat_0000", "sat_0001"}
    assert len(sha1s) == 2  # different masks -> different hashes
    for rec in result.candidates:
        assert len(rec["candidate_rle_sha1"]) == 16


def test_diagnostics_contains_no_geometry(runner: SatellitePipelineRunner) -> None:
    """Sanity: every candidate record only has identity + status + history;
    no segmentation / bbox / contours / rle payload sneaks in."""
    sats = [_make_square_mask(cx=110, cy=110, size=40, score=0.70)]
    result = runner.run(sats, _empty_streams_map(), H, W, base_key="test-no-geom")
    for rec in result.candidates:
        assert set(rec.keys()) == {
            "candidate_id",
            "candidate_rle_sha1",
            "final_status",
            "matched_stream_id",
            "history",
        }
