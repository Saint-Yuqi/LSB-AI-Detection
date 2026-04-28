"""Tests for apply_satellite_post_with_trace and stage attribution analysis.

Run:
    conda run -n sam3 pytest tests/test_satellite_stage_trace.py -v
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.checkpoint_eval import (
    SATELLITE_POST_STAGES,
    apply_post_pred_only,
    apply_satellite_post_with_trace,
)
from scripts.analysis.plot_checkpoint_satellite_tradeoff import (
    STAGE_ATTRIBUTION_COLUMNS,
    STAGE_ORDER,
    STAGE_TRACE_PLOT_FILENAMES,
    STAGE_TRACE_SIDECAR_NAME,
    _join_trace_to_matched,
    collect_stage_attribution,
    plot_first_drop_stage_profile,
    plot_label_reason_first_drop_stage,
    plot_stage_reason_heatmap,
    write_stage_attribution_csv,
)


# =========================================================================== #
#  Helpers for building minimal mask fixtures
# =========================================================================== #


def _make_sat_mask(
    *,
    raw_index: int = 0,
    area_clean: int = 500,
    score: float = 0.5,
    solidity: float = 0.9,
    aspect_sym_moment: float = 1.1,
    dist_to_center: float = 200.0,
    candidate_id: str | None = None,
    candidate_rle_sha1: str | None = None,
) -> dict:
    """Build a minimal satellite mask dict for testing."""
    seg = np.zeros((1024, 1024), dtype=np.uint8)
    seg[100:110, 100:110] = 1  # dummy segmentation
    if candidate_id is None:
        candidate_id = f"sat_{raw_index:04d}"
    if candidate_rle_sha1 is None:
        candidate_rle_sha1 = f"sha_{raw_index}"
    return {
        "type_label": "satellites",
        "raw_index": raw_index,
        "candidate_id": candidate_id,
        "candidate_rle_sha1": candidate_rle_sha1,
        "segmentation": seg,
        "area_clean": area_clean,
        "score": score,
        "solidity": solidity,
        "aspect_sym_moment": aspect_sym_moment,
        "aspect_sym": aspect_sym_moment,
        "dist_to_center": dist_to_center,
    }


# Default config: all 3 stages enabled.
_DEFAULT_CFG = {
    "enable_score_gate": True,
    "score_gate": {},
    "enable_prior_filter": True,
    "prior_filter": {},
    "enable_core_policy": True,
    "core_policy": {},
}


# =========================================================================== #
#  Trace helper unit tests
# =========================================================================== #


class TestTraceScoreGateDrop:
    """Candidate dropped at score_gate → subsequent stages not_reached."""

    def test_small_area_low_score(self):
        mask = _make_sat_mask(area_clean=50, score=0.1)
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 0
        assert len(traces) == 1

        t = traces[0]
        assert t["final_status"] == "removed"
        assert t["first_drop_stage"] == "score_gate"
        assert t["first_drop_reason"] == "drop_small_score"

        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["score_gate"]["outcome"] == "drop"
        assert sr["prior_filter"]["outcome"] == "not_reached"
        assert sr["core_policy"]["outcome"] == "not_reached"


class TestTracePriorFilterDrop:
    """Candidate passes score_gate but dropped at prior_filter."""

    def test_low_solidity(self):
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.5,  # below default 0.83
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 0

        t = traces[0]
        assert t["final_status"] == "removed"
        assert t["first_drop_stage"] == "prior_filter"
        assert t["first_drop_reason"] == "prior_solidity"

        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["score_gate"]["outcome"] == "pass"
        assert sr["prior_filter"]["outcome"] == "drop"
        assert sr["core_policy"]["outcome"] == "not_reached"

    def test_small_area_prior(self):
        mask = _make_sat_mask(
            area_clean=200, score=0.7,  # passes score_gate (medium tier)
            solidity=0.95, aspect_sym_moment=1.1,
        )
        # area_clean=200 >= default score_gate medium_area_max_px? No, 200 < 600.
        # score_gate medium_min_score=0.20, so 0.7 passes.
        # prior_filter area_min=30, so 200 passes.
        # This should pass prior_filter too. Let's use area=10 instead.
        mask = _make_sat_mask(
            area_clean=10, score=0.95,  # passes score_gate (small tier, score>=0.60)
            solidity=0.95, aspect_sym_moment=1.1,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 0

        t = traces[0]
        assert t["first_drop_stage"] == "prior_filter"
        assert t["first_drop_reason"] == "prior_area_low"


class TestTraceCorePolicy:
    """Candidate dropped or rescued at core_policy."""

    def test_hard_core_drop_is_now_at_prior_filter(self):
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.9, aspect_sym_moment=1.1,
            dist_to_center=10.0,  # 10/1024 ≈ 0.0098 < 0.03 hard_core_radius_frac
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 0

        t = traces[0]
        assert t["final_status"] == "removed"
        assert t["first_drop_stage"] == "prior_filter"
        assert t["first_drop_reason"] == "prior_hard_center"

        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["score_gate"]["outcome"] == "pass"
        assert sr["prior_filter"]["outcome"] == "drop"
        assert sr["prior_filter"]["reason"] == "prior_hard_center"
        assert sr["core_policy"]["outcome"] == "not_reached"

    def test_soft_core_rescue(self):
        # In soft core: 0.03 <= dist_frac < 0.08
        # dist_to_center = 50 → 50/1024 ≈ 0.0488 → in soft core.
        # Rescue requires: area>=600, iou>=0.18, solidity>=0.90, aspect<=1.80
        mask = _make_sat_mask(
            area_clean=700, score=0.5,
            solidity=0.95, aspect_sym_moment=1.2,
            dist_to_center=50.0,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 1

        t = traces[0]
        assert t["final_status"] == "kept"
        assert t["first_drop_stage"] is None
        assert t["first_drop_reason"] is None

        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["core_policy"]["outcome"] == "rescue"
        assert sr["core_policy"]["reason"] == "rescue_soft_core"

    def test_soft_core_drop(self):
        # In soft core but NOT meeting rescue criteria (small area).
        mask = _make_sat_mask(
            area_clean=300, score=0.5,
            solidity=0.95, aspect_sym_moment=1.2,
            dist_to_center=50.0,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 0

        t = traces[0]
        assert t["first_drop_stage"] == "core_policy"
        assert t["first_drop_reason"] == "drop_soft_core"


class TestTraceAllPass:
    """Candidate passes all 3 stages."""

    def test_clean_satellite(self):
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.9, aspect_sym_moment=1.1,
            dist_to_center=300.0,  # outside core
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, _DEFAULT_CFG)
        assert len(kept) == 1

        t = traces[0]
        assert t["final_status"] == "kept"
        assert t["first_drop_stage"] is None
        assert t["first_drop_reason"] is None

        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["score_gate"]["outcome"] == "pass"
        assert sr["prior_filter"]["outcome"] == "pass"
        assert sr["prior_filter"]["reason"] == "pass_prior_filter"
        assert sr["core_policy"]["outcome"] == "pass"
        assert sr["core_policy"]["reason"] == "pass_outside_core"


class TestTraceDisabledStage:
    """Disabled stages recorded as outcome='disabled'."""

    def test_prior_filter_disabled(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["enable_prior_filter"] = False
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.5,  # would normally fail prior_filter
            dist_to_center=300.0,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, cfg)
        # solidity is bad but prior_filter is disabled, so it passes through.
        assert len(kept) == 1

        t = traces[0]
        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["prior_filter"]["outcome"] == "disabled"
        assert sr["prior_filter"]["reason"] is None
        # core_policy should still run.
        assert sr["core_policy"]["outcome"] == "pass"

    def test_core_policy_disabled_but_prior_still_filters_hard_center(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["enable_core_policy"] = False
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.95, aspect_sym_moment=1.1,
            dist_to_center=10.0,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, cfg)
        assert len(kept) == 0
        t = traces[0]
        assert t["first_drop_stage"] == "prior_filter"
        assert t["first_drop_reason"] == "prior_hard_center"
        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["prior_filter"]["outcome"] == "drop"
        assert sr["core_policy"]["outcome"] == "not_reached"

    def test_prior_filter_disabled_core_policy_still_drops_hard_center(self):
        cfg = dict(_DEFAULT_CFG)
        cfg["enable_prior_filter"] = False
        mask = _make_sat_mask(
            area_clean=500, score=0.5,
            solidity=0.95, aspect_sym_moment=1.1,
            dist_to_center=10.0,
        )
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, cfg)
        assert len(kept) == 0
        t = traces[0]
        assert t["first_drop_stage"] == "core_policy"
        assert t["first_drop_reason"] == "drop_hard_core"
        sr = {r["stage"]: r for r in t["stage_results"]}
        assert sr["prior_filter"]["outcome"] == "disabled"
        assert sr["core_policy"]["outcome"] == "drop"

    def test_all_disabled(self):
        cfg = {
            "enable_score_gate": False,
            "enable_prior_filter": False,
            "enable_core_policy": False,
        }
        mask = _make_sat_mask(area_clean=5, score=0.01,
                             solidity=0.1, dist_to_center=1.0)
        kept, traces = apply_satellite_post_with_trace([mask], 1024, 1024, cfg)
        assert len(kept) == 1
        t = traces[0]
        assert t["final_status"] == "kept"
        for sr in t["stage_results"]:
            assert sr["outcome"] == "disabled"


class TestTraceReconciliation:
    """Trace helper must agree with apply_post_pred_only on kept list."""

    def test_multi_candidate_reconciliation(self):
        masks = [
            _make_sat_mask(raw_index=0, area_clean=50, score=0.1),   # score_gate drop
            _make_sat_mask(raw_index=1, area_clean=500, score=0.5,
                          solidity=0.5),                                     # prior_filter drop
            _make_sat_mask(raw_index=2, area_clean=500, score=0.5,
                          solidity=0.9, dist_to_center=10.0),                # prior_filter hard-center drop
            _make_sat_mask(raw_index=3, area_clean=500, score=0.5,
                          solidity=0.9, dist_to_center=300.0),               # all pass
            _make_sat_mask(raw_index=4, area_clean=700, score=0.5,
                          solidity=0.95, dist_to_center=50.0),               # core_policy rescue
        ]
        cfg = _DEFAULT_CFG

        # apply_post_pred_only (streams=[], so only sat stages matter)
        _, po_sats = apply_post_pred_only([], masks, 1024, 1024, cfg)
        traced_sats, traces = apply_satellite_post_with_trace(masks, 1024, 1024, cfg)

        assert len(traced_sats) == len(po_sats)
        assert len(traces) == 5

        # Verify per-candidate trace status.
        statuses = [t["first_drop_stage"] for t in traces]
        assert statuses == ["score_gate", "prior_filter", "prior_filter", None, None]


class TestTraceStageConstants:
    def test_stage_order(self):
        assert SATELLITE_POST_STAGES == ("score_gate", "prior_filter", "core_policy")


# =========================================================================== #
#  Analysis pipeline tests
# =========================================================================== #


def _make_trace_sidecar(records: list[dict]) -> dict:
    n_kept = sum(1 for r in records if r["final_status"] == "kept")
    return {
        "schema_version": "1.0.0",
        "benchmark_mode": "fbox_gold_satellites",
        "layer": "post_pred_only",
        "stage_order": list(STAGE_ORDER),
        "n_raw_satellites": len(records),
        "n_kept_post": n_kept,
        "n_removed_post": len(records) - n_kept,
        "records": records,
    }


def _make_trace_record(
    raw_index: int,
    final_status: str = "kept",
    first_drop_stage: str | None = None,
    first_drop_reason: str | None = None,
    score_gate: tuple[str, str | None] = ("pass", "pass_medium"),
    prior_filter: tuple[str, str | None] = ("pass", "pass_prior_filter"),
    core_policy: tuple[str, str | None] = ("pass", "pass_outside_core"),
) -> dict:
    return {
        "raw_index": raw_index,
        "candidate_id": f"sat_{raw_index:04d}",
        "candidate_rle_sha1": f"sha_{raw_index}",
        "final_status": final_status,
        "first_drop_stage": first_drop_stage,
        "first_drop_reason": first_drop_reason,
        "stage_results": [
            {"stage": "score_gate", "outcome": score_gate[0], "reason": score_gate[1]},
            {"stage": "prior_filter", "outcome": prior_filter[0], "reason": prior_filter[1]},
            {"stage": "core_policy", "outcome": core_policy[0], "reason": core_policy[1]},
        ],
    }


def _write_trace_sidecar(sample_dir, records):
    doc = _make_trace_sidecar(records)
    (sample_dir / STAGE_TRACE_SIDECAR_NAME).write_text(json.dumps(doc, indent=2))


def _write_diag_and_predictions(
    sample_dir,
    n_sats: int = 4,
    kept_raw_indices: tuple[int, ...] = (0, 2, 3),
):
    """Write minimal diagnostics.json + predictions_raw/post for n_sats candidates."""
    diag_rows = []
    raw_preds = []
    for i in range(n_sats):
        matched = i < 3  # first 3 are matched
        diag_rows.append({
            "raw_index": i,
            "candidate_id": f"sat_{i:04d}",
            "seed_area": 10,
            "confidence_score": 0.8,
            "matched_gt_id": i + 1 if matched else None,
            "matched_gt_area": 10 if matched else None,
            "overlap_px": 10 if matched else 0,
            "purity": 1.0 if matched else 0.0,
            "completeness": 1.0 if matched else None,
            "seed_gt_ratio": 1.0 if matched else None,
            "is_one_to_one": matched,
            "taxonomy_label": "compact_complete" if matched else "reject_unmatched",
            "label_reason": "pure_and_complete" if matched else "no_gt_overlap",
            "intersects_roi": True,
            "annulus_excess": 1.0,
            "radial_monotonicity": 1.0,
        })
        raw_preds.append({
            "type_label": "satellites",
            "raw_index": i,
            "candidate_id": f"sat_{i:04d}",
            "candidate_rle_sha1": f"sha_{i}",
        })

    post_preds = [
        {
            "type_label": "satellites",
            "raw_index": pi,
            "candidate_id": f"sat_{pi:04d}",
            "candidate_rle_sha1": raw_preds[ri]["candidate_rle_sha1"],
        }
        for pi, ri in enumerate(kept_raw_indices)
    ]

    (sample_dir / "diagnostics.json").write_text(
        json.dumps({"per_candidate": diag_rows})
    )
    (sample_dir / "predictions_raw.json").write_text(
        json.dumps({"predictions": raw_preds})
    )
    (sample_dir / "predictions_post_pred_only.json").write_text(
        json.dumps({"predictions": post_preds})
    )


def _make_test_report_and_sidecars(tmp_path):
    """Set up a 1-sample report with trace sidecar + diag/pred sidecars."""
    base_key = "g_trace_eo"
    sample_dir = tmp_path / base_key
    sample_dir.mkdir()

    # 4 raw sats: 0=kept, 1=score_gate drop, 2=kept, 3=prior_filter drop
    trace_records = [
        _make_trace_record(0, final_status="kept"),
        _make_trace_record(
            1, final_status="removed", first_drop_stage="score_gate",
            first_drop_reason="drop_small_score",
            score_gate=("drop", "drop_small_score"),
            prior_filter=("not_reached", None),
            core_policy=("not_reached", None),
        ),
        _make_trace_record(2, final_status="kept"),
        _make_trace_record(
            3, final_status="removed", first_drop_stage="prior_filter",
            first_drop_reason="prior_solidity",
            prior_filter=("drop", "prior_solidity"),
            core_policy=("not_reached", None),
        ),
    ]
    _write_trace_sidecar(sample_dir, trace_records)
    _write_diag_and_predictions(sample_dir, n_sats=4, kept_raw_indices=(0, 2))

    sample = {
        "base_key": base_key,
        "galaxy_id": "g_trace",
        "view": "eo",
        "benchmark_mode": "fbox_gold_satellites",
        "num_gt_satellites": 3,
        "diagnostics": None,
        "layers": {
            "raw": {"satellites": {
                "roi": {
                    "num_gt": 3, "num_pred": 4,
                    "matched_candidates": 3, "unmatched_candidates": 1,
                    "unique_gt_covered": 3, "gt_uncovered": 0,
                    "precision": 0.75, "recall": 1.0, "f1": None,
                    "counts_by_label": {
                        "compact_complete": 3, "diffuse_core": 0,
                        "reject_unmatched": 1, "reject_low_purity": 0,
                    },
                    "pixel": {"dice": 0.5},
                    "is_empty_trivial": False,
                },
                "full_frame": {
                    "num_gt": 3, "num_pred": 4,
                    "matched_candidates": 3, "unmatched_candidates": 1,
                    "unique_gt_covered": 3, "gt_uncovered": 0,
                    "precision": 0.75, "recall": 1.0, "f1": None,
                    "counts_by_label": {
                        "compact_complete": 3, "diffuse_core": 0,
                        "reject_unmatched": 1, "reject_low_purity": 0,
                    },
                    "pixel": {"dice": 0.5},
                    "is_empty_trivial": False,
                },
            }},
            "post_pred_only": {"satellites": {
                "roi": {
                    "num_gt": 3, "num_pred": 2,
                    "matched_candidates": 2, "unmatched_candidates": 0,
                    "unique_gt_covered": 2, "gt_uncovered": 1,
                    "precision": 1.0, "recall": 2/3, "f1": None,
                    "counts_by_label": {
                        "compact_complete": 2, "diffuse_core": 0,
                        "reject_unmatched": 0, "reject_low_purity": 0,
                    },
                    "pixel": {"dice": 0.6},
                    "is_empty_trivial": False,
                },
                "full_frame": {
                    "num_gt": 3, "num_pred": 2,
                    "matched_candidates": 2, "unmatched_candidates": 0,
                    "unique_gt_covered": 2, "gt_uncovered": 1,
                    "precision": 1.0, "recall": 2/3, "f1": None,
                    "counts_by_label": {
                        "compact_complete": 2, "diffuse_core": 0,
                        "reject_unmatched": 0, "reject_low_purity": 0,
                    },
                    "pixel": {"dice": 0.6},
                    "is_empty_trivial": False,
                },
            }},
        },
    }

    report = {
        "per_sample": [sample],
        "config": {},
        "benchmark_mode": "fbox_gold_satellites",
        "n_samples": 1,
        "elapsed_seconds": 0.0,
        "summary": {},
        "diagnostics_summary": {},
    }
    return report, base_key


class TestJoinTraceToMatched:
    def test_basic_join(self):
        matched_records = [
            {"raw_index": 0, "candidate_id": "sat_0000", "candidate_rle_sha1": "sha_0",
             "taxonomy_label": "compact_complete", "label_reason": "pure_and_complete",
             "kept_in_post": True, "post_status": "kept", "base_key": "test"},
        ]
        trace_records = [
            _make_trace_record(0, final_status="kept"),
        ]
        joined = _join_trace_to_matched(matched_records, trace_records)
        assert len(joined) == 1
        assert joined[0]["first_drop_stage"] is None
        assert joined[0]["score_gate_outcome"] == "pass"
        assert joined[0]["prior_filter_outcome"] == "pass"
        assert joined[0]["core_policy_outcome"] == "pass"

    def test_missing_trace_skipped(self):
        matched_records = [
            {"raw_index": 99, "candidate_id": "sat_0099", "kept_in_post": True},
        ]
        trace_records = [_make_trace_record(0)]
        joined = _join_trace_to_matched(matched_records, trace_records)
        assert len(joined) == 0


class TestCollectStageAttribution:
    def test_basic_collection(self, tmp_path):
        report, _ = _make_test_report_and_sidecars(tmp_path)
        rows, summary = collect_stage_attribution(report, samples_root=tmp_path)

        assert len(rows) > 0
        assert summary is not None
        assert summary["n_samples_with_trace"] == 1

        # all_raw_satellites covers all 4 candidates.
        assert summary["all_raw_satellites"]["total_candidates"] == 4

        # raw_matched_satellites covers 3 matched candidates.
        assert summary["raw_matched_satellites"]["total_candidates"] == 3

    def test_no_trace_returns_none(self, tmp_path):
        report = {
            "per_sample": [{"base_key": "nonexistent", "benchmark_mode": "fbox_gold_satellites"}],
        }
        rows, summary = collect_stage_attribution(report, samples_root=tmp_path)
        assert rows == []
        assert summary is None


class TestStageAttributionCSV:
    def test_columns(self, tmp_path):
        report, _ = _make_test_report_and_sidecars(tmp_path)
        rows, _ = collect_stage_attribution(report, samples_root=tmp_path)
        csv_path = tmp_path / "attribution.csv"
        write_stage_attribution_csv(rows, str(csv_path))

        with open(csv_path) as f:
            read_rows = list(csv.DictReader(f))

        assert len(read_rows) > 0
        assert set(read_rows[0].keys()) == set(STAGE_ATTRIBUTION_COLUMNS)


class TestReconciliation:
    """Sum of by_first_drop_stage for raw_matched_removed must equal
    the number of removed matched candidates."""

    def test_first_drop_stage_sums_to_removed(self, tmp_path):
        report, _ = _make_test_report_and_sidecars(tmp_path)
        rows, summary = collect_stage_attribution(report, samples_root=tmp_path)
        assert summary is not None

        removed_rows = [r for r in rows if not bool(r.get("kept_in_post"))]
        by_fds = summary["raw_matched_removed"]["by_first_drop_stage"]
        # Sum of by_first_drop_stage counts should equal total removed matched.
        fds_total = sum(by_fds.values())
        assert fds_total == len(removed_rows)


class TestStageEnteredDroppedMath:
    def test_entered_minus_dropped_equals_survived(self, tmp_path):
        report, _ = _make_test_report_and_sidecars(tmp_path)
        _, summary = collect_stage_attribution(report, samples_root=tmp_path)
        assert summary is not None

        for group_name in ("all_raw_satellites", "raw_matched_satellites", "raw_matched_removed"):
            group = summary[group_name]
            for stage, stats in group["by_stage_entered_dropped"].items():
                assert stats["entered"] - stats["dropped"] == stats["survived_stage"], (
                    f"{group_name}/{stage}: {stats}"
                )


class TestPlotsSmokeStageTrace:
    @pytest.fixture(autouse=True)
    def _mpl_agg(self):
        import matplotlib
        matplotlib.use("Agg")

    def test_three_new_plots(self, tmp_path):
        report, _ = _make_test_report_and_sidecars(tmp_path)
        rows, summary = collect_stage_attribution(report, samples_root=tmp_path)
        assert summary is not None

        outdir = tmp_path / "plots"
        outdir.mkdir()

        plot_first_drop_stage_profile(rows, summary, str(outdir))
        plot_stage_reason_heatmap(rows, str(outdir))
        plot_label_reason_first_drop_stage(rows, str(outdir))

        for fname in STAGE_TRACE_PLOT_FILENAMES.values():
            assert os.path.isfile(outdir / fname), f"Missing plot: {fname}"


class TestMissingTraceGraceful:
    """When no trace sidecars exist, analysis degrades gracefully."""

    @pytest.fixture(autouse=True)
    def _mpl_agg(self):
        import matplotlib
        matplotlib.use("Agg")

    def test_no_crash_no_files(self, tmp_path):
        base_key = "g_notrace_eo"
        sample_dir = tmp_path / base_key
        sample_dir.mkdir()
        # Write diag/pred but NOT trace sidecar.
        _write_diag_and_predictions(sample_dir)

        sample = {
            "base_key": base_key,
            "galaxy_id": "g_notrace",
            "view": "eo",
            "benchmark_mode": "fbox_gold_satellites",
        }
        report = {"per_sample": [sample]}

        rows, summary = collect_stage_attribution(report, samples_root=tmp_path)
        assert rows == []
        assert summary is None

        # Plots should not crash with empty data.
        outdir = tmp_path / "plots"
        outdir.mkdir()
        plot_first_drop_stage_profile([], {}, str(outdir))
        plot_stage_reason_heatmap([], str(outdir))
        plot_label_reason_first_drop_stage([], str(outdir))

        # No plot files should be created.
        for fname in STAGE_TRACE_PLOT_FILENAMES.values():
            assert not os.path.isfile(outdir / fname)
