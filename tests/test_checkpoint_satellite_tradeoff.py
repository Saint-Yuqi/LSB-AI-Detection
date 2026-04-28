"""Tests for scripts/analysis/plot_checkpoint_satellite_tradeoff.py.

Run:
    conda run -n sam3 pytest tests/test_checkpoint_satellite_tradeoff.py -v
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.analysis.plot_checkpoint_satellite_tradeoff import (
    CSV_COLUMNS,
    PLOT_FILENAMES,
    REMOVED_MATCHED_COLUMNS,
    STAGE_TRACE_SIDECAR_NAME,
    UNMATCHED_POST_DISPOSITION_COLUMNS,
    _compute_taxonomy_coverage,
    _official_satellite_scope,
    build_summary,
    collect_matched_candidate_dispositions,
    collect_removed_matched_candidates,
    collect_unmatched_candidate_dispositions,
    collect_unmatched_stage_attribution,
    extract_all,
    extract_row,
    plot_delta_precision_recall_scatter,
    plot_global_matrix,
    plot_matched_purity_completeness_post_disposition,
    plot_matched_reason_post_disposition_flow,
    plot_matched_reason_removal_rate,
    plot_matched_loss_vs_num_gt,
    plot_precision_recall_raw_vs_post,
    plot_precision_recall_raw_vs_post_scatter,
    plot_rate_boxplot,
    plot_tradeoff_scatter,
    write_removed_matched_candidates_csv,
    write_unmatched_candidate_post_disposition_csv,
    write_csv,
)


def _taxonomy_block(
    *,
    num_gt: int,
    counts: dict[str, int],
    unique_gt_covered: int,
    pixel_dice: float,
) -> dict:
    matched = counts.get("compact_complete", 0) + counts.get("diffuse_core", 0)
    unmatched = counts.get("reject_unmatched", 0) + counts.get("reject_low_purity", 0)
    num_pred = matched + unmatched
    return {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "matched_candidates": matched,
        "unmatched_candidates": unmatched,
        "unique_gt_covered": unique_gt_covered,
        "gt_uncovered": max(0, num_gt - unique_gt_covered),
        "precision": (matched / num_pred) if num_pred else None,
        "recall": (unique_gt_covered / num_gt) if num_gt else None,
        "f1": None,
        "counts_by_label": counts,
        "pixel": {"dice": pixel_dice},
        "is_empty_trivial": num_gt == 0 and num_pred == 0,
    }


def _make_new_schema_fbox_sample(
    *,
    base_key: str = "gnew_001",
    roi_num_gt: int = 3,
    full_num_gt: int = 3,
    raw_counts: dict[str, int] | None = None,
    post_counts: dict[str, int] | None = None,
    unique_gt_covered_raw: int = 3,
    unique_gt_covered_post: int = 3,
) -> dict:
    raw_counts = raw_counts or {
        "compact_complete": 2, "diffuse_core": 1,
        "reject_unmatched": 1, "reject_low_purity": 0,
    }
    post_counts = post_counts or {
        "compact_complete": 2, "diffuse_core": 1,
        "reject_unmatched": 0, "reject_low_purity": 0,
    }
    return {
        "base_key": base_key,
        "galaxy_id": "gnew",
        "view": "eo",
        "benchmark_mode": "fbox_gold_satellites",
        "num_gt_satellites": full_num_gt,
        "diagnostics": None,
        "layers": {
            "raw": {"satellites": {
                "roi": _taxonomy_block(
                    num_gt=roi_num_gt, counts=raw_counts,
                    unique_gt_covered=unique_gt_covered_raw, pixel_dice=0.55,
                ),
                "full_frame": _taxonomy_block(
                    num_gt=full_num_gt, counts=raw_counts,
                    unique_gt_covered=unique_gt_covered_raw, pixel_dice=0.55,
                ),
            }},
            "post_pred_only": {"satellites": {
                "roi": _taxonomy_block(
                    num_gt=roi_num_gt, counts=post_counts,
                    unique_gt_covered=unique_gt_covered_post, pixel_dice=0.6,
                ),
                "full_frame": _taxonomy_block(
                    num_gt=full_num_gt, counts=post_counts,
                    unique_gt_covered=unique_gt_covered_post, pixel_dice=0.6,
                ),
            }},
        },
    }


def _make_new_schema_gt_canonical_sample(
    *,
    base_key: str = "gcan_new_001",
    full_num_gt: int = 3,
    raw_counts: dict[str, int] | None = None,
    post_counts: dict[str, int] | None = None,
    unique_gt_covered_raw: int = 3,
    unique_gt_covered_post: int = 3,
) -> dict:
    raw_counts = raw_counts or {
        "compact_complete": 2, "diffuse_core": 1,
        "reject_unmatched": 1, "reject_low_purity": 0,
    }
    post_counts = post_counts or {
        "compact_complete": 2, "diffuse_core": 1,
        "reject_unmatched": 0, "reject_low_purity": 0,
    }
    return {
        "base_key": base_key,
        "galaxy_id": "gcan_new",
        "view": "eo",
        "benchmark_mode": "gt_canonical",
        "num_gt_satellites": full_num_gt,
        "num_gt_streams": 1,
        "diagnostics": None,
        "layers": {
            "raw": {
                "satellites": {
                    "full_frame": _taxonomy_block(
                        num_gt=full_num_gt, counts=raw_counts,
                        unique_gt_covered=unique_gt_covered_raw, pixel_dice=0.4,
                    ),
                    "roi": None,
                },
            },
            "post_pred_only": {
                "satellites": {
                    "full_frame": _taxonomy_block(
                        num_gt=full_num_gt, counts=post_counts,
                        unique_gt_covered=unique_gt_covered_post, pixel_dice=0.45,
                    ),
                    "roi": None,
                },
            },
        },
    }


def _make_sample(
    *,
    base_key: str = "g001_eo",
    galaxy_id: str = "g001",
    view: str = "eo",
    full_num_gt: int = 3,
    roi_num_gt: int = 3,
    raw_tp: int = 2,
    raw_fp: int = 2,
    raw_fn: int = 1,
    raw_precision: float | None = 0.5,
    raw_recall: float | None = 2 / 3,
    raw_f1: float | None = 4 / 7,
    post_tp: int = 2,
    post_fp: int = 1,
    post_fn: int = 1,
    post_precision: float | None = 2 / 3,
    post_recall: float | None = 2 / 3,
    post_f1: float | None = 2 / 3,
    raw_counts_roi: dict[str, int] | None = None,
    post_counts_roi: dict[str, int] | None = None,
    include_diagnostics: bool = True,
) -> dict:
    raw_counts_roi = raw_counts_roi or {
        "compact_complete": 2,
        "diffuse_core": 1,
        "reject_unmatched": 1,
        "reject_low_purity": 0,
    }
    post_counts_roi = post_counts_roi or {
        "compact_complete": 2,
        "diffuse_core": 1,
        "reject_unmatched": 0,
        "reject_low_purity": 0,
    }

    diagnostics = None
    if include_diagnostics:
        diagnostics = {
            "satellites_raw": {
                "summary": {
                    "counts_by_label_roi": raw_counts_roi,
                    "counts_post_by_label_roi": post_counts_roi,
                    "thresholds_used": {"min_purity_for_match": 0.5},
                }
            }
        }

    return {
        "base_key": base_key,
        "galaxy_id": galaxy_id,
        "view": view,
        "benchmark_mode": "fbox_gold_satellites",
        "num_gt_satellites": full_num_gt,
        "diagnostics": diagnostics,
        "layers": {
            "raw": {
                "satellites": {
                    "roi": {
                        "num_gt": roi_num_gt,
                        "num_pred": sum(raw_counts_roi.values()),
                        "tp": raw_tp,
                        "fp": raw_fp,
                        "fn": raw_fn,
                        "detection": {
                            "precision": raw_precision,
                            "recall": raw_recall,
                            "f1": raw_f1,
                        },
                        "matched_iou_mean": 0.65,
                        "pixel": {"dice": 0.55},
                    }
                }
            },
            "post_pred_only": {
                "satellites": {
                    "roi": {
                        "num_gt": roi_num_gt,
                        "num_pred": sum(post_counts_roi.values()),
                        "tp": post_tp,
                        "fp": post_fp,
                        "fn": post_fn,
                        "detection": {
                            "precision": post_precision,
                            "recall": post_recall,
                            "f1": post_f1,
                        },
                        "matched_iou_mean": 0.7,
                        "pixel": {"dice": 0.6},
                    }
                }
            },
        },
    }


def _make_report(samples: list[dict]) -> dict:
    return {
        "per_sample": samples,
        "config": {},
        "benchmark_mode": "fbox_gold_satellites",
        "n_samples": len(samples),
        "elapsed_seconds": 0.0,
        "summary": {},
        "diagnostics_summary": {},
    }


def _write_sample_sidecars(
    tmp_path,
    base_key: str,
    kept_raw_indices: tuple[int, ...] = (0, 2, 3),
    write_stage_trace: bool = False,
) -> None:
    sample_dir = tmp_path / base_key
    sample_dir.mkdir()

    diag_rows = [
        {
            "raw_index": 0,
            "candidate_id": "sat_0000",
            "seed_area": 10,
            "confidence_score": 0.8,
            "matched_gt_id": 1,
            "matched_gt_area": 10,
            "overlap_px": 10,
            "purity": 1.0,
            "completeness": 1.0,
            "seed_gt_ratio": 1.0,
            "is_one_to_one": True,
            "taxonomy_label": "diffuse_core",
            "label_reason": "pure_but_core_only",
            "intersects_roi": True,
            "annulus_excess": 1.0,
            "radial_monotonicity": 1.0,
        },
        {
            "raw_index": 1,
            "candidate_id": "sat_0001",
            "seed_area": 10,
            "confidence_score": 0.2,
            "matched_gt_id": None,
            "matched_gt_area": None,
            "overlap_px": 0,
            "purity": 0.0,
            "completeness": None,
            "seed_gt_ratio": None,
            "is_one_to_one": False,
            "taxonomy_label": "reject_unmatched",
            "label_reason": "no_gt_overlap",
            "intersects_roi": True,
            "annulus_excess": 1.0,
            "radial_monotonicity": 1.0,
        },
        {
            "raw_index": 2,
            "candidate_id": "sat_0002",
            "seed_area": 10,
            "confidence_score": 0.7,
            "matched_gt_id": 3,
            "matched_gt_area": 10,
            "overlap_px": 10,
            "purity": 1.0,
            "completeness": 1.0,
            "seed_gt_ratio": 1.0,
            "is_one_to_one": True,
            "taxonomy_label": "compact_complete",
            "label_reason": "pure_and_complete",
            "intersects_roi": True,
            "annulus_excess": 1.0,
            "radial_monotonicity": 1.0,
        },
        {
            "raw_index": 3,
            "candidate_id": "sat_0003",
            "seed_area": 10,
            "confidence_score": 0.9,
            "matched_gt_id": 2,
            "matched_gt_area": 10,
            "overlap_px": 10,
            "purity": 1.0,
            "completeness": 1.0,
            "seed_gt_ratio": 1.0,
            "is_one_to_one": True,
            "taxonomy_label": "compact_complete",
            "label_reason": "pure_and_complete",
            "intersects_roi": True,
            "annulus_excess": 1.0,
            "radial_monotonicity": 1.0,
        },
    ]
    raw_preds = [
        {"type_label": "satellites", "raw_index": 0, "candidate_id": "sat_0000", "candidate_rle_sha1": "sha0"},
        {"type_label": "satellites", "raw_index": 1, "candidate_id": "sat_0001", "candidate_rle_sha1": "sha1"},
        {"type_label": "satellites", "raw_index": 2, "candidate_id": "sat_0002", "candidate_rle_sha1": "sha2"},
        {"type_label": "satellites", "raw_index": 3, "candidate_id": "sat_0003", "candidate_rle_sha1": "sha3"},
    ]
    # Post predictions are intentionally re-numbered to match the real bug.
    post_preds = [
        {
            "type_label": "satellites",
            "raw_index": post_idx,
            "candidate_id": f"sat_{post_idx:04d}",
            "candidate_rle_sha1": raw_preds[raw_idx]["candidate_rle_sha1"],
        }
        for post_idx, raw_idx in enumerate(kept_raw_indices)
    ]

    with open(sample_dir / "diagnostics.json", "w") as f:
        json.dump({"per_candidate": diag_rows}, f)
    with open(sample_dir / "predictions_raw.json", "w") as f:
        json.dump({"predictions": raw_preds}, f)
    with open(sample_dir / "predictions_post_pred_only.json", "w") as f:
        json.dump({"predictions": post_preds}, f)

    if write_stage_trace:
        def _trace_record(
            raw_idx: int,
            *,
            final_status: str,
            first_drop_stage: str | None = None,
            first_drop_reason: str | None = None,
        ) -> dict:
            if final_status == "kept":
                stage_results = [
                    {"stage": "score_gate", "outcome": "pass", "reason": "pass_small"},
                    {"stage": "prior_filter", "outcome": "pass", "reason": "pass_prior_filter"},
                    {"stage": "core_policy", "outcome": "pass", "reason": "pass_outside_core"},
                ]
            elif first_drop_stage == "score_gate":
                stage_results = [
                    {"stage": "score_gate", "outcome": "drop", "reason": first_drop_reason},
                    {"stage": "prior_filter", "outcome": "not_reached", "reason": None},
                    {"stage": "core_policy", "outcome": "not_reached", "reason": None},
                ]
            elif first_drop_stage == "prior_filter":
                stage_results = [
                    {"stage": "score_gate", "outcome": "pass", "reason": "pass_small"},
                    {"stage": "prior_filter", "outcome": "drop", "reason": first_drop_reason},
                    {"stage": "core_policy", "outcome": "not_reached", "reason": None},
                ]
            else:
                stage_results = [
                    {"stage": "score_gate", "outcome": "pass", "reason": "pass_small"},
                    {"stage": "prior_filter", "outcome": "pass", "reason": "pass_prior_filter"},
                    {"stage": "core_policy", "outcome": "drop", "reason": first_drop_reason},
                ]
            return {
                "raw_index": raw_idx,
                "candidate_id": raw_preds[raw_idx]["candidate_id"],
                "candidate_rle_sha1": raw_preds[raw_idx]["candidate_rle_sha1"],
                "final_status": final_status,
                "first_drop_stage": first_drop_stage,
                "first_drop_reason": first_drop_reason,
                "stage_results": stage_results,
            }

        trace_records = []
        for raw_idx in range(len(raw_preds)):
            if raw_idx in kept_raw_indices:
                trace_records.append(_trace_record(raw_idx, final_status="kept"))
            elif raw_idx == 1:
                trace_records.append(
                    _trace_record(
                        raw_idx,
                        final_status="removed",
                        first_drop_stage="score_gate",
                        first_drop_reason="drop_small_score",
                    )
                )
            else:
                trace_records.append(
                    _trace_record(
                        raw_idx,
                        final_status="removed",
                        first_drop_stage="prior_filter",
                        first_drop_reason="prior_solidity" if raw_idx == 3 else "prior_aspect",
                    )
                )

        with open(sample_dir / STAGE_TRACE_SIDECAR_NAME, "w") as f:
            json.dump({"records": trace_records}, f)


class TestCoverageHelper:
    def test_sha1_mapping_recovers_post_reindexed_candidates(self):
        diag_rows = [
            {"raw_index": 0, "candidate_id": "sat_0000", "matched_gt_id": 1, "taxonomy_label": "diffuse_core", "intersects_roi": True},
            {"raw_index": 1, "candidate_id": "sat_0001", "matched_gt_id": None, "taxonomy_label": "reject_unmatched", "intersects_roi": True},
            {"raw_index": 2, "candidate_id": "sat_0002", "matched_gt_id": 3, "taxonomy_label": "compact_complete", "intersects_roi": True},
            {"raw_index": 3, "candidate_id": "sat_0003", "matched_gt_id": 2, "taxonomy_label": "compact_complete", "intersects_roi": True},
        ]
        raw_preds = [
            {"type_label": "satellites", "raw_index": 0, "candidate_id": "sat_0000", "candidate_rle_sha1": "sha0"},
            {"type_label": "satellites", "raw_index": 1, "candidate_id": "sat_0001", "candidate_rle_sha1": "sha1"},
            {"type_label": "satellites", "raw_index": 2, "candidate_id": "sat_0002", "candidate_rle_sha1": "sha2"},
            {"type_label": "satellites", "raw_index": 3, "candidate_id": "sat_0003", "candidate_rle_sha1": "sha3"},
        ]
        post_preds = [
            {"type_label": "satellites", "raw_index": 0, "candidate_id": "sat_0000", "candidate_rle_sha1": "sha0"},
            {"type_label": "satellites", "raw_index": 1, "candidate_id": "sat_0001", "candidate_rle_sha1": "sha2"},
            {"type_label": "satellites", "raw_index": 2, "candidate_id": "sat_0002", "candidate_rle_sha1": "sha3"},
        ]

        cov = _compute_taxonomy_coverage(diag_rows, raw_preds, post_preds)
        assert cov["matched_candidates_raw"] == 3
        assert cov["matched_candidates_post"] == 3
        assert cov["unique_gt_covered_raw"] == 3
        assert cov["unique_gt_covered_post"] == 3
        assert cov["used_sha_links"] == 3


class TestExtractRow:
    def test_taxonomy_recall_uses_unique_gt_coverage(self, tmp_path):
        sample = _make_sample()
        _write_sample_sidecars(tmp_path, sample["base_key"])
        row = extract_row(sample, samples_root=tmp_path)

        assert row["num_gt_satellites_full"] == 3
        assert row["num_gt_satellites_roi"] == 3
        assert row["matched_candidates_raw"] == 3
        assert row["matched_candidates_post"] == 3
        assert row["unique_gt_covered_raw"] == 3
        assert row["unique_gt_covered_post"] == 3
        assert row["tax_precision_raw"] == pytest.approx(3 / 4)
        assert row["tax_precision_post"] == pytest.approx(1.0)
        assert row["tax_recall_raw"] == pytest.approx(1.0)
        assert row["tax_recall_post"] == pytest.approx(1.0)

    def test_missing_sidecars_gives_nan_taxonomy_recall(self, tmp_path):
        sample = _make_sample()
        row = extract_row(sample, samples_root=tmp_path)

        assert math.isnan(row["tax_recall_raw"])
        assert math.isnan(row["tax_recall_post"])
        assert row["matched_candidates_raw"] == 3
        assert row["matched_candidates_post"] == 3


class TestOfficialSatelliteScope:
    def test_fbox_gold_maps_to_roi(self):
        assert _official_satellite_scope("fbox_gold_satellites") == "roi"

    def test_gt_canonical_maps_to_full_frame(self):
        assert _official_satellite_scope("gt_canonical") == "full_frame"

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError):
            _official_satellite_scope("firebox_dr1_streams")


class TestExtractRowNewSchema:
    def test_fbox_reads_taxonomy_fields_directly_without_sidecars(self, tmp_path):
        sample = _make_new_schema_fbox_sample(
            unique_gt_covered_raw=3,
            unique_gt_covered_post=3,
        )
        # samples_root is provided but there are no sidecars; new schema
        # short-circuits the sidecar path entirely.
        row = extract_row(sample, samples_root=tmp_path)

        assert row["num_gt_satellites_full"] == 3
        assert row["num_gt_satellites_roi"] == 3
        assert row["matched_candidates_raw"] == 3
        assert row["unmatched_candidates_raw"] == 1
        assert row["matched_candidates_post"] == 3
        assert row["unmatched_candidates_post"] == 0
        assert row["unique_gt_covered_raw"] == 3
        assert row["unique_gt_covered_post"] == 3
        assert row["tax_precision_raw"] == pytest.approx(3 / 4)
        assert row["tax_precision_post"] == pytest.approx(1.0)
        assert row["tax_recall_raw"] == pytest.approx(1.0)
        assert row["tax_recall_post"] == pytest.approx(1.0)
        # Counts_by_label pulled directly off the block.
        assert row["compact_complete_raw"] == 2
        assert row["diffuse_core_raw"] == 1
        assert row["reject_unmatched_raw"] == 1

    def test_gt_canonical_reads_from_full_frame_scope(self, tmp_path):
        sample = _make_new_schema_gt_canonical_sample()
        row = extract_row(sample, samples_root=tmp_path)

        # `_roi` columns carry the full_frame value for gt_canonical (plan §
        # "Tradeoff script behavior" — CSV column names are stable).
        assert row["num_gt_satellites_roi"] == 3
        assert row["num_pred_raw_roi"] == 4
        assert row["matched_candidates_raw"] == 3
        assert row["pixel_dice_raw"] == pytest.approx(0.4)
        assert row["pixel_dice_post"] == pytest.approx(0.45)


class TestExtractAllAndCSV:
    def test_roundtrip(self, tmp_path):
        s1 = _make_sample()
        s2 = _make_sample(base_key="g002_fo", view="fo")
        _write_sample_sidecars(tmp_path, s1["base_key"])
        _write_sample_sidecars(tmp_path, s2["base_key"])
        rows = extract_all(_make_report([s1, s2]), samples_root=tmp_path)
        csv_path = tmp_path / "tradeoff.csv"
        write_csv(rows, str(csv_path))

        with open(csv_path) as f:
            read_rows = list(csv.DictReader(f))

        assert len(read_rows) == 2
        assert set(read_rows[0].keys()) == set(CSV_COLUMNS)

    def test_nan_written_as_blank(self, tmp_path):
        rows = extract_all(_make_report([_make_sample()]), samples_root=tmp_path)
        csv_path = tmp_path / "tradeoff.csv"
        write_csv(rows, str(csv_path))

        with open(csv_path) as f:
            row = next(csv.DictReader(f))

        assert row["tax_recall_raw"] == ""
        assert row["tax_recall_post"] == ""


class TestSummary:
    def test_global_taxonomy_micro(self, tmp_path):
        s1 = _make_sample(base_key="a")
        s2 = _make_sample(base_key="b")
        _write_sample_sidecars(tmp_path, "a")
        _write_sample_sidecars(tmp_path, "b")
        rows = extract_all(_make_report([s1, s2]), samples_root=tmp_path)
        summary = build_summary(rows, _make_report([s1, s2]))

        assert summary["n_samples"] == 2
        assert summary["global_taxonomy_micro"]["raw"]["unique_gt_covered"] == 6
        assert summary["global_taxonomy_micro"]["post_pred_only"]["unique_gt_covered"] == 6
        assert summary["global_taxonomy_micro"]["post_pred_only"]["recall"] == pytest.approx(1.0)
        assert "taxonomy_stats" in summary
        json.dumps(summary)


class TestRemovedMatchedCandidates:
    def test_collect_matched_candidate_dispositions(self, tmp_path):
        sample = _make_sample(base_key="g_matched_eo")
        _write_sample_sidecars(tmp_path, sample["base_key"], kept_raw_indices=(0, 2))

        rows, summary = collect_matched_candidate_dispositions(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        assert len(rows) == 3
        assert summary["raw_matched_candidate_total"] == 3
        assert summary["kept_in_post"] == 2
        assert summary["removed_in_post"] == 1
        assert summary["stage_trace_available"] is False
        assert summary["by_taxonomy_label"]["compact_complete"]["removed"] == 1
        assert summary["by_label_reason"]["pure_but_core_only"]["kept"] == 1
        json.dumps(summary)

    def test_collect_removed_matched_candidates(self, tmp_path):
        sample = _make_sample(base_key="g_removed_eo")
        _write_sample_sidecars(tmp_path, sample["base_key"], kept_raw_indices=(0, 2))

        rows, summary = collect_removed_matched_candidates(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        assert len(rows) == 1
        assert rows[0]["taxonomy_label"] == "compact_complete"
        assert rows[0]["matched_gt_id"] == 2
        assert rows[0]["candidate_rle_sha1"] == "sha3"
        assert summary["n_removed_matched_candidates"] == 1
        assert summary["removed_by_taxonomy_label"] == {"compact_complete": 1}
        assert summary["per_sample"][0]["base_key"] == "g_removed_eo"
        json.dumps(summary)

    def test_write_removed_matched_candidates_csv(self, tmp_path):
        sample = _make_sample(base_key="g_removed_csv")
        _write_sample_sidecars(tmp_path, sample["base_key"], kept_raw_indices=(0, 2))
        rows, _summary = collect_removed_matched_candidates(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        csv_path = tmp_path / "removed.csv"
        write_removed_matched_candidates_csv(rows, str(csv_path))

        with open(csv_path) as f:
            read_rows = list(csv.DictReader(f))

        assert len(read_rows) == 1
        assert set(read_rows[0].keys()) == set(REMOVED_MATCHED_COLUMNS)

    def test_collect_unmatched_candidate_dispositions(self, tmp_path):
        sample = _make_sample(base_key="g_unmatched_eo")
        _write_sample_sidecars(tmp_path, sample["base_key"], kept_raw_indices=(0, 1, 2))

        rows, summary = collect_unmatched_candidate_dispositions(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        assert len(rows) == 1
        assert rows[0]["taxonomy_label"] == "reject_unmatched"
        assert rows[0]["matched_gt_id"] is None
        assert rows[0]["kept_in_post"] is True
        assert summary["raw_unmatched_candidate_total"] == 1
        assert summary["kept_in_post"] == 1
        assert summary["removed_in_post"] == 0
        assert summary["by_taxonomy_label"]["reject_unmatched"]["kept"] == 1
        json.dumps(summary)

    def test_write_unmatched_candidate_post_disposition_csv(self, tmp_path):
        sample = _make_sample(base_key="g_unmatched_csv")
        _write_sample_sidecars(tmp_path, sample["base_key"], kept_raw_indices=(0, 2))
        rows, _summary = collect_unmatched_candidate_dispositions(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        csv_path = tmp_path / "unmatched.csv"
        write_unmatched_candidate_post_disposition_csv(rows, str(csv_path))

        with open(csv_path) as f:
            read_rows = list(csv.DictReader(f))

        assert len(read_rows) == 1
        assert set(read_rows[0].keys()) == set(UNMATCHED_POST_DISPOSITION_COLUMNS)

    def test_collect_unmatched_stage_attribution(self, tmp_path):
        sample = _make_sample(base_key="g_unmatched_trace")
        _write_sample_sidecars(
            tmp_path,
            sample["base_key"],
            kept_raw_indices=(0, 2),
            write_stage_trace=True,
        )

        rows, summary = collect_unmatched_stage_attribution(
            _make_report([sample]),
            samples_root=tmp_path,
        )

        assert len(rows) == 1
        assert rows[0]["candidate_id"] == "sat_0001"
        assert rows[0]["first_drop_stage"] == "score_gate"
        assert summary["raw_unmatched_satellites"]["total_candidates"] == 1
        assert summary["raw_unmatched_removed"]["by_first_drop_stage"] == {"score_gate": 1}
        assert summary["raw_unmatched_kept"]["total_candidates"] == 0
        json.dumps(summary)


class TestPlotSmoke:
    @pytest.fixture(autouse=True)
    def _mpl_agg(self):
        import matplotlib

        matplotlib.use("Agg")

    def test_all_plots(self, tmp_path):
        samples = []
        for i in range(5):
            base_key = f"g{i:03d}_eo"
            sample = _make_sample(base_key=base_key, galaxy_id=f"g{i:03d}")
            samples.append(sample)
            _write_sample_sidecars(tmp_path, base_key)
        rows = extract_all(_make_report(samples), samples_root=tmp_path)
        outdir = tmp_path / "plots"
        outdir.mkdir()

        plot_tradeoff_scatter(rows, str(outdir))
        plot_rate_boxplot(rows, str(outdir))
        plot_global_matrix(rows, str(outdir))
        plot_delta_precision_recall_scatter(rows, str(outdir))
        plot_precision_recall_raw_vs_post(rows, str(outdir))
        plot_precision_recall_raw_vs_post_scatter(rows, str(outdir))
        plot_matched_loss_vs_num_gt(rows, str(outdir))
        matched_rows, matched_summary = collect_matched_candidate_dispositions(
            _make_report(samples),
            samples_root=tmp_path,
        )
        plot_matched_reason_post_disposition_flow(matched_rows, str(outdir))
        plot_matched_purity_completeness_post_disposition(matched_rows, str(outdir))
        plot_matched_reason_removal_rate(matched_summary, str(outdir))

        for name in PLOT_FILENAMES.values():
            assert os.path.isfile(outdir / name)
