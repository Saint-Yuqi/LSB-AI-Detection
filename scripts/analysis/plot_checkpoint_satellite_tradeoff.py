#!/usr/bin/env python3
"""Satellite checkpoint tradeoff analysis from checkpoint ``report.json``.

Usage:
    conda run -n sam3 python scripts/analysis/plot_checkpoint_satellite_tradeoff.py \
        --report outputs/eval_checkpoint/fbox_gold_satellites/current/linear_magnitude/report.json

This script follows the taxonomy interpretation used in the satellite
diagnostics:

- ``compact_complete`` and ``diffuse_core`` both count as a successful GT
  match.
- ``reject_unmatched`` and ``reject_low_purity`` count as false candidates.
- Taxonomy recall is ``unique_gt_covered / num_gt``.
- Taxonomy precision is ``matched_candidates / total_candidates``.

Because some historical ``predictions_post_pred_only.json`` files do not
preserve the original ``raw_index`` / ``candidate_id``, post survivors are
matched back to raw candidates primarily via ``candidate_rle_sha1``.

Outputs:
    per_sample_tradeoff.csv
    summary.json
    false_candidate_removal_vs_gt_coverage_loss.png
    gt_centric_rate_boxplot.png
    global_gt_coverage_false_candidate_matrix.png
    delta_taxonomy_precision_recall_scatter.png
    taxonomy_precision_recall_raw_vs_post.png
    taxonomy_precision_recall_raw_vs_post_scatter.png
    gt_coverage_loss_vs_num_gt.png
    matched_candidate_reason_post_disposition_flow.png
    matched_candidate_purity_completeness_post_disposition.png
    matched_candidate_reason_removal_rate.png
    removed_matched_candidates.csv
    matched_candidate_post_disposition_summary.json
    removed_matched_candidates_summary.json
    unmatched_candidate_post_disposition.csv
    unmatched_candidate_post_disposition_summary.json
    unmatched_candidate_stage_attribution.csv
    unmatched_candidate_stage_attribution_summary.json
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

_MATCHED_LABELS = ("compact_complete", "diffuse_core")
_UNMATCHED_LABELS = ("reject_unmatched", "reject_low_purity")
_MATCHED_LABEL_SET = set(_MATCHED_LABELS)

PLOT_FILENAMES = {
    "tradeoff_scatter": "false_candidate_removal_vs_gt_coverage_loss.png",
    "rate_boxplot": "gt_centric_rate_boxplot.png",
    "global_matrix": "global_gt_coverage_false_candidate_matrix.png",
    "delta_precision_recall_scatter": "delta_taxonomy_precision_recall_scatter.png",
    "precision_recall_raw_vs_post": "taxonomy_precision_recall_raw_vs_post.png",
    "precision_recall_raw_vs_post_scatter": "taxonomy_precision_recall_raw_vs_post_scatter.png",
    "matched_loss_vs_num_gt": "gt_coverage_loss_vs_num_gt.png",
    "matched_reason_post_disposition_flow": "matched_candidate_reason_post_disposition_flow.png",
    "matched_purity_completeness_post_disposition": "matched_candidate_purity_completeness_post_disposition.png",
    "matched_reason_removal_rate": "matched_candidate_reason_removal_rate.png",
}

LEGACY_PLOT_FILENAMES = (
    "tradeoff_scatter.png",
    "rate_boxplot.png",
    "global_2x2_matrix.png",
    "delta_precision_recall_scatter.png",
    "precision_recall_raw_vs_post.png",
    "precision_recall_raw_vs_post_scatter.png",
    "matched_loss_vs_num_gt.png",
    "matched_reason_post_disposition_flow.png",
    "matched_purity_completeness_post_disposition.png",
    "matched_reason_removal_rate.png",
)

REMOVED_MATCHED_CANDIDATES_CSV = "removed_matched_candidates.csv"
MATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY = "matched_candidate_post_disposition_summary.json"
REMOVED_MATCHED_CANDIDATES_SUMMARY = "removed_matched_candidates_summary.json"
UNMATCHED_CANDIDATE_POST_DISPOSITION_CSV = "unmatched_candidate_post_disposition.csv"
UNMATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY = "unmatched_candidate_post_disposition_summary.json"

STAGE_ATTRIBUTION_CSV = "matched_candidate_stage_attribution.csv"
STAGE_ATTRIBUTION_SUMMARY = "matched_candidate_stage_attribution_summary.json"
UNMATCHED_STAGE_ATTRIBUTION_CSV = "unmatched_candidate_stage_attribution.csv"
UNMATCHED_STAGE_ATTRIBUTION_SUMMARY = "unmatched_candidate_stage_attribution_summary.json"
STAGE_TRACE_SIDECAR_NAME = "post_pred_only_satellite_stage_trace.json"
STAGE_TRACE_PLOT_FILENAMES = {
    "first_drop_stage_profile": "matched_candidate_first_drop_stage_profile.png",
    "stage_reason_heatmap": "matched_candidate_stage_reason_heatmap.png",
    "label_reason_first_drop_stage": "matched_candidate_label_reason_first_drop_stage.png",
}
STAGE_ORDER = ("score_gate", "prior_filter", "core_policy")

STAGE_ATTRIBUTION_COLUMNS = [
    "base_key",
    "galaxy_id",
    "view",
    "benchmark_mode",
    "raw_index",
    "candidate_id",
    "candidate_rle_sha1",
    "taxonomy_label",
    "label_reason",
    "kept_in_post",
    "first_drop_stage",
    "first_drop_reason",
    "score_gate_outcome",
    "score_gate_reason",
    "prior_filter_outcome",
    "prior_filter_reason",
    "core_policy_outcome",
    "core_policy_reason",
]

MATCHED_REASON_ORDER = (
    "one_to_one_complete_seed_ratio_ok",
    "pure_and_complete",
    "pure_but_core_only",
)
MATCHED_REASON_LABELS = {
    "one_to_one_complete_seed_ratio_ok": "1:1 near-complete\nratio OK",
    "pure_and_complete": "Pure + complete",
    "pure_but_core_only": "Pure but\ncore-only",
}
UNMATCHED_REASON_ORDER = (
    "no_gt_overlap",
    "mixed_coverage",
)
MATCHED_POST_STATUS_ORDER = ("kept", "removed")
MATCHED_POST_STATUS_LABELS = {
    "kept": "Kept in Post",
    "removed": "Removed in Post",
}
MATCHED_DISPOSITION_NUMERIC_FIELDS = (
    "purity",
    "completeness",
    "seed_gt_ratio",
    "confidence_score",
    "seed_area",
    "annulus_excess",
    "radial_monotonicity",
)

REMOVED_MATCHED_COLUMNS = [
    "base_key",
    "galaxy_id",
    "view",
    "benchmark_mode",
    "raw_index",
    "candidate_id",
    "candidate_rle_sha1",
    "taxonomy_label",
    "label_reason",
    "matched_gt_id",
    "matched_gt_area",
    "seed_area",
    "confidence_score",
    "overlap_px",
    "purity",
    "completeness",
    "seed_gt_ratio",
    "is_one_to_one",
    "intersects_roi",
    "annulus_excess",
    "radial_monotonicity",
    "raw_prediction_score",
    "raw_prediction_area",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
]

UNMATCHED_POST_DISPOSITION_COLUMNS = [
    "base_key",
    "galaxy_id",
    "view",
    "benchmark_mode",
    "raw_index",
    "candidate_id",
    "candidate_rle_sha1",
    "taxonomy_label",
    "label_reason",
    "matched_gt_id",
    "matched_gt_area",
    "seed_area",
    "confidence_score",
    "overlap_px",
    "purity",
    "completeness",
    "seed_gt_ratio",
    "is_one_to_one",
    "intersects_roi",
    "annulus_excess",
    "radial_monotonicity",
    "raw_prediction_score",
    "raw_prediction_area",
    "bbox_x",
    "bbox_y",
    "bbox_w",
    "bbox_h",
    "kept_in_post",
    "post_status",
]

REMOVED_MATCHED_NUMERIC_FIELDS = (
    "seed_area",
    "confidence_score",
    "matched_gt_area",
    "overlap_px",
    "purity",
    "completeness",
    "seed_gt_ratio",
    "annulus_excess",
    "radial_monotonicity",
    "raw_prediction_score",
    "raw_prediction_area",
    "bbox_w",
    "bbox_h",
)
UNMATCHED_DISPOSITION_NUMERIC_FIELDS = REMOVED_MATCHED_NUMERIC_FIELDS

CSV_COLUMNS = [
    "base_key",
    "galaxy_id",
    "view",
    "benchmark_mode",
    "num_gt_satellites_full",
    "num_gt_satellites_roi",
    "num_pred_raw_roi",
    "num_pred_post_roi",
    "compact_complete_raw",
    "diffuse_core_raw",
    "reject_unmatched_raw",
    "reject_low_purity_raw",
    "compact_complete_post",
    "diffuse_core_post",
    "reject_unmatched_post",
    "reject_low_purity_post",
    "matched_candidates_raw",
    "unmatched_candidates_raw",
    "matched_candidates_post",
    "unmatched_candidates_post",
    "unique_gt_covered_raw",
    "unique_gt_covered_post",
    "unique_gt_lost_post",
    "matched_candidates_removed",
    "unmatched_candidates_removed",
    "gt_coverage_retention",
    "gt_coverage_loss_rate",
    "matched_candidate_retention",
    "matched_candidate_loss_rate",
    "unmatched_candidate_removal_rate",
    "residual_unmatched_candidate_fraction",
    "tax_precision_raw",
    "tax_recall_raw",
    "tax_precision_post",
    "tax_recall_post",
    "delta_tax_precision",
    "delta_tax_recall",
    "pixel_dice_raw",
    "pixel_dice_post",
]


def _safe_div(a: int | float | None, b: int | float | None) -> float:
    if a is None or b in (None, 0):
        return float("nan")
    return float(a) / float(b)


def _safe_sub(a: int | float | None, b: int | float | None) -> float:
    if a is None or b is None:
        return float("nan")
    if isinstance(a, float) and math.isnan(a):
        return float("nan")
    if isinstance(b, float) and math.isnan(b):
        return float("nan")
    return float(a) - float(b)


def _is_finite_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not math.isnan(v)


def _fmt(v: Any) -> Any:
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def _plot_path(outdir: str, key: str) -> str:
    return os.path.join(outdir, PLOT_FILENAMES[key])


def _cleanup_legacy_plot_outputs(outdir: str) -> None:
    for name in LEGACY_PLOT_FILENAMES:
        path = os.path.join(outdir, name)
        if os.path.exists(path):
            os.remove(path)
            logger.info("Removed legacy plot %s", name)


def _warn_soft_invariant(base_key: str, message: str) -> None:
    warnings.warn(f"{base_key}: {message}", stacklevel=2)


def _get_diag_summary(sample: dict[str, Any]) -> dict[str, Any] | None:
    diagnostics = sample.get("diagnostics") or {}
    sat = diagnostics.get("satellites_raw") or {}
    return sat.get("summary")


def _official_satellite_scope(benchmark_mode: str) -> str:
    """Map a benchmark mode to the scope used for official satellite metrics.

    ``fbox_gold_satellites`` → ``"roi"`` (GT curated inside the ROI).
    ``gt_canonical`` → ``"full_frame"`` (no ROI).
    """
    if benchmark_mode == "fbox_gold_satellites":
        return "roi"
    if benchmark_mode == "gt_canonical":
        return "full_frame"
    raise ValueError(
        f"no official satellite scope for benchmark_mode={benchmark_mode!r}"
    )


def _get_satellite_block(
    sample: dict[str, Any], layer_name: str, scope: str,
) -> dict[str, Any]:
    """Fetch the per-scope satellite block for a layer. Raises KeyError if
    the sample predates the typed-block schema.
    """
    return sample["layers"][layer_name]["satellites"][scope]


def _is_new_taxonomy_schema(block: dict[str, Any]) -> bool:
    """A satellite block written by the new taxonomy pipeline carries
    ``matched_candidates``; legacy detection blocks carry ``tp``.
    """
    return "matched_candidates" in block


def _count_or_none(counts: dict[str, int] | None, label: str) -> int | None:
    if counts is None:
        return None
    return int(counts.get(label, 0))


def _sum_labels(counts: dict[str, int] | None, labels: tuple[str, ...]) -> int | None:
    if counts is None:
        return None
    return int(sum(int(counts.get(label, 0)) for label in labels))


def _load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def _load_predictions_list(path: Path) -> list[dict[str, Any]]:
    doc = _load_json(path)
    if isinstance(doc, dict) and "predictions" in doc:
        return list(doc["predictions"])
    if isinstance(doc, list):
        return list(doc)
    raise ValueError(f"Unsupported predictions JSON schema: {path}")


def _has_stage_trace(predictions: list[dict[str, Any]]) -> bool:
    trace_keys = {"history", "final_status", "stage"}
    return any(any(key in pred for key in trace_keys) for pred in predictions)


def _identity_from_row(row: dict[str, Any]) -> tuple[int, str]:
    return int(row["raw_index"]), str(row["candidate_id"])


def _identity_from_prediction(pred: dict[str, Any]) -> tuple[int, str] | None:
    if "raw_index" not in pred or "candidate_id" not in pred:
        return None
    return int(pred["raw_index"]), str(pred["candidate_id"])


def _in_official_scope(row: dict[str, Any], benchmark_mode: str) -> bool:
    scope = _official_satellite_scope(benchmark_mode)
    if scope == "full_frame":
        return True
    return bool(row.get("intersects_roi", False))


def _link_post_predictions_to_raw(
    diag_rows: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
) -> dict[str, Any]:
    diag_by_identity = {
        _identity_from_row(row): row for row in diag_rows
    }
    raw_pred_by_identity = {
        identity: pred
        for pred in raw_predictions
        for identity in [_identity_from_prediction(pred)]
        if identity is not None
    }
    raw_by_sha = {
        str(pred["candidate_rle_sha1"]): identity
        for pred in raw_predictions
        for identity in [_identity_from_prediction(pred)]
        if pred.get("candidate_rle_sha1") is not None and identity is not None
    }

    linked_identities: set[tuple[int, str]] = set()
    linked_rows: list[dict[str, Any]] = []
    missing_links = 0
    used_sha_links = 0
    used_identity_links = 0

    for pred in post_predictions:
        row = None
        identity = None
        sha = pred.get("candidate_rle_sha1")
        if sha is not None:
            identity = raw_by_sha.get(str(sha))
            if identity is not None:
                row = diag_by_identity.get(identity)
                if row is not None:
                    used_sha_links += 1
        if row is None:
            identity = _identity_from_prediction(pred)
            if identity is not None:
                row = diag_by_identity.get(identity)
                if row is not None:
                    used_identity_links += 1
        if row is None or identity is None:
            missing_links += 1
            continue
        linked_identities.add(identity)
        linked_rows.append(row)

    return {
        "diag_by_identity": diag_by_identity,
        "raw_pred_by_identity": raw_pred_by_identity,
        "linked_post_identities": linked_identities,
        "linked_post_rows": linked_rows,
        "missing_post_links": missing_links,
        "used_sha_links": used_sha_links,
        "used_identity_links": used_identity_links,
    }


def _compute_taxonomy_coverage(
    diag_rows: list[dict[str, Any]],
    raw_predictions: list[dict[str, Any]],
    post_predictions: list[dict[str, Any]],
    benchmark_mode: str = "fbox_gold_satellites",
) -> dict[str, Any]:
    """Compute taxonomy coverage using raw diagnostics + post survivor remap.

    Post predictions are mapped back to raw diagnostics by:
    1. ``candidate_rle_sha1`` when available
    2. fallback ``(raw_index, candidate_id)``
    """
    linkage = _link_post_predictions_to_raw(diag_rows, raw_predictions, post_predictions)

    roi_rows = [row for row in diag_rows if _in_official_scope(row, benchmark_mode)]
    matched_raw_rows = [
        row
        for row in roi_rows
        if row.get("taxonomy_label") in _MATCHED_LABEL_SET
        and row.get("matched_gt_id") is not None
    ]
    raw_gt_ids = {int(row["matched_gt_id"]) for row in matched_raw_rows}

    post_roi_rows = [
        row for row in linkage["linked_post_rows"] if _in_official_scope(row, benchmark_mode)
    ]
    post_matched_rows = [
        row
        for row in post_roi_rows
        if row.get("taxonomy_label") in _MATCHED_LABEL_SET
        and row.get("matched_gt_id") is not None
    ]
    post_gt_ids = {int(row["matched_gt_id"]) for row in post_matched_rows}

    return {
        "raw_candidates_roi": len(roi_rows),
        "matched_candidates_raw": len(matched_raw_rows),
        "unique_gt_covered_raw": len(raw_gt_ids),
        "raw_gt_ids": raw_gt_ids,
        "post_candidates_roi": len(post_roi_rows),
        "matched_candidates_post": len(post_matched_rows),
        "unique_gt_covered_post": len(post_gt_ids),
        "post_gt_ids": post_gt_ids,
        "missing_post_links": linkage["missing_post_links"],
        "used_sha_links": linkage["used_sha_links"],
        "used_identity_links": linkage["used_identity_links"],
    }


def _compute_taxonomy_coverage_from_sidecars(
    sample_dir: Path,
    benchmark_mode: str = "fbox_gold_satellites",
) -> dict[str, Any] | None:
    diag_path = sample_dir / "diagnostics.json"
    raw_path = sample_dir / "predictions_raw.json"
    post_path = sample_dir / "predictions_post_pred_only.json"
    if not (diag_path.exists() and raw_path.exists() and post_path.exists()):
        return None

    diag_doc = _load_json(diag_path)
    diag_rows = list(diag_doc["per_candidate"])
    raw_predictions = [
        pred
        for pred in _load_predictions_list(raw_path)
        if pred.get("type_label", "satellites") == "satellites"
    ]
    post_predictions = [
        pred
        for pred in _load_predictions_list(post_path)
        if pred.get("type_label", "satellites") == "satellites"
    ]
    return _compute_taxonomy_coverage(
        diag_rows,
        raw_predictions,
        post_predictions,
        benchmark_mode=benchmark_mode,
    )


def extract_row(
    sample: dict[str, Any],
    samples_root: str | Path | None = None,
) -> dict[str, Any]:
    """Extract one per-sample row from the checkpoint report.

    Resolution order:
      1. New taxonomy schema — read ``matched_candidates``, ``unique_gt_covered``,
         ``num_pred``, ``counts_by_label`` directly from the per-scope satellite
         block resolved via :func:`_official_satellite_scope`.
      2. Legacy schema (IoU/Hungarian block + diagnostics summary + sidecar
         sha-remap) — kept for back-compat with older ``report.json`` files.

    CSV column suffixes stay ``_roi`` regardless of the resolved scope so
    downstream consumers don't need to migrate.
    """
    base_key = sample["base_key"]
    scope = _official_satellite_scope(sample["benchmark_mode"])

    raw_block = _get_satellite_block(sample, "raw", scope)
    post_block = _get_satellite_block(sample, "post_pred_only", scope)

    full_num_gt = sample.get("num_gt_satellites")
    raw_num_gt_scope = raw_block.get("num_gt")
    post_num_gt_scope = post_block.get("num_gt")
    if (
        raw_num_gt_scope is not None
        and post_num_gt_scope is not None
        and int(raw_num_gt_scope) != int(post_num_gt_scope)
    ):
        _warn_soft_invariant(
            base_key,
            f"{scope} GT mismatch between layers: raw={raw_num_gt_scope}, post={post_num_gt_scope}",
        )
    num_gt_roi = int(
        raw_num_gt_scope if raw_num_gt_scope is not None else post_num_gt_scope or 0
    )

    new_schema = _is_new_taxonomy_schema(raw_block) and _is_new_taxonomy_schema(post_block)

    if new_schema:
        raw_counts_roi = raw_block.get("counts_by_label")
        post_counts_roi = post_block.get("counts_by_label")
    else:
        diag_summary = _get_diag_summary(sample)
        raw_counts_roi = (
            None if diag_summary is None else diag_summary.get("counts_by_label_roi")
        )
        post_counts_roi = (
            None if diag_summary is None else diag_summary.get("counts_post_by_label_roi")
        )

    if new_schema:
        matched_raw = int(raw_block["matched_candidates"])
        unmatched_raw = int(raw_block["unmatched_candidates"])
        matched_post = int(post_block["matched_candidates"])
        unmatched_post = int(post_block["unmatched_candidates"])
        raw_num_pred_roi = int(raw_block["num_pred"])
        post_num_pred_roi = int(post_block["num_pred"])
        unique_gt_covered_raw = int(raw_block["unique_gt_covered"])
        unique_gt_covered_post = int(post_block["unique_gt_covered"])
    else:
        matched_raw_summary = _sum_labels(raw_counts_roi, _MATCHED_LABELS)
        unmatched_raw_summary = _sum_labels(raw_counts_roi, _UNMATCHED_LABELS)
        matched_post_summary = _sum_labels(post_counts_roi, _MATCHED_LABELS)
        unmatched_post_summary = _sum_labels(post_counts_roi, _UNMATCHED_LABELS)

        coverage = None
        if samples_root is not None:
            coverage = _compute_taxonomy_coverage_from_sidecars(
                Path(samples_root) / base_key,
                benchmark_mode=sample["benchmark_mode"],
            )

        if coverage is not None:
            matched_raw = int(coverage["matched_candidates_raw"])
            matched_post = int(coverage["matched_candidates_post"])
            raw_num_pred_roi = int(coverage["raw_candidates_roi"])
            post_num_pred_roi = int(coverage["post_candidates_roi"])
            unique_gt_covered_raw = int(coverage["unique_gt_covered_raw"])
            unique_gt_covered_post = int(coverage["unique_gt_covered_post"])
            unmatched_raw = raw_num_pred_roi - matched_raw
            unmatched_post = post_num_pred_roi - matched_post
            if matched_raw_summary is not None and matched_raw_summary != matched_raw:
                _warn_soft_invariant(
                    base_key,
                    f"raw matched count drift: summary={matched_raw_summary}, sidecar={matched_raw}",
                )
            if matched_post_summary is not None and matched_post_summary != matched_post:
                _warn_soft_invariant(
                    base_key,
                    f"post matched count drift: summary={matched_post_summary}, sidecar={matched_post}",
                )
            if coverage["missing_post_links"] > 0:
                _warn_soft_invariant(
                    base_key,
                    f"{coverage['missing_post_links']} post predictions could not be linked back to raw diagnostics",
                )
        else:
            matched_raw = matched_raw_summary
            unmatched_raw = unmatched_raw_summary
            matched_post = matched_post_summary
            unmatched_post = unmatched_post_summary
            raw_num_pred_roi = int(raw_block.get("num_pred", 0))
            post_num_pred_roi = int(post_block.get("num_pred", 0))
            unique_gt_covered_raw = None
            unique_gt_covered_post = None

    matched_removed = _safe_sub(matched_raw, matched_post)
    unmatched_removed = _safe_sub(unmatched_raw, unmatched_post)
    if _is_finite_number(matched_removed) and matched_removed < 0:
        _warn_soft_invariant(base_key, f"matched_candidates_removed={matched_removed} < 0")
    if _is_finite_number(unmatched_removed) and unmatched_removed < 0:
        _warn_soft_invariant(base_key, f"unmatched_candidates_removed={unmatched_removed} < 0")

    unique_gt_lost_post = _safe_sub(unique_gt_covered_raw, unique_gt_covered_post)
    if _is_finite_number(unique_gt_lost_post) and unique_gt_lost_post < 0:
        _warn_soft_invariant(base_key, f"unique_gt_lost_post={unique_gt_lost_post} < 0")

    tax_precision_raw = _safe_div(matched_raw, raw_num_pred_roi)
    tax_precision_post = _safe_div(matched_post, post_num_pred_roi)
    tax_recall_raw = _safe_div(unique_gt_covered_raw, num_gt_roi)
    tax_recall_post = _safe_div(unique_gt_covered_post, num_gt_roi)

    return {
        "base_key": base_key,
        "galaxy_id": sample["galaxy_id"],
        "view": sample["view"],
        "benchmark_mode": sample["benchmark_mode"],
        "num_gt_satellites_full": full_num_gt,
        "num_gt_satellites_roi": num_gt_roi,
        "num_pred_raw_roi": raw_num_pred_roi,
        "num_pred_post_roi": post_num_pred_roi,
        "compact_complete_raw": _count_or_none(raw_counts_roi, "compact_complete"),
        "diffuse_core_raw": _count_or_none(raw_counts_roi, "diffuse_core"),
        "reject_unmatched_raw": _count_or_none(raw_counts_roi, "reject_unmatched"),
        "reject_low_purity_raw": _count_or_none(raw_counts_roi, "reject_low_purity"),
        "compact_complete_post": _count_or_none(post_counts_roi, "compact_complete"),
        "diffuse_core_post": _count_or_none(post_counts_roi, "diffuse_core"),
        "reject_unmatched_post": _count_or_none(post_counts_roi, "reject_unmatched"),
        "reject_low_purity_post": _count_or_none(post_counts_roi, "reject_low_purity"),
        "matched_candidates_raw": matched_raw,
        "unmatched_candidates_raw": unmatched_raw,
        "matched_candidates_post": matched_post,
        "unmatched_candidates_post": unmatched_post,
        "unique_gt_covered_raw": unique_gt_covered_raw,
        "unique_gt_covered_post": unique_gt_covered_post,
        "unique_gt_lost_post": unique_gt_lost_post,
        "matched_candidates_removed": matched_removed,
        "unmatched_candidates_removed": unmatched_removed,
        "gt_coverage_retention": _safe_div(unique_gt_covered_post, unique_gt_covered_raw),
        "gt_coverage_loss_rate": _safe_div(unique_gt_lost_post, unique_gt_covered_raw),
        "matched_candidate_retention": _safe_div(matched_post, matched_raw),
        "matched_candidate_loss_rate": _safe_div(matched_removed, matched_raw),
        "unmatched_candidate_removal_rate": _safe_div(unmatched_removed, unmatched_raw),
        "residual_unmatched_candidate_fraction": _safe_div(unmatched_post, post_num_pred_roi),
        "tax_precision_raw": tax_precision_raw,
        "tax_recall_raw": tax_recall_raw,
        "tax_precision_post": tax_precision_post,
        "tax_recall_post": tax_recall_post,
        "delta_tax_precision": _safe_sub(tax_precision_post, tax_precision_raw),
        "delta_tax_recall": _safe_sub(tax_recall_post, tax_recall_raw),
        "pixel_dice_raw": (raw_block.get("pixel") or {}).get("dice"),
        "pixel_dice_post": (post_block.get("pixel") or {}).get("dice"),
    }


def extract_all(
    report: dict[str, Any],
    samples_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    return [extract_row(sample, samples_root=samples_root) for sample in report["per_sample"]]


def write_csv(rows: list[dict[str, Any]], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in CSV_COLUMNS})
    logger.info("Wrote %s (%d rows)", path, len(rows))


def _finite_vec(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.asarray([row[key] for row in rows if _is_finite_number(row.get(key))], dtype=float)


def _finite_pair(
    rows: list[dict[str, Any]],
    xkey: str,
    ykey: str,
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[float] = []
    ys: list[float] = []
    for row in rows:
        xv = row.get(xkey)
        yv = row.get(ykey)
        if _is_finite_number(xv) and _is_finite_number(yv):
            xs.append(float(xv))
            ys.append(float(yv))
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def _paired_metric(
    rows: list[dict[str, Any]],
    raw_key: str,
    post_key: str,
) -> tuple[np.ndarray, np.ndarray]:
    raw_vals: list[float] = []
    post_vals: list[float] = []
    for row in rows:
        raw_val = row.get(raw_key)
        post_val = row.get(post_key)
        if _is_finite_number(raw_val) and _is_finite_number(post_val):
            raw_vals.append(float(raw_val))
            post_vals.append(float(post_val))
    return np.asarray(raw_vals, dtype=float), np.asarray(post_vals, dtype=float)


def _paired_taxonomy_precision(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Paired taxonomy precision using the strict symmetric zero-division rule.

    For each side X in {raw, post} (denominator = ``num_pred_X_roi``, counter on
    the opposite side = ``num_gt_satellites_roi``):

      - ``num_pred_X > 0`` -> side_P := matched_X / num_pred_X
      - ``num_pred_X = 0`` AND ``num_gt = 0`` -> side_P := 1.0 (vacuous: nothing
        predicted, nothing to predict — legitimate empty-vs-empty agreement).
      - ``num_pred_X = 0`` AND ``num_gt > 0`` -> this side is undefined; the
        whole paired point is excluded. Forcing it to 1.0 would reward a
        collapse; forcing it to 0.0 mixes a recall failure into precision.
        The recall plot carries that failure explicitly.

    A sample is classified as:
      - ``natural``  : both sides natural (no zero-division anywhere)
      - ``vacuous``  : at least one side was 1.0 by the vacuous rule, and the
                       sample was not excluded
      - ``excluded`` : any side hit the ``pred=0 & GT>0`` case
    """
    raw_vals: list[float] = []
    post_vals: list[float] = []
    natural = vacuous = excluded = 0
    for row in rows:
        num_gt = int(row.get("num_gt_satellites_roi") or 0)
        num_raw = int(row.get("num_pred_raw_roi") or 0)
        num_post = int(row.get("num_pred_post_roi") or 0)
        matched_raw = int(row.get("matched_candidates_raw") or 0)
        matched_post = int(row.get("matched_candidates_post") or 0)

        raw_excluded = num_raw == 0 and num_gt > 0
        post_excluded = num_post == 0 and num_gt > 0
        if raw_excluded or post_excluded:
            excluded += 1
            continue

        raw_p = 1.0 if num_raw == 0 else matched_raw / num_raw
        post_p = 1.0 if num_post == 0 else matched_post / num_post
        raw_vals.append(raw_p)
        post_vals.append(post_p)
        if num_raw == 0 or num_post == 0:
            vacuous += 1
        else:
            natural += 1
    breakdown = {
        "natural": natural,
        "vacuous": vacuous,
        "excluded": excluded,
        "paired": natural + vacuous,
        "total": len(rows),
    }
    return (
        np.asarray(raw_vals, dtype=float),
        np.asarray(post_vals, dtype=float),
        breakdown,
    )


def _paired_taxonomy_recall(
    rows: list[dict[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Paired taxonomy recall using the strict symmetric zero-division rule.

    Mirror image of the precision rule. For each side X in {raw, post}
    (denominator = ``num_gt_satellites_roi``, counter on the opposite side =
    ``num_pred_X_roi``):

      - ``num_gt > 0`` -> side_R := unique_gt_covered_X / num_gt
      - ``num_gt = 0`` AND ``num_pred_X = 0`` -> side_R := 1.0 (vacuous: empty
        GT matched by empty predictions — legitimate empty-vs-empty agreement).
      - ``num_gt = 0`` AND ``num_pred_X > 0`` -> this side is undefined; the
        paired point is excluded. Forcing it to 1.0 would reward hallucination
        on an empty-GT sample with a perfect recall score; the precision plot
        is where that hallucination is visible.

    Classification uses the same ``natural / vacuous / excluded`` labels as
    the precision helper above.
    """
    raw_vals: list[float] = []
    post_vals: list[float] = []
    natural = vacuous = excluded = 0
    for row in rows:
        num_gt = int(row.get("num_gt_satellites_roi") or 0)
        num_raw = int(row.get("num_pred_raw_roi") or 0)
        num_post = int(row.get("num_pred_post_roi") or 0)
        covered_raw = int(row.get("unique_gt_covered_raw") or 0)
        covered_post = int(row.get("unique_gt_covered_post") or 0)

        raw_excluded = num_gt == 0 and num_raw > 0
        post_excluded = num_gt == 0 and num_post > 0
        if raw_excluded or post_excluded:
            excluded += 1
            continue

        raw_r = 1.0 if num_gt == 0 else covered_raw / num_gt
        post_r = 1.0 if num_gt == 0 else covered_post / num_gt
        raw_vals.append(raw_r)
        post_vals.append(post_r)
        if num_gt == 0:
            vacuous += 1
        else:
            natural += 1
    breakdown = {
        "natural": natural,
        "vacuous": vacuous,
        "excluded": excluded,
        "paired": natural + vacuous,
        "total": len(rows),
    }
    return (
        np.asarray(raw_vals, dtype=float),
        np.asarray(post_vals, dtype=float),
        breakdown,
    )


def _paired_breakdown_subtitle(kind: str, breakdown: dict[str, int]) -> str:
    """Two-line subtitle summarizing the strict symmetric pairing for one metric."""
    nat = breakdown["natural"]
    vac = breakdown["vacuous"]
    exc = breakdown["excluded"]
    paired = breakdown["paired"]
    total = breakdown["total"]
    line1 = f"paired n={paired} / {total}  (natural={nat} + vacuous={vac}, excluded={exc})"
    if kind == "precision":
        line2 = "pred=0 & GT=0 -> 1.0 (vacuous);  pred=0 & GT>0 -> excluded"
    else:
        line2 = "GT=0 & pred=0 -> 1.0 (vacuous);  GT=0 & pred>0 -> excluded"
    return line1 + "\n" + line2


def _sum_int(rows: list[dict[str, Any]], key: str) -> int:
    total = 0
    for row in rows:
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and math.isnan(value):
            continue
        total += int(value)
    return total


def _rate_stats(rows: list[dict[str, Any]], key: str) -> dict[str, int | float | None]:
    vals = _finite_vec(rows, key)
    if vals.size == 0:
        return {"valid": 0, "omitted": len(rows), "mean": None, "median": None, "std": None}
    return {
        "valid": int(vals.size),
        "omitted": int(len(rows) - vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
    }


def _numeric_stats(rows: list[dict[str, Any]], key: str) -> dict[str, int | float | None]:
    vals = _finite_vec(rows, key)
    if vals.size == 0:
        return {
            "valid": 0,
            "omitted": len(rows),
            "mean": None,
            "median": None,
            "std": None,
            "min": None,
            "max": None,
        }
    return {
        "valid": int(vals.size),
        "omitted": int(len(rows) - vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "std": float(np.std(vals)),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
    }


def _json_group_key(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _count_by_key(rows: list[dict[str, Any]], key: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        group_key = _json_group_key(row.get(key))
        counts[group_key] = counts.get(group_key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: (-item[1], item[0])))


def _group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        group_key = _json_group_key(row.get(key))
        groups.setdefault(group_key, []).append(row)
    return groups


def _summarize_removed_group(
    rows: list[dict[str, Any]],
    total_removed: int,
) -> dict[str, Any]:
    matched_gt_ids = {
        int(row["matched_gt_id"])
        for row in rows
        if row.get("matched_gt_id") is not None
    }
    return {
        "count": len(rows),
        "fraction_of_removed_matched_candidates": (
            float(len(rows)) / float(total_removed) if total_removed else None
        ),
        "unique_matched_gt_ids": len(matched_gt_ids),
        "is_one_to_one_counts": _count_by_key(rows, "is_one_to_one"),
        "numeric_stats": {
            field: _numeric_stats(rows, field) for field in REMOVED_MATCHED_NUMERIC_FIELDS
        },
    }


def _summarize_post_disposition_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    kept_rows = [row for row in rows if bool(row.get("kept_in_post"))]
    removed_rows = [row for row in rows if not bool(row.get("kept_in_post"))]
    return {
        "raw_total": len(rows),
        "kept": len(kept_rows),
        "removed": len(removed_rows),
        "removal_rate": (float(len(removed_rows)) / float(len(rows)) if rows else None),
    }


def _candidate_record_from_diag_row(
    sample: dict[str, Any],
    row: dict[str, Any],
    raw_pred: dict[str, Any],
    *,
    kept_in_post: bool,
) -> dict[str, Any]:
    identity = _identity_from_row(row)
    bbox = raw_pred.get("bbox_xywh") or [None, None, None, None]
    return {
        "base_key": sample["base_key"],
        "galaxy_id": sample["galaxy_id"],
        "view": sample["view"],
        "benchmark_mode": sample["benchmark_mode"],
        "raw_index": identity[0],
        "candidate_id": identity[1],
        "candidate_rle_sha1": raw_pred.get("candidate_rle_sha1"),
        "taxonomy_label": row.get("taxonomy_label"),
        "label_reason": row.get("label_reason"),
        "matched_gt_id": row.get("matched_gt_id"),
        "matched_gt_area": row.get("matched_gt_area"),
        "seed_area": row.get("seed_area"),
        "confidence_score": row.get("confidence_score"),
        "overlap_px": row.get("overlap_px"),
        "purity": row.get("purity"),
        "completeness": row.get("completeness"),
        "seed_gt_ratio": row.get("seed_gt_ratio"),
        "is_one_to_one": row.get("is_one_to_one"),
        "intersects_roi": row.get("intersects_roi"),
        "annulus_excess": row.get("annulus_excess"),
        "radial_monotonicity": row.get("radial_monotonicity"),
        "raw_prediction_score": raw_pred.get("score"),
        "raw_prediction_area": raw_pred.get("area"),
        "bbox_x": bbox[0],
        "bbox_y": bbox[1],
        "bbox_w": bbox[2],
        "bbox_h": bbox[3],
        "kept_in_post": kept_in_post,
        "post_status": "kept" if kept_in_post else "removed",
    }


def _collect_candidate_dispositions_from_sample(
    sample: dict[str, Any],
    sample_dir: Path,
    *,
    label_set: set[str],
    require_matched_gt: bool,
) -> dict[str, Any] | None:
    diag_path = sample_dir / "diagnostics.json"
    raw_path = sample_dir / "predictions_raw.json"
    post_path = sample_dir / "predictions_post_pred_only.json"
    if not (diag_path.exists() and raw_path.exists() and post_path.exists()):
        return None

    diag_doc = _load_json(diag_path)
    diag_rows = list(diag_doc["per_candidate"])
    raw_predictions = [
        pred
        for pred in _load_predictions_list(raw_path)
        if pred.get("type_label", "satellites") == "satellites"
    ]
    post_predictions = [
        pred
        for pred in _load_predictions_list(post_path)
        if pred.get("type_label", "satellites") == "satellites"
    ]

    linkage = _link_post_predictions_to_raw(diag_rows, raw_predictions, post_predictions)
    records: list[dict[str, Any]] = []

    for row in diag_rows:
        if not _in_official_scope(row, sample["benchmark_mode"]):
            continue
        if row.get("taxonomy_label") not in label_set:
            continue
        if require_matched_gt and row.get("matched_gt_id") is None:
            continue

        identity = _identity_from_row(row)
        raw_pred = linkage["raw_pred_by_identity"].get(identity, {})
        kept_in_post = identity in linkage["linked_post_identities"]
        records.append(
            _candidate_record_from_diag_row(
                sample,
                row,
                raw_pred,
                kept_in_post=kept_in_post,
            )
        )

    return {
        "records": records,
        "missing_post_links": linkage["missing_post_links"],
        "used_sha_links": linkage["used_sha_links"],
        "used_identity_links": linkage["used_identity_links"],
        "stage_trace_available": (
            _has_stage_trace(raw_predictions) or _has_stage_trace(post_predictions)
        ),
    }


def _collect_matched_candidate_dispositions_from_sample(
    sample: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any] | None:
    return _collect_candidate_dispositions_from_sample(
        sample,
        sample_dir,
        label_set=_MATCHED_LABEL_SET,
        require_matched_gt=True,
    )


def _collect_unmatched_candidate_dispositions_from_sample(
    sample: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any] | None:
    return _collect_candidate_dispositions_from_sample(
        sample,
        sample_dir,
        label_set=set(_UNMATCHED_LABELS),
        require_matched_gt=False,
    )


def _build_reason_disposition_summary(
    records: list[dict[str, Any]],
    *,
    reason_order: tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    reason_groups = _group_rows(records, "label_reason")
    by_label_reason: dict[str, dict[str, Any]] = {}
    if reason_order is not None:
        for reason in reason_order:
            group_rows = reason_groups.get(reason, [])
            if group_rows:
                by_label_reason[reason] = _summarize_post_disposition_group(group_rows)
    for reason, group_rows in sorted(reason_groups.items(), key=lambda item: (-len(item[1]), item[0])):
        if reason not in by_label_reason:
            by_label_reason[reason] = _summarize_post_disposition_group(group_rows)
    return by_label_reason


def _build_post_status_summary(
    records: list[dict[str, Any]],
    *,
    numeric_fields: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    by_post_status: dict[str, dict[str, Any]] = {}
    for status in MATCHED_POST_STATUS_ORDER:
        status_rows = [row for row in records if row.get("post_status") == status]
        by_post_status[status] = {
            "count": len(status_rows),
            "numeric_stats": {
                field: _numeric_stats(status_rows, field)
                for field in numeric_fields
            },
            "taxonomy_label_counts": _count_by_key(status_rows, "taxonomy_label"),
            "label_reason_counts": _count_by_key(status_rows, "label_reason"),
        }
    return by_post_status


def collect_matched_candidate_dispositions(
    report: dict[str, Any],
    samples_root: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = list(report.get("per_sample", []))
    if samples_root is None:
        summary = {
            "n_samples_in_report": len(samples),
            "n_samples_with_sidecars": 0,
            "n_samples_missing_sidecars": len(samples),
            "samples_missing_sidecars": [sample["base_key"] for sample in samples],
            "raw_matched_candidate_total": 0,
            "kept_in_post": 0,
            "removed_in_post": 0,
            "stage_trace_available": False,
            "missing_post_links_total": 0,
            "used_sha_links_total": 0,
            "used_identity_links_total": 0,
            "by_taxonomy_label": {},
            "by_label_reason": {},
            "by_post_status": {},
        }
        return [], summary

    records: list[dict[str, Any]] = []
    missing_samples: list[str] = []
    stage_trace_available = False
    missing_post_links_total = 0
    used_sha_links_total = 0
    used_identity_links_total = 0
    samples_root = Path(samples_root)

    for sample in samples:
        payload = _collect_matched_candidate_dispositions_from_sample(
            sample,
            samples_root / sample["base_key"],
        )
        if payload is None:
            missing_samples.append(sample["base_key"])
            continue
        records.extend(payload["records"])
        stage_trace_available = stage_trace_available or bool(payload["stage_trace_available"])
        missing_post_links_total += int(payload["missing_post_links"])
        used_sha_links_total += int(payload["used_sha_links"])
        used_identity_links_total += int(payload["used_identity_links"])

    by_taxonomy_label = {
        key: _summarize_post_disposition_group(group_rows)
        for key, group_rows in sorted(
            _group_rows(records, "taxonomy_label").items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }
    by_label_reason = _build_reason_disposition_summary(
        records,
        reason_order=MATCHED_REASON_ORDER,
    )
    by_post_status = _build_post_status_summary(
        records,
        numeric_fields=MATCHED_DISPOSITION_NUMERIC_FIELDS,
    )
    raw_total = len(records)
    kept_total = sum(1 for row in records if bool(row.get("kept_in_post")))
    removed_total = raw_total - kept_total
    summary = {
        "n_samples_in_report": len(samples),
        "n_samples_with_sidecars": len(samples) - len(missing_samples),
        "n_samples_missing_sidecars": len(missing_samples),
        "samples_missing_sidecars": missing_samples,
        "raw_matched_candidate_total": raw_total,
        "kept_in_post": kept_total,
        "removed_in_post": removed_total,
        "stage_trace_available": stage_trace_available,
        "missing_post_links_total": missing_post_links_total,
        "used_sha_links_total": used_sha_links_total,
        "used_identity_links_total": used_identity_links_total,
        "by_taxonomy_label": by_taxonomy_label,
        "by_label_reason": by_label_reason,
        "by_post_status": by_post_status,
    }
    return records, summary


def collect_unmatched_candidate_dispositions(
    report: dict[str, Any],
    samples_root: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = list(report.get("per_sample", []))
    if samples_root is None:
        summary = {
            "n_samples_in_report": len(samples),
            "n_samples_with_sidecars": 0,
            "n_samples_missing_sidecars": len(samples),
            "samples_missing_sidecars": [sample["base_key"] for sample in samples],
            "raw_unmatched_candidate_total": 0,
            "kept_in_post": 0,
            "removed_in_post": 0,
            "stage_trace_available": False,
            "missing_post_links_total": 0,
            "used_sha_links_total": 0,
            "used_identity_links_total": 0,
            "by_taxonomy_label": {},
            "by_label_reason": {},
            "by_post_status": {},
        }
        return [], summary

    records: list[dict[str, Any]] = []
    missing_samples: list[str] = []
    stage_trace_available = False
    missing_post_links_total = 0
    used_sha_links_total = 0
    used_identity_links_total = 0
    samples_root = Path(samples_root)

    for sample in samples:
        payload = _collect_unmatched_candidate_dispositions_from_sample(
            sample,
            samples_root / sample["base_key"],
        )
        if payload is None:
            missing_samples.append(sample["base_key"])
            continue
        records.extend(payload["records"])
        stage_trace_available = stage_trace_available or bool(payload["stage_trace_available"])
        missing_post_links_total += int(payload["missing_post_links"])
        used_sha_links_total += int(payload["used_sha_links"])
        used_identity_links_total += int(payload["used_identity_links"])

    by_taxonomy_label = {
        key: _summarize_post_disposition_group(group_rows)
        for key, group_rows in sorted(
            _group_rows(records, "taxonomy_label").items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }
    by_label_reason = _build_reason_disposition_summary(
        records,
        reason_order=UNMATCHED_REASON_ORDER,
    )
    by_post_status = _build_post_status_summary(
        records,
        numeric_fields=UNMATCHED_DISPOSITION_NUMERIC_FIELDS,
    )

    raw_total = len(records)
    kept_total = sum(1 for row in records if bool(row.get("kept_in_post")))
    removed_total = raw_total - kept_total
    summary = {
        "n_samples_in_report": len(samples),
        "n_samples_with_sidecars": len(samples) - len(missing_samples),
        "n_samples_missing_sidecars": len(missing_samples),
        "samples_missing_sidecars": missing_samples,
        "raw_unmatched_candidate_total": raw_total,
        "kept_in_post": kept_total,
        "removed_in_post": removed_total,
        "stage_trace_available": stage_trace_available,
        "missing_post_links_total": missing_post_links_total,
        "used_sha_links_total": used_sha_links_total,
        "used_identity_links_total": used_identity_links_total,
        "by_taxonomy_label": by_taxonomy_label,
        "by_label_reason": by_label_reason,
        "by_post_status": by_post_status,
    }
    return records, summary


def _collect_removed_matched_candidates_from_sample(
    sample: dict[str, Any],
    sample_dir: Path,
) -> dict[str, Any] | None:
    payload = _collect_matched_candidate_dispositions_from_sample(sample, sample_dir)
    if payload is None:
        return None

    removed_records = [
        {
            key: value
            for key, value in row.items()
            if key not in {"kept_in_post", "post_status"}
        }
        for row in payload["records"]
        if not bool(row.get("kept_in_post"))
    ]
    removed_gt_ids = {
        int(row["matched_gt_id"])
        for row in removed_records
        if row.get("matched_gt_id") is not None
    }
    return {
        "records": removed_records,
        "missing_post_links": payload["missing_post_links"],
        "used_sha_links": payload["used_sha_links"],
        "used_identity_links": payload["used_identity_links"],
        "unique_removed_gt_ids": removed_gt_ids,
    }


def collect_removed_matched_candidates(
    report: dict[str, Any],
    samples_root: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    samples = list(report.get("per_sample", []))
    if samples_root is None:
        summary = {
            "n_samples_in_report": len(samples),
            "n_samples_with_sidecars": 0,
            "n_samples_missing_sidecars": len(samples),
            "samples_missing_sidecars": [sample["base_key"] for sample in samples],
            "n_samples_with_removed_matched_candidates": 0,
            "n_removed_matched_candidates": 0,
            "unique_removed_matched_gt_ids": 0,
            "missing_post_links_total": 0,
            "used_sha_links_total": 0,
            "used_identity_links_total": 0,
            "removed_by_taxonomy_label": {},
            "removed_by_label_reason": {},
            "overall_numeric_stats": {
                field: _numeric_stats([], field) for field in REMOVED_MATCHED_NUMERIC_FIELDS
            },
            "by_taxonomy_label": {},
            "by_label_reason": {},
            "per_sample": [],
        }
        return [], summary

    records: list[dict[str, Any]] = []
    missing_samples: list[str] = []
    per_sample_summary: list[dict[str, Any]] = []
    unique_removed_gt_ids: set[int] = set()
    missing_post_links_total = 0
    used_sha_links_total = 0
    used_identity_links_total = 0
    samples_root = Path(samples_root)

    for sample in samples:
        payload = _collect_removed_matched_candidates_from_sample(
            sample,
            samples_root / sample["base_key"],
        )
        if payload is None:
            missing_samples.append(sample["base_key"])
            continue

        sample_records = payload["records"]
        records.extend(sample_records)
        unique_removed_gt_ids.update(payload["unique_removed_gt_ids"])
        missing_post_links_total += int(payload["missing_post_links"])
        used_sha_links_total += int(payload["used_sha_links"])
        used_identity_links_total += int(payload["used_identity_links"])
        if sample_records:
            per_sample_summary.append(
                {
                    "base_key": sample["base_key"],
                    "galaxy_id": sample["galaxy_id"],
                    "view": sample["view"],
                    "benchmark_mode": sample["benchmark_mode"],
                    "n_removed_matched_candidates": len(sample_records),
                    "removed_taxonomy_labels": _count_by_key(sample_records, "taxonomy_label"),
                    "removed_label_reasons": _count_by_key(sample_records, "label_reason"),
                    "unique_removed_matched_gt_ids": len(
                        {
                            int(record["matched_gt_id"])
                            for record in sample_records
                            if record.get("matched_gt_id") is not None
                        }
                    ),
                }
            )

    total_removed = len(records)
    by_taxonomy_label = {
        key: _summarize_removed_group(group_rows, total_removed)
        for key, group_rows in sorted(
            _group_rows(records, "taxonomy_label").items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }
    by_label_reason = {
        key: _summarize_removed_group(group_rows, total_removed)
        for key, group_rows in sorted(
            _group_rows(records, "label_reason").items(),
            key=lambda item: (-len(item[1]), item[0]),
        )
    }
    per_sample_summary.sort(key=lambda row: (-row["n_removed_matched_candidates"], row["base_key"]))

    summary = {
        "n_samples_in_report": len(samples),
        "n_samples_with_sidecars": len(samples) - len(missing_samples),
        "n_samples_missing_sidecars": len(missing_samples),
        "samples_missing_sidecars": missing_samples,
        "n_samples_with_removed_matched_candidates": len(per_sample_summary),
        "n_removed_matched_candidates": total_removed,
        "unique_removed_matched_gt_ids": len(unique_removed_gt_ids),
        "missing_post_links_total": missing_post_links_total,
        "used_sha_links_total": used_sha_links_total,
        "used_identity_links_total": used_identity_links_total,
        "removed_by_taxonomy_label": _count_by_key(records, "taxonomy_label"),
        "removed_by_label_reason": _count_by_key(records, "label_reason"),
        "overall_numeric_stats": {
            field: _numeric_stats(records, field) for field in REMOVED_MATCHED_NUMERIC_FIELDS
        },
        "by_taxonomy_label": by_taxonomy_label,
        "by_label_reason": by_label_reason,
        "per_sample": per_sample_summary,
    }
    return records, summary


def write_removed_matched_candidates_csv(
    rows: list[dict[str, Any]],
    path: str,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REMOVED_MATCHED_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in REMOVED_MATCHED_COLUMNS})
    logger.info("Wrote %s (%d rows)", path, len(rows))


def write_unmatched_candidate_post_disposition_csv(
    rows: list[dict[str, Any]],
    path: str,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNMATCHED_POST_DISPOSITION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in UNMATCHED_POST_DISPOSITION_COLUMNS})
    logger.info("Wrote %s (%d rows)", path, len(rows))


def build_summary(rows: list[dict[str, Any]], report: dict[str, Any]) -> dict[str, Any]:
    thresholds = None
    if report.get("per_sample"):
        diag_summary = _get_diag_summary(report["per_sample"][0])
        thresholds = None if diag_summary is None else diag_summary.get("thresholds_used")

    gt_total_tax = _sum_int(rows, "num_gt_satellites_roi")
    taxonomy_micro = {
        "raw": {
            "precision": _safe_div(_sum_int(rows, "matched_candidates_raw"), _sum_int(rows, "num_pred_raw_roi")),
            "recall": _safe_div(_sum_int(rows, "unique_gt_covered_raw"), gt_total_tax),
            "unique_gt_covered": _sum_int(rows, "unique_gt_covered_raw"),
            "num_gt": gt_total_tax,
        },
        "post_pred_only": {
            "precision": _safe_div(_sum_int(rows, "matched_candidates_post"), _sum_int(rows, "num_pred_post_roi")),
            "recall": _safe_div(_sum_int(rows, "unique_gt_covered_post"), gt_total_tax),
            "unique_gt_covered": _sum_int(rows, "unique_gt_covered_post"),
            "num_gt": gt_total_tax,
        },
    }

    raw_prec_vals, post_prec_vals, prec_breakdown = _paired_taxonomy_precision(rows)
    raw_rec_vals, post_rec_vals, rec_breakdown = _paired_taxonomy_recall(rows)
    paired_tax_precision = int(raw_prec_vals.size)
    paired_tax_recall = int(raw_rec_vals.size)

    def _pair_means(raw_vals: np.ndarray, post_vals: np.ndarray) -> dict[str, float | None]:
        if raw_vals.size == 0:
            return {"mean_raw": None, "mean_post": None, "delta_mean": None}
        mean_raw = float(np.mean(raw_vals))
        mean_post = float(np.mean(post_vals))
        return {
            "mean_raw": mean_raw,
            "mean_post": mean_post,
            "delta_mean": mean_post - mean_raw,
        }

    paired_breakdown = {
        "tax_precision": {
            **prec_breakdown,
            **_pair_means(raw_prec_vals, post_prec_vals),
        },
        "tax_recall": {
            **rec_breakdown,
            **_pair_means(raw_rec_vals, post_rec_vals),
        },
    }

    return {
        "n_samples": len(rows),
        "global_candidate_totals": {
            "matched_candidates_raw": _sum_int(rows, "matched_candidates_raw"),
            "unmatched_candidates_raw": _sum_int(rows, "unmatched_candidates_raw"),
            "matched_candidates_post": _sum_int(rows, "matched_candidates_post"),
            "unmatched_candidates_post": _sum_int(rows, "unmatched_candidates_post"),
            "matched_candidates_removed": _sum_int(rows, "matched_candidates_removed"),
            "unmatched_candidates_removed": _sum_int(rows, "unmatched_candidates_removed"),
            "num_pred_raw_roi": _sum_int(rows, "num_pred_raw_roi"),
            "num_pred_post_roi": _sum_int(rows, "num_pred_post_roi"),
            "unique_gt_covered_raw": _sum_int(rows, "unique_gt_covered_raw"),
            "unique_gt_covered_post": _sum_int(rows, "unique_gt_covered_post"),
            "unique_gt_lost_post": _sum_int(rows, "unique_gt_lost_post"),
        },
        "global_taxonomy_micro": taxonomy_micro,
        "candidate_rate_stats": {
            "matched_candidate_loss_rate": _rate_stats(rows, "matched_candidate_loss_rate"),
            "unmatched_candidate_removal_rate": _rate_stats(rows, "unmatched_candidate_removal_rate"),
            "residual_unmatched_candidate_fraction": _rate_stats(
                rows,
                "residual_unmatched_candidate_fraction",
            ),
            "gt_coverage_loss_rate": _rate_stats(rows, "gt_coverage_loss_rate"),
            "gt_coverage_retention": _rate_stats(rows, "gt_coverage_retention"),
        },
        "taxonomy_stats": {
            "tax_precision_raw": _rate_stats(rows, "tax_precision_raw"),
            "tax_recall_raw": _rate_stats(rows, "tax_recall_raw"),
            "tax_precision_post": _rate_stats(rows, "tax_precision_post"),
            "tax_recall_post": _rate_stats(rows, "tax_recall_post"),
            "delta_tax_precision": _rate_stats(rows, "delta_tax_precision"),
            "delta_tax_recall": _rate_stats(rows, "delta_tax_recall"),
        },
        "paired_breakdown": paired_breakdown,
        "plot_valid_counts": {
            PLOT_FILENAMES["tradeoff_scatter"]: int(
                _finite_pair(
                    rows,
                    "unmatched_candidate_removal_rate",
                    "gt_coverage_loss_rate",
                )[0].size
            ),
            PLOT_FILENAMES["rate_boxplot"]: len(rows),
            PLOT_FILENAMES["global_matrix"]: len(rows),
            PLOT_FILENAMES["precision_recall_raw_vs_post"]: int(min(paired_tax_precision, paired_tax_recall)),
            PLOT_FILENAMES["precision_recall_raw_vs_post_scatter"]: int(
                min(paired_tax_precision, paired_tax_recall)
            ),
            PLOT_FILENAMES["matched_loss_vs_num_gt"]: int(
                _finite_pair(rows, "num_gt_satellites_roi", "gt_coverage_loss_rate")[0].size
            ),
            PLOT_FILENAMES["delta_precision_recall_scatter"]: int(
                _finite_pair(rows, "delta_tax_precision", "delta_tax_recall")[0].size
            ),
        },
        "thresholds_used": thresholds,
    }


def _setup_mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 150,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
        }
    )
    return plt


def plot_tradeoff_scatter(rows: list[dict[str, Any]], outdir: str) -> None:
    plt = _setup_mpl()
    x, y = _finite_pair(
        rows,
        "unmatched_candidate_removal_rate",
        "gt_coverage_loss_rate",
    )
    if x.size == 0:
        logger.warning("tradeoff_scatter: no valid data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=30, alpha=0.6, edgecolors="k", linewidths=0.4, c="#4C72B0")
    ax.set_xlabel("Unmatched Candidate Removal Rate")
    ax.set_ylabel("GT Coverage Loss Rate")
    ax.set_title("ROI Tradeoff: False Candidate Removal vs GT Coverage Loss")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "tradeoff_scatter"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["tradeoff_scatter"])


def plot_rate_boxplot(rows: list[dict[str, Any]], outdir: str) -> None:
    plt = _setup_mpl()
    keys = [
        "gt_coverage_loss_rate",
        "unmatched_candidate_removal_rate",
        "residual_unmatched_candidate_fraction",
        "delta_tax_recall",
    ]
    labels = [
        "GT Cover\nLoss",
        "Unmatched\nCand Removal",
        "Residual\nUnmatched",
        "Delta Tax\nRecall",
    ]
    data = [_finite_vec(rows, key) for key in keys]
    if all(arr.size == 0 for arr in data):
        logger.warning("rate_boxplot: no valid data")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        list(data),
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanprops={"marker": "D", "markerfacecolor": "red", "markersize": 5},
    )
    for patch, color in zip(bp["boxes"], ["#55A868", "#4C72B0", "#C44E52", "#8172B2"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.5)
    ax.set_ylabel("Rate")
    ax.set_title("GT-Centric Tradeoff and Taxonomy Recall Shift")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "rate_boxplot"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["rate_boxplot"])


def plot_global_matrix(rows: list[dict[str, Any]], outdir: str) -> None:
    plt = _setup_mpl()
    import matplotlib.colors as mcolors

    gt_kept = _sum_int(rows, "unique_gt_covered_post")
    gt_lost = _sum_int(rows, "unique_gt_lost_post")
    unmatched_kept = _sum_int(rows, "unmatched_candidates_post")
    unmatched_removed = _sum_int(rows, "unmatched_candidates_removed")

    mat = np.asarray(
        [
            [gt_kept, gt_lost],
            [unmatched_kept, unmatched_removed],
        ],
        dtype=float,
    )
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = mat.max() * 1.1 if mat.max() > 0 else 1.0
    im = ax.imshow(mat, cmap=plt.cm.YlOrRd, norm=mcolors.Normalize(vmin=0, vmax=vmax), aspect="auto")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Kept", "Removed"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Unique GT", "Unmatched Cand"])
    for i in range(2):
        for j in range(2):
            color = "white" if mat[i, j] > mat.max() * 0.6 else "black"
            ax.text(
                j,
                i,
                f"{int(mat[i, j])}",
                ha="center",
                va="center",
                fontsize=16,
                fontweight="bold",
                color=color,
            )
    ax.set_title("Global GT Coverage / False Candidate Matrix")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "global_matrix"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["global_matrix"])


def plot_delta_precision_recall_scatter(rows: list[dict[str, Any]], outdir: str) -> None:
    plt = _setup_mpl()
    x, y = _finite_pair(rows, "delta_tax_precision", "delta_tax_recall")
    if x.size == 0:
        logger.warning("delta_precision_recall_scatter: no valid data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=30, alpha=0.65, edgecolors="k", linewidths=0.4, c="#DD8452")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    ax.axvline(0, color="grey", lw=0.5, ls="--")
    ax.set_xlabel("Delta Taxonomy Precision")
    ax.set_ylabel("Delta Taxonomy Recall")
    ax.set_title("Taxonomy Precision vs Recall Shift")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "delta_precision_recall_scatter"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["delta_precision_recall_scatter"])


def plot_precision_recall_raw_vs_post(rows: list[dict[str, Any]], outdir: str) -> None:
    """Plot taxonomy precision / recall as paired raw -> post points."""
    plt = _setup_mpl()

    pair_fns = [
        (_paired_taxonomy_precision, "Taxonomy Precision", "precision"),
        (_paired_taxonomy_recall, "Taxonomy Recall", "recall"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.6), sharey=True)
    raw_color = "#4C72B0"
    post_color = "#DD8452"

    any_valid = False
    for ax, (pair_fn, title, kind) in zip(axes, pair_fns):
        raw_vals, post_vals, breakdown = pair_fn(rows)
        if raw_vals.size == 0:
            ax.set_visible(False)
            continue
        any_valid = True
        for raw_val, post_val in zip(raw_vals, post_vals):
            ax.plot([0, 1], [raw_val, post_val], color="#BBBBBB", alpha=0.25, lw=0.8, zorder=1)
        ax.scatter(np.zeros(raw_vals.size), raw_vals, s=18, alpha=0.65, c=raw_color, zorder=3)
        ax.scatter(np.ones(post_vals.size), post_vals, s=18, alpha=0.65, c=post_color, zorder=3)
        mean_raw = float(np.mean(raw_vals))
        mean_post = float(np.mean(post_vals))
        ax.plot(
            [0, 1],
            [mean_raw, mean_post],
            color="black",
            lw=2.2,
            marker="D",
            ms=6,
            zorder=4,
        )
        ax.text(0, mean_raw + 0.03, f"{mean_raw:.3f}", ha="center", va="bottom", fontsize=9)
        ax.text(1, mean_post + 0.03, f"{mean_post:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Raw", "Post"])
        ax.set_ylim(-0.05, 1.05)
        ax.axhline(0.5, color="grey", lw=0.4, ls=":")
        ax.axhline(1.0, color="grey", lw=0.4, ls=":")
        ax.set_title(
            f"{title}\n{_paired_breakdown_subtitle(kind, breakdown)}",
            fontsize=9,
        )

    if not any_valid:
        logger.warning("precision_recall_raw_vs_post: no valid data")
        plt.close(fig)
        return

    axes[0].set_ylabel("Rate")
    fig.suptitle("Taxonomy Precision / Recall: Raw vs Post", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "precision_recall_raw_vs_post"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["precision_recall_raw_vs_post"])


def plot_precision_recall_raw_vs_post_scatter(
    rows: list[dict[str, Any]],
    outdir: str,
) -> None:
    """Plot raw-vs-post bubble scatter so repeated pairs are visible."""
    plt = _setup_mpl()

    pair_fns = [
        (_paired_taxonomy_precision, "Taxonomy Precision", "precision"),
        (_paired_taxonomy_recall, "Taxonomy Recall", "recall"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 5.8), sharex=True, sharey=True)
    point_color = "#55A868"

    any_valid = False
    for ax, (pair_fn, title, kind) in zip(axes, pair_fns):
        raw_vals, post_vals, breakdown = pair_fn(rows)
        if raw_vals.size == 0:
            ax.set_visible(False)
            continue

        any_valid = True
        pair_counts: dict[tuple[float, float], int] = {}
        for raw_val, post_val in zip(raw_vals, post_vals):
            key = (float(raw_val), float(post_val))
            pair_counts[key] = pair_counts.get(key, 0) + 1

        xs = np.asarray([pair[0] for pair in pair_counts], dtype=float)
        ys = np.asarray([pair[1] for pair in pair_counts], dtype=float)
        counts = np.asarray([pair_counts[pair] for pair in pair_counts], dtype=float)
        sizes = 40.0 + 35.0 * np.sqrt(counts)

        ax.scatter(
            xs,
            ys,
            s=sizes,
            alpha=0.7,
            c=point_color,
            edgecolors="k",
            linewidths=0.4,
            zorder=3,
        )
        for x, y, count in zip(xs, ys, counts):
            if count >= 4:
                ax.text(x, y, str(int(count)), ha="center", va="center", fontsize=8, zorder=4)

        ax.plot([0, 1], [0, 1], color="grey", lw=1.0, ls="--", zorder=1)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Raw")
        ax.set_title(
            f"{title}\n{_paired_breakdown_subtitle(kind, breakdown)}"
            f"\nunique pairs={len(pair_counts)}",
            fontsize=9,
        )

    if not any_valid:
        logger.warning("precision_recall_raw_vs_post_scatter: no valid data")
        plt.close(fig)
        return

    axes[0].set_ylabel("Post")
    fig.suptitle(
        "Taxonomy Precision / Recall: Raw vs Post (bubble size = repeated sample count)",
        fontsize=11,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "precision_recall_raw_vs_post_scatter"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["precision_recall_raw_vs_post_scatter"])


def plot_matched_reason_post_disposition_flow(
    matched_rows: list[dict[str, Any]],
    outdir: str,
) -> None:
    """Show how raw matched reasons split into kept vs removed post outcomes."""
    plt = _setup_mpl()
    import matplotlib.patches as mpatches

    rows = [row for row in matched_rows if row.get("label_reason") is not None]
    if not rows:
        logger.warning("matched_reason_post_disposition_flow: no matched rows")
        return

    reason_order = [
        reason
        for reason in MATCHED_REASON_ORDER
        if any(row.get("label_reason") == reason for row in rows)
    ]
    if not reason_order:
        logger.warning("matched_reason_post_disposition_flow: no supported reasons")
        return

    status_order = list(MATCHED_POST_STATUS_ORDER)
    total = len(rows)
    left_totals = {
        reason: sum(1 for row in rows if row.get("label_reason") == reason)
        for reason in reason_order
    }
    right_totals = {
        status: sum(1 for row in rows if row.get("post_status") == status)
        for status in status_order
    }
    flow_counts = {
        (reason, status): sum(
            1
            for row in rows
            if row.get("label_reason") == reason and row.get("post_status") == status
        )
        for reason in reason_order
        for status in status_order
    }

    def _layout_segments(order: list[str], totals_by_key: dict[str, int]) -> dict[str, tuple[float, float]]:
        padding = 0.05
        nonzero_order = [key for key in order if totals_by_key.get(key, 0) > 0]
        if not nonzero_order:
            return {}
        available = 1.0 - padding * (len(nonzero_order) - 1)
        current_top = 1.0
        layout: dict[str, tuple[float, float]] = {}
        for key in nonzero_order:
            height = available * (float(totals_by_key[key]) / float(total))
            y1 = current_top
            y0 = y1 - height
            layout[key] = (y0, y1)
            current_top = y0 - padding
        return layout

    left_layout = _layout_segments(reason_order, left_totals)
    right_layout = _layout_segments(status_order, right_totals)
    if not left_layout or not right_layout:
        logger.warning("matched_reason_post_disposition_flow: empty layout")
        return

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    left_x0, left_x1 = 0.08, 0.24
    right_x0, right_x1 = 0.76, 0.92
    reason_colors = {
        "one_to_one_complete_seed_ratio_ok": "#4C72B0",
        "pure_and_complete": "#55A868",
        "pure_but_core_only": "#DD8452",
    }
    status_colors = {
        "kept": "#D8EEDB",
        "removed": "#F5D6C6",
    }

    for status, (y0, y1) in right_layout.items():
        ax.add_patch(
            mpatches.Rectangle(
                (right_x0, y0),
                right_x1 - right_x0,
                y1 - y0,
                facecolor=status_colors[status],
                edgecolor="black",
                lw=0.8,
                zorder=2,
            )
        )
        ax.text(
            right_x0 + 0.01,
            0.5 * (y0 + y1),
            f"{MATCHED_POST_STATUS_LABELS[status]}\n n={right_totals[status]}",
            ha="left",
            va="center",
            fontsize=10,
            zorder=3,
        )

    for reason, (y0, y1) in left_layout.items():
        ax.add_patch(
            mpatches.Rectangle(
                (left_x0, y0),
                left_x1 - left_x0,
                y1 - y0,
                facecolor=reason_colors.get(reason, "#999999"),
                edgecolor="black",
                lw=0.8,
                alpha=0.9,
                zorder=2,
            )
        )
        ax.text(
            left_x0 - 0.01,
            0.5 * (y0 + y1),
            f"{MATCHED_REASON_LABELS.get(reason, reason)}\n n={left_totals[reason]}",
            ha="right",
            va="center",
            fontsize=10,
            zorder=3,
        )

    left_offsets = {reason: left_layout[reason][1] for reason in left_layout}
    right_offsets = {status: right_layout[status][1] for status in right_layout}
    for reason in reason_order:
        for status in status_order:
            count = flow_counts[(reason, status)]
            if count <= 0:
                continue
            flow_height = (left_layout[reason][1] - left_layout[reason][0]) * (
                float(count) / float(left_totals[reason])
            )
            ly1 = left_offsets[reason]
            ly0 = ly1 - flow_height
            left_offsets[reason] = ly0

            flow_height_right = (right_layout[status][1] - right_layout[status][0]) * (
                float(count) / float(right_totals[status])
            )
            ry1 = right_offsets[status]
            ry0 = ry1 - flow_height_right
            right_offsets[status] = ry0

            verts = [
                (left_x1, ly1),
                (right_x0, ry1),
                (right_x0, ry0),
                (left_x1, ly0),
            ]
            ax.add_patch(
                mpatches.Polygon(
                    verts,
                    closed=True,
                    facecolor=reason_colors.get(reason, "#999999"),
                    edgecolor="none",
                    alpha=0.45,
                    zorder=1,
                )
            )
            if count >= 8:
                ax.text(
                    0.5 * (left_x1 + right_x0),
                    0.25 * (ly0 + ly1 + ry0 + ry1),
                    str(count),
                    ha="center",
                    va="center",
                    fontsize=9,
                    zorder=4,
                )

    removed_total = right_totals.get("removed", 0)
    ax.set_title(
        "Raw Matched Candidate Reason -> Post Outcome\n"
        f"raw matched n={total}, removed={removed_total} ({removed_total / total:.1%})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "matched_reason_post_disposition_flow"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["matched_reason_post_disposition_flow"])


def plot_matched_purity_completeness_post_disposition(
    matched_rows: list[dict[str, Any]],
    outdir: str,
) -> None:
    """Plot raw matched candidates in purity-completeness space, colored by post outcome."""
    plt = _setup_mpl()

    rows = [
        row
        for row in matched_rows
        if _is_finite_number(row.get("purity")) and _is_finite_number(row.get("completeness"))
    ]
    if not rows:
        logger.warning("matched_purity_completeness_post_disposition: no valid rows")
        return

    kept_rows = [row for row in rows if bool(row.get("kept_in_post"))]
    removed_rows = [row for row in rows if not bool(row.get("kept_in_post"))]

    fig, ax = plt.subplots(figsize=(7, 6))
    if kept_rows:
        ax.scatter(
            [float(row["purity"]) for row in kept_rows],
            [float(row["completeness"]) for row in kept_rows],
            s=28,
            alpha=0.65,
            c="#55A868",
            edgecolors="k",
            linewidths=0.3,
            label=f"Kept in post (n={len(kept_rows)})",
        )
    if removed_rows:
        ax.scatter(
            [float(row["purity"]) for row in removed_rows],
            [float(row["completeness"]) for row in removed_rows],
            s=34,
            alpha=0.75,
            c="#C44E52",
            edgecolors="k",
            linewidths=0.35,
            label=f"Removed in post (n={len(removed_rows)})",
        )

    ax.axvline(0.5, color="grey", lw=0.8, ls="--")
    ax.axhline(0.5, color="grey", lw=0.8, ls="--")
    ax.axhline(0.95, color="grey", lw=0.8, ls=":")
    ax.text(0.505, 0.03, "purity=0.5", fontsize=8, color="grey")
    ax.text(0.02, 0.505, "completeness=0.5", fontsize=8, color="grey")
    ax.text(0.02, 0.955, "completeness=0.95", fontsize=8, color="grey")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Raw Candidate Purity")
    ax.set_ylabel("Raw Candidate Completeness")
    ax.set_title("Raw Matched Candidates: Purity vs Completeness by Post Outcome")
    ax.legend(loc="lower left", frameon=True)
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "matched_purity_completeness_post_disposition"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["matched_purity_completeness_post_disposition"])


def plot_matched_reason_removal_rate(
    matched_summary: dict[str, Any],
    outdir: str,
) -> None:
    """Plot post-removal rates for raw matched groups."""
    plt = _setup_mpl()

    reason_items = [
        (reason, matched_summary["by_label_reason"][reason])
        for reason in MATCHED_REASON_ORDER
        if reason in matched_summary.get("by_label_reason", {})
    ]
    taxonomy_items = [
        (label, matched_summary["by_taxonomy_label"][label])
        for label in _MATCHED_LABELS
        if label in matched_summary.get("by_taxonomy_label", {})
    ]
    if not reason_items and not taxonomy_items:
        logger.warning("matched_reason_removal_rate: no summary data")
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), sharey=True)
    panels = [
        (
            axes[0],
            reason_items,
            [MATCHED_REASON_LABELS.get(key, key) for key, _ in reason_items],
            "By Raw Label Reason",
        ),
        (
            axes[1],
            taxonomy_items,
            [key.replace("_", "\n") for key, _ in taxonomy_items],
            "By Raw Taxonomy Label",
        ),
    ]
    colors = ["#4C72B0", "#55A868", "#DD8452", "#8172B2"]
    for ax, items, labels, title in panels:
        if not items:
            ax.set_visible(False)
            continue
        rates = [float(item["removal_rate"]) for _, item in items]
        xs = np.arange(len(items), dtype=float)
        ax.bar(xs, rates, color=colors[: len(items)], alpha=0.85, edgecolor="k", linewidth=0.4)
        for x, (_, item) in zip(xs, items):
            ax.text(
                x,
                min(0.98, float(item["removal_rate"]) + 0.03),
                f"{item['removed']}/{item['raw_total']}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.set_ylim(0.0, 1.05)
        ax.axhline(0.0, color="grey", lw=0.5)
        ax.set_title(title)
        ax.set_ylabel("Post Removal Rate")

    fig.suptitle("How Often Raw Matched Candidates Are Removed by Post", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "matched_reason_removal_rate"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["matched_reason_removal_rate"])


def plot_matched_loss_vs_num_gt(rows: list[dict[str, Any]], outdir: str) -> None:
    plt = _setup_mpl()
    x, y = _finite_pair(rows, "num_gt_satellites_roi", "gt_coverage_loss_rate")
    if x.size == 0:
        logger.warning("matched_loss_vs_num_gt: no valid data")
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(x, y, s=30, alpha=0.6, edgecolors="k", linewidths=0.4, c="#8172B2")
    ax.set_xlabel("ROI GT Satellite Count")
    ax.set_ylabel("GT Coverage Loss Rate")
    ax.set_title("GT Coverage Loss Rate vs ROI GT Count")
    ax.axhline(0, color="grey", lw=0.5, ls="--")
    fig.tight_layout()
    fig.savefig(_plot_path(outdir, "matched_loss_vs_num_gt"))
    plt.close(fig)
    logger.info("Wrote %s", PLOT_FILENAMES["matched_loss_vs_num_gt"])


# =========================================================================== #
#  Stage attribution pipeline (from trace sidecars)
# =========================================================================== #


def _load_stage_trace(sample_dir: Path) -> dict[str, Any] | None:
    """Load the stage trace sidecar; return None if missing."""
    path = sample_dir / STAGE_TRACE_SIDECAR_NAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _join_trace_to_matched(
    matched_records: list[dict[str, Any]],
    trace_records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join stage trace onto matched candidate records by (raw_index, candidate_id).

    Candidates in matched_records that cannot be found in trace_records are
    skipped with a warning.
    """
    trace_by_identity: dict[tuple[int, str], dict[str, Any]] = {}
    for rec in trace_records:
        if rec.get("raw_index") is not None and rec.get("candidate_id") is not None:
            key = (int(rec["raw_index"]), str(rec["candidate_id"]))
            trace_by_identity[key] = rec

    joined: list[dict[str, Any]] = []
    for row in matched_records:
        identity = (int(row["raw_index"]), str(row["candidate_id"]))
        trace = trace_by_identity.get(identity)
        if trace is None:
            continue
        # Audit sha1 match.
        if (
            row.get("candidate_rle_sha1") is not None
            and trace.get("candidate_rle_sha1") is not None
            and str(row["candidate_rle_sha1"]) != str(trace["candidate_rle_sha1"])
        ):
            _warn_soft_invariant(
                str(row.get("base_key", "?")),
                f"sha1 mismatch for raw_index={identity[0]}: "
                f"diag={row['candidate_rle_sha1']} trace={trace['candidate_rle_sha1']}",
            )

        stage_map: dict[str, dict[str, str | None]] = {}
        for sr in trace.get("stage_results", []):
            stage_map[sr["stage"]] = sr

        enriched = dict(row)
        enriched["first_drop_stage"] = trace.get("first_drop_stage")
        enriched["first_drop_reason"] = trace.get("first_drop_reason")
        for stage in STAGE_ORDER:
            sr = stage_map.get(stage, {})
            enriched[f"{stage}_outcome"] = sr.get("outcome")
            enriched[f"{stage}_reason"] = sr.get("reason")
        joined.append(enriched)

    return joined


def _build_stage_entered_dropped(
    records: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """For each stage compute entered/dropped/survived/conditional_drop_rate."""
    result: dict[str, dict[str, Any]] = {}
    for stage in STAGE_ORDER:
        outcome_key = f"{stage}_outcome"
        entered = sum(
            1 for r in records
            if r.get(outcome_key) in ("pass", "rescue", "drop")
        )
        dropped = sum(
            1 for r in records
            if r.get(outcome_key) == "drop"
        )
        survived = entered - dropped
        result[stage] = {
            "entered": entered,
            "dropped": dropped,
            "survived_stage": survived,
            "conditional_drop_rate": (
                float(dropped) / float(entered) if entered > 0 else None
            ),
        }
    return result


def _build_stage_attribution_summary_group(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build one summary group (all_raw / matched / matched_removed)."""
    return {
        "total_candidates": len(records),
        "by_first_drop_stage": _count_by_key(records, "first_drop_stage"),
        "by_stage_reason": {
            stage: _count_by_key(records, f"{stage}_reason")
            for stage in STAGE_ORDER
        },
        "by_stage_entered_dropped": _build_stage_entered_dropped(records),
        "by_label_reason_and_stage": {
            reason: _count_by_key(group_rows, "first_drop_stage")
            for reason, group_rows in sorted(
                _group_rows(records, "label_reason").items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        },
        "by_taxonomy_label_and_stage": {
            label: _count_by_key(group_rows, "first_drop_stage")
            for label, group_rows in sorted(
                _group_rows(records, "taxonomy_label").items(),
                key=lambda item: (-len(item[1]), item[0]),
            )
        },
    }


def collect_stage_attribution(
    report: dict[str, Any],
    samples_root: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Collect stage attribution by joining matched candidate dispositions
    with per-sample stage trace sidecars.

    Returns (attribution_rows, summary) or ([], None) when no trace data.
    """
    if samples_root is None:
        return [], None

    samples = list(report.get("per_sample", []))
    samples_root = Path(samples_root)
    all_attribution_rows: list[dict[str, Any]] = []
    all_raw_trace_records: list[dict[str, Any]] = []
    n_with_trace = 0

    for sample in samples:
        sample_dir = samples_root / sample["base_key"]
        trace_doc = _load_stage_trace(sample_dir)
        if trace_doc is None:
            continue
        n_with_trace += 1
        trace_records = trace_doc.get("records", [])

        # Enrich all raw trace records with sample identity + placeholder
        # taxonomy fields for the all_raw_satellites group.
        for rec in trace_records:
            enriched = {
                "base_key": sample["base_key"],
                "galaxy_id": sample.get("galaxy_id"),
                "view": sample.get("view"),
                "benchmark_mode": sample.get("benchmark_mode"),
                "raw_index": rec.get("raw_index"),
                "candidate_id": rec.get("candidate_id"),
                "candidate_rle_sha1": rec.get("candidate_rle_sha1"),
                "taxonomy_label": None,
                "label_reason": None,
                "kept_in_post": rec.get("final_status") == "kept",
                "first_drop_stage": rec.get("first_drop_stage"),
                "first_drop_reason": rec.get("first_drop_reason"),
            }
            stage_map = {sr["stage"]: sr for sr in rec.get("stage_results", [])}
            for stage in STAGE_ORDER:
                sr = stage_map.get(stage, {})
                enriched[f"{stage}_outcome"] = sr.get("outcome")
                enriched[f"{stage}_reason"] = sr.get("reason")
            all_raw_trace_records.append(enriched)

        # Get matched candidate dispositions for this sample.
        payload = _collect_matched_candidate_dispositions_from_sample(
            sample, sample_dir,
        )
        if payload is None:
            continue

        joined = _join_trace_to_matched(payload["records"], trace_records)
        all_attribution_rows.extend(joined)

    if n_with_trace == 0:
        logger.warning(
            "collect_stage_attribution: no stage trace sidecars found; "
            "skipping stage attribution"
        )
        return [], None

    # Build summary with 3 groups.
    matched_removed = [r for r in all_attribution_rows if not bool(r.get("kept_in_post"))]
    summary: dict[str, Any] = {
        "n_samples_with_trace": n_with_trace,
        "n_samples_total": len(samples),
        "all_raw_satellites": _build_stage_attribution_summary_group(all_raw_trace_records),
        "raw_matched_satellites": _build_stage_attribution_summary_group(all_attribution_rows),
        "raw_matched_removed": _build_stage_attribution_summary_group(matched_removed),
    }
    return all_attribution_rows, summary


def collect_unmatched_stage_attribution(
    report: dict[str, Any],
    samples_root: str | Path | None,
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    """Collect stage attribution for unmatched raw candidates (false positives).

    Returns (attribution_rows, summary) or ([], None) when no trace data.
    """
    if samples_root is None:
        return [], None

    samples = list(report.get("per_sample", []))
    samples_root = Path(samples_root)
    all_attribution_rows: list[dict[str, Any]] = []
    all_raw_trace_records: list[dict[str, Any]] = []
    n_with_trace = 0

    for sample in samples:
        sample_dir = samples_root / sample["base_key"]
        trace_doc = _load_stage_trace(sample_dir)
        if trace_doc is None:
            continue
        n_with_trace += 1
        trace_records = trace_doc.get("records", [])

        for rec in trace_records:
            enriched = {
                "base_key": sample["base_key"],
                "galaxy_id": sample.get("galaxy_id"),
                "view": sample.get("view"),
                "benchmark_mode": sample.get("benchmark_mode"),
                "raw_index": rec.get("raw_index"),
                "candidate_id": rec.get("candidate_id"),
                "candidate_rle_sha1": rec.get("candidate_rle_sha1"),
                "taxonomy_label": None,
                "label_reason": None,
                "kept_in_post": rec.get("final_status") == "kept",
                "first_drop_stage": rec.get("first_drop_stage"),
                "first_drop_reason": rec.get("first_drop_reason"),
            }
            stage_map = {sr["stage"]: sr for sr in rec.get("stage_results", [])}
            for stage in STAGE_ORDER:
                sr = stage_map.get(stage, {})
                enriched[f"{stage}_outcome"] = sr.get("outcome")
                enriched[f"{stage}_reason"] = sr.get("reason")
            all_raw_trace_records.append(enriched)

        payload = _collect_unmatched_candidate_dispositions_from_sample(sample, sample_dir)
        if payload is None:
            continue

        joined = _join_trace_to_matched(payload["records"], trace_records)
        all_attribution_rows.extend(joined)

    if n_with_trace == 0:
        logger.warning(
            "collect_unmatched_stage_attribution: no stage trace sidecars found; "
            "skipping stage attribution"
        )
        return [], None

    unmatched_kept = [r for r in all_attribution_rows if bool(r.get("kept_in_post"))]
    unmatched_removed = [r for r in all_attribution_rows if not bool(r.get("kept_in_post"))]
    summary: dict[str, Any] = {
        "n_samples_with_trace": n_with_trace,
        "n_samples_total": len(samples),
        "all_raw_satellites": _build_stage_attribution_summary_group(all_raw_trace_records),
        "raw_unmatched_satellites": _build_stage_attribution_summary_group(all_attribution_rows),
        "raw_unmatched_kept": _build_stage_attribution_summary_group(unmatched_kept),
        "raw_unmatched_removed": _build_stage_attribution_summary_group(unmatched_removed),
    }
    return all_attribution_rows, summary


def write_stage_attribution_csv(
    rows: list[dict[str, Any]],
    path: str,
) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=STAGE_ATTRIBUTION_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _fmt(row.get(k)) for k in STAGE_ATTRIBUTION_COLUMNS})
    logger.info("Wrote %s (%d rows)", path, len(rows))


# =========================================================================== #
#  Stage attribution plots
# =========================================================================== #


def _stage_trace_plot_path(outdir: str, key: str) -> str:
    return os.path.join(outdir, STAGE_TRACE_PLOT_FILENAMES[key])


def plot_first_drop_stage_profile(
    attribution_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    outdir: str,
) -> None:
    """Left: first_drop_stage counts for raw matched removed.
    Right: conditional_drop_rate per stage.
    """
    plt = _setup_mpl()

    removed = [r for r in attribution_rows if not bool(r.get("kept_in_post"))]
    if not removed:
        logger.warning("first_drop_stage_profile: no removed matched candidates")
        return

    by_fds = summary.get("raw_matched_removed", {}).get("by_first_drop_stage", {})
    stage_entered_dropped = summary.get("raw_matched_removed", {}).get(
        "by_stage_entered_dropped", {}
    )

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(11, 5))
    colors = {"score_gate": "#4C72B0", "prior_filter": "#55A868", "core_policy": "#DD8452"}

    # Left panel: first_drop_stage counts.
    stages_present = [s for s in STAGE_ORDER if by_fds.get(s, 0) > 0]
    if stages_present:
        y_pos = np.arange(len(stages_present), dtype=float)
        counts = [int(by_fds.get(s, 0)) for s in stages_present]
        bar_colors = [colors.get(s, "#999999") for s in stages_present]
        ax_left.barh(y_pos, counts, color=bar_colors, edgecolor="k", linewidth=0.4)
        ax_left.set_yticks(y_pos)
        ax_left.set_yticklabels([s.replace("_", "\n") for s in stages_present])
        for y, c in zip(y_pos, counts):
            ax_left.text(c + 0.3, y, str(c), va="center", fontsize=9)
        ax_left.set_xlabel("Count")
        ax_left.set_title("First Drop Stage\n(raw matched removed)")
    else:
        ax_left.set_visible(False)

    # Right panel: conditional_drop_rate.
    cdr_stages = list(STAGE_ORDER)
    cdr_vals = [
        float(stage_entered_dropped.get(s, {}).get("conditional_drop_rate", 0) or 0)
        for s in cdr_stages
    ]
    y_pos2 = np.arange(len(cdr_stages), dtype=float)
    bar_colors2 = [colors.get(s, "#999999") for s in cdr_stages]
    ax_right.barh(y_pos2, cdr_vals, color=bar_colors2, edgecolor="k", linewidth=0.4, alpha=0.85)
    ax_right.set_yticks(y_pos2)
    ax_right.set_yticklabels([s.replace("_", "\n") for s in cdr_stages])
    for y, v, s in zip(y_pos2, cdr_vals, cdr_stages):
        entered = stage_entered_dropped.get(s, {}).get("entered", 0)
        ax_right.text(
            min(v + 0.02, 0.95), y,
            f"{v:.1%} (n={entered})",
            va="center", fontsize=9,
        )
    ax_right.set_xlim(0, 1.05)
    ax_right.set_xlabel("Conditional Drop Rate")
    ax_right.set_title("Stage Drop Rate\n(entered → dropped)")

    fig.suptitle(
        f"Matched Candidate Stage Attribution (n_removed={len(removed)})",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(_stage_trace_plot_path(outdir, "first_drop_stage_profile"))
    plt.close(fig)
    logger.info("Wrote %s", STAGE_TRACE_PLOT_FILENAMES["first_drop_stage_profile"])


def plot_stage_reason_heatmap(
    attribution_rows: list[dict[str, Any]],
    outdir: str,
) -> None:
    """Heatmap: rows=stages, cols=drop reasons, values=removed matched count."""
    plt = _setup_mpl()
    import matplotlib.colors as mcolors

    removed = [r for r in attribution_rows if not bool(r.get("kept_in_post"))]
    if not removed:
        logger.warning("stage_reason_heatmap: no removed matched candidates")
        return

    # Collect all observed (stage, reason) pairs.
    stage_reason_counts: dict[tuple[str, str], int] = {}
    for r in removed:
        fds = r.get("first_drop_stage")
        fdr = r.get("first_drop_reason")
        if fds and fdr:
            key = (fds, fdr)
            stage_reason_counts[key] = stage_reason_counts.get(key, 0) + 1

    if not stage_reason_counts:
        logger.warning("stage_reason_heatmap: no stage/reason data")
        return

    # Build axes.
    all_reasons = sorted({r for (_, r) in stage_reason_counts})
    mat = np.zeros((len(STAGE_ORDER), len(all_reasons)), dtype=float)
    for (stage, reason), count in stage_reason_counts.items():
        if stage in STAGE_ORDER:
            si = STAGE_ORDER.index(stage)
            ri = all_reasons.index(reason)
            mat[si, ri] = count

    fig, ax = plt.subplots(figsize=(max(6, 1.5 * len(all_reasons)), 4))
    vmax = mat.max() * 1.1 if mat.max() > 0 else 1.0
    im = ax.imshow(
        mat, cmap=plt.cm.YlOrRd,
        norm=mcolors.Normalize(vmin=0, vmax=vmax),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(all_reasons)))
    ax.set_xticklabels([r.replace("_", "\n") for r in all_reasons], fontsize=8, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(STAGE_ORDER)))
    ax.set_yticklabels([s.replace("_", " ") for s in STAGE_ORDER])
    for si in range(len(STAGE_ORDER)):
        for ri in range(len(all_reasons)):
            val = int(mat[si, ri])
            if val > 0:
                color = "white" if val > mat.max() * 0.6 else "black"
                ax.text(ri, si, str(val), ha="center", va="center", fontsize=10, fontweight="bold", color=color)
    ax.set_title(f"Stage × Drop Reason Heatmap (raw matched removed, n={len(removed)})")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    fig.savefig(_stage_trace_plot_path(outdir, "stage_reason_heatmap"))
    plt.close(fig)
    logger.info("Wrote %s", STAGE_TRACE_PLOT_FILENAMES["stage_reason_heatmap"])


def plot_label_reason_first_drop_stage(
    attribution_rows: list[dict[str, Any]],
    outdir: str,
) -> None:
    """Stacked bar: x=label_reason, stacked colors=first_drop_stage."""
    plt = _setup_mpl()

    removed = [r for r in attribution_rows if not bool(r.get("kept_in_post"))]
    if not removed:
        logger.warning("label_reason_first_drop_stage: no removed matched candidates")
        return

    reason_groups = _group_rows(removed, "label_reason")
    reason_order = sorted(reason_groups.keys(), key=lambda k: (-len(reason_groups[k]), k))
    if not reason_order:
        return

    stage_colors = {"score_gate": "#4C72B0", "prior_filter": "#55A868", "core_policy": "#DD8452"}
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(reason_order)), 5))
    x_pos = np.arange(len(reason_order), dtype=float)
    bottom = np.zeros(len(reason_order), dtype=float)

    for stage in STAGE_ORDER:
        vals = np.asarray(
            [
                sum(1 for r in reason_groups.get(reason, []) if r.get("first_drop_stage") == stage)
                for reason in reason_order
            ],
            dtype=float,
        )
        if vals.sum() > 0:
            ax.bar(
                x_pos, vals, bottom=bottom,
                color=stage_colors.get(stage, "#999999"),
                edgecolor="k", linewidth=0.3,
                label=stage.replace("_", " "),
            )
            bottom += vals

    ax.set_xticks(x_pos)
    ax.set_xticklabels([r.replace("_", "\n") for r in reason_order], fontsize=8)
    ax.set_ylabel("Count")
    ax.set_title(f"First Drop Stage by Label Reason (raw matched removed, n={len(removed)})")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(_stage_trace_plot_path(outdir, "label_reason_first_drop_stage"))
    plt.close(fig)
    logger.info("Wrote %s", STAGE_TRACE_PLOT_FILENAMES["label_reason_first_drop_stage"])



def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--report", required=True, help="Path to checkpoint report.json")
    parser.add_argument("--outdir", default=None, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report_path = Path(args.report)
    if not report_path.exists():
        raise FileNotFoundError(f"Report not found: {report_path}")

    outdir = args.outdir or os.path.join(str(report_path.parent), "satellite_tradeoff")
    os.makedirs(outdir, exist_ok=True)
    _cleanup_legacy_plot_outputs(outdir)

    with open(report_path) as f:
        report = json.load(f)

    rows = extract_all(report, samples_root=report_path.parent)
    logger.info("Loaded report with %d samples", len(rows))
    matched_rows, matched_summary = collect_matched_candidate_dispositions(
        report,
        samples_root=report_path.parent,
    )
    unmatched_rows, unmatched_summary = collect_unmatched_candidate_dispositions(
        report,
        samples_root=report_path.parent,
    )

    write_csv(rows, os.path.join(outdir, "per_sample_tradeoff.csv"))

    summary = build_summary(rows, report)
    summary["plot_valid_counts"][PLOT_FILENAMES["matched_reason_post_disposition_flow"]] = len(matched_rows)
    summary["plot_valid_counts"][PLOT_FILENAMES["matched_purity_completeness_post_disposition"]] = len(matched_rows)
    summary["plot_valid_counts"][PLOT_FILENAMES["matched_reason_removal_rate"]] = len(matched_rows)
    with open(os.path.join(outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("Wrote summary.json")

    plot_tradeoff_scatter(rows, outdir)
    plot_rate_boxplot(rows, outdir)
    plot_global_matrix(rows, outdir)
    plot_delta_precision_recall_scatter(rows, outdir)
    plot_precision_recall_raw_vs_post(rows, outdir)
    plot_precision_recall_raw_vs_post_scatter(rows, outdir)
    plot_matched_loss_vs_num_gt(rows, outdir)
    plot_matched_reason_post_disposition_flow(matched_rows, outdir)
    plot_matched_purity_completeness_post_disposition(matched_rows, outdir)
    plot_matched_reason_removal_rate(matched_summary, outdir)

    with open(os.path.join(outdir, MATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY), "w") as f:
        json.dump(matched_summary, f, indent=2)
    logger.info("Wrote %s", MATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY)

    write_unmatched_candidate_post_disposition_csv(
        unmatched_rows,
        os.path.join(outdir, UNMATCHED_CANDIDATE_POST_DISPOSITION_CSV),
    )
    with open(os.path.join(outdir, UNMATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY), "w") as f:
        json.dump(unmatched_summary, f, indent=2)
    logger.info("Wrote %s", UNMATCHED_CANDIDATE_POST_DISPOSITION_SUMMARY)

    removed_rows, removed_summary = collect_removed_matched_candidates(
        report,
        samples_root=report_path.parent,
    )
    write_removed_matched_candidates_csv(
        removed_rows,
        os.path.join(outdir, REMOVED_MATCHED_CANDIDATES_CSV),
    )
    with open(os.path.join(outdir, REMOVED_MATCHED_CANDIDATES_SUMMARY), "w") as f:
        json.dump(removed_summary, f, indent=2)
    logger.info("Wrote %s", REMOVED_MATCHED_CANDIDATES_SUMMARY)

    # --- Stage attribution (from trace sidecars) ---
    attribution_rows, attribution_summary = collect_stage_attribution(
        report,
        samples_root=report_path.parent,
    )
    if attribution_summary is not None:
        write_stage_attribution_csv(
            attribution_rows,
            os.path.join(outdir, STAGE_ATTRIBUTION_CSV),
        )
        with open(os.path.join(outdir, STAGE_ATTRIBUTION_SUMMARY), "w") as f:
            json.dump(attribution_summary, f, indent=2)
        logger.info("Wrote %s", STAGE_ATTRIBUTION_SUMMARY)

        plot_first_drop_stage_profile(attribution_rows, attribution_summary, outdir)
        plot_stage_reason_heatmap(attribution_rows, outdir)
        plot_label_reason_first_drop_stage(attribution_rows, outdir)

        for key, fname in STAGE_TRACE_PLOT_FILENAMES.items():
            summary["plot_valid_counts"][fname] = len(attribution_rows)

    unmatched_attribution_rows, unmatched_attribution_summary = collect_unmatched_stage_attribution(
        report,
        samples_root=report_path.parent,
    )
    if unmatched_attribution_summary is not None:
        write_stage_attribution_csv(
            unmatched_attribution_rows,
            os.path.join(outdir, UNMATCHED_STAGE_ATTRIBUTION_CSV),
        )
        with open(os.path.join(outdir, UNMATCHED_STAGE_ATTRIBUTION_SUMMARY), "w") as f:
            json.dump(unmatched_attribution_summary, f, indent=2)
        logger.info("Wrote %s", UNMATCHED_STAGE_ATTRIBUTION_SUMMARY)

    logger.info("All outputs written to %s", outdir)


if __name__ == "__main__":
    main()
