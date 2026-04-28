"""Evaluation metrics and utilities."""

from .metrics import (
    calculate_iou,
    calculate_instance_iou,
    calculate_matched_metrics,
    calculate_pixel_metrics,
    calculate_optimal_instance_metrics,
    compute_one_to_one_flags,
    derive_purity_completeness,
    primary_gt_match,
)
from .satellite_diagnostics import (
    CandidateRow,
    DiagnosticCfg,
    SatelliteDiagnosticReport,
    TAXONOMY_LABELS,
    aggregate_diagnostics,
    build_candidate_table,
    classify,
    matched_unmatched_counts,
)

__all__ = [
    "calculate_iou",
    "calculate_instance_iou",
    "calculate_matched_metrics",
    "calculate_pixel_metrics",
    "calculate_optimal_instance_metrics",
    "compute_one_to_one_flags",
    "derive_purity_completeness",
    "primary_gt_match",
    "CandidateRow",
    "DiagnosticCfg",
    "SatelliteDiagnosticReport",
    "TAXONOMY_LABELS",
    "aggregate_diagnostics",
    "build_candidate_table",
    "classify",
    "matched_unmatched_counts",
]
