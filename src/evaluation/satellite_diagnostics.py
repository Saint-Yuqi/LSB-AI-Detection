"""
Satellite diagnostic taxonomy — Phase 1 (GT-driven, offline).

Builds a prediction-indexed table from raw SAM3 satellite candidates by
matching each prediction to a primary GT satellite via instance-map argmax
(NO IoU threshold) and labelling it from GT overlap, completeness, purity,
one-to-one ownership, and seed/GT area ratio:

    A  matched_gt_id is None -> reject_unmatched
    B  completeness >= complete_one_to_one_min_completeness AND
       is_one_to_one AND
       seed_gt_ratio <= complete_one_to_one_max_seed_ratio -> compact_complete
    C  purity < min_purity_for_match -> reject_low_purity
    D  purity ok AND completeness >= completeness_complete -> compact_complete
    E  purity ok AND completeness <  completeness_complete -> diffuse_core

``reject_host_background`` is reserved for Phase 2 (needs a host-support
loader that does not exist yet).

Public API:
    DiagnosticCfg            dataclass of thresholds.
    CandidateRow             TypedDict, one row per raw satellite prediction.
    SatelliteDiagnosticReport TypedDict, full output for one sample.
    build_candidate_table(...)
    aggregate_diagnostics(...)

Invariants:
    - Matching never uses an IoU threshold.
    - Rows cover EVERY raw satellite prediction. ROI is a secondary slice
      recorded via the legacy ``intersects_roi`` field — never a primary
      filter. ROI membership is mask-centroid-in-ROI, not any-pixel touch.
    - Each row is keyed by (raw_index, candidate_id), the same fields
      ``save_predictions_json`` writes to ``predictions_raw.json``. The
      caller must assign these BEFORE calling ``build_candidate_table``.
    - ``confidence_score`` in the row is SAM3's native mask ``score``
      renamed to avoid confusion with "prediction vs GT IoU".
    - All signal metrics run on the (H, W) float32 ``render_signal`` — the
      same intensity array SAM3 saw at inference. No magnitude-to-flux
      inversion.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Optional, TypedDict

import numpy as np

from src.analysis.mask_metrics import annulus_excess, radial_monotonicity
from src.evaluation.metrics import (
    compute_one_to_one_flags,
    derive_purity_completeness,
    primary_gt_match,
)

__all__ = [
    "DiagnosticCfg",
    "CandidateRow",
    "SatelliteDiagnosticReport",
    "TAXONOMY_LABELS",
    "MATCHED_LABELS",
    "UNMATCHED_LABELS",
    "TaxonomyEntry",
    "matched_unmatched_counts",
    "build_candidate_table",
    "classify",
    "classify_candidates",
    "aggregate_diagnostics",
]


TAXONOMY_LABELS: tuple[str, ...] = (
    "compact_complete",
    "diffuse_core",
    "reject_unmatched",
    "reject_low_purity",
)

#: Labels that count as a successful GT match (matched candidate).
MATCHED_LABELS: tuple[str, ...] = ("compact_complete", "diffuse_core")

#: Labels that count as a false/unmatched candidate.
UNMATCHED_LABELS: tuple[str, ...] = ("reject_unmatched", "reject_low_purity")


@dataclass(frozen=True)
class TaxonomyEntry:
    """Minimal per-candidate classification used by both the diagnostics
    sidecar and the official satellite eval block.

    One entry per predicted satellite. ``intersects_roi`` is always populated
    (``True`` when ``roi_bbox`` is None, matching the ROI-is-optional
    contract of the full frame).
    """

    raw_index: Optional[int]
    candidate_id: Optional[str]
    matched_gt_id: Optional[int]
    taxonomy_label: str
    label_reason: str
    intersects_roi: bool


@dataclass(frozen=True)
class DiagnosticCfg:
    """Thresholds and image-evidence parameters for the Phase-1 taxonomy.

    Defaults match the plan; all tunable from ``diagnostics.satellites`` in
    ``configs/eval_checkpoint.yaml``.
    """
    min_purity_for_match: float = 0.50
    completeness_complete: float = 0.50
    complete_one_to_one_min_completeness: float = 0.95
    complete_one_to_one_max_seed_ratio: float = 3.0
    annulus_r_in_frac: float = 1.2
    annulus_r_out_frac: float = 2.0
    radial_n_rings: int = 6


class CandidateRow(TypedDict, total=True):
    raw_index: int
    candidate_id: str
    seed_area: int
    confidence_score: float
    matched_gt_id: Optional[int]
    matched_gt_area: Optional[int]
    overlap_px: int
    purity: float
    completeness: Optional[float]
    seed_gt_ratio: Optional[float]
    is_one_to_one: bool
    host_background_frac: Optional[float]
    intersects_roi: bool
    annulus_excess: Optional[float]
    radial_monotonicity: Optional[float]
    taxonomy_label: str
    label_reason: str


class SatelliteDiagnosticReport(TypedDict, total=True):
    per_candidate: list[CandidateRow]
    counts_by_label: dict[str, int]
    counts_by_label_roi: Optional[dict[str, int]]
    counts_post_by_label: Optional[dict[str, int]]
    counts_post_by_label_roi: Optional[dict[str, int]]
    thresholds_used: dict[str, float]
    host_support_available: bool


def matched_unmatched_counts(
    raw_counts: Optional[dict[str, int]],
    post_counts: Optional[dict[str, int]] = None,
) -> dict[str, Optional[int]]:
    """Collapse taxonomy counts into matched/unmatched raw/post buckets.

    The caller decides whether the counts represent full-frame or ROI slices.
    """

    def _sum(counts: Optional[dict[str, int]], labels: tuple[str, str]) -> Optional[int]:
        if counts is None:
            return None
        return int(sum(counts.get(label, 0) for label in labels))

    matched_labels = ("compact_complete", "diffuse_core")
    unmatched_labels = ("reject_unmatched", "reject_low_purity")
    return {
        "matched_raw": _sum(raw_counts, matched_labels),
        "unmatched_raw": _sum(raw_counts, unmatched_labels),
        "matched_post": _sum(post_counts, matched_labels),
        "unmatched_post": _sum(post_counts, unmatched_labels),
    }


# =========================================================================== #
#  Classifier (A/B/C/D in the plan)
# =========================================================================== #


def classify(
    matched_gt_id: Optional[int],
    purity: float,
    completeness: Optional[float],
    seed_gt_ratio: Optional[float],
    is_one_to_one: bool,
    cfg: DiagnosticCfg,
) -> tuple[str, str]:
    """Return ``(taxonomy_label, label_reason)``. See module docstring."""
    if matched_gt_id is None:
        return "reject_unmatched", "no_gt_overlap"
    # Past this point we have a valid primary match, so completeness is not None.
    if completeness is None or seed_gt_ratio is None:
        raise AssertionError(
            "classify(): completeness/seed_gt_ratio is None but matched_gt_id is set; "
            "this indicates a bug in build_candidate_table."
        )
    if (
        completeness >= cfg.complete_one_to_one_min_completeness
        and is_one_to_one
        and seed_gt_ratio <= cfg.complete_one_to_one_max_seed_ratio
    ):
        return "compact_complete", "one_to_one_complete_seed_ratio_ok"
    if purity < cfg.min_purity_for_match:
        return "reject_low_purity", "mixed_coverage"
    if completeness >= cfg.completeness_complete:
        return "compact_complete", "pure_and_complete"
    return "diffuse_core", "pure_but_core_only"


# =========================================================================== #
#  Build per-sample table
# =========================================================================== #


def _mask_centroid_in_roi(
    seg: np.ndarray, roi_bbox: Optional[tuple[int, int, int, int]]
) -> bool:
    if roi_bbox is None:
        return True
    ys, xs = np.nonzero(seg)
    if len(xs) == 0:
        return False
    cy = float(ys.mean())
    cx = float(xs.mean())
    y0, x0, y1, x1 = roi_bbox
    return bool(y0 <= cy < y1 and x0 <= cx < x1)


def _empty_counts() -> dict[str, int]:
    return {label: 0 for label in TAXONOMY_LABELS}


class _ClassifiedCandidate(TypedDict):
    """Internal rich per-candidate classification result.

    Shared between :func:`classify_candidates` (public minimal projection) and
    :func:`_build_candidate_rows_and_counts` (feature-augmented sidecar rows)
    so matching / classification happens exactly once per candidate list.
    """

    raw_index: Optional[int]
    candidate_id: Optional[str]
    pred_bool: np.ndarray
    pred_area: int
    matched_gt_id: Optional[int]
    matched_gt_area: Optional[int]
    overlap_px: int
    purity: float
    completeness: Optional[float]
    seed_gt_ratio: Optional[float]
    is_one_to_one: bool
    intersects_roi: bool
    taxonomy_label: str
    label_reason: str


def _classify_candidates_detailed(
    sats: list[dict[str, Any]],
    gt_sat_map: np.ndarray,
    H: int,
    W: int,
    cfg: DiagnosticCfg,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    source_name: str = "sats",
) -> list[_ClassifiedCandidate]:
    """One matching + classification pass over ``sats``.

    This is the single source of truth for prediction-to-GT matching and
    taxonomy labelling. ``segmentation`` is the only required key on each
    mask dict; ``raw_index`` / ``candidate_id`` are copied through when
    present. ``score`` / ``area_clean`` are NOT required here —
    feature-column augmentation is done in
    :func:`_build_candidate_rows_and_counts`.
    """
    if gt_sat_map.shape != (H, W):
        raise ValueError(
            f"gt_sat_map shape {gt_sat_map.shape} != (H, W)=({H}, {W})"
        )

    pred_bins: list[np.ndarray] = []
    for i, m in enumerate(sats):
        if "segmentation" not in m:
            raise KeyError(f"{source_name}[{i}] missing 'segmentation'")
        seg = m["segmentation"]
        if seg.shape != (H, W):
            raise ValueError(
                f"{source_name}[{i}].segmentation shape {seg.shape} != (H, W)=({H}, {W})"
            )
        pred_bins.append(seg.astype(bool))

    one_to_one_flags = compute_one_to_one_flags(pred_bins, gt_sat_map)

    out: list[_ClassifiedCandidate] = []
    for i, m in enumerate(sats):
        pred_bool = pred_bins[i]

        match_info = primary_gt_match(pred_bool, gt_sat_map)
        overlap_px = match_info["overlap_px"]
        pred_area = match_info["pred_area"]
        matched_gt_id = match_info["matched_gt_id"]
        matched_gt_area = match_info["matched_gt_area"]

        pc = derive_purity_completeness(overlap_px, pred_area, matched_gt_area)
        purity = pc["purity"]
        completeness = pc["completeness"]
        seed_gt_ratio = pc["seed_gt_ratio"]

        label, reason = classify(
            matched_gt_id,
            purity,
            completeness,
            seed_gt_ratio,
            bool(one_to_one_flags[i]),
            cfg,
        )

        in_roi = _mask_centroid_in_roi(pred_bool, roi_bbox)

        raw_idx = m.get("raw_index")
        cand_id = m.get("candidate_id")
        out.append({
            "raw_index": int(raw_idx) if raw_idx is not None else None,
            "candidate_id": str(cand_id) if cand_id is not None else None,
            "pred_bool": pred_bool,
            "pred_area": int(pred_area),
            "matched_gt_id": matched_gt_id,
            "matched_gt_area": matched_gt_area,
            "overlap_px": int(overlap_px),
            "purity": float(purity),
            "completeness": (float(completeness) if completeness is not None else None),
            "seed_gt_ratio": (float(seed_gt_ratio) if seed_gt_ratio is not None else None),
            "is_one_to_one": bool(one_to_one_flags[i]),
            "intersects_roi": in_roi,
            "taxonomy_label": label,
            "label_reason": reason,
        })
    return out


def classify_candidates(
    sats: list[dict[str, Any]],
    gt_sat_map: np.ndarray,
    H: int,
    W: int,
    cfg: DiagnosticCfg,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
) -> list[TaxonomyEntry]:
    """Classify a satellite prediction list against the GT satellites map.

    Returns one :class:`TaxonomyEntry` per input mask. Requires only
    ``segmentation`` on each input dict; ``raw_index`` / ``candidate_id`` are
    forwarded when present (both ``None`` otherwise). No dependency on
    ``render_signal`` or feature-column inputs.

    This is the shared primitive consumed by both the official satellite
    eval block (``checkpoint_eval.compute_sample_report``) and the diagnostics
    sidecar (``build_candidate_table``).
    """
    detailed = _classify_candidates_detailed(
        sats, gt_sat_map, H, W, cfg, roi_bbox=roi_bbox, source_name="sats"
    )
    return [
        TaxonomyEntry(
            raw_index=d["raw_index"],
            candidate_id=d["candidate_id"],
            matched_gt_id=d["matched_gt_id"],
            taxonomy_label=d["taxonomy_label"],
            label_reason=d["label_reason"],
            intersects_roi=d["intersects_roi"],
        )
        for d in detailed
    ]


def _counts_from_detailed(
    detailed: list[_ClassifiedCandidate],
    *,
    roi: bool,
) -> dict[str, int]:
    counts = _empty_counts()
    for d in detailed:
        if roi and not d["intersects_roi"]:
            continue
        counts[d["taxonomy_label"]] += 1
    return counts


def _build_candidate_rows_and_counts(
    sats: list[dict[str, Any]],
    gt_sat_map: np.ndarray,
    render_signal: Optional[np.ndarray],
    H: int,
    W: int,
    cfg: DiagnosticCfg,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    source_name: str = "sats",
    include_rows: bool = True,
) -> tuple[list[CandidateRow], dict[str, int], Optional[dict[str, int]]]:
    """Build taxonomy rows/counts for one satellite candidate list.

    Delegates matching + classification to
    :func:`_classify_candidates_detailed` (single-pass) and only adds
    feature columns (``annulus_excess`` / ``radial_monotonicity``) and
    row-level bookkeeping on top.

    ``include_rows=False`` is used for post-filtered aggregate counts where
    per-candidate feature rows are not emitted. When ``include_rows=True``,
    ``render_signal`` must be a (H, W) float array; when False it may be
    ``None``.
    """
    detailed = _classify_candidates_detailed(
        sats, gt_sat_map, H, W, cfg, roi_bbox=roi_bbox, source_name=source_name,
    )

    rows: list[CandidateRow] = []
    if include_rows:
        if render_signal is None:
            raise ValueError(
                "render_signal is required when include_rows=True"
            )
        for i, (m, d) in enumerate(zip(sats, detailed)):
            if "raw_index" not in m or "candidate_id" not in m:
                raise KeyError(
                    f"{source_name}[{i}] missing 'raw_index' or 'candidate_id'; "
                    "call assign_stable_ids() on the full raw_masks list first."
                )
            if "score" not in m:
                raise KeyError(f"{source_name}[{i}] missing 'score'")

            pred_bool = d["pred_bool"]
            seed_area = int(m.get("area_clean", d["pred_area"]))

            ae = annulus_excess(
                render_signal,
                pred_bool.astype(np.uint8),
                r_in_frac=cfg.annulus_r_in_frac,
                r_out_frac=cfg.annulus_r_out_frac,
            )
            rm = radial_monotonicity(
                render_signal,
                pred_bool.astype(np.uint8),
                n_rings=cfg.radial_n_rings,
                r_out_frac=cfg.annulus_r_out_frac,
            )

            row: CandidateRow = {
                "raw_index": int(m["raw_index"]),
                "candidate_id": str(m["candidate_id"]),
                "seed_area": seed_area,
                "confidence_score": float(m["score"]),
                "matched_gt_id": d["matched_gt_id"],
                "matched_gt_area": d["matched_gt_area"],
                "overlap_px": d["overlap_px"],
                "purity": d["purity"],
                "completeness": d["completeness"],
                "seed_gt_ratio": d["seed_gt_ratio"],
                "is_one_to_one": d["is_one_to_one"],
                # Phase 2+: populated when a host_support mask is supplied.
                "host_background_frac": None,
                "intersects_roi": d["intersects_roi"],
                "annulus_excess": ae,
                "radial_monotonicity": rm,
                "taxonomy_label": d["taxonomy_label"],
                "label_reason": d["label_reason"],
            }
            rows.append(row)

    counts = _counts_from_detailed(detailed, roi=False)
    counts_roi: Optional[dict[str, int]] = (
        _counts_from_detailed(detailed, roi=True) if roi_bbox is not None else None
    )
    return rows, counts, counts_roi


def build_candidate_table(
    raw_sats: list[dict[str, Any]],
    gt_sat_map: np.ndarray,
    render_signal: np.ndarray,
    H: int,
    W: int,
    cfg: DiagnosticCfg,
    roi_bbox: Optional[tuple[int, int, int, int]] = None,
    host_support: Optional[np.ndarray] = None,
    post_sats: Optional[list[dict[str, Any]]] = None,
) -> SatelliteDiagnosticReport:
    """Build the per-sample satellite diagnostic table.

    Args:
        raw_sats:        list of raw satellite mask dicts. Each dict MUST
                         carry ``raw_index`` (int) and ``candidate_id`` (str)
                         — stamp these via ``assign_stable_ids`` before
                         calling this function. Other expected keys:
                         ``segmentation`` ((H, W) bool), ``score``
                         (float; SAM3 native confidence), and ``area_clean``
                         (int; produced by ``append_metrics_to_masks``).
        gt_sat_map:      (H, W) int — satellites-only GT instance map, with
                         positive IDs and 0 = background.
        render_signal:   (H, W) float32 — intensity the model saw.
        H, W:            frame size (checked for consistency).
        cfg:             ``DiagnosticCfg`` with thresholds.
        roi_bbox:        optional ``(y0, x0, y1, x1)`` for the legacy
                         ``intersects_roi`` column and the
                         ``counts_by_label_roi`` slice. Membership uses the
                         mask centroid, so edge-touching objects outside the
                         ROI are not counted as ROI objects. None leaves all
                         rows marked ``intersects_roi=True`` and
                         ``counts_by_label_roi=None``.
        host_support:    optional (H, W) bool host/galaxy-support mask.
                         Reserved for Phase 2; in Phase 1 ``host_background_frac``
                         stays None and ``host_support_available`` is False.
        post_sats:       optional post-filtered satellite predictions. When
                         provided, the same taxonomy is applied to them and
                         only aggregate ``counts_post_*`` are emitted.

    Returns:
        SatelliteDiagnosticReport — see TypedDict above.

    Raises:
        ValueError: on shape mismatches between render_signal, gt_sat_map,
                    or any prediction segmentation.
        KeyError:   if a mask is missing raw_index / candidate_id /
                    segmentation / score / area_clean.
    """
    if gt_sat_map.shape != (H, W):
        raise ValueError(
            f"gt_sat_map shape {gt_sat_map.shape} != (H, W)=({H}, {W})"
        )
    if render_signal.shape != (H, W):
        raise ValueError(
            f"render_signal shape {render_signal.shape} != (H, W)=({H}, {W})"
        )
    if render_signal.ndim != 2:
        raise ValueError(
            f"render_signal must be (H, W); got shape {render_signal.shape}"
        )
    if host_support is not None and host_support.shape != (H, W):
        raise ValueError(
            f"host_support shape {host_support.shape} != (H, W)=({H}, {W})"
        )

    per_candidate, counts, counts_roi = _build_candidate_rows_and_counts(
        raw_sats,
        gt_sat_map,
        render_signal,
        H,
        W,
        cfg,
        roi_bbox=roi_bbox,
        source_name="raw_sats",
    )
    if post_sats is not None:
        _post_rows, counts_post, counts_post_roi = _build_candidate_rows_and_counts(
            post_sats,
            gt_sat_map,
            render_signal,
            H,
            W,
            cfg,
            roi_bbox=roi_bbox,
            source_name="post_sats",
            include_rows=False,
        )
    else:
        counts_post = None
        counts_post_roi = None

    return {
        "per_candidate": per_candidate,
        "counts_by_label": counts,
        "counts_by_label_roi": counts_roi,
        "counts_post_by_label": counts_post,
        "counts_post_by_label_roi": counts_post_roi,
        "thresholds_used": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in asdict(cfg).items()
        },
        "host_support_available": host_support is not None,
    }


# =========================================================================== #
#  Aggregation across samples
# =========================================================================== #


_SUMMARY_FEATURE_KEYS: tuple[str, ...] = (
    "purity",
    "completeness",
    "seed_gt_ratio",
    "annulus_excess",
    "radial_monotonicity",
)


def _quantile_summary(values: list[float]) -> Optional[dict[str, float]]:
    if not values:
        return None
    arr = np.asarray(values, dtype=np.float64)
    return {
        "n": int(arr.size),
        "min": float(arr.min()),
        "p25": float(np.percentile(arr, 25)),
        "median": float(np.median(arr)),
        "p75": float(np.percentile(arr, 75)),
        "max": float(arr.max()),
    }


def aggregate_diagnostics(
    per_sample_rows: list[list[CandidateRow]],
    per_sample_post_counts: Optional[list[Optional[dict[str, int]]]] = None,
    per_sample_post_counts_roi: Optional[list[Optional[dict[str, int]]]] = None,
) -> dict[str, Any]:
    """Aggregate per-sample candidate rows into a global summary.

    Returns a dict with:
        n_samples: int
        n_candidates: int
        counts_by_label: dict[str, int]
        counts_by_label_roi: dict[str, int] | None  (None if no sample had ROI counts)
        counts_post_by_label: dict[str, int] | None
        counts_post_by_label_roi: dict[str, int] | None
        roi_matched_unmatched: matched/unmatched raw/post rollup from ROI counts
        feature_summary: dict[label, dict[feature, {n,min,p25,median,p75,max}]]
    """
    counts = _empty_counts()
    counts_roi = _empty_counts()
    counts_post = _empty_counts()
    counts_post_roi = _empty_counts()
    any_roi = False
    any_post = False
    any_post_roi = False

    # Gather values per (label, feature).
    buckets: dict[str, dict[str, list[float]]] = {
        label: {k: [] for k in _SUMMARY_FEATURE_KEYS} for label in TAXONOMY_LABELS
    }

    n_candidates = 0
    for rows in per_sample_rows:
        for r in rows:
            n_candidates += 1
            label = r["taxonomy_label"]
            if label not in counts:
                counts[label] = 0
            counts[label] += 1
            if r["intersects_roi"]:
                any_roi = True
                if label not in counts_roi:
                    counts_roi[label] = 0
                counts_roi[label] += 1
            for key in _SUMMARY_FEATURE_KEYS:
                v = r.get(key)
                if v is not None:
                    buckets.setdefault(label, {k: [] for k in _SUMMARY_FEATURE_KEYS})
                    buckets[label].setdefault(key, [])
                    buckets[label][key].append(float(v))

    if per_sample_post_counts:
        for sample_counts in per_sample_post_counts:
            if sample_counts is None:
                continue
            any_post = True
            for label, n in sample_counts.items():
                if label not in counts_post:
                    counts_post[label] = 0
                counts_post[label] += int(n)

    if per_sample_post_counts_roi:
        for sample_counts in per_sample_post_counts_roi:
            if sample_counts is None:
                continue
            any_post_roi = True
            for label, n in sample_counts.items():
                if label not in counts_post_roi:
                    counts_post_roi[label] = 0
                counts_post_roi[label] += int(n)

    feature_summary: dict[str, dict[str, Optional[dict[str, float]]]] = {}
    for label in TAXONOMY_LABELS:
        per_feature = {}
        for key in _SUMMARY_FEATURE_KEYS:
            per_feature[key] = _quantile_summary(buckets[label][key])
        feature_summary[label] = per_feature

    return {
        "n_samples": len(per_sample_rows),
        "n_candidates": n_candidates,
        "counts_by_label": counts,
        "counts_by_label_roi": counts_roi if any_roi else None,
        "counts_post_by_label": counts_post if any_post else None,
        "counts_post_by_label_roi": counts_post_roi if any_post_roi else None,
        "roi_matched_unmatched": matched_unmatched_counts(
            counts_roi if any_roi else None,
            counts_post_roi if any_post_roi else None,
        ),
        "feature_summary": feature_summary,
    }
