"""
DR1 auto-derived silver label generation.

Uses ``candidate_matcher`` (NOT ``calculate_matched_metrics``) for
pred-centric signals.  Outputs lightweight intermediate
``silver_labels_{family}.jsonl``, not the final business JSONL.

V1 constraints:
  - ``minor_fix`` and ``redraw`` are gold-only (never auto-generated).
  - Ambiguous candidates are dropped (abstain).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.review.candidate_matcher import (
    CandidateMatchResult,
    ImageExhaustivityResult,
)


# ---------------------------------------------------------------------------
#  Policy config (loaded from silver_policy_v1.yaml at runtime)
# ---------------------------------------------------------------------------

@dataclass
class SilverPolicyConfig:
    """Thresholds for silver derivation across verifier families."""
    accept_iou_thresh: float = 0.7
    reject_iou_thresh: float = 0.1
    min_area_ratio: float = 0.5
    max_area_ratio: float = 2.0
    boundary_f1_thresh: float = 0.3
    confidence_thresh: float = 0.5
    ambiguity_gap: float = 0.15
    ev_match_iou_thresh: float = 0.3
    ev_recall_thresh: float = 0.9
    ev_precision_thresh: float = 0.9
    ev_mixed_area_dominance_ratio: float = 3.0


def load_silver_policy(
    path: Path | str,
    family: str | None = None,
) -> SilverPolicyConfig:
    import yaml
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}

    cfg_dict: dict[str, Any] = {}
    common = raw.get("common", {})
    if isinstance(common, dict):
        cfg_dict.update(common)

    if family is not None and isinstance(raw.get(family), dict):
        cfg_dict.update(raw[family])
    elif isinstance(raw.get("satellite_mv"), dict):
        cfg_dict.update(raw["satellite_mv"])
    elif isinstance(raw, dict):
        cfg_dict.update(raw)

    return SilverPolicyConfig(**{
        k: v for k, v in cfg_dict.items()
        if k in SilverPolicyConfig.__dataclass_fields__
    })


# ---------------------------------------------------------------------------
#  Satellite MV silver
# ---------------------------------------------------------------------------

@dataclass
class SilverLabel:
    """One lightweight silver label record (intermediate output)."""
    sample_id: str
    candidate_key: str
    decision_label: str
    confidence: float
    signals: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "candidate_key": self.candidate_key,
            "decision_label": self.decision_label,
            "confidence": round(self.confidence, 4),
            "signals": self.signals,
        }


def label_satellite_mv_candidates(
    sample_id: str,
    candidate_results: list[CandidateMatchResult],
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
    candidate_keys: list[str] | None = None,
) -> list[SilverLabel]:
    """Derive silver labels for satellite_mv candidates on one image.

    Returns only non-abstained labels.  ``minor_fix`` is never emitted.
    """
    labels: list[SilverLabel] = []

    for i, cr in enumerate(candidate_results):
        ckey = candidate_keys[i] if candidate_keys else f"pred_{cr.pred_idx:03d}"
        label = _label_satellite_mv_candidate(
            sample_id, ckey, cr, exhaustivity, policy,
        )
        if label is not None:
            labels.append(label)

    return labels


def _satellite_mv_candidate_signals(
    cr: CandidateMatchResult,
) -> dict[str, Any]:
    """Serialize the pred-centric matching signals for one candidate."""
    return {
        "best_iou": cr.best_iou,
        "second_best_iou": cr.second_best_iou,
        "area_ratio": cr.area_ratio,
        "boundary_f1": cr.boundary_f1,
        "confidence_score": cr.confidence_score,
        "filter_status": cr.filter_status,
        "overlaps_multiple_gt": cr.overlaps_multiple_gt,
    }


def _label_satellite_mv_candidate(
    sample_id: str,
    candidate_key: str,
    cr: CandidateMatchResult,
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
) -> SilverLabel | None:
    """Classify one pred-centric satellite_mv candidate.

    Returns ``None`` for abstains.
    """
    signals = _satellite_mv_candidate_signals(cr)

    if cr.filter_status in ("prior_rejected", "core_rejected"):
        return SilverLabel(
            sample_id=sample_id,
            candidate_key=candidate_key,
            decision_label="reject",
            confidence=0.95,
            signals=signals,
        )

    # Clear reject: low IoU to all GT
    if cr.best_iou < policy.reject_iou_thresh:
        return SilverLabel(
            sample_id=sample_id,
            candidate_key=candidate_key,
            decision_label="reject",
            confidence=0.90,
            signals=signals,
        )

    # Ambiguity check → abstain
    if cr.overlaps_multiple_gt:
        return None
    if (
        cr.second_best_iou > 0
        and (cr.best_iou - cr.second_best_iou) < policy.ambiguity_gap
    ):
        return None

    # Clear accept: high IoU + ratio + boundary + confidence
    if (
        cr.best_iou >= policy.accept_iou_thresh
        and cr.area_ratio is not None
        and policy.min_area_ratio <= cr.area_ratio <= policy.max_area_ratio
        and cr.boundary_f1 is not None
        and cr.boundary_f1 >= policy.boundary_f1_thresh
        and cr.confidence_score >= policy.confidence_thresh
    ):
        return SilverLabel(
            sample_id=sample_id,
            candidate_key=candidate_key,
            decision_label="accept",
            confidence=cr.best_iou,
            signals=signals,
        )

    # Image-level route_to_ev if exhaustivity is uncertain
    if not exhaustivity.exhaustivity_confident:
        return SilverLabel(
            sample_id=sample_id,
            candidate_key=candidate_key,
            decision_label="route_to_ev",
            confidence=0.5,
            signals=signals,
        )

    # Default: abstain (dropped from silver set)
    return None


def label_satellite_mv_authoritative_candidates(
    sample_id: str,
    candidate_results: list[CandidateMatchResult],
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
) -> list[SilverLabel]:
    """Derive one silver label per authoritative satellite instance.

    The stage-A matcher is pred-centric, but stage B expects silver labels
    keyed by authoritative instance IDs (``inst_{id}``). We therefore
    collapse predictions onto their best-matched GT instance and keep the
    strongest non-reject label per instance.

    Reject labels are intentionally omitted here because pre-merge rejected
    examples are built separately by ``example_builder``.
    """
    winners: dict[int, SilverLabel] = {}

    for cr in candidate_results:
        if cr.best_gt_id is None:
            continue

        candidate_key = f"inst_{cr.best_gt_id:03d}"
        label = _label_satellite_mv_candidate(
            sample_id, candidate_key, cr, exhaustivity, policy,
        )
        if label is None or label.decision_label == "reject":
            continue

        prev = winners.get(cr.best_gt_id)
        if prev is None or _satellite_mv_label_rank(label) > _satellite_mv_label_rank(prev):
            winners[cr.best_gt_id] = label
            continue
        if (
            _satellite_mv_label_rank(label) == _satellite_mv_label_rank(prev)
            and label.confidence > prev.confidence
        ):
            winners[cr.best_gt_id] = label

    return [winners[gid] for gid in sorted(winners)]


def _satellite_mv_label_rank(label: SilverLabel) -> int:
    """Order authoritative satellite_mv labels by preference."""
    if label.decision_label == "accept":
        return 2
    if label.decision_label == "route_to_ev":
        return 1
    return 0


# ---------------------------------------------------------------------------
#  Satellite EV silver (GT-driven synthetic variants)
# ---------------------------------------------------------------------------

def _sort_instances_by_area(
    instance_map: np.ndarray,
    instances: list[dict[str, Any]],
    type_label: str,
) -> list[dict[str, Any]]:
    """Return instances of *type_label* sorted largest-area-first.

    Ties broken by ascending instance id for determinism.
    """
    typed = [i for i in instances if i["type"] == type_label]
    areas: list[tuple[int, int, dict[str, Any]]] = []
    for inst in typed:
        area = int((instance_map == inst["id"]).sum())
        areas.append((-area, inst["id"], inst))  # neg area for descending
    areas.sort()
    return [t[2] for t in areas]

def label_satellite_ev(
    sample_id: str,
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
) -> SilverLabel | None:
    """DEPRECATED: pred-vs-GT exhaustivity labeler. Use label_satellite_ev_gt."""
    import warnings
    warnings.warn(
        "label_satellite_ev is deprecated; use label_satellite_ev_gt",
        DeprecationWarning, stacklevel=2,
    )
    """Derive silver label for satellite_ev (one per image).

    Returns ``None`` on abstain.  ``redraw`` is gold-only.
    """
    signals: dict[str, Any] = {
        "num_gt": exhaustivity.num_gt,
        "num_pred_matched": exhaustivity.num_pred_matched,
        "num_gt_unmatched": exhaustivity.num_gt_unmatched,
        "num_pred_unmatched": exhaustivity.num_pred_unmatched,
        "mean_matched_iou": exhaustivity.mean_matched_iou,
        "total_gt_area": exhaustivity.total_gt_area,
        "total_pred_area": exhaustivity.total_pred_area,
        "unmatched_gt_area": exhaustivity.unmatched_gt_area,
        "unmatched_pred_area": exhaustivity.unmatched_pred_area,
        "recall": exhaustivity.recall,
        "precision": exhaustivity.precision,
    }

    if exhaustivity.num_gt == 0 and exhaustivity.num_pred_unmatched == 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="confirm_empty", confidence=0.95, signals=signals,
        )

    if (
        exhaustivity.num_gt_unmatched == 0
        and exhaustivity.num_pred_unmatched == 0
    ):
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="confirm_complete", confidence=exhaustivity.mean_matched_iou,
            signals=signals,
        )

    if exhaustivity.num_pred_unmatched > 0 and exhaustivity.num_gt_unmatched == 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="remove_fp", confidence=0.6, signals=signals,
        )

    if exhaustivity.num_gt_unmatched > 0 and exhaustivity.num_pred_unmatched == 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="add_missing", confidence=0.6, signals=signals,
        )

    mixed_label = _resolve_satellite_ev_mixed_label(exhaustivity, policy)
    if mixed_label is not None:
        return SilverLabel(
            sample_id=sample_id,
            candidate_key="image",
            decision_label=mixed_label,
            confidence=0.55,
            signals=signals,
        )

    return None  # abstain


def _resolve_satellite_ev_mixed_label(
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
) -> str | None:
    """Resolve mixed missing+false-positive EV cases by dominant area."""
    if (
        exhaustivity.num_gt_unmatched == 0
        or exhaustivity.num_pred_unmatched == 0
    ):
        return None

    unmatched_gt_frac = (
        exhaustivity.unmatched_gt_area / exhaustivity.total_gt_area
        if exhaustivity.total_gt_area > 0
        else 0.0
    )
    unmatched_pred_frac = (
        exhaustivity.unmatched_pred_area / exhaustivity.total_pred_area
        if exhaustivity.total_pred_area > 0
        else 0.0
    )

    if unmatched_pred_frac >= unmatched_gt_frac * policy.ev_mixed_area_dominance_ratio:
        return "remove_fp"
    if unmatched_gt_frac >= unmatched_pred_frac * policy.ev_mixed_area_dominance_ratio:
        return "add_missing"
    return None


# ---------------------------------------------------------------------------
#  Stream EV silver
# ---------------------------------------------------------------------------

def label_stream_ev(
    sample_id: str,
    exhaustivity: ImageExhaustivityResult,
    policy: SilverPolicyConfig,
) -> SilverLabel | None:
    """DEPRECATED: pred-vs-GT exhaustivity labeler. Use label_stream_ev_gt."""
    import warnings
    warnings.warn(
        "label_stream_ev is deprecated; use label_stream_ev_gt",
        DeprecationWarning, stacklevel=2,
    )
    """Derive silver label for stream_ev (one per image).

    Returns ``None`` on abstain.  ``redraw`` is gold-only;
    ``uncertain`` is flagged but not a hard label.
    """
    signals: dict[str, Any] = {
        "num_gt": exhaustivity.num_gt,
        "num_pred_matched": exhaustivity.num_pred_matched,
        "num_gt_unmatched": exhaustivity.num_gt_unmatched,
        "num_pred_unmatched": exhaustivity.num_pred_unmatched,
        "mean_matched_iou": exhaustivity.mean_matched_iou,
    }

    if exhaustivity.num_gt == 0 and exhaustivity.num_pred_unmatched == 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="confirm_empty", confidence=0.95, signals=signals,
        )

    if exhaustivity.exhaustivity_confident:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="confirm_complete",
            confidence=exhaustivity.mean_matched_iou, signals=signals,
        )

    if exhaustivity.num_gt_unmatched > 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="add_missing_fragment", confidence=0.5,
            signals=signals,
        )

    if exhaustivity.num_pred_unmatched > 0:
        return SilverLabel(
            sample_id=sample_id, candidate_key="image",
            decision_label="delete_fragment", confidence=0.5, signals=signals,
        )

    return None  # abstain


# ---------------------------------------------------------------------------
#  File I/O helpers
# ---------------------------------------------------------------------------

def write_silver_labels(labels: list[SilverLabel], path: Path) -> None:
    """Write labels to JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for lab in labels:
            fh.write(json.dumps(lab.to_dict(), sort_keys=True) + "\n")


def read_silver_labels(path: Path) -> list[SilverLabel]:
    """Read silver labels from JSONL."""
    labels: list[SilverLabel] = []
    with open(path) as fh:
        for line in fh:
            d = json.loads(line)
            labels.append(SilverLabel(**d))
    return labels


# ---------------------------------------------------------------------------
#  GT-driven labelers (synthetic variants)
# ---------------------------------------------------------------------------

def label_satellite_ev_gt(
    sample_id: str,
    instance_map: np.ndarray,
    instances: list[dict[str, Any]],
) -> list[SilverLabel]:
    """Deterministic GT-driven silver labels for ``satellite_ev``.

    Variants:
      - ``gt_complete``: all satellite instances visible  → confirm_complete
      - ``gt_empty``:    zero satellites (empty field)     → confirm_empty
      - ``drop_topK``:   hide the largest up to 3 satellites
                         in one synthetic-missing variant   → add_missing

    Returns one label per retained variant, keyed ``image:<variant_id>``.
    """
    sorted_sats = _sort_instances_by_area(instance_map, instances, "satellites")
    all_ids = tuple(s["id"] for s in sorted_sats)
    labels: list[SilverLabel] = []

    if len(sorted_sats) == 0:
        # Only gt_empty variant
        labels.append(SilverLabel(
            sample_id=sample_id,
            candidate_key="image:gt_empty",
            decision_label="confirm_empty",
            confidence=1.0,
            signals={
                "synthetic_variant_id": "gt_empty",
                "visible_instance_ids": [],
                "hidden_instance_ids": [],
                "num_satellites": 0,
            },
        ))
        return labels

    # gt_complete: everything visible
    labels.append(SilverLabel(
        sample_id=sample_id,
        candidate_key="image:gt_complete",
        decision_label="confirm_complete",
        confidence=1.0,
        signals={
            "synthetic_variant_id": "gt_complete",
            "visible_instance_ids": list(all_ids),
            "hidden_instance_ids": [],
            "num_satellites": len(all_ids),
        },
    ))

    # Single synthetic-missing variant: hide the largest up to 3 satellites.
    hidden_ids = [sat["id"] for sat in sorted_sats[:3]]
    if hidden_ids:
        hidden_set = set(hidden_ids)
        visible_ids = [sid for sid in all_ids if sid not in hidden_set]
        variant_id = f"drop_top{len(hidden_ids)}"
        labels.append(SilverLabel(
            sample_id=sample_id,
            candidate_key=f"image:{variant_id}",
            decision_label="add_missing",
            confidence=1.0,
            signals={
                "synthetic_variant_id": variant_id,
                "visible_instance_ids": visible_ids,
                "hidden_instance_ids": hidden_ids,
                "num_satellites": len(all_ids),
            },
        ))

    return labels


def label_stream_ev_gt(
    sample_id: str,
    instance_map: np.ndarray,
    instances: list[dict[str, Any]],
) -> list[SilverLabel]:
    """Deterministic GT-driven silver labels for ``stream_ev``.

    Variants:
      - ``gt_empty``:    zero stream fragments             → confirm_empty
      - ``gt_complete``: all stream fragments visible      → confirm_complete
      - ``drop_top1``:   hide the single largest fragment  → add_missing_fragment

    Returns one label per retained variant, keyed ``image:<variant_id>``.
    """
    sorted_streams = _sort_instances_by_area(instance_map, instances, "streams")
    all_ids = tuple(s["id"] for s in sorted_streams)
    labels: list[SilverLabel] = []

    if len(sorted_streams) == 0:
        labels.append(SilverLabel(
            sample_id=sample_id,
            candidate_key="image:gt_empty",
            decision_label="confirm_empty",
            confidence=1.0,
            signals={
                "synthetic_variant_id": "gt_empty",
                "visible_instance_ids": [],
                "hidden_instance_ids": [],
                "num_streams": 0,
            },
        ))
        return labels

    # gt_complete
    labels.append(SilverLabel(
        sample_id=sample_id,
        candidate_key="image:gt_complete",
        decision_label="confirm_complete",
        confidence=1.0,
        signals={
            "synthetic_variant_id": "gt_complete",
            "visible_instance_ids": list(all_ids),
            "hidden_instance_ids": [],
            "num_streams": len(all_ids),
        },
    ))

    # Single synthetic-missing variant: hide only the largest fragment.
    hidden_ids = [stream["id"] for stream in sorted_streams[:1]]
    if hidden_ids:
        hidden_set = set(hidden_ids)
        visible_ids = [sid for sid in all_ids if sid not in hidden_set]
        variant_id = "drop_top1"
        labels.append(SilverLabel(
            sample_id=sample_id,
            candidate_key=f"image:{variant_id}",
            decision_label="add_missing_fragment",
            confidence=1.0,
            signals={
                "synthetic_variant_id": variant_id,
                "visible_instance_ids": visible_ids,
                "hidden_instance_ids": hidden_ids,
                "num_streams": len(all_ids),
            },
        ))

    return labels
