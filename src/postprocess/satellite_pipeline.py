"""
Satellite pipeline runner for DR1 canonical evaluate.

Design:
    - Thin orchestrator. Takes already-assembled components and drives
      raw satellite candidates through an 8-stage fixed-order state
      machine, recording one StageEvent per stage per candidate it
      actually touches (once a candidate is dropped, later filter
      stages don't revisit it).
    - Every filter stage delegates its decision to its component; the
      runner only emits StageEvents and updates `status`.
    - Metrics snapshots are strictly thin: only scalar keys in
      `_THIN_KEYS` are allowed, enforced by `build_thin()`.

Stages (fixed order):
    1. raw_retrieval
    2. metrics_completion
    3. size_aware_score_gate
    4. satellite_prior_filter
    5. core_exclusion_or_soft_core_rescue
    6. stream_conflict_resolution
    7. final_gt_write
    8. diagnostics_emit       (image-level, not per-candidate)

The runtime pipeline is pure: reviewed exceptions are NOT injected or
resurrected here. They live as explicit human-adopted instances in the
Shadow GT flow (see ``scripts/review/migrate_satellite_overrides.py``).

Public API:
    SatellitePipelineRunner(score_gate, prior_filter, core_policy,
                            conflict_resolver).run(
        raw_sats, streams_gt_map, H, W, base_key=None)
    -> SatellitePipelineResult

StageEvent fields:
    stage, input_state, rule_name, threshold_version, threshold_values,
    decision, reason, output_state, metrics_snapshot_thin
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from src.utils.coco_utils import mask_to_rle

from src.pipelines.unified_dataset.taxonomy import INNER_GALAXY, SATELLITES, normalize_type_label

from .satellite_conflict_resolver import SatelliteConflictResolver
from .satellite_core_policy import SatelliteCorePolicy
from .satellite_prior_filter import SatellitePriorFilter, _compute_aspect_sym, _compute_solidity
from .satellite_score_gate import SatelliteScoreGate

STAGE_ORDER: tuple[str, ...] = (
    "raw_retrieval",
    "metrics_completion",
    "size_aware_score_gate",
    "satellite_prior_filter",
    "core_exclusion_or_soft_core_rescue",
    "stream_conflict_resolution",
    "final_gt_write",
    "diagnostics_emit",
)

_THIN_KEYS: frozenset[str] = frozenset(
    {
        "score",
        "area_clean_px",
        "solidity",
        "aspect_sym_moment",
        "dist_to_center_px",
        "dist_to_center_frac",
        "overlap_ratio_satellite",
        "overlap_ratio_stream",
    }
)


def build_thin(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return a new dict containing only whitelisted thin metric keys.

    Values that are None are dropped. Raises AssertionError if any
    non-whitelisted key sneaks in (defensive; callers build from a
    known shape).
    """
    thin = {k: _to_scalar(v) for k, v in metrics.items() if k in _THIN_KEYS and v is not None}
    forbidden = set(thin.keys()) - _THIN_KEYS
    assert not forbidden, f"thin metrics contain forbidden keys: {forbidden}"
    return thin


def _to_scalar(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


@dataclass
class StageEvent:
    stage: str
    input_state: str
    rule_name: str
    threshold_version: str
    threshold_values: dict[str, Any]
    decision: str
    reason: str
    output_state: str
    metrics_snapshot_thin: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SatelliteCandidateState:
    candidate_id: str
    candidate_rle_sha1: str
    mask: dict[str, Any]
    status: str = "alive"  # alive | dropped | rescued | kept
    matched_stream_id: int | None = None
    history: list[StageEvent] = field(default_factory=list)

    def append(self, event: StageEvent) -> None:
        self.history.append(event)

    def thin_metrics(
        self,
        H: int,
        W: int,
        overlap_ratio_satellite: float | None = None,
        overlap_ratio_stream: float | None = None,
    ) -> dict[str, Any]:
        """Project the mask dict into a thin metrics snapshot."""
        m = self.mask
        area_clean_px = m.get("area_clean")
        if area_clean_px is None:
            seg = m.get("segmentation")
            area_clean_px = int(seg.sum()) if seg is not None else 0

        dist_px = m.get("dist_to_center")
        dist_frac: float | None = None
        if dist_px is not None:
            dist_frac = float(dist_px) / float(min(H, W))

        raw = {
            "score": m.get("score"),
            "area_clean_px": area_clean_px,
            "solidity": m.get("solidity"),
            "aspect_sym_moment": m.get("aspect_sym_moment") or m.get("aspect_sym"),
            "dist_to_center_px": dist_px,
            "dist_to_center_frac": dist_frac,
            "overlap_ratio_satellite": overlap_ratio_satellite,
            "overlap_ratio_stream": overlap_ratio_stream,
        }
        return build_thin(raw)


@dataclass
class SatellitePipelineResult:
    final_sats: list[dict[str, Any]]
    candidates: list[dict[str, Any]]
    image_summary: dict[str, Any]
    # New-path additions (empty on legacy default behavior).
    final_inner_galaxy: list[dict[str, Any]] = field(default_factory=list)


class SatellitePipelineRunner:
    """Orchestrator for DR1 satellite post-processing stages.

    On the legacy path the runner walks the full 8-stage chain. On the new
    path callers disable the core stage and the conflict stage by passing
    ``enable_core_policy=False`` / ``enable_conflict_resolution=False``;
    those stages then emit a ``StageEvent("skipped", ...)`` and pass the
    candidate through unchanged.
    """

    def __init__(
        self,
        score_gate: SatelliteScoreGate,
        prior_filter: SatellitePriorFilter,
        core_policy: SatelliteCorePolicy | None = None,
        conflict_resolver: SatelliteConflictResolver | None = None,
        *,
        enable_core_policy: bool = True,
        enable_conflict_resolution: bool = True,
    ) -> None:
        self.score_gate = score_gate
        self.prior_filter = prior_filter
        self.core_policy = core_policy
        self.conflict_resolver = conflict_resolver
        self.enable_core_policy = enable_core_policy
        self.enable_conflict_resolution = enable_conflict_resolution

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(
        self,
        raw_sats: list[dict[str, Any]],
        streams_gt_map: np.ndarray,
        H: int,
        W: int,
        base_key: str | None = None,
    ) -> SatellitePipelineResult:
        """Drive raw satellite masks through the fixed 8-stage pipeline."""
        candidates = [
            self._stage_raw_retrieval(idx, mask, H, W)
            for idx, mask in enumerate(raw_sats)
        ]

        for cand in candidates:
            if cand.status == "alive":
                self._stage_metrics_completion(cand, H, W)

        for cand in candidates:
            if cand.status == "alive":
                self._stage_score_gate(cand, H, W)

        for cand in candidates:
            if cand.status == "alive":
                self._stage_prior_filter(cand, H, W)

        for cand in candidates:
            if cand.status == "alive":
                self._stage_core_policy(cand, H, W)

        for cand in candidates:
            if cand.status in {"alive", "rescued"}:
                self._stage_conflict_resolution(cand, streams_gt_map, H, W)

        final_sats: list[dict[str, Any]] = []
        final_inner_galaxy: list[dict[str, Any]] = []
        for cand in candidates:
            if cand.status in {"alive", "rescued", "kept", "relabeled_inner_galaxy"}:
                self._stage_final_gt_write(cand, H, W)
                # Classify by type_label so the new path can route relabeled
                # candidates into the inner_galaxy bucket; legacy candidates
                # default to satellites.
                bucket = normalize_type_label(cand.mask.get("type_label", SATELLITES))
                if bucket == INNER_GALAXY:
                    final_inner_galaxy.append(cand.mask)
                else:
                    final_sats.append(cand.mask)

        image_summary = self._stage_diagnostics_emit(candidates, base_key=base_key)

        candidate_records = [
            {
                "candidate_id": c.candidate_id,
                "candidate_rle_sha1": c.candidate_rle_sha1,
                "final_status": c.status,
                "matched_stream_id": c.matched_stream_id,
                "history": [ev.to_json() for ev in c.history],
            }
            for c in candidates
        ]

        return SatellitePipelineResult(
            final_sats=final_sats,
            candidates=candidate_records,
            image_summary=image_summary,
            final_inner_galaxy=final_inner_galaxy,
        )

    # ------------------------------------------------------------------
    # Individual stages
    # ------------------------------------------------------------------

    def _stage_raw_retrieval(
        self,
        idx: int,
        mask: dict[str, Any],
        H: int,
        W: int,
    ) -> SatelliteCandidateState:
        # Prefer upstream-stamped identity (assign_stable_ids on the combined
        # raw mask list); fall back to local generation for legacy callers
        # and tests that bypass the stamp.
        candidate_id = mask.get("candidate_id") or f"sat_{idx:04d}"
        rle_sha1 = mask.get("candidate_rle_sha1") or _rle_sha1(mask)
        # Mirror the stamp back onto the mask so downstream consumers
        # (predictions.json, instances.json) read identical values.
        mask.setdefault("candidate_id", candidate_id)
        mask.setdefault("candidate_rle_sha1", rle_sha1)
        mask.setdefault("raw_index", idx)
        cand = SatelliteCandidateState(
            candidate_id=candidate_id,
            candidate_rle_sha1=rle_sha1,
            mask=mask,
            status="alive",
        )
        cand.append(
            StageEvent(
                stage="raw_retrieval",
                input_state="raw",
                rule_name="raw_intake",
                threshold_version="raw_v1",
                threshold_values={},
                decision="pass",
                reason="raw_accept",
                output_state="alive",
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )
        return cand

    def _stage_metrics_completion(
        self,
        cand: SatelliteCandidateState,
        H: int,
        W: int,
    ) -> None:
        m = cand.mask
        seg = m.get("segmentation")
        if seg is None:
            self._mark_dropped(cand, "metrics_completion", "metrics_fill", "metrics_v1", {}, "no_segmentation", H, W)
            return

        if "area_clean" not in m or m["area_clean"] is None:
            m["area_clean"] = int(seg.astype(bool).sum())

        if m.get("solidity") is None:
            m["solidity"] = float(_compute_solidity(seg.astype(np.uint8)))

        if m.get("aspect_sym_moment") is None and m.get("aspect_sym") is None:
            aspect = float(_compute_aspect_sym(seg.astype(np.uint8)))
            m["aspect_sym_moment"] = aspect
            m["aspect_sym"] = aspect

        if m.get("dist_to_center") is None:
            rows, cols = np.where(seg.astype(bool))
            if len(rows):
                cen_y, cen_x = float(rows.mean()), float(cols.mean())
                m["centroid_y"] = cen_y
                m["centroid_x"] = cen_x
                m["dist_to_center"] = float(np.hypot(cen_x - W / 2.0, cen_y - H / 2.0))
            else:
                m["dist_to_center"] = float("inf")

        cand.append(
            StageEvent(
                stage="metrics_completion",
                input_state="alive",
                rule_name="metrics_fill",
                threshold_version="metrics_v1",
                threshold_values={},
                decision="pass",
                reason="metrics_filled",
                output_state="alive",
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )

    def _stage_score_gate(
        self,
        cand: SatelliteCandidateState,
        H: int,
        W: int,
    ) -> None:
        m = cand.mask
        area = int(m.get("area_clean", 0))
        score = float(m.get("score", 0.0))
        decision, reason = self.score_gate.decide(area, score)
        output = "alive" if decision == "pass" else "dropped"
        cand.status = output
        cand.append(
            StageEvent(
                stage="size_aware_score_gate",
                input_state="alive",
                rule_name="size_tier_score",
                threshold_version=self.score_gate.threshold_version,
                threshold_values=self.score_gate.threshold_values(),
                decision=decision,
                reason=reason,
                output_state=output,
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )

    def _stage_prior_filter(
        self,
        cand: SatelliteCandidateState,
        H: int,
        W: int,
    ) -> None:
        decision, reason, target = self.prior_filter.decide_with_target(cand.mask)
        cfg = self.prior_filter.cfg

        if decision == "relabel" and target == INNER_GALAXY:
            cand.mask["type_label"] = INNER_GALAXY
            cand.mask["relabel_target"] = INNER_GALAXY
            output = "relabeled_inner_galaxy"
        elif decision == "pass":
            output = "alive"
        else:
            output = "dropped"
        cand.status = output
        cand.append(
            StageEvent(
                stage="satellite_prior_filter",
                input_state="alive",
                rule_name=(
                    "hard_center_area_solidity_aspect"
                    if cfg.get("hard_center_radius_frac") is not None
                    else "area_solidity_aspect"
                ),
                threshold_version=self.prior_filter.threshold_version,
                threshold_values=self.prior_filter.threshold_values(),
                decision=decision,
                reason=reason,
                output_state=output,
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )

    def _stage_core_policy(
        self,
        cand: SatelliteCandidateState,
        H: int,
        W: int,
    ) -> None:
        if not self.enable_core_policy or self.core_policy is None:
            cand.append(
                StageEvent(
                    stage="core_exclusion_or_soft_core_rescue",
                    input_state="alive",
                    rule_name="core_radius_with_soft_rescue",
                    threshold_version="disabled",
                    threshold_values={},
                    decision="skipped",
                    reason="core_disabled",
                    output_state=cand.status,
                    metrics_snapshot_thin=cand.thin_metrics(H, W),
                )
            )
            return

        m = cand.mask
        dist_px = m.get("dist_to_center")
        if dist_px is None:
            dist_frac = float("inf")
        else:
            dist_frac = float(dist_px) / float(min(H, W))

        area = int(m.get("area_clean", 0))
        score = float(m.get("score", 0.0))
        solidity = float(m.get("solidity") or 0.0)
        aspect = float(m.get("aspect_sym_moment") or m.get("aspect_sym") or 0.0)

        decision, reason = self.core_policy.decide(
            dist_to_center_frac=dist_frac,
            area_clean_px=area,
            score=score,
            solidity=solidity,
            aspect_sym_moment=aspect,
        )

        if decision == "pass":
            output = "alive"
        elif decision == "rescue":
            output = "rescued"
        else:
            output = "dropped"
        cand.status = output
        cand.append(
            StageEvent(
                stage="core_exclusion_or_soft_core_rescue",
                input_state="alive",
                rule_name="core_radius_with_soft_rescue",
                threshold_version=self.core_policy.threshold_version,
                threshold_values=self.core_policy.threshold_values(),
                decision=decision,
                reason=reason,
                output_state=output,
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )

    def _stage_conflict_resolution(
        self,
        cand: SatelliteCandidateState,
        streams_gt_map: np.ndarray,
        H: int,
        W: int,
    ) -> None:
        if not self.enable_conflict_resolution or self.conflict_resolver is None:
            cand.append(
                StageEvent(
                    stage="stream_conflict_resolution",
                    input_state="alive",
                    rule_name="gt_stream_overlap",
                    threshold_version="disabled",
                    threshold_values={},
                    decision="skipped",
                    reason="conflict_disabled",
                    output_state=cand.status,
                    metrics_snapshot_thin=cand.thin_metrics(H, W),
                )
            )
            return

        seg = cand.mask.get("segmentation")
        if seg is None:
            self._mark_dropped(
                cand,
                "stream_conflict_resolution",
                "gt_stream_overlap",
                self.conflict_resolver.threshold_version,
                self.conflict_resolver.threshold_values(),
                "no_segmentation",
                H,
                W,
            )
            return

        matched_id, _overlap_px, ratio_sat, ratio_stream = self.conflict_resolver.match_stream(
            seg, streams_gt_map
        )
        area = int(cand.mask.get("area_clean", 0))
        solidity = float(cand.mask.get("solidity") or 0.0)
        aspect = float(cand.mask.get("aspect_sym_moment") or cand.mask.get("aspect_sym") or 0.0)

        decision, reason, extras = self.conflict_resolver.decide(
            matched_stream_id=matched_id,
            overlap_ratio_satellite=ratio_sat,
            area_clean_px=area,
            solidity=solidity,
            aspect_sym_moment=aspect,
        )
        cand.matched_stream_id = extras.get("matched_stream_id")

        if decision == "drop":
            output = "dropped"
        elif decision == "win":
            output = "kept"
        else:
            output = cand.status if cand.status == "rescued" else "alive"
        cand.status = output

        thresholds = dict(self.conflict_resolver.threshold_values())
        thresholds["matched_stream_id"] = cand.matched_stream_id
        cand.append(
            StageEvent(
                stage="stream_conflict_resolution",
                input_state="alive" if cand.history[-1].output_state == "alive" else cand.history[-1].output_state,
                rule_name="gt_stream_overlap",
                threshold_version=self.conflict_resolver.threshold_version,
                threshold_values=thresholds,
                decision=decision,
                reason=reason,
                output_state=output,
                metrics_snapshot_thin=cand.thin_metrics(
                    H,
                    W,
                    overlap_ratio_satellite=ratio_sat if matched_id is not None else None,
                    overlap_ratio_stream=ratio_stream if matched_id is not None else None,
                ),
            )
        )

    def _stage_final_gt_write(
        self,
        cand: SatelliteCandidateState,
        H: int,
        W: int,
    ) -> None:
        # A relabeled inner-galaxy candidate keeps its terminal status so the
        # caller can route it to the right output bucket; everything else
        # consolidates to "kept" for downstream filters that expect that label.
        prior_status = cand.status
        if prior_status != "relabeled_inner_galaxy":
            cand.status = "kept"
        reason = (
            "write_relabeled_inner_galaxy"
            if prior_status == "relabeled_inner_galaxy"
            else "write_kept"
        )
        cand.append(
            StageEvent(
                stage="final_gt_write",
                input_state=prior_status,
                rule_name="persist_to_final",
                threshold_version="final_v1",
                threshold_values={},
                decision="pass",
                reason=reason,
                output_state=cand.status,
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )

    def _stage_diagnostics_emit(
        self,
        candidates: list[SatelliteCandidateState],
        base_key: str | None = None,
    ) -> dict[str, Any]:
        stage_drop_counts: dict[str, int] = {}
        final_status_counts: dict[str, int] = {}
        for cand in candidates:
            final_status_counts[cand.status] = final_status_counts.get(cand.status, 0) + 1
            for ev in cand.history:
                if ev.decision == "drop":
                    stage_drop_counts[ev.stage] = stage_drop_counts.get(ev.stage, 0) + 1

        n_inner_galaxy = sum(
            1
            for c in candidates
            if c.status == "relabeled_inner_galaxy"
            or normalize_type_label(c.mask.get("type_label", SATELLITES)) == INNER_GALAXY
        )
        image_summary = {
            "base_key": base_key,
            "n_raw_satellites": len(candidates),
            "n_final_satellites": sum(1 for c in candidates if c.status == "kept"),
            "n_final_inner_galaxy": n_inner_galaxy,
            "final_status_counts": final_status_counts,
            "stage_drop_counts": stage_drop_counts,
            "thresholds_version": {
                "score_gate": self.score_gate.threshold_version,
                "prior_filter": self.prior_filter.threshold_version,
                "core_policy": (
                    self.core_policy.threshold_version
                    if (self.enable_core_policy and self.core_policy is not None)
                    else "disabled"
                ),
                "conflict_policy": (
                    self.conflict_resolver.threshold_version
                    if (self.enable_conflict_resolution and self.conflict_resolver is not None)
                    else "disabled"
                ),
            },
        }
        return image_summary

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _mark_dropped(
        self,
        cand: SatelliteCandidateState,
        stage: str,
        rule_name: str,
        threshold_version: str,
        threshold_values: dict[str, Any],
        reason: str,
        H: int,
        W: int,
    ) -> None:
        cand.status = "dropped"
        cand.append(
            StageEvent(
                stage=stage,
                input_state="alive",
                rule_name=rule_name,
                threshold_version=threshold_version,
                threshold_values=threshold_values,
                decision="drop",
                reason=reason,
                output_state="dropped",
                metrics_snapshot_thin=cand.thin_metrics(H, W),
            )
        )


def _rle_sha1(mask: dict[str, Any]) -> str:
    seg = mask.get("segmentation")
    if seg is None:
        return "0" * 16
    rle = mask_to_rle(np.asarray(seg).astype(np.uint8))
    blob = json.dumps(rle, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
