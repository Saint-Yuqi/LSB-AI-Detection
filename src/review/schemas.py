"""
Frozen dataclasses and enums for the AI Verifier Protocol V1.1.

All verifier data contracts live here: VerifierExample, AssetRefs variants,
CropSpec, CorrectionLink, label-space enforcement, and revision-hash
computation.  Nothing in this module performs I/O.
"""
from __future__ import annotations

import enum
import hashlib
import json
from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
#  Enums
# ---------------------------------------------------------------------------

class TaskFamily(str, enum.Enum):
    SATELLITE_MV = "satellite_mv"
    SATELLITE_EV = "satellite_ev"
    STREAM_EV = "stream_ev"


class LabelSource(str, enum.Enum):
    SILVER = "silver"
    GOLD = "gold"


# ---------------------------------------------------------------------------
#  Label spaces (frozen per family)
# ---------------------------------------------------------------------------

LABEL_SPACES: dict[TaskFamily, frozenset[str]] = {
    TaskFamily.SATELLITE_MV: frozenset({
        "accept", "minor_fix", "reject", "route_to_ev",
    }),
    TaskFamily.SATELLITE_EV: frozenset({
        "confirm_complete", "add_missing", "confirm_empty",
    }),
    TaskFamily.STREAM_EV: frozenset({
        "confirm_complete", "add_missing_fragment", "delete_fragment",
        "redraw", "confirm_empty", "uncertain",
    }),
}

STREAM_EV_REASON_CODES = frozenset({
    "uncertain_extension", "ambiguous_link", "halo_confusion", "escalate",
})


def validate_label(family: TaskFamily, label: str) -> None:
    """Raise ValueError if *label* is not in the family's allowed set."""
    allowed = LABEL_SPACES[family]
    if label not in allowed:
        raise ValueError(
            f"Label {label!r} invalid for {family.value}; "
            f"allowed: {sorted(allowed)}"
        )


# ---------------------------------------------------------------------------
#  CropSpec
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CropSpec:
    """Spatial crop window centred on a candidate."""
    center_x: int
    center_y: int
    size: int = 384
    pad_mode: str = "zero"

    @property
    def content_key(self) -> str:
        """Content-based cache key (no example_id dependency)."""
        return f"{self.center_x}_{self.center_y}_{self.size}_{self.pad_mode}"


# ---------------------------------------------------------------------------
#  AssetRefs (per-family discriminated union)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SatMvAssetRefs:
    """Asset references for a satellite mask-verification candidate.

    ``bare_context_path`` is the shared unannotated full-image;
    the per-candidate bbox is stamped onto a per-candidate copy at ETL time.
    """
    bare_context_path: str
    bare_context_sha1: str
    crop_spec: CropSpec
    candidate_bbox_xywh: tuple[int, int, int, int]
    candidate_rle: dict[str, Any]
    candidate_id: str
    authoritative_instance_id: int | None
    candidate_source: str  # "authoritative" | "pre_merge_rejected"


@dataclass(frozen=True)
class EvAssetRefs:
    """Asset references for image-level exhaustivity verification."""
    review_image_path: str
    review_image_sha1: str
    annotation_state_hash: str
    fragment_hints: tuple[dict[str, Any], ...] | None = None
    synthetic_variant_id: str | None = None
    hidden_instance_ids: tuple[int, ...] | None = None
    visible_instance_ids: tuple[int, ...] | None = None


AssetRefs = SatMvAssetRefs | EvAssetRefs


# ---------------------------------------------------------------------------
#  VerifierExample (business JSONL record)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class VerifierExample:
    """One row in ``verifier_examples_{family}.jsonl``."""
    example_id: str
    task_family: TaskFamily
    sample_id: str
    halo_id: int
    view_id: str
    task_revision_id: str
    asset_refs: AssetRefs
    render_spec_id: str
    fixed_prompt_id: str
    decision_label: str
    label_source: LabelSource
    source_round: str
    source_checkpoint: str
    revision_hash: str

    parent_example_id: str | None = None
    uncertainty_flag: bool | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        validate_label(self.task_family, self.decision_label)
        if self.reason_code is not None:
            if self.task_family != TaskFamily.STREAM_EV:
                raise ValueError("reason_code only valid for stream_ev")
            if self.reason_code not in STREAM_EV_REASON_CODES:
                raise ValueError(
                    f"Unknown reason_code {self.reason_code!r}; "
                    f"allowed: {sorted(STREAM_EV_REASON_CODES)}"
                )


# ---------------------------------------------------------------------------
#  CorrectionLink
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CorrectionLink:
    """Tracks a redraw correction linking an original to a new revision."""
    original_task_revision_id: str
    correction_task_revision_id: str
    original_revision_hash: str
    new_revision_hash: str
    status: str  # "superseded" | "active"

    def __post_init__(self) -> None:
        if self.status not in ("superseded", "active"):
            raise ValueError(f"Invalid status {self.status!r}")


# ---------------------------------------------------------------------------
#  Revision-hash helpers
# ---------------------------------------------------------------------------

def _canonical_bytes(*parts: Any) -> bytes:
    """Deterministic serialisation of heterogeneous hash inputs."""
    tokens: list[str] = []
    for p in parts:
        if isinstance(p, dict):
            tokens.append(json.dumps(p, sort_keys=True, separators=(",", ":")))
        elif isinstance(p, (list, tuple)):
            tokens.append(json.dumps(p, sort_keys=True, separators=(",", ":")))
        elif isinstance(p, enum.Enum):
            tokens.append(p.value)
        elif p is None:
            tokens.append("")
        else:
            tokens.append(str(p))
    return "\x00".join(tokens).encode("utf-8")


def compute_revision_hash_sat_mv(
    *,
    task_family: TaskFamily,
    sample_id: str,
    task_revision_id: str,
    render_spec_id: str,
    fixed_prompt_id: str,
    candidate_rle: dict[str, Any],
    candidate_bbox_xywh: tuple[int, int, int, int],
    crop_spec: CropSpec,
    bare_context_sha1: str,
    candidate_source: str,
    authoritative_instance_id: int | None,
) -> str:
    data = _canonical_bytes(
        task_family, sample_id, task_revision_id,
        render_spec_id, fixed_prompt_id,
        candidate_rle,
        candidate_bbox_xywh,
        (crop_spec.center_x, crop_spec.center_y, crop_spec.size, crop_spec.pad_mode),
        bare_context_sha1,
        candidate_source,
        authoritative_instance_id if authoritative_instance_id is not None else "",
    )
    return hashlib.sha256(data).hexdigest()


def compute_revision_hash_ev(
    *,
    task_family: TaskFamily,
    sample_id: str,
    task_revision_id: str,
    render_spec_id: str,
    fixed_prompt_id: str,
    review_image_sha1: str,
    annotation_state_hash: str,
    fragment_hints_hash: str | None = None,
    synthetic_variant_id: str | None = None,
) -> str:
    parts: list[Any] = [
        task_family, sample_id, task_revision_id,
        render_spec_id, fixed_prompt_id,
        review_image_sha1,
        annotation_state_hash,
    ]
    if fragment_hints_hash is not None:
        parts.append(fragment_hints_hash)
    # Backward compat: None never enters hash bytes → old records unchanged.
    if synthetic_variant_id is not None:
        parts.append(synthetic_variant_id)
    return hashlib.sha256(_canonical_bytes(*parts)).hexdigest()
