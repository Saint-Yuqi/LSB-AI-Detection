"""
Pipeline artifacts → business JSONL assembly.

Dual-source design:
  - **Source A (authoritative)**: kept candidates from
    ``instance_map_uint8.png`` + ``instances.json``.
  - **Source B (pre-merge)**: rejected candidates from re-filtered
    ``sam3_predictions_raw.json``.

Output: ``verifier_examples_{family}.jsonl``
"""
from __future__ import annotations

import hashlib
import json
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.review.asset_manager import AssetManager
from src.review.key_adapter import (
    base_key_to_halo_id,
    base_key_to_sample_id,
)
from src.review.render_spec import RenderSpec
from src.review.schemas import (
    CropSpec,
    EvAssetRefs,
    LabelSource,
    SatMvAssetRefs,
    TaskFamily,
    VerifierExample,
    compute_revision_hash_ev,
    compute_revision_hash_sat_mv,
)
from src.review.silver_labeler import SilverLabel
from src.pipelines.unified_dataset.keys import BaseKey
from src.utils.coco_utils import decode_rle, get_bbox_from_mask, mask_to_rle


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _annotation_state_hash(
    instance_map: np.ndarray, instances: list[dict],
) -> str:
    """Content hash over the full annotation state of an image."""
    h = hashlib.sha256()
    h.update(instance_map.tobytes())
    h.update(json.dumps(instances, sort_keys=True).encode())
    return h.hexdigest()


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def _new_example_id(family: TaskFamily, counter: int) -> str:
    return f"{family.value[:6]}_{counter:06d}"


# ---------------------------------------------------------------------------
#  Source A: kept candidates from authoritative layer
# ---------------------------------------------------------------------------

def _build_sat_mv_authoritative(
    bk: BaseKey,
    gt_dir: Path,
    instance_map: np.ndarray,
    instances: list[dict],
    silver_lookup: dict[str, SilverLabel],
    asset_mgr: AssetManager,
    spec: RenderSpec,
    prompt_id: str,
    full_image: np.ndarray,
    source_round: str,
    source_checkpoint: str,
    counter: int,
) -> tuple[list[VerifierExample], int]:
    """Build examples from kept satellite instances (authoritative layer)."""
    sample_id = base_key_to_sample_id(bk)
    halo_id = base_key_to_halo_id(bk)
    view_id = bk.view_id

    ctx_path, ctx_sha1 = asset_mgr.ensure_bare_context(
        sample_id, spec.input_variant, full_image,
    )

    sat_instances = [inst for inst in instances if inst["type"] == "satellites"]
    examples: list[VerifierExample] = []

    for inst in sat_instances:
        inst_id = inst["id"]
        candidate_mask = (instance_map == inst_id).astype(np.uint8)
        if candidate_mask.sum() == 0:
            continue

        bbox = get_bbox_from_mask(candidate_mask)
        bbox_xywh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rle = mask_to_rle(candidate_mask)
        cx = bbox_xywh[0] + bbox_xywh[2] // 2
        cy = bbox_xywh[1] + bbox_xywh[3] // 2
        crop_spec = CropSpec(center_x=cx, center_y=cy, size=spec.crop_size)
        cand_id = f"inst_{inst_id:03d}"

        asset_mgr.ensure_crop(
            sample_id, crop_spec, spec, full_image, rle,
        )

        silver_key = f"inst_{inst_id:03d}"
        label_rec = silver_lookup.get(silver_key)
        if label_rec is None:
            continue
        decision_label = label_rec.decision_label

        rev_hash = compute_revision_hash_sat_mv(
            task_family=TaskFamily.SATELLITE_MV,
            sample_id=sample_id,
            task_revision_id="rev_00",
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            candidate_rle=rle,
            candidate_bbox_xywh=bbox_xywh,
            crop_spec=crop_spec,
            bare_context_sha1=ctx_sha1,
            candidate_source="authoritative",
            authoritative_instance_id=inst_id,
        )

        examples.append(VerifierExample(
            example_id=_new_example_id(TaskFamily.SATELLITE_MV, counter),
            task_family=TaskFamily.SATELLITE_MV,
            sample_id=sample_id,
            halo_id=halo_id,
            view_id=view_id,
            task_revision_id="rev_00",
            asset_refs=SatMvAssetRefs(
                bare_context_path=ctx_path,
                bare_context_sha1=ctx_sha1,
                crop_spec=crop_spec,
                candidate_bbox_xywh=bbox_xywh,
                candidate_rle=rle,
                candidate_id=cand_id,
                authoritative_instance_id=inst_id,
                candidate_source="authoritative",
            ),
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            decision_label=decision_label,
            label_source=LabelSource.SILVER,
            source_round=source_round,
            source_checkpoint=source_checkpoint,
            revision_hash=rev_hash,
        ))
        counter += 1

    return examples, counter


# ---------------------------------------------------------------------------
#  Source B: rejected candidates from pre-merge artifacts
# ---------------------------------------------------------------------------

def _load_reject_candidates_sam3(
    gt_dir: Path, bk: BaseKey,
) -> list[dict[str, Any]]:
    """Load rejected satellite candidates from sam3_predictions_raw.json.

    Re-filters using the same type_label == 'satellites' constraint.
    Candidates kept in post-filtering are excluded via RLE set match.
    Candidates that overlap authoritative GT satellites (IoU > 0.3) are
    also excluded to prevent false pre_merge_rejected examples.
    """
    raw_path = gt_dir / str(bk) / "sam3_predictions_raw.json"
    if not raw_path.exists():
        return []

    with open(raw_path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw["predictions"]

    post_path = gt_dir / str(bk) / "sam3_predictions_post.json"
    kept_rle_set: set[str] = set()
    if post_path.exists():
        with open(post_path) as f:
            post = json.load(f)
        if isinstance(post, dict):
            post = post["predictions"]
        for m in post:
            seg = m.get("segmentation", m.get("rle", {}))
            if isinstance(seg, dict):
                kept_rle_set.add(json.dumps(seg, sort_keys=True))

    gt_sat_masks: list[np.ndarray] | None = None
    imap_path = gt_dir / str(bk) / "instance_map_uint8.png"
    inst_path = gt_dir / str(bk) / "instances.json"
    if imap_path.exists() and inst_path.exists():
        imap = np.array(Image.open(imap_path))
        with open(inst_path) as f:
            instances = json.load(f)
        gt_sat_masks = [
            (imap == inst["id"]).astype(np.uint8)
            for inst in instances if inst["type"] == "satellites"
        ]

    rejects: list[dict[str, Any]] = []
    for idx, m in enumerate(raw):
        if m.get("type_label") != "satellites":
            continue
        seg = m.get("segmentation", m.get("rle", {}))
        if isinstance(seg, dict):
            if json.dumps(seg, sort_keys=True) in kept_rle_set:
                continue
            mask = decode_rle(seg)
        elif isinstance(seg, np.ndarray):
            mask = seg.astype(np.uint8)
        else:
            continue

        if gt_sat_masks:
            overlaps_gt = False
            pred_area = float(mask.sum())
            for gt_m in gt_sat_masks:
                inter = float(np.logical_and(mask, gt_m).sum())
                gt_area = float(gt_m.sum())
                union = pred_area + gt_area - inter
                if union > 0 and (inter / union) > 0.3:
                    overlaps_gt = True
                    break
            if overlaps_gt:
                continue

        rejects.append({
            "segmentation": mask,
            "rle": mask_to_rle(mask) if not isinstance(seg, dict) else seg,
            "bucket": "raw_rejected",
            "index": idx,
            "score": m.get("score", 0.0),
            "stability_score": m.get("stability_score", 0.0),
        })
    return rejects


def _build_sat_mv_rejected(
    bk: BaseKey,
    gt_dir: Path,
    reject_candidates: list[dict[str, Any]],
    asset_mgr: AssetManager,
    spec: RenderSpec,
    prompt_id: str,
    full_image: np.ndarray,
    source_round: str,
    source_checkpoint: str,
    counter: int,
) -> tuple[list[VerifierExample], int]:
    """Build examples from rejected candidates (pre-merge artifacts)."""
    sample_id = base_key_to_sample_id(bk)
    halo_id = base_key_to_halo_id(bk)
    view_id = bk.view_id

    ctx_path, ctx_sha1 = asset_mgr.ensure_bare_context(
        sample_id, spec.input_variant, full_image,
    )

    examples: list[VerifierExample] = []

    for cand in reject_candidates:
        mask = cand["segmentation"]
        if mask.sum() == 0:
            continue

        bbox = get_bbox_from_mask(mask)
        bbox_xywh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))
        rle = cand["rle"]
        cx = bbox_xywh[0] + bbox_xywh[2] // 2
        cy = bbox_xywh[1] + bbox_xywh[3] // 2
        crop_spec = CropSpec(center_x=cx, center_y=cy, size=spec.crop_size)
        bucket = cand["bucket"]
        idx = cand["index"]
        cand_id = f"rej_{bucket}_{idx:03d}"

        asset_mgr.ensure_crop(sample_id, crop_spec, spec, full_image, rle)

        rev_hash = compute_revision_hash_sat_mv(
            task_family=TaskFamily.SATELLITE_MV,
            sample_id=sample_id,
            task_revision_id="rev_00",
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            candidate_rle=rle,
            candidate_bbox_xywh=bbox_xywh,
            crop_spec=crop_spec,
            bare_context_sha1=ctx_sha1,
            candidate_source="pre_merge_rejected",
            authoritative_instance_id=None,
        )

        examples.append(VerifierExample(
            example_id=_new_example_id(TaskFamily.SATELLITE_MV, counter),
            task_family=TaskFamily.SATELLITE_MV,
            sample_id=sample_id,
            halo_id=halo_id,
            view_id=view_id,
            task_revision_id="rev_00",
            asset_refs=SatMvAssetRefs(
                bare_context_path=ctx_path,
                bare_context_sha1=ctx_sha1,
                crop_spec=crop_spec,
                candidate_bbox_xywh=bbox_xywh,
                candidate_rle=rle,
                candidate_id=cand_id,
                authoritative_instance_id=None,
                candidate_source="pre_merge_rejected",
            ),
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            decision_label="reject",
            label_source=LabelSource.SILVER,
            source_round=source_round,
            source_checkpoint=source_checkpoint,
            revision_hash=rev_hash,
        ))
        counter += 1

    return examples, counter


# ---------------------------------------------------------------------------
#  EV example builders
# ---------------------------------------------------------------------------

def _build_ev_examples(
    bk: BaseKey,
    family: TaskFamily,
    gt_dir: Path,
    instance_map: np.ndarray,
    instances: list[dict],
    silver_lookup: dict[str, SilverLabel],
    asset_mgr: AssetManager,
    spec: RenderSpec,
    prompt_id: str,
    full_image: np.ndarray,
    source_round: str,
    source_checkpoint: str,
    counter: int,
) -> tuple[list[VerifierExample], int]:
    """Build satellite_ev or stream_ev examples (multi-variant GT-driven).

    Iterates all ``image:*`` keys in *silver_lookup*, building one
    ``VerifierExample`` per synthetic variant.  ``fragment_hints`` are
    constructed only from ``visible_instance_ids``.
    """
    sample_id = base_key_to_sample_id(bk)
    halo_id = base_key_to_halo_id(bk)
    view_id = bk.view_id

    # Collect all image:* variant records (sorted for determinism)
    variant_keys = sorted(k for k in silver_lookup if k.startswith("image"))
    if not variant_keys:
        # Backward compat: try bare "image" key (legacy single-variant)
        label_rec = silver_lookup.get("image")
        if label_rec is None:
            return [], counter
        variant_keys = ["image"]

    type_label = "satellites" if family == TaskFamily.SATELLITE_EV else "streams"

    ann_hash = _annotation_state_hash(instance_map, instances)
    examples: list[VerifierExample] = []

    for vkey in variant_keys:
        label_rec = silver_lookup[vkey]
        signals = label_rec.signals

        # Parse variant metadata
        variant_id = signals.get("synthetic_variant_id")
        visible_ids = signals.get("visible_instance_ids")
        hidden_ids = signals.get("hidden_instance_ids")

        # Build fragment hints only from visible instances
        fragment_hints: list[dict] | None = None
        if visible_ids is not None and len(visible_ids) > 0:
            hints: list[dict] = []
            for inst in instances:
                if inst["id"] in visible_ids and inst["type"] == type_label:
                    smask = (instance_map == inst["id"]).astype(np.uint8)
                    if smask.sum() == 0:
                        continue
                    hints.append({"rle": mask_to_rle(smask), "instance_id": inst["id"]})
            fragment_hints = hints or None
        elif visible_ids is None:
            # Legacy path: use all instances of this type
            typed_insts = [i for i in instances if i["type"] == type_label]
            if typed_insts:
                hints_legacy: list[dict] = []
                for si in typed_insts:
                    smask = (instance_map == si["id"]).astype(np.uint8)
                    if smask.sum() == 0:
                        continue
                    hints_legacy.append({"rle": mask_to_rle(smask), "instance_id": si["id"]})
                fragment_hints = hints_legacy or None

        state_key = variant_id  # None keeps old filename
        img_path, img_sha1 = asset_mgr.ensure_ev_image(
            sample_id, spec.input_variant, spec,
            full_image,
            fragment_hints=fragment_hints,
            state_key=state_key,
        )

        frag_hints_tuple = None
        if fragment_hints:
            frag_hints_tuple = tuple(
                {k: v for k, v in h.items()} for h in fragment_hints
            )
        sorted_frag_hash = None
        if fragment_hints:
            sorted_frag_hash = hashlib.sha256(
                json.dumps(fragment_hints, sort_keys=True).encode()
            ).hexdigest()

        rev_hash = compute_revision_hash_ev(
            task_family=family,
            sample_id=sample_id,
            task_revision_id="rev_00",
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            review_image_sha1=img_sha1,
            annotation_state_hash=ann_hash,
            fragment_hints_hash=sorted_frag_hash,
            synthetic_variant_id=variant_id,
        )

        hidden_ids_tuple = tuple(hidden_ids) if hidden_ids is not None else None
        visible_ids_tuple = tuple(visible_ids) if visible_ids is not None else None

        ex = VerifierExample(
            example_id=_new_example_id(family, counter),
            task_family=family,
            sample_id=sample_id,
            halo_id=halo_id,
            view_id=view_id,
            task_revision_id="rev_00",
            asset_refs=EvAssetRefs(
                review_image_path=img_path,
                review_image_sha1=img_sha1,
                annotation_state_hash=ann_hash,
                fragment_hints=frag_hints_tuple,
                synthetic_variant_id=variant_id,
                hidden_instance_ids=hidden_ids_tuple,
                visible_instance_ids=visible_ids_tuple,
            ),
            render_spec_id=spec.spec_id,
            fixed_prompt_id=prompt_id,
            decision_label=label_rec.decision_label,
            label_source=LabelSource.SILVER,
            source_round=source_round,
            source_checkpoint=source_checkpoint,
            revision_hash=rev_hash,
        )
        examples.append(ex)
        counter += 1

    return examples, counter


# ---------------------------------------------------------------------------
#  Serialization helpers
# ---------------------------------------------------------------------------

def _example_to_dict(ex: VerifierExample) -> dict[str, Any]:
    """Convert VerifierExample to a JSON-serialisable dict."""
    d: dict[str, Any] = {
        "example_id": ex.example_id,
        "task_family": ex.task_family.value,
        "sample_id": ex.sample_id,
        "halo_id": ex.halo_id,
        "view_id": ex.view_id,
        "task_revision_id": ex.task_revision_id,
        "render_spec_id": ex.render_spec_id,
        "fixed_prompt_id": ex.fixed_prompt_id,
        "decision_label": ex.decision_label,
        "label_source": ex.label_source.value,
        "source_round": ex.source_round,
        "source_checkpoint": ex.source_checkpoint,
        "revision_hash": ex.revision_hash,
    }
    if ex.parent_example_id is not None:
        d["parent_example_id"] = ex.parent_example_id
    if ex.uncertainty_flag is not None:
        d["uncertainty_flag"] = ex.uncertainty_flag
    if ex.reason_code is not None:
        d["reason_code"] = ex.reason_code

    arefs = ex.asset_refs
    if isinstance(arefs, SatMvAssetRefs):
        d["asset_refs"] = {
            "bare_context_path": arefs.bare_context_path,
            "bare_context_sha1": arefs.bare_context_sha1,
            "crop_spec": {
                "center_x": arefs.crop_spec.center_x,
                "center_y": arefs.crop_spec.center_y,
                "size": arefs.crop_spec.size,
                "pad_mode": arefs.crop_spec.pad_mode,
            },
            "candidate_bbox_xywh": list(arefs.candidate_bbox_xywh),
            "candidate_rle": arefs.candidate_rle,
            "candidate_id": arefs.candidate_id,
            "authoritative_instance_id": arefs.authoritative_instance_id,
            "candidate_source": arefs.candidate_source,
        }
    elif isinstance(arefs, EvAssetRefs):
        ad: dict[str, Any] = {
            "review_image_path": arefs.review_image_path,
            "review_image_sha1": arefs.review_image_sha1,
            "annotation_state_hash": arefs.annotation_state_hash,
        }
        if arefs.fragment_hints is not None:
            ad["fragment_hints"] = list(arefs.fragment_hints)
        if arefs.synthetic_variant_id is not None:
            ad["synthetic_variant_id"] = arefs.synthetic_variant_id
        if arefs.hidden_instance_ids is not None:
            ad["hidden_instance_ids"] = list(arefs.hidden_instance_ids)
        if arefs.visible_instance_ids is not None:
            ad["visible_instance_ids"] = list(arefs.visible_instance_ids)
        d["asset_refs"] = ad

    return d


def write_examples(examples: list[VerifierExample], path: Path) -> None:
    """Write examples to JSONL."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        for ex in examples:
            fh.write(json.dumps(_example_to_dict(ex), sort_keys=True) + "\n")


def read_examples_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read business JSONL as raw dicts (no deserialization to dataclass)."""
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            records.append(json.loads(line))
    return records
