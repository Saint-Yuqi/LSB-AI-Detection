"""
Authoritative GT editing helpers for manual satellite adoption/deletion.

This module intentionally keeps ``instances.json`` as a plain list.  Any
monotonic-ID recovery state is derived from the append-only
``manual_corrections.jsonl`` audit log.
"""
from __future__ import annotations

import json
import os
import socket
import tempfile
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from PIL import Image

from src.pipelines.unified_dataset.keys import BaseKey
from src.pipelines.unified_dataset.paths import PathResolver
from src.postprocess.satellite_prior_filter import load_filter_cfg
from src.utils.coco_utils import decode_rle, get_bbox_from_mask, mask_to_rle


PROVENANCE_EXPORT_FIELDS: tuple[str, ...] = (
    "label_source",
    "human_fix",
    "human_fix_kind",
    "source_candidate_id",
    "source_raw_index",
    "source_candidate_rle_sha1",
    "source_prediction_path",
    "final_mask_rle_sha1",
    "manual_note",
)

_LOCK_DIRNAME = ".update.lock"
_LOCK_OWNER_FILENAME = "owner.json"
_AUDIT_FILENAME = "manual_corrections.jsonl"
_ID_HEX_LEN = 16
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class AuthoritativeGTError(RuntimeError):
    """Base class for manual authoritative GT edit failures."""


class LockAcquisitionError(AuthoritativeGTError):
    """Raised when a per-image pessimistic lock cannot be acquired."""


class DuplicateAdoptionError(AuthoritativeGTError):
    """Raised when the same source candidate is already present in GT."""


class MissingAuthoritativeInstanceError(AuthoritativeGTError):
    """Raised when a requested authoritative instance is missing."""


class AlreadyDeletedError(AuthoritativeGTError):
    """Raised when a delete request targets an already-deleted instance."""


class UnsupportedInstanceTypeError(AuthoritativeGTError):
    """Raised when an operation targets a non-satellite instance."""


class EmptyCropError(AuthoritativeGTError):
    """Raised when cropping a source candidate to GT background removes all pixels."""


class AreaTooSmallError(AuthoritativeGTError):
    """Raised when a cropped mask fails the minimum-area guard."""


def parse_base_key(value: str) -> BaseKey:
    """Parse ``00011_eo``-style strings into ``BaseKey``."""
    try:
        gid_str, view_id = value.split("_", 1)
    except ValueError as exc:
        raise ValueError(f"Invalid BaseKey {value!r}; expected '00011_eo'") from exc
    if not gid_str.isdigit():
        raise ValueError(f"Invalid BaseKey {value!r}; galaxy prefix must be numeric")
    return BaseKey(galaxy_id=int(gid_str), view_id=view_id)


def rle_sha1(rle: dict[str, Any]) -> str:
    """Return a short SHA1 derived directly from COCO RLE payload."""
    counts = rle.get("counts")
    if isinstance(counts, str):
        blob = counts.encode("ascii")
    else:
        blob = json.dumps(rle, sort_keys=True, separators=(",", ":")).encode("utf-8")
    import hashlib

    return hashlib.sha1(blob).hexdigest()[:_ID_HEX_LEN]


def extract_annotation_provenance(inst: dict[str, Any]) -> dict[str, Any]:
    """Return sparse COCO annotation provenance fields for one GT instance."""
    prov = inst.get("provenance")
    if not isinstance(prov, dict):
        return {}
    out: dict[str, Any] = {}
    for key in PROVENANCE_EXPORT_FIELDS:
        if key in prov and prov[key] is not None:
            out[key] = prov[key]
    return out


def rebuild_id_map(instances: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Rebuild type-aware id_map without mutating surviving instance IDs.

    Tolerates both legacy (``streams``/``satellites``) and tidal_v1
    (``tidal_features``/``satellites``/``inner_galaxy``) type labels.
    Always emits a ``satellites`` bucket even when empty for backward
    compatibility with consumers that index it directly.
    """
    id_map: dict[str, dict[str, int]] = {}

    stream_ids = sorted(inst["id"] for inst in instances if inst.get("type") == "streams")
    id_map["streams"] = {str(inst_id): inst_id for inst_id in stream_ids}

    type_to_ids: dict[str, list[int]] = {}
    for inst in instances:
        type_label = inst.get("type") or inst.get("type_label", "unknown")
        if type_label == "streams":
            continue
        type_to_ids.setdefault(type_label, []).append(int(inst["id"]))

    for type_label, ids in type_to_ids.items():
        id_map[type_label] = {
            str(idx): inst_id for idx, inst_id in enumerate(sorted(ids))
        }

    if "satellites" not in id_map:
        id_map["satellites"] = {}

    return id_map


def resolve_authoritative_gt_dir(
    config: dict[str, Any],
    key: BaseKey,
    *,
    dataset: str | None = None,
    condition: str | None = None,
) -> tuple[Path, str, str]:
    """Resolve the authoritative GT directory for one BaseKey."""
    if dataset is not None:
        config = dict(config)
        config["dataset_name"] = dataset

    resolver = PathResolver(config)
    dataset_name = resolver.dataset_name

    if dataset_name == "dr1":
        return resolver.get_gt_dir(key), dataset_name, "clean"

    effective_condition = condition or resolver.default_condition
    if resolver.get_label_mode(effective_condition) != "authoritative":
        raise ValueError(
            f"Condition {effective_condition!r} is not authoritative for dataset {dataset_name!r}"
        )
    return (
        resolver.get_pseudo_gt_dir(key, dataset=dataset_name, condition=effective_condition),
        dataset_name,
        effective_condition,
    )


def load_instances(path: Path) -> list[dict[str, Any]]:
    """Read ``instances.json`` and assert the legacy list contract."""
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"{path} must remain a JSON list, got {type(payload).__name__}")
    return payload


def load_manual_corrections(path: Path) -> list[dict[str, Any]]:
    """Load an append-only corrections log."""
    if not path.exists():
        return []

    entries: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            text = line.strip()
            if not text:
                continue
            try:
                entry = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Malformed JSONL in {path} at line {lineno}") from exc
            if not isinstance(entry, dict):
                raise ValueError(f"{path} line {lineno} must be a JSON object")
            entries.append(entry)
    return entries


def historical_reserved_ids(entries: list[dict[str, Any]]) -> set[int]:
    """Return IDs reserved by prepared/committed manual edits."""
    reserved: set[int] = set()
    for entry in entries:
        if entry.get("status") not in {"prepared", "committed"}:
            continue
        for key in ("assigned_instance_id", "deleted_instance_id"):
            value = entry.get(key)
            if value is not None:
                reserved.add(int(value))
    return reserved


def max_reserved_id(entries: list[dict[str, Any]]) -> int:
    reserved = historical_reserved_ids(entries)
    return max(reserved) if reserved else 0


def append_jsonl_entry(path: Path, entry: dict[str, Any]) -> None:
    """Append one JSONL record and fsync it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, sort_keys=True) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_json_atomic(path: Path, payload: Any) -> None:
    """Write JSON atomically via temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def write_png_atomic(path: Path, array: np.ndarray) -> None:
    """Write a PNG atomically via temp file + ``os.replace``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.stem}.",
        suffix=f"{path.suffix}.tmp",
    )
    os.close(fd)
    tmp_path = Path(tmp_name)
    try:
        Image.fromarray(array.astype(np.uint8)).save(tmp_path, format="PNG")
        os.replace(tmp_path, path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise


def save_authoritative_state(
    gt_dir: Path,
    instance_map: np.ndarray,
    instances: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Persist the authoritative GT trio and return the rebuilt id_map."""
    id_map = rebuild_id_map(instances)
    write_png_atomic(gt_dir / "instance_map_uint8.png", instance_map.astype(np.uint8))
    write_json_atomic(gt_dir / "instances.json", instances)
    write_json_atomic(gt_dir / "id_map.json", id_map)
    return id_map


def derive_candidate_entries(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return in-memory enriched prediction records without mutating the source doc."""
    type_counts: dict[str, int] = {}
    enriched: list[dict[str, Any]] = []
    prefix_map = {"satellites": "sat", "streams": "stream"}

    for raw_index, pred in enumerate(predictions):
        type_label = pred.get("type_label", "unknown")
        type_index = type_counts.get(type_label, 0)
        prefix = prefix_map.get(type_label, type_label.replace(" ", "_")[:8] or "cand")
        candidate_id = pred.get("candidate_id", f"{prefix}_{type_index:04d}")
        rle = pred.get("rle")
        if not isinstance(rle, dict):
            raise ValueError("Prediction entries must contain an RLE payload")

        enriched.append(
            {
                **pred,
                "candidate_id": candidate_id,
                "raw_index": raw_index,
                "candidate_rle_sha1": rle_sha1(rle),
            }
        )
        type_counts[type_label] = type_index + 1

    return enriched


def select_prediction_candidate(
    predictions: list[dict[str, Any]],
    *,
    candidate_id: str | None = None,
    raw_index: int | None = None,
    candidate_rle_sha1: str | None = None,
) -> dict[str, Any]:
    """Select one in-memory enriched prediction by id, raw index, or RLE SHA1.

    All provided selectors must agree with the chosen record. ``candidate_rle_sha1``
    is typically the most stable identity across re-inference runs (it survives
    checkpoint-specific candidate ordering changes), so it is resolved first and
    then cross-checked against the other selectors when given.
    """
    if candidate_id is None and raw_index is None and candidate_rle_sha1 is None:
        raise ValueError("Either candidate_id, raw_index, or candidate_rle_sha1 is required")

    matches = predictions
    if candidate_rle_sha1 is not None:
        matches = [pred for pred in matches if pred.get("candidate_rle_sha1") == candidate_rle_sha1]
    if candidate_id is not None:
        matches = [pred for pred in matches if pred.get("candidate_id") == candidate_id]
    if raw_index is not None:
        matches = [pred for pred in matches if int(pred.get("raw_index", -1)) == int(raw_index)]

    if not matches:
        selector = (
            f"candidate_id={candidate_id!r}, raw_index={raw_index!r}, "
            f"candidate_rle_sha1={candidate_rle_sha1!r}"
        )
        raise ValueError(f"No prediction matches {selector}")
    if len(matches) > 1:
        selector = (
            f"candidate_id={candidate_id!r}, raw_index={raw_index!r}, "
            f"candidate_rle_sha1={candidate_rle_sha1!r}"
        )
        raise ValueError(f"Prediction selector is ambiguous: {selector}")
    return matches[0]


def operation_timestamp() -> str:
    """UTC ISO timestamp for logs and provenance."""
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def directory_update_lock(
    gt_dir: Path,
    *,
    operation: str,
    retries: int = 8,
    initial_delay_s: float = 0.25,
    max_delay_s: float = 4.0,
) -> Iterator[dict[str, Any]]:
    """Acquire a cross-node pessimistic lock via atomic directory creation."""
    lock_dir = gt_dir / _LOCK_DIRNAME
    owner_meta = {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "operation": operation,
        "acquired_at": operation_timestamp(),
    }

    delay_s = max(initial_delay_s, 0.0)
    last_owner: dict[str, Any] | None = None

    for attempt in range(retries + 1):
        try:
            os.mkdir(lock_dir)
            break
        except FileExistsError:
            owner_path = lock_dir / _LOCK_OWNER_FILENAME
            if owner_path.exists():
                try:
                    last_owner = json.loads(owner_path.read_text())
                except Exception:
                    last_owner = None

            if attempt >= retries:
                suffix = f"; owner={last_owner}" if last_owner is not None else ""
                raise LockAcquisitionError(
                    f"Could not acquire {lock_dir} after {retries + 1} attempts{suffix}"
                )
            time.sleep(min(delay_s, max_delay_s))
            delay_s = min(max_delay_s, delay_s * 2 if delay_s > 0 else max_delay_s)
    else:
        raise LockAcquisitionError(f"Could not acquire {lock_dir}")

    owner_path = lock_dir / _LOCK_OWNER_FILENAME
    try:
        owner_path.write_text(json.dumps(owner_meta, indent=2))
        yield owner_meta
    finally:
        try:
            if owner_path.exists():
                owner_path.unlink()
        finally:
            os.rmdir(lock_dir)


def default_min_area_px(config: dict[str, Any]) -> int:
    """Load the default area guard from the configured satellite prior stats."""
    prior_cfg = config.get("satellites", {}).get("prior", {})
    stats_json = Path(prior_cfg.get("stats_json", "outputs/mask_stats/mask_stats_summary.json"))
    if not stats_json.is_absolute():
        stats_json = _PROJECT_ROOT / stats_json
    return int(load_filter_cfg(stats_json)["area_min"])


def _surviving_duplicate_exists(
    instances: list[dict[str, Any]],
    *,
    source_prediction_path: str,
    source_candidate_id: str,
    source_candidate_rle_sha1: str,
) -> bool:
    for inst in instances:
        if inst.get("type") != "satellites":
            continue
        prov = inst.get("provenance")
        if not isinstance(prov, dict):
            continue
        if (
            prov.get("source_prediction_path") == source_prediction_path
            and prov.get("source_candidate_id") == source_candidate_id
            and prov.get("source_candidate_rle_sha1") == source_candidate_rle_sha1
        ):
            return True
    return False


def _manual_note_with_warning(note: str | None, warning: str | None) -> str | None:
    if not warning:
        return note
    if not note:
        return warning
    return f"{note} | {warning}"


def _bbox_fill_warning(mask: np.ndarray) -> tuple[float, str | None]:
    bbox = get_bbox_from_mask(mask.astype(np.uint8))
    bbox_area = int(bbox[2] * bbox[3])
    area = int(mask.sum())
    bbox_fill = (float(area) / float(bbox_area)) if bbox_area > 0 else 0.0
    if bbox_area > 0 and bbox_fill < 0.10:
        warning = f"warning:bbox_fill_lt_0.10 ({bbox_fill:.4f})"
        return bbox_fill, warning
    return bbox_fill, None


def _find_instance(
    instances: list[dict[str, Any]],
    *,
    instance_id: int | None = None,
    source_candidate_id: str | None = None,
) -> dict[str, Any] | None:
    if instance_id is not None:
        for inst in instances:
            if int(inst["id"]) == int(instance_id):
                return inst
        return None

    if source_candidate_id is None:
        return None

    matched = [
        inst
        for inst in instances
        if isinstance(inst.get("provenance"), dict)
        and inst["provenance"].get("source_candidate_id") == source_candidate_id
    ]
    if len(matched) > 1:
        raise ValueError(f"Multiple surviving instances share source_candidate_id={source_candidate_id!r}")
    return matched[0] if matched else None


def _instance_was_already_deleted(
    audit_entries: list[dict[str, Any]],
    *,
    instance_id: int | None = None,
    source_candidate_id: str | None = None,
) -> bool:
    for entry in audit_entries:
        if entry.get("operation") != "delete_authoritative_satellite":
            continue
        if entry.get("status") != "committed":
            continue
        if instance_id is not None and int(entry.get("deleted_instance_id", -1)) == int(instance_id):
            return True
        prov = entry.get("deleted_instance_provenance") or {}
        if source_candidate_id is not None and prov.get("source_candidate_id") == source_candidate_id:
            return True
    return False


def adopt_raw_candidate(
    config: dict[str, Any],
    *,
    key: BaseKey,
    dataset: str | None = None,
    condition: str | None = None,
    source_json: Path | None = None,
    candidate_id: str | None = None,
    raw_index: int | None = None,
    candidate_rle_sha1: str | None = None,
    min_area_px: int | None = None,
    manual_note: str | None = None,
    lock_retries: int = 8,
) -> dict[str, Any]:
    """Adopt one raw satellite candidate into authoritative GT."""
    gt_dir, dataset_name, effective_condition = resolve_authoritative_gt_dir(
        config,
        key,
        dataset=dataset,
        condition=condition,
    )
    source_path = (source_json or (gt_dir / "sam3_predictions_raw.json")).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source prediction JSON not found: {source_path}")

    min_area = int(min_area_px) if min_area_px is not None else default_min_area_px(config)
    operation_id = uuid.uuid4().hex
    audit_path = gt_dir / _AUDIT_FILENAME

    prepared_entry: dict[str, Any] | None = None
    committed = False
    resolved_source_path = str(source_path)
    new_id = -1

    try:
        with directory_update_lock(
            gt_dir,
            operation=f"adopt_raw:{key}",
            retries=lock_retries,
        ):
            audit_entries = load_manual_corrections(audit_path)
            instances_path = gt_dir / "instances.json"
            instance_map_path = gt_dir / "instance_map_uint8.png"
            if not instance_map_path.exists():
                raise FileNotFoundError(f"Authoritative instance_map missing: {instance_map_path}")

            instances = load_instances(instances_path)
            instance_map = np.array(Image.open(instance_map_path))

            raw_doc = json.loads(source_path.read_text())
            predictions = raw_doc["predictions"] if isinstance(raw_doc, dict) else raw_doc
            enriched_predictions = derive_candidate_entries(predictions)
            selected = select_prediction_candidate(
                enriched_predictions,
                candidate_id=candidate_id,
                raw_index=raw_index,
                candidate_rle_sha1=candidate_rle_sha1,
            )

            if selected.get("type_label") != "satellites":
                raise UnsupportedInstanceTypeError(
                    f"Only satellite candidates can be adopted, got type={selected.get('type_label')!r}"
                )

            source_candidate_id = str(selected["candidate_id"])
            source_candidate_sha = str(selected["candidate_rle_sha1"])
            source_raw_index = int(selected["raw_index"])

            if _surviving_duplicate_exists(
                instances,
                source_prediction_path=resolved_source_path,
                source_candidate_id=source_candidate_id,
                source_candidate_rle_sha1=source_candidate_sha,
            ):
                raise DuplicateAdoptionError(
                    "Source candidate is already present in authoritative GT: "
                    f"{resolved_source_path}::{source_candidate_id}::{source_candidate_sha}"
                )

            current_ids = [int(inst["id"]) for inst in instances]
            current_max_id = max(current_ids) if current_ids else 0
            historical_max_id = max_reserved_id(audit_entries)
            if current_max_id >= 255 or historical_max_id >= 255:
                raise OverflowError(
                    f"Cannot allocate a new uint8 instance id (current_max={current_max_id}, historical_max={historical_max_id})"
                )
            new_id = max(current_max_id, historical_max_id) + 1

            source_mask = decode_rle(selected["rle"]).astype(bool)
            cropped_mask = source_mask & (instance_map == 0)
            cropped_area_px = int(cropped_mask.sum())
            if cropped_area_px == 0:
                raise EmptyCropError(
                    f"Candidate {source_candidate_id} becomes empty after background cropping"
                )
            if cropped_area_px < min_area:
                raise AreaTooSmallError(
                    f"Cropped candidate area {cropped_area_px} is below min_area_px={min_area}"
                )

            bbox_fill, warning = _bbox_fill_warning(cropped_mask.astype(np.uint8))
            final_manual_note = _manual_note_with_warning(manual_note, warning)
            final_rle = mask_to_rle(cropped_mask.astype(np.uint8))
            provenance = {
                "label_source": "manual_adopted",
                "human_fix": True,
                "human_fix_kind": "adopt_from_raw",
                "source_candidate_id": source_candidate_id,
                "source_raw_index": source_raw_index,
                "source_candidate_rle_sha1": source_candidate_sha,
                "source_prediction_path": resolved_source_path,
                "final_mask_rle_sha1": rle_sha1(final_rle),
                "manual_note": final_manual_note,
                "operation_id": operation_id,
                "updated_at": operation_timestamp(),
            }

            prepared_entry = {
                "operation_id": operation_id,
                "status": "prepared",
                "timestamp": operation_timestamp(),
                "operation": "adopt_raw_satellite",
                "dataset": dataset_name,
                "condition": effective_condition,
                "base_key": str(key),
                "source_prediction_path": resolved_source_path,
                "source_candidate_id": source_candidate_id,
                "source_raw_index": source_raw_index,
                "source_candidate_rle_sha1": source_candidate_sha,
                "assigned_instance_id": new_id,
                "cropped_area_px": cropped_area_px,
                "bbox_fill": bbox_fill,
                "final_mask_rle_sha1": provenance["final_mask_rle_sha1"],
                "manual_note": final_manual_note,
            }
            append_jsonl_entry(audit_path, prepared_entry)

            new_map = instance_map.copy()
            new_map[cropped_mask] = new_id
            new_instances = list(instances) + [{"id": new_id, "type": "satellites", "provenance": provenance}]
            save_authoritative_state(gt_dir, new_map, new_instances)

            committed_entry = {**prepared_entry, "status": "committed", "timestamp": operation_timestamp()}
            append_jsonl_entry(audit_path, committed_entry)
            committed = True
    except Exception:
        if prepared_entry is not None and not committed:
            try:
                append_jsonl_entry(
                    audit_path,
                    {
                        **prepared_entry,
                        "status": "failed",
                        "timestamp": operation_timestamp(),
                    },
                )
            except Exception:
                pass
        raise

    return {
        "operation_id": operation_id,
        "assigned_instance_id": new_id,
        "base_key": str(key),
        "dataset": dataset_name,
        "condition": effective_condition,
        "source_prediction_path": resolved_source_path,
    }


def delete_authoritative_instance(
    config: dict[str, Any],
    *,
    key: BaseKey,
    dataset: str | None = None,
    condition: str | None = None,
    instance_id: int | None = None,
    source_candidate_id: str | None = None,
    manual_note: str | None = None,
    lock_retries: int = 8,
) -> dict[str, Any]:
    """Delete one surviving authoritative satellite instance."""
    gt_dir, dataset_name, effective_condition = resolve_authoritative_gt_dir(
        config,
        key,
        dataset=dataset,
        condition=condition,
    )
    audit_path = gt_dir / _AUDIT_FILENAME
    operation_id = uuid.uuid4().hex
    prepared_entry: dict[str, Any] | None = None
    committed = False

    target_id = -1

    try:
        with directory_update_lock(
            gt_dir,
            operation=f"delete_instance:{key}",
            retries=lock_retries,
        ):
            audit_entries = load_manual_corrections(audit_path)
            instances = load_instances(gt_dir / "instances.json")
            instance_map = np.array(Image.open(gt_dir / "instance_map_uint8.png"))

            target = _find_instance(
                instances,
                instance_id=instance_id,
                source_candidate_id=source_candidate_id,
            )
            if target is None:
                if _instance_was_already_deleted(
                    audit_entries,
                    instance_id=instance_id,
                    source_candidate_id=source_candidate_id,
                ):
                    raise AlreadyDeletedError(
                        f"Instance already deleted for selector instance_id={instance_id!r}, "
                        f"source_candidate_id={source_candidate_id!r}"
                    )
                raise MissingAuthoritativeInstanceError(
                    f"No surviving authoritative instance matches instance_id={instance_id!r}, "
                    f"source_candidate_id={source_candidate_id!r}"
                )

            if target.get("type") != "satellites":
                raise UnsupportedInstanceTypeError(
                    f"Only satellite instances can be deleted, got type={target.get('type')!r}"
                )

            target_id = int(target["id"])
            deleted_mask = (instance_map == target_id).astype(np.uint8)
            deleted_area_px = int(deleted_mask.sum())
            deleted_bbox = get_bbox_from_mask(deleted_mask)
            deleted_rle = mask_to_rle(deleted_mask)

            prepared_entry = {
                "operation_id": operation_id,
                "status": "prepared",
                "timestamp": operation_timestamp(),
                "operation": "delete_authoritative_satellite",
                "dataset": dataset_name,
                "condition": effective_condition,
                "base_key": str(key),
                "deleted_instance_id": target_id,
                "deleted_mask_rle": deleted_rle,
                "deleted_bbox_xywh": deleted_bbox,
                "deleted_area_px": deleted_area_px,
                "deleted_instance_provenance": target.get("provenance"),
                "manual_note": manual_note,
            }
            append_jsonl_entry(audit_path, prepared_entry)

            new_map = instance_map.copy()
            new_map[new_map == target_id] = 0
            new_instances = [inst for inst in instances if int(inst["id"]) != target_id]
            save_authoritative_state(gt_dir, new_map, new_instances)

            committed_entry = {**prepared_entry, "status": "committed", "timestamp": operation_timestamp()}
            append_jsonl_entry(audit_path, committed_entry)
            committed = True
    except Exception:
        if prepared_entry is not None and not committed:
            try:
                append_jsonl_entry(
                    audit_path,
                    {
                        **prepared_entry,
                        "status": "failed",
                        "timestamp": operation_timestamp(),
                    },
                )
            except Exception:
                pass
        raise

    return {
        "operation_id": operation_id,
        "deleted_instance_id": target_id,
        "base_key": str(key),
        "dataset": dataset_name,
        "condition": effective_condition,
    }
