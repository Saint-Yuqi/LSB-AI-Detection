#!/usr/bin/env python3
"""
One-shot migration: legacy satellite override YAML -> Shadow GT explicit edits.

This script translates ``configs/archive/sam3_satellite_overrides_legacy.yaml`` into
concrete ``adopt_raw_candidate`` and ``delete_authoritative_instance`` calls
against a Shadow GT root that was produced by a pure SAM3 evaluate run.

force_keep:
    - Select the shadow raw candidate by ``candidate_rle_sha1`` (preferred),
      falling back to ``candidate_id`` / ``raw_index`` if a sha is not given.
    - If the entry's sha is listed in ``inject_from_json`` for the same base
      key, adopt from that external JSON path; otherwise adopt from the
      shadow image's native ``sam3_predictions_raw.json``.
    - Full provenance is recorded in ``instances.json`` and
      ``manual_corrections.jsonl`` by ``adopt_raw_candidate``.

force_drop:
    - Resolve the targeted surviving shadow instance to a concrete
      ``instance_id`` by reading the shadow
      ``sam3_predictions_post.json`` and replaying the configured final
      satellite sort policy against the shadow ``streams_instance_map.npy``.
    - Call ``delete_authoritative_instance(instance_id=...)`` on the shadow GT.

The legacy YAML is treated as input-only archived review history; it is never
consumed by inference runtime.

Usage:
    conda run --no-capture-output -n sam3 python scripts/review/migrate_satellite_overrides.py \\
        --config configs/unified_data_prep.yaml \\
        --shadow-root scratch/gt_shadow
    conda run --no-capture-output -n sam3 python scripts/review/migrate_satellite_overrides.py \\
        --config ... --shadow-root ... --overrides-yaml configs/archive/sam3_satellite_overrides_legacy.yaml --dry-run
    conda run --no-capture-output -n sam3 python scripts/review/migrate_satellite_overrides.py \\
        --config ... --shadow-root ... --base-keys 00066_fo,00049_eo
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.artifacts import load_predictions_json
from src.pipelines.unified_dataset.config import load_config
from src.review.authoritative_gt import (
    adopt_raw_candidate,
    delete_authoritative_instance,
    parse_base_key,
    resolve_authoritative_gt_dir,
)
from src.utils.logger import setup_logger
from src.utils.runtime_env import assert_expected_conda_env


DEFAULT_SORT_POLICY: tuple[str, ...] = ("area_desc", "centroid_x_asc", "centroid_y_asc")
DEFAULT_OVERRIDES_YAML = (
    PROJECT_ROOT / "configs" / "archive" / "sam3_satellite_overrides_legacy.yaml"
)


def _override_config_with_shadow_root(config: dict[str, Any], shadow_root: Path) -> dict[str, Any]:
    """Return a shallow-copied config that resolves GT/render paths under ``shadow_root``."""
    patched = dict(config)
    paths_cfg = dict(patched.get("paths", {}))
    paths_cfg["output_root"] = str(shadow_root)
    patched["paths"] = paths_cfg
    return patched


def _resolve_inject_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"inject_from_json source missing: {path}")
    return path


def _sort_key_for_policy(policy: list[str]):
    def _key(mask: dict[str, Any]) -> tuple[Any, ...]:
        parts: list[Any] = []
        for step in policy:
            if step == "area_desc":
                parts.append(-int(mask.get("area", 0)))
            elif step == "area_asc":
                parts.append(int(mask.get("area", 0)))
            elif step == "centroid_x_asc":
                parts.append(float(mask.get("centroid_x", 0.0)))
            elif step == "centroid_y_asc":
                parts.append(float(mask.get("centroid_y", 0.0)))
            else:
                raise ValueError(f"Unknown satellite sort policy step: {step!r}")
        return tuple(parts)

    return _key


def _build_force_drop_resolver(
    shadow_gt_dir: Path,
    sort_policy: list[str],
) -> dict[str, int]:
    """Map shadow post satellite identities -> final instance_id.

    Uses the same ordering convention as ``merge_instances``: kept satellites
    are indexed in ``_sort_key_for_policy`` order after the max stream id.
    """
    post_path = shadow_gt_dir / "sam3_predictions_post.json"
    streams_path = shadow_gt_dir / "streams_instance_map.npy"
    if not post_path.exists():
        raise FileNotFoundError(f"Shadow post predictions missing: {post_path}")
    if not streams_path.exists():
        raise FileNotFoundError(f"Shadow streams_instance_map missing: {streams_path}")

    doc = json.loads(post_path.read_text())
    predictions = doc.get("predictions", []) if isinstance(doc, dict) else doc
    sat_entries: list[dict[str, Any]] = [
        p for p in predictions if p.get("type_label") == "satellites"
    ]

    streams_map = np.load(streams_path)
    max_stream_id = int(streams_map.max()) if streams_map.size else 0

    sort_fn = _sort_key_for_policy(sort_policy)
    sorted_sats = sorted(sat_entries, key=sort_fn)

    resolver: dict[str, int] = {}
    for i, pred in enumerate(sorted_sats):
        instance_id = max_stream_id + i + 1
        sha = pred.get("candidate_rle_sha1")
        cid = pred.get("candidate_id")
        if sha:
            resolver[f"sha:{sha}"] = instance_id
        if cid:
            resolver[f"cid:{cid}"] = instance_id
    return resolver


def _resolve_force_drop_instance_id(
    shadow_gt_dir: Path,
    sort_policy: list[str],
    entry: dict[str, Any],
) -> int:
    resolver = _build_force_drop_resolver(shadow_gt_dir, sort_policy)
    sha = entry.get("candidate_rle_sha1")
    cid = entry.get("candidate_id")
    if sha and f"sha:{sha}" in resolver:
        return resolver[f"sha:{sha}"]
    if cid and f"cid:{cid}" in resolver:
        return resolver[f"cid:{cid}"]
    raise ValueError(
        f"force_drop could not resolve to a surviving shadow instance: "
        f"candidate_id={cid!r}, candidate_rle_sha1={sha!r}"
    )


def _apply_force_keep(
    *,
    shadow_config: dict[str, Any],
    base_key_str: str,
    shadow_gt_dir: Path,
    entry: dict[str, Any],
    inject_lookup: dict[str, dict[str, Any]],
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    sha = entry.get("candidate_rle_sha1")
    cid = entry.get("candidate_id")
    raw_index = entry.get("raw_index")
    note = entry.get("note")

    inject_entry = inject_lookup.get(sha) if sha else None
    if inject_entry is not None:
        source_json = _resolve_inject_path(inject_entry["path"])
        logger.info(
            "[force_keep] %s sha=%s -> adopt from external %s",
            base_key_str, sha, source_json,
        )
    else:
        source_json = shadow_gt_dir / "sam3_predictions_raw.json"
        if not source_json.exists():
            raise FileNotFoundError(
                f"Shadow raw predictions missing for {base_key_str}: {source_json}"
            )
        logger.info(
            "[force_keep] %s sha=%s cid=%s -> adopt from native shadow raw %s",
            base_key_str, sha, cid, source_json,
        )

    if dry_run:
        return None

    return adopt_raw_candidate(
        shadow_config,
        key=parse_base_key(base_key_str),
        source_json=source_json,
        candidate_id=cid,
        raw_index=raw_index,
        candidate_rle_sha1=sha,
        manual_note=note,
    )


def _apply_force_drop(
    *,
    shadow_config: dict[str, Any],
    base_key_str: str,
    shadow_gt_dir: Path,
    sort_policy: list[str],
    entry: dict[str, Any],
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, Any] | None:
    instance_id = _resolve_force_drop_instance_id(shadow_gt_dir, sort_policy, entry)
    logger.info(
        "[force_drop] %s sha=%s cid=%s -> instance_id=%d",
        base_key_str,
        entry.get("candidate_rle_sha1"),
        entry.get("candidate_id"),
        instance_id,
    )
    if dry_run:
        return None
    return delete_authoritative_instance(
        shadow_config,
        key=parse_base_key(base_key_str),
        instance_id=instance_id,
        manual_note=entry.get("note"),
    )


def _sort_policy_from_config(config: dict[str, Any]) -> list[str]:
    return list(
        config.get("satellites", {}).get("satellite_sort_policy") or DEFAULT_SORT_POLICY
    )


def migrate(
    *,
    config: dict[str, Any],
    shadow_root: Path,
    overrides: dict[str, dict[str, Any]],
    base_key_filter: set[str] | None,
    dry_run: bool,
    logger: logging.Logger,
) -> dict[str, Any]:
    shadow_config = _override_config_with_shadow_root(config, shadow_root)
    sort_policy = _sort_policy_from_config(config)

    summary = {
        "n_base_keys": 0,
        "force_keep": {"ok": 0, "failed": 0, "skipped_dry_run": 0},
        "force_drop": {"ok": 0, "failed": 0, "skipped_dry_run": 0},
        "failures": [],
    }

    for base_key_str, rules in sorted(overrides.items()):
        if base_key_filter is not None and base_key_str not in base_key_filter:
            continue
        summary["n_base_keys"] += 1

        shadow_gt_dir, _, _ = resolve_authoritative_gt_dir(
            shadow_config, parse_base_key(base_key_str),
        )
        inject_entries = rules.get("inject_from_json") or []
        inject_lookup: dict[str, dict[str, Any]] = {}
        for inj in inject_entries:
            sha = inj.get("candidate_rle_sha1")
            if sha:
                inject_lookup.setdefault(sha, inj)

        for entry in rules.get("force_keep") or []:
            try:
                _apply_force_keep(
                    shadow_config=shadow_config,
                    base_key_str=base_key_str,
                    shadow_gt_dir=shadow_gt_dir,
                    entry=entry,
                    inject_lookup=inject_lookup,
                    dry_run=dry_run,
                    logger=logger,
                )
                summary["force_keep"]["skipped_dry_run" if dry_run else "ok"] += 1
            except Exception as exc:  # noqa: BLE001 - summarize per-entry failures
                logger.exception(
                    "[force_keep] %s failed for entry=%s", base_key_str, entry,
                )
                summary["force_keep"]["failed"] += 1
                summary["failures"].append({
                    "base_key": base_key_str,
                    "rule": "force_keep",
                    "entry": entry,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

        for entry in rules.get("force_drop") or []:
            try:
                _apply_force_drop(
                    shadow_config=shadow_config,
                    base_key_str=base_key_str,
                    shadow_gt_dir=shadow_gt_dir,
                    sort_policy=sort_policy,
                    entry=entry,
                    dry_run=dry_run,
                    logger=logger,
                )
                summary["force_drop"]["skipped_dry_run" if dry_run else "ok"] += 1
            except Exception as exc:  # noqa: BLE001 - summarize per-entry failures
                logger.exception(
                    "[force_drop] %s failed for entry=%s", base_key_str, entry,
                )
                summary["force_drop"]["failed"] += 1
                summary["failures"].append({
                    "base_key": base_key_str,
                    "rule": "force_drop",
                    "entry": entry,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                })

    return summary


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", type=Path, required=True,
                    help="Dataset config YAML (used only for path resolution semantics).")
    ap.add_argument("--shadow-root", type=Path, required=True,
                    help="Shadow root containing gt_canonical/current with pure SAM3 artifacts.")
    ap.add_argument("--overrides-yaml", type=Path, default=DEFAULT_OVERRIDES_YAML,
                    help="Archived sam3_satellite_overrides legacy YAML to migrate.")
    ap.add_argument("--base-keys", type=str, default=None,
                    help="Comma-separated base keys to restrict migration to.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Resolve and log every action without calling adopt/delete.")
    return ap.parse_args()


def main() -> int:
    assert_expected_conda_env(context="scripts/review/migrate_satellite_overrides.py")
    args = parse_args()
    args.shadow_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("migrate_satellite_overrides", args.shadow_root / "_migration_logs")

    config = load_config(args.config)
    overrides_doc = yaml.safe_load(args.overrides_yaml.read_text()) or {}
    overrides = overrides_doc.get("base_keys", {})
    if not isinstance(overrides, dict):
        raise ValueError("overrides YAML must contain a mapping at 'base_keys'")

    base_key_filter: set[str] | None = None
    if args.base_keys:
        base_key_filter = {b.strip() for b in args.base_keys.split(",") if b.strip()}

    summary = migrate(
        config=config,
        shadow_root=args.shadow_root,
        overrides=overrides,
        base_key_filter=base_key_filter,
        dry_run=args.dry_run,
        logger=logger,
    )

    logger.info("Migration summary: %s", json.dumps(summary, indent=2, default=str))
    summary_path = args.shadow_root / "_migration_logs" / "migration_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    logger.info("Summary written to %s", summary_path)

    if summary["force_keep"]["failed"] or summary["force_drop"]["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
