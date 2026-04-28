"""
Phase 2: Load SB mask -> per-class instance maps (GT canonical).

Legacy path output: gt_canonical/current/{BaseKey}/streams_instance_map.npy
New (tidal_v1) path output:
    gt_canonical_tidal_v1/current/{BaseKey}/tidal_features_instance_map.npy
    gt_canonical_tidal_v1/current/{BaseKey}/instances.json   (FITS-derived rows pre-built)
    gt_canonical_tidal_v1/current/{BaseKey}/manifest.json    (provenance stamps)

The branch is chosen by ``PathResolver.is_new_path()``; legacy configs
keep emitting the 2-class ``streams_instance_map.npy`` exactly as before.
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

import cv2
import numpy as np

from src.data.io import load_fits_gz
from .keys import BaseKey
from .paths import PathResolver
from .fs_utils import sha1_file


def run_gt_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Load SB32 streams mask -> streams_instance_map.npy."""
    if not config.get("gt_phase", {}).get("enabled", True):
        logger.info("GT phase disabled in config — skipping")
        return

    logger.info("=" * 60)
    logger.info("PHASE 2: GT (Streams)")
    logger.info("=" * 60)

    resolver = PathResolver(config)
    sb_threshold = config["data_selection"]["canonical_sb_threshold"]
    target_size = tuple(config["processing"]["target_size"])
    on_new_path = resolver.is_new_path()

    # Snapshot the prior-filter threshold source as a sha1 for manifest
    # provenance on the new path; harmless on the legacy branch.
    prior_threshold_sha1 = _hash_prior_filter_cfg(config)

    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_mask": 0}

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        npy_legacy = gt_dir / "streams_instance_map.npy"
        npy_tidal = gt_dir / "tidal_features_instance_map.npy"
        existing_marker = npy_tidal if on_new_path else npy_legacy

        if existing_marker.exists():
            stats["skipped_exists"] += 1
            continue

        mask_path = resolver.get_mask_path(key, sb_threshold)

        if mask_path is None or not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            stats["skipped_no_mask"] += 1
            continue

        # Writers create the directory tree on demand (F15).
        gt_dir.mkdir(parents=True, exist_ok=True)

        mask_data = load_fits_gz(mask_path)
        instance_map = np.round(mask_data).astype(np.int32)
        instance_map_resized = cv2.resize(
            instance_map, target_size, interpolation=cv2.INTER_NEAREST
        )

        if on_new_path:
            # Remap source instance IDs to a contiguous 1..N local-id space,
            # sorted-stable so the mapping is deterministic across runs.
            source_ids = sorted(int(x) for x in np.unique(instance_map_resized) if x != 0)
            local_map = np.zeros_like(instance_map_resized, dtype=np.int32)
            local_to_source: dict[int, int] = {}
            for local_id, src_id in enumerate(source_ids, start=1):
                local_map[instance_map_resized == src_id] = local_id
                local_to_source[local_id] = src_id

            np.save(npy_tidal, local_map)

            # Pre-build FITS-derived instances.json rows. Inference will
            # read this file, append SAM-derived rows, and re-write through
            # save_per_class_instance_maps (which preserves these entries).
            instances_rows = [
                {
                    "id": local_id,                 # transitional alias (F13)
                    "global_id": local_id,
                    "local_id": local_id,
                    "type": "tidal_features",
                    "type_label": "tidal_features",
                    "map_file": "tidal_features_instance_map.npy",
                    "source": "firebox_sb31.5_fits",
                    "source_instance_id": int(src),
                    "raw_index": None,
                }
                for local_id, src in local_to_source.items()
            ]
            (gt_dir / "instances.json").write_text(json.dumps(instances_rows, indent=2))

            manifest = {
                "gt_path_version": "tidal_v1",
                "sb_threshold_used": sb_threshold,
                "source_mask": str(mask_path),
                "source_mask_sha1": sha1_file(mask_path),
                "n_tidal_features": len(local_to_source),
                "tidal_features_local_to_source": {
                    str(k): v for k, v in local_to_source.items()
                },
                "target_size": list(target_size),
                "prior_filter_thresholds_frozen": True,
                "prior_filter_threshold_source": prior_threshold_sha1,
                "fits_phase_created_at": datetime.now().isoformat(),
            }
            (gt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        else:
            np.save(npy_legacy, instance_map_resized)
            manifest = {
                "sb_threshold_used": sb_threshold,
                "source_mask": str(mask_path),
                "source_mask_sha1": sha1_file(mask_path),
                "max_stream_id": int(instance_map_resized.max()),
                "n_stream_instances": int(len(np.unique(instance_map_resized)) - 1),
                "target_size": list(target_size),
                "created_at": datetime.now().isoformat(),
            }
            (gt_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        stats["processed"] += 1

    logger.info(f"Processed: {stats['processed']}, Skipped (exists): {stats['skipped_exists']}, Skipped (no mask): {stats['skipped_no_mask']}")


def _hash_prior_filter_cfg(config: dict[str, Any]) -> str:
    """SHA1 of the configured prior_filter values (new path) or empty string."""
    sam3_cfg = config.get("inference_phase", {}).get("sam3", {})
    pf = sam3_cfg.get("prior_filter")
    if not pf:
        return ""
    blob = json.dumps(pf, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]
