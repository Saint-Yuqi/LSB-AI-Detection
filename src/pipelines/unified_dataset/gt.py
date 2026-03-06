"""
Phase 2: Load SB mask -> streams_instance_map.npy (GT canonical).

Output: gt_canonical/current/{BaseKey}/streams_instance_map.npy
"""
from __future__ import annotations

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
    logger.info("=" * 60)
    logger.info("PHASE 2: GT (Streams)")
    logger.info("=" * 60)

    resolver = PathResolver(config)
    sb_threshold = config["data_selection"]["canonical_sb_threshold"]
    target_size = tuple(config["processing"]["target_size"])

    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_mask": 0}

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        npy_path = gt_dir / "streams_instance_map.npy"

        if npy_path.exists():
            stats["skipped_exists"] += 1
            continue

        mask_path = resolver.get_mask_path(key, sb_threshold)

        if not mask_path.exists():
            logger.warning(f"Mask not found: {mask_path}")
            stats["skipped_no_mask"] += 1
            continue

        gt_dir.mkdir(parents=True, exist_ok=True)

        mask_data = load_fits_gz(mask_path)
        instance_map = np.round(mask_data).astype(np.int32)

        instance_map_resized = cv2.resize(
            instance_map, target_size, interpolation=cv2.INTER_NEAREST
        )

        np.save(npy_path, instance_map_resized)

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
