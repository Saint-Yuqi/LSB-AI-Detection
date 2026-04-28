"""
Phase 1: Render FITS -> RGB for each preprocessing variant.

Output: renders/current/{preprocessing}/{BaseKey}/0000.png
"""
from __future__ import annotations

import logging
from typing import Any

import cv2

from src.data.io import load_fits_gz
from .keys import BaseKey
from .paths import PathResolver
from .preprocessor_factory import create_preprocessor


def run_render_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Render FITS -> RGB for each preprocessing variant."""
    logger.info("=" * 60)
    logger.info("PHASE 1: RENDER")
    logger.info("=" * 60)

    resolver = PathResolver(config)
    variants = config["preprocessing_variants"]
    target_size = tuple(config["processing"]["target_size"])

    preprocessors = {}
    for v in variants:
        preprocessors[v["name"]] = create_preprocessor(
            v["name"], v.get("params", {}), target_size
        )

    stats = {"rendered": 0, "skipped_exists": 0, "skipped_no_fits": 0}
    conditions = resolver.get_active_conditions()

    for condition in conditions:
        logger.info("Render condition: %s", condition)
        for key in base_keys:
            fits_path = resolver.get_condition_fits_path(
                key,
                dataset=resolver.dataset_name,
                condition=condition,
            )

            if not fits_path.exists():
                logger.warning(f"FITS not found: {fits_path}")
                stats["skipped_no_fits"] += 1
                continue

            sb_map = load_fits_gz(fits_path)

            for name, proc in preprocessors.items():
                out_dir = resolver.get_render_dir(
                    name,
                    key,
                    dataset=resolver.dataset_name,
                    condition=condition,
                )
                out_path = out_dir / "0000.png"

                if out_path.exists() and force_variants and name in force_variants:
                    out_path.unlink()
                    logger.info(f"Force rebuild: {condition}/{name}/{key}")
                elif out_path.exists():
                    stats["skipped_exists"] += 1
                    continue

                out_dir.mkdir(parents=True, exist_ok=True)

                rgb = proc.process(sb_map)
                cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                stats["rendered"] += 1

    logger.info(f"Rendered: {stats['rendered']}, Skipped (exists): {stats['skipped_exists']}, Skipped (no FITS): {stats['skipped_no_fits']}")
