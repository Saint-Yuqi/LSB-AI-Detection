#!/usr/bin/env python3
"""
Unified Dataset Preparation Pipeline

4-Phase Architecture:
    render     -> renders/current/{preprocessing}/{BaseKey}/0000.png
    gt         -> gt_canonical/.../streams_instance_map.npy
    inference  -> SAM2 (AutoMask merge) or SAM3 (evaluate: predictions JSON + QA overlay)
    export     -> SAM2 symlinks + SAM3 annotations.json

Usage:
    python scripts/prepare_unified_dataset.py --config configs/unified_data_prep.yaml
    python scripts/prepare_unified_dataset.py --config ... --phase inference
    python scripts/prepare_unified_dataset.py --config ... --phase satellites  # alias
    python scripts/prepare_unified_dataset.py --config ... --galaxies 11,13
    python scripts/prepare_unified_dataset.py --config ... --force
    python scripts/prepare_unified_dataset.py --config ... --force-variants asinh_stretch

Env:
    CUDA, PyTorch with bf16 support for inference phase.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.config import load_config, generate_base_keys
from src.pipelines.unified_dataset.render import run_render_phase
from src.pipelines.unified_dataset.gt import run_gt_phase
from src.pipelines.unified_dataset.inference import run_inference_phase
from src.pipelines.unified_dataset.export import run_export_phase

# Re-exports for backward compatibility (tests import from this script)
from src.pipelines.unified_dataset.keys import BaseKey, VariantKey  # noqa: F401
from src.pipelines.unified_dataset.paths import PathResolver  # noqa: F401
from src.pipelines.unified_dataset.preprocessor_factory import create_preprocessor  # noqa: F401
from src.pipelines.unified_dataset.artifacts import (  # noqa: F401
    save_predictions_json as _save_predictions_json,
    merge_instances as _merge_instances,
)


def main():
    parser = argparse.ArgumentParser(description="Unified Dataset Preparation")
    parser.add_argument("--config", "-c", type=Path, required=True, help="Config YAML path")
    parser.add_argument(
        "--phase", type=str,
        choices=["render", "gt", "inference", "satellites", "export", "all"],
        default="all",
    )
    parser.add_argument("--galaxies", type=str, default=None,
                        help="Comma-separated galaxy IDs subset")
    parser.add_argument("--force", action="store_true",
                        help="Force rebuild all variants in selected phase")
    parser.add_argument("--force-variants", type=str, default=None,
                        help="Comma-separated variant names to force rebuild")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger(__name__)

    config = load_config(args.config)

    galaxy_filter = None
    if args.galaxies:
        galaxy_filter = [int(g.strip()) for g in args.galaxies.split(",")]

    force_variants: set[str] | None = None
    if args.force:
        force_variants = {v["name"] for v in config["preprocessing_variants"]}
        logger.info(f"Force rebuild ALL variants: {force_variants}")
    elif args.force_variants:
        force_variants = {v.strip() for v in args.force_variants.split(",")}
        logger.info(f"Force rebuild variants: {force_variants}")

    base_keys = generate_base_keys(config, galaxy_filter)
    logger.info(f"Processing {len(base_keys)} BaseKeys")

    phases = ["render", "gt", "inference", "export"] if args.phase == "all" else [args.phase]

    for phase in phases:
        if phase == "render":
            run_render_phase(config, base_keys, logger, force_variants)
        elif phase == "gt":
            if force_variants is not None:
                logger.warning("GT is variant-independent; --force-variants is a no-op for this phase")
            run_gt_phase(config, base_keys, logger, force_variants)
        elif phase in ("inference", "satellites"):
            run_inference_phase(config, base_keys, logger, force_variants=force_variants)
        elif phase == "export":
            run_export_phase(config, base_keys, logger, force_variants)

    logger.info("Done.")


if __name__ == "__main__":
    main()
