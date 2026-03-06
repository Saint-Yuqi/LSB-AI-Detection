"""
Phase 3 thin dispatcher: routes to SAM2 or SAM3 engine based on config.
"""
from __future__ import annotations

import logging
from typing import Any

from .keys import BaseKey


def run_inference_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """Run inference engine -> type-aware filter -> merge (SAM2) or evaluate (SAM3)."""
    logger.info("=" * 60)
    logger.info("PHASE 3: INFERENCE")
    logger.info("=" * 60)

    inf_cfg = config.get("inference_phase", {})
    engine = inf_cfg.get("engine", "sam2")
    run_mode = inf_cfg.get("run_mode", "evaluate")
    logger.info(f"Engine: {engine}, run_mode: {run_mode}")

    if engine == "sam3":
        from .inference_sam3 import run_inference_sam3
        run_inference_sam3(config, base_keys, logger, force, force_variants)
    else:
        from .inference_sam2 import run_inference_sam2
        run_inference_sam2(config, base_keys, logger, force, force_variants)
