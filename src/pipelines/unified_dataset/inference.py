"""
Phase 3 thin dispatcher: SAM3 engine only (SAM2 was purged).
"""
from __future__ import annotations

import logging
from typing import Any

from .inference_sam3 import run_inference_sam3
from .keys import BaseKey


def run_inference_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """Run SAM3 inference + type-aware filtering + QA overlay generation."""
    logger.info("=" * 60)
    logger.info("PHASE 3: INFERENCE (SAM3)")
    logger.info("=" * 60)

    inf_cfg = config.get("inference_phase", {})
    engine = inf_cfg.get("engine", "sam3")
    if engine != "sam3":
        raise ValueError(
            f"Only engine='sam3' is supported; got {engine!r}. "
            f"SAM2 inference was removed in the eval refactor."
        )
    run_mode = inf_cfg.get("run_mode", "evaluate")
    logger.info("Engine: sam3, run_mode: %s", run_mode)

    run_inference_sam3(config, base_keys, logger, force, force_variants)
