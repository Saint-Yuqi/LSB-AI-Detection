"""
SAM2 AutoMask Runner – wrapper for SAM2AutomaticMaskGenerator.

Usage:
    from src.inference.sam2_automask_runner import AutoMaskRunner
    runner = AutoMaskRunner(checkpoint_path, model_cfg)
    masks = runner.run(image_rgb, config)

Env:
    Uses bf16 precision on Ampere+ GPUs by default.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

# SAM2 imports (adjust to your installation path if needed)
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from sam2.build_sam import build_sam2

# Default SAM2 checkpoint & config ---------------------------------
DEFAULT_CHECKPOINT = "/home/yuqyan/Yuqi/sam2/scratch/sam2_finetuning_20260205_210313/checkpoints/checkpoint.pt"
DEFAULT_MODEL_CFG = "configs/sam2.1/sam2.1_hiera_b+.yaml"


class AutoMaskRunner:
    """Thin wrapper around SAM2AutomaticMaskGenerator with bf16 + CUDA sync timing."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        model_cfg: str = DEFAULT_MODEL_CFG,
        device: str = "cuda",
        use_bf16: bool = True,
    ):
        self.device = device
        self.use_bf16 = use_bf16 and torch.cuda.is_available()
        self.checkpoint = Path(checkpoint)

        # Build model once ------------------------------------------------
        self.sam2_model = build_sam2(
            model_cfg,
            str(self.checkpoint),
            device=self.device,
        )
        # NOTE: Do NOT cast model to bf16 here. SAM2AutomaticMaskGenerator
        # uses torch.autocast internally. Pre-casting causes dtype mismatch.
        self.sam2_model.eval()

    # ------------------------------------------------------------------ #
    def run(
        self,
        image: np.ndarray,
        config: dict[str, Any] | None = None,
        warmup: bool = False,
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Generate masks for an 8-bit RGB image.

        Args:
            image: (H, W, 3) uint8 RGB numpy array.
            config: SAM2AutomaticMaskGenerator kwargs (points_per_side, pred_iou_thresh, ...).
            warmup: If True, run once without timing (for JIT warm-up).

        Returns:
            (masks, time_ms): masks list with keys segmentation, area, predicted_iou, stability_score, etc.
        """
        config = config or {}
        generator = SAM2AutomaticMaskGenerator(
            model=self.sam2_model,
            **config,
        )

        # CUDA sync timing -------------------------------------------------
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            masks = generator.generate(image)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        if warmup:
            return masks, 0.0
        return masks, elapsed_ms

    # ------------------------------------------------------------------ #
    def warmup(self, image: np.ndarray, config: dict[str, Any] | None = None, n: int = 2):
        """Run n warm-up passes (JIT compile) before actual timing."""
        for _ in range(n):
            self.run(image, config, warmup=True)
