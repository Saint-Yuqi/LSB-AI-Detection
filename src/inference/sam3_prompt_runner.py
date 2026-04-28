"""
SAM3 Text-Prompt Runner – wraps Sam3Processor for text-prompt grounding.

Usage:
    from src.inference.sam3_prompt_runner import SAM3PromptRunner
    runner = SAM3PromptRunner(checkpoint, bpe_path, confidence_threshold=0.55)
    masks, time_ms = runner.run(image_pil, prompts)

Args:
    checkpoint: path to fine-tuned SAM3 checkpoint (.pt)
    bpe_path:   path to BPE vocab file
    prompts:    [{"text": "stellar stream", "type_label": "streams"}, ...]

Returns:
    masks: list[dict] aligned with the legacy AutoMask mask contract:
        segmentation: np.ndarray(H_work, W_work, bool)
        area:         int
        score:        float (SAM3 native; from inference_state["scores"])
        stability_score: float (= score, SAM3 has no stability metric)
        type_label:   str
        bbox:         [x, y, w, h]  (XYWH, on working grid)
    time_ms: float

Env:
    CUDA, bf16 Ampere+. SAM3 package must be importable.

Contract:
    - Every mask is resized to (H_work, W_work) via NEAREST before return.
    - reset_all_prompts called before each text prompt (VRAM safety).
    - RLE self-check on first mask per run.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image

from src.utils.coco_utils import mask_to_rle, decode_rle


# Default paths ----------------------------------------------------------------
DEFAULT_SAM3_CHECKPOINT = (
    "/home/yuqyan/Yuqi/sam3/scratch/sam3_finetuning/checkpoints/checkpoint_best.pt"
)
DEFAULT_BPE_PATH = "/home/yuqyan/Yuqi/sam3/sam3/assets/bpe_simple_vocab_16e6.txt.gz"


class SAM3PromptRunner:
    """Thin wrapper: image + text prompts → list[dict] aligned with the legacy mask contract."""

    def __init__(
        self,
        checkpoint: str | Path = DEFAULT_SAM3_CHECKPOINT,
        bpe_path: str | Path = DEFAULT_BPE_PATH,
        confidence_threshold: float = 0.55,
        resolution: int = 1008,
        device: str = "cuda",
        target_size: tuple[int, int] = (1024, 1024),
    ):
        self.device = device
        self.target_size = target_size          # (H_work, W_work)
        self._rle_checked = False

        # Lazy import to avoid loading sam3 at module level
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        self._model = build_sam3_image_model(
            bpe_path=str(bpe_path),
            checkpoint_path=str(checkpoint),
            device=device,
            eval_mode=True,
            enable_segmentation=True,
            enable_inst_interactivity=False,
            load_from_HF=False,
        )
        self._processor = Sam3Processor(
            self._model,
            resolution=resolution,
            device=device,
            confidence_threshold=confidence_threshold,
        )

    # ---------------------------------------------------------------------- #
    def run(
        self,
        image_pil: Image.Image,
        prompts: list[dict[str, str]],
    ) -> tuple[list[dict[str, Any]], float]:
        """
        Run text-prompt grounding on a single image.

        Args:
            image_pil: PIL.Image (RGB).
            prompts:   [{"text": ..., "type_label": ...,
                          "confidence_threshold": 0.55}, ...]
                       Per-prompt confidence_threshold is optional; if absent
                       the mask is kept unconditionally (processor-level
                       threshold already applied).

        Returns:
            (masks, time_ms)
        """
        H_work, W_work = self.target_size

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            state = self._processor.set_image(image_pil)
            all_masks: list[dict[str, Any]] = []

            for prompt in prompts:
                # VRAM safety: clear previous text / masks / scores
                self._processor.reset_all_prompts(state)
                state = self._processor.set_text_prompt(prompt["text"], state)

                converted = self._convert_state(
                    state,
                    type_label=prompt["type_label"],
                    H_work=H_work,
                    W_work=W_work,
                )

                # Per-prompt confidence filter (above processor floor)
                thresh = prompt.get("confidence_threshold")
                if thresh is not None:
                    converted = [
                        m for m in converted
                        if m["score"] >= thresh
                    ]

                all_masks.extend(converted)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return all_masks, elapsed_ms

    # ---------------------------------------------------------------------- #
    def _convert_state(
        self,
        state: dict,
        type_label: str,
        H_work: int,
        W_work: int,
    ) -> list[dict[str, Any]]:
        """Convert Sam3Processor state → list of unified mask dicts."""
        masks_tensor = state.get("masks")     # (N, 1, H_orig, W_orig) bool
        scores_tensor = state.get("scores")   # (N,)
        boxes_tensor = state.get("boxes")     # (N, 4)  x1y1x2y2

        if masks_tensor is None or len(masks_tensor) == 0:
            return []

        result: list[dict[str, Any]] = []

        for i in range(len(scores_tensor)):
            # --- segmentation: squeeze + CPU + canonical grid resize ---
            seg = masks_tensor[i].squeeze(0).cpu().numpy()  # (H_orig, W_orig) bool
            if seg.shape != (H_work, W_work):
                seg = cv2.resize(
                    seg.astype(np.uint8), (W_work, H_work),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(bool)

            # RLE self-check (once per run)
            if not self._rle_checked:
                self._rle_selfcheck(seg)
                self._rle_checked = True

            area = int(seg.sum())
            score = float(scores_tensor[i].item())

            # --- bbox: x1y1x2y2 → xywh on working grid ---
            box = boxes_tensor[i].cpu().numpy()  # x1, y1, x2, y2
            # Scale box to working grid if image was different size
            orig_h = state.get("original_height", H_work)
            orig_w = state.get("original_width", W_work)
            scale_x = W_work / orig_w if orig_w != W_work else 1.0
            scale_y = H_work / orig_h if orig_h != H_work else 1.0
            x1 = int(box[0] * scale_x)
            y1 = int(box[1] * scale_y)
            x2 = int(box[2] * scale_x)
            y2 = int(box[3] * scale_y)
            bbox_xywh = [x1, y1, x2 - x1, y2 - y1]

            result.append({
                "segmentation": seg,
                "area": area,
                "score": score,
                "stability_score": score,     # SAM3 has no stability metric
                "type_label": type_label,
                "bbox": bbox_xywh,
            })

        return result

    # ---------------------------------------------------------------------- #
    @staticmethod
    def _rle_selfcheck(seg: np.ndarray) -> None:
        """Assert RLE roundtrip fidelity (once per run, near-zero cost)."""
        rle = mask_to_rle(seg.astype(np.uint8))
        decoded = decode_rle(rle).astype(bool)
        assert np.array_equal(seg, decoded), (
            f"RLE roundtrip failed: seg.shape={seg.shape}, "
            f"decoded.shape={decoded.shape}, mismatched pixels="
            f"{int((seg != decoded).sum())}"
        )
