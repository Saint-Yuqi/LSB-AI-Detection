#!/usr/bin/env python3
"""
Run low-confidence SAM3 raw inference for selected DR1 base keys and save raw JSON.

This is a non-destructive probe:
    - uses current renders
    - does not touch gt_canonical/current
    - writes raw predictions into a separate probe directory

Usage:
    conda run -n sam3 python scripts/data/probe_sam3_low_conf_raw.py \
        --config configs/unified_data_prep.yaml \
        --base-keys 00019_fo,00049_eo,00066_fo \
        --satellite-threshold 0.06
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.artifacts import save_predictions_json
from src.pipelines.unified_dataset.config import load_config
from src.pipelines.unified_dataset.inference_sam3 import _prepare_sam3_context
from src.pipelines.unified_dataset.keys import BaseKey
from src.pipelines.unified_dataset.paths import PathResolver


def _parse_base_keys(text: str) -> list[BaseKey]:
    keys: list[BaseKey] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        gid_str, view_id = token.split("_", 1)
        keys.append(BaseKey(int(gid_str), view_id))
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Unified data prep config")
    parser.add_argument(
        "--base-keys",
        type=str,
        required=True,
        help="Comma-separated base keys, e.g. 00019_fo,00049_eo",
    )
    parser.add_argument(
        "--satellite-threshold",
        type=float,
        default=0.06,
        help="Temporary low confidence threshold applied to the satellite prompt",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/02_processed/probes/sam3_low_conf_raw"),
        help="Probe output directory root",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger = logging.getLogger("probe_sam3_low_conf_raw")

    config = load_config(args.config)
    sam3_cfg = config["inference_phase"]["sam3"]
    sam3_cfg["confidence_threshold"] = min(
        float(args.satellite_threshold),
        float(sam3_cfg.get("confidence_threshold", 1.0)),
    )
    for prompt in sam3_cfg.get("prompts", []):
        if prompt.get("type_label") == "satellites":
            prompt["confidence_threshold"] = float(args.satellite_threshold)

    ctx = _prepare_sam3_context(config)
    runner = ctx["runner"]
    prompts = ctx["prompts"]
    append_metrics_to_masks = ctx["append_metrics_to_masks"]
    H_work, W_work = tuple(config["processing"]["target_size"])
    resolver = PathResolver(config)
    input_variant = config["inference_phase"].get("input_image_variant", "linear_magnitude")

    args.output_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict] = []
    for key in _parse_base_keys(args.base_keys):
        render_path = resolver.get_render_dir(input_variant, key) / "0000.png"
        if not render_path.exists():
            raise FileNotFoundError(f"Render not found for {key}: {render_path}")

        logger.info("Running low-conf raw probe for %s", key)
        image_pil = Image.open(render_path).convert("RGB")
        masks, time_ms = runner.run(image_pil, prompts)
        if masks:
            append_metrics_to_masks(masks, H_work, W_work, compute_hull=True)

        out_dir = args.output_root / str(key)
        out_dir.mkdir(parents=True, exist_ok=True)
        save_predictions_json(
            out_dir / "sam3_predictions_raw.json",
            masks,
            H_work,
            W_work,
            engine="sam3",
            layer="raw_probe",
        )

        sat_masks = [m for m in masks if m.get("type_label") == "satellites"]
        summary.append(
            {
                "base_key": str(key),
                "satellite_threshold": float(args.satellite_threshold),
                "time_ms": round(time_ms, 2),
                "n_raw_total": len(masks),
                "n_raw_satellites": len(sat_masks),
                "top_satellites_by_area": [
                    {
                        "score": round(float(m.get("score", 0.0)), 6),
                        "area_clean": int(m.get("area_clean", m.get("area", 0))),
                        "dist_to_center": round(float(m.get("dist_to_center", 0.0)), 2),
                        "solidity": round(float(m.get("solidity", 0.0)), 4),
                        "aspect_sym_moment": round(
                            float(m.get("aspect_sym_moment", m.get("aspect_sym", 0.0))),
                            4,
                        ),
                    }
                    for m in sorted(
                        sat_masks,
                        key=lambda x: (
                            -int(x.get("area_clean", x.get("area", 0))),
                            -float(x.get("score", 0.0)),
                        ),
                    )[:15]
                ],
            }
        )

    (args.output_root / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
