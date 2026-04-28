#!/usr/bin/env python3
"""Render crop + bare context + EV full-image assets for a review round.

Thin CLI wrapper around ``src.review.asset_manager`` and
``src.review.review_render``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.review.asset_manager import AssetManager
from src.review.render_spec import RenderSpecRegistry
from src.review.schemas import CropSpec, SatMvAssetRefs, TaskFamily


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True,
                        help="verifier_examples_{family}.jsonl")
    parser.add_argument("--render-dir", type=Path, required=True,
                        help="Root of existing rendered images")
    parser.add_argument("--render-spec-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True,
                        help="Output root for review assets")
    args = parser.parse_args(argv)

    registry = RenderSpecRegistry.from_yaml(args.render_spec_config)
    mgr = AssetManager(args.asset_root)

    with open(args.examples) as f:
        for line in f:
            rec = json.loads(line)
            family = TaskFamily(rec["task_family"])
            sample_id = rec["sample_id"]
            spec = registry.get(rec["render_spec_id"])

            render_path = args.render_dir / sample_id / "0000.png"
            if not render_path.exists():
                print(f"WARN: render not found for {sample_id}: {render_path}",
                      file=sys.stderr)
                continue
            full_image = _load_image(render_path)

            if family == TaskFamily.SATELLITE_MV:
                arefs = rec["asset_refs"]
                mgr.ensure_bare_context(sample_id, spec.input_variant, full_image)
                crop_spec = CropSpec(**arefs["crop_spec"])
                mgr.ensure_crop(
                    sample_id, crop_spec, spec, full_image,
                    arefs["candidate_rle"],
                )
            else:
                hints = rec["asset_refs"].get("fragment_hints")
                state_key = rec["asset_refs"].get("synthetic_variant_id")
                mgr.ensure_ev_image(
                    sample_id, spec.input_variant, spec,
                    full_image, hints,
                    state_key=state_key,
                )

    print(f"Assets written to {args.asset_root}")


if __name__ == "__main__":
    main()
