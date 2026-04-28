#!/usr/bin/env python3
"""Pipeline artifacts + silver labels → business JSONL.

Dual-source: kept candidates from authoritative layer, rejected
candidates from pre-merge artifacts.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from src.review.asset_manager import AssetManager
from src.review.example_builder import (
    _build_ev_examples,
    _build_sat_mv_authoritative,
    _build_sat_mv_rejected,
    _load_reject_candidates_sam3,
    write_examples,
)
from src.review.key_adapter import base_key_to_sample_id
from src.review.render_spec import RenderSpecRegistry
from src.review.schemas import TaskFamily, VerifierExample
from src.review.silver_labeler import SilverLabel, read_silver_labels
from src.pipelines.unified_dataset.keys import BaseKey


def _load_image(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("RGB"))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=[f.value for f in TaskFamily])
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--render-dir", type=Path, required=True)
    parser.add_argument("--silver-labels", type=Path, required=True,
                        help="silver_labels_{family}.jsonl from Phase 6")
    parser.add_argument("--render-spec-config", type=Path, required=True)
    parser.add_argument("--render-spec-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--keys-file", type=Path, required=True)
    parser.add_argument("--source-round", default="round_00")
    parser.add_argument("--source-checkpoint", default="initial")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    family = TaskFamily(args.family)
    registry = RenderSpecRegistry.from_yaml(args.render_spec_config)
    spec = registry.get(args.render_spec_id)
    asset_mgr = AssetManager(args.asset_root)

    silver_labels = read_silver_labels(args.silver_labels)
    silver_by_sample: dict[str, dict[str, SilverLabel]] = {}
    for lab in silver_labels:
        silver_by_sample.setdefault(lab.sample_id, {})[lab.candidate_key] = lab

    with open(args.keys_file) as f:
        keys_raw = json.load(f)
    base_keys = [BaseKey(galaxy_id=k[0], view_id=k[1]) for k in keys_raw]

    all_examples: list[VerifierExample] = []
    counter = 0

    for bk in base_keys:
        sample_id = base_key_to_sample_id(bk)
        bk_dir = args.gt_dir / str(bk)

        imap_path = bk_dir / "instance_map_uint8.png"
        inst_path = bk_dir / "instances.json"
        if not imap_path.exists():
            continue

        instance_map = np.array(Image.open(imap_path))
        with open(inst_path) as f:
            instances = json.load(f)

        render_path = args.render_dir / str(bk) / "0000.png"
        if not render_path.exists():
            print(f"WARN: render not found for {bk}: {render_path}", file=sys.stderr)
            continue
        full_image = _load_image(render_path)

        lookup = silver_by_sample.get(sample_id, {})

        if family == TaskFamily.SATELLITE_MV:
            exs, counter = _build_sat_mv_authoritative(
                bk, args.gt_dir, instance_map, instances, lookup,
                asset_mgr, spec, args.prompt_id, full_image,
                args.source_round, args.source_checkpoint, counter,
            )
            all_examples.extend(exs)

            rejects = _load_reject_candidates_sam3(args.gt_dir, bk)

            exs_rej, counter = _build_sat_mv_rejected(
                bk, args.gt_dir, rejects,
                asset_mgr, spec, args.prompt_id, full_image,
                args.source_round, args.source_checkpoint, counter,
            )
            all_examples.extend(exs_rej)

        else:
            exs, counter = _build_ev_examples(
                bk, family, args.gt_dir, instance_map, instances,
                lookup, asset_mgr, spec, args.prompt_id, full_image,
                args.source_round, args.source_checkpoint, counter,
            )
            all_examples.extend(exs)

    write_examples(all_examples, args.output)
    print(f"Wrote {len(all_examples)} examples to {args.output}")


if __name__ == "__main__":
    main()
