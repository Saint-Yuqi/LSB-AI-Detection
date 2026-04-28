#!/usr/bin/env python3
"""
Build noise-augmented COCO annotations from a clean split.

Usage:
    python scripts/data/build_noise_augmented_annotations.py \
      --base-annotations data/02_processed/sam3_prepared/annotations_train.json \
      --noisy-root data/02_processed/renders/noisy \
      --dataset-root data/02_processed/sam3_prepared \
      --noise-tags sb30 sb31.5 \
      --target-split train \
      --noisy-variants linear_magnitude \
      --split-manifest data/02_processed/sam3_prepared/split_manifest.json \
      --output data/02_processed/sam3_prepared/annotations_train_noise_augmented.json

Output:
    {output}                   (augmented COCO JSON)
    {output_dir}/noise_aug_manifest_{split}.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.noise_aug import build_noise_augmented_coco


def main():
    parser = argparse.ArgumentParser(
        description="Build noise-augmented COCO annotations for a split",
    )
    parser.add_argument(
        "--base-annotations", type=Path, required=True,
        help="Path to clean annotations_{split}.json",
    )
    parser.add_argument(
        "--noisy-root", type=Path, required=True,
        help="Root of pre-rendered noisy PNGs "
             "(layout: {variant}/{noise_tag}/{base_key}/0000.png)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="SAM3 dataset root (symlinks created under images/)",
    )
    parser.add_argument(
        "--noise-tags", nargs="+", required=True,
        help="Noise profiles to include, e.g. sb30 sb31.5",
    )
    parser.add_argument(
        "--target-split", choices=["train", "val"], required=True,
        help="Which clean split is being augmented",
    )
    parser.add_argument(
        "--noisy-variants", nargs="+", default=["linear_magnitude"],
        help="Variants that receive noisy clones (default: linear_magnitude)",
    )
    parser.add_argument(
        "--split-manifest", type=Path, required=True,
        help="Path to split_manifest.json (used for leakage check)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for augmented annotations JSON",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite divergent symlinks instead of raising",
    )
    args = parser.parse_args()

    coco_split = json.loads(args.base_annotations.read_text())
    split_manifest = json.loads(args.split_manifest.read_text())

    augmented_coco, stats = build_noise_augmented_coco(
        coco_split=coco_split,
        noisy_root=args.noisy_root,
        noise_tags=args.noise_tags,
        dataset_root=args.dataset_root,
        split_manifest=split_manifest,
        target_split=args.target_split,
        noisy_variants=set(args.noisy_variants),
        force=args.force,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented_coco, indent=2))

    manifest = {
        **stats,
        "source_annotations": str(args.base_annotations),
        "split_manifest": str(args.split_manifest),
    }
    manifest_path = (
        args.output.parent / f"noise_aug_manifest_{args.target_split}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"Clean:  {stats['n_images_clean']} images, "
          f"{stats['n_annotations_clean']} annotations")
    print(f"Noisy:  {stats['n_images_noisy']} images, "
          f"{stats['n_annotations_noisy']} annotations")
    print(f"Total:  {stats['n_images_total']} images, "
          f"{stats['n_annotations_total']} annotations")
    print(f"Output: {args.output}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
