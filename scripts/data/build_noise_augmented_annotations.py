#!/usr/bin/env python3
"""
Build noise-augmented COCO annotations from a clean train split.

Usage:
    python scripts/data/build_noise_augmented_annotations.py \
      --base-annotations data/02_processed/sam3_prepared/annotations_train.json \
      --noisy-root data/02_processed/renders/noisy \
      --dataset-root data/02_processed/sam3_prepared \
      --snr-tags snr20 snr50 \
      --split-manifest data/02_processed/sam3_prepared/split_manifest.json \
      --output data/02_processed/sam3_prepared/annotations_train_noise_augmented.json

Output:
    {output}                   (augmented COCO JSON)
    {output_dir}/noise_aug_manifest.json
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
        description="Build noise-augmented COCO annotations (train-only)",
    )
    parser.add_argument(
        "--base-annotations", type=Path, required=True,
        help="Path to clean annotations_train.json",
    )
    parser.add_argument(
        "--noisy-root", type=Path, required=True,
        help="Root of pre-rendered noisy PNGs "
             "(layout: {variant}/{snr}/{base_key}/0000.png)",
    )
    parser.add_argument(
        "--dataset-root", type=Path, required=True,
        help="SAM3 dataset root (symlinks created under images/)",
    )
    parser.add_argument(
        "--snr-tags", nargs="+", required=True,
        help="SNR profiles to include, e.g. snr20 snr50",
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

    coco_train = json.loads(args.base_annotations.read_text())
    split_manifest = json.loads(args.split_manifest.read_text())

    augmented_coco, stats = build_noise_augmented_coco(
        coco_train=coco_train,
        noisy_root=args.noisy_root,
        snr_tags=args.snr_tags,
        dataset_root=args.dataset_root,
        split_manifest=split_manifest,
        force=args.force,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(augmented_coco, indent=2))

    manifest = {
        **stats,
        "source_annotations": str(args.base_annotations),
        "split_manifest": str(args.split_manifest),
    }
    manifest_path = args.output.parent / "noise_aug_manifest.json"
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
