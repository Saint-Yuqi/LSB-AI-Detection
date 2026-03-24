#!/usr/bin/env python3
"""
Galaxy-level train/val split for COCO annotations.

Usage:
    python scripts/split_annotations.py
    python scripts/split_annotations.py --annotations path/to/annotations.json
    python scripts/split_annotations.py --train-ratio 0.8 --seed 42
    python scripts/split_annotations.py --reuse-manifest path/to/split_manifest.json

Output:
    {output_dir}/annotations_train.json
    {output_dir}/annotations_val.json
    {output_dir}/split_manifest.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.split import galaxy_split_coco


def main():
    parser = argparse.ArgumentParser(description="Galaxy-level COCO train/val split")
    parser.add_argument("--annotations", type=Path,
                        default=Path("data/02_processed/sam3_prepared/annotations.json"))
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--reuse-manifest", type=Path, default=None,
                        help="Lock prior galaxy assignments from existing manifest")
    parser.add_argument("--output-prefix", type=str, default="annotations",
                        help="Prefix for output filenames (default: 'annotations')")
    args = parser.parse_args()

    coco = json.loads(args.annotations.read_text())

    reuse = None
    if args.reuse_manifest:
        reuse = json.loads(args.reuse_manifest.read_text())

    coco_train, coco_val, manifest = galaxy_split_coco(
        coco, train_ratio=args.train_ratio, seed=args.seed, reuse_manifest=reuse,
    )

    out_dir = args.output_dir or args.annotations.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = args.output_prefix
    (out_dir / f"{prefix}_train.json").write_text(json.dumps(coco_train, indent=2))
    (out_dir / f"{prefix}_val.json").write_text(json.dumps(coco_val, indent=2))
    (out_dir / "split_manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"Train: {manifest['n_train_images']} images, {manifest['n_train_annotations']} annotations "
          f"({len(manifest['train_galaxy_ids'])} galaxies)")
    print(f"Val:   {manifest['n_val_images']} images, {manifest['n_val_annotations']} annotations "
          f"({len(manifest['val_galaxy_ids'])} galaxies)")
    print(f"Output: {out_dir}")


if __name__ == "__main__":
    main()
