#!/usr/bin/env python3
"""
Compose a training dataset from one or more annotation sources.

Single source  -> symlink mode (O(1), no rewrite).
Multiple sources -> merge mode (concatenate + renumber + tag dataset_source).

Usage:
    # Noise-augmented (symlink mode)
    python scripts/data/build_training_dataset.py \
      --include noise_aug:data/02_processed/sam3_prepared/annotations_train_noise_augmented.json \
      --output data/02_processed/sam3_prepared/annotations_train_active.json

    # Clean-only (symlink mode)
    python scripts/data/build_training_dataset.py \
      --include clean:data/02_processed/sam3_prepared/annotations_train.json \
      --output data/02_processed/sam3_prepared/annotations_train_active.json

    # Multi-source (merge mode)
    python scripts/data/build_training_dataset.py \
      --include clean:annotations_train.json \
      --include betterdata:annotations_train_betterdata.json \
      --output annotations_train_active.json

Output:
    {output}                            (COCO JSON or symlink)
    {output_dir}/compose_manifest.json
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.compose import compose_training_coco


def _parse_include(value: str) -> tuple[str, Path]:
    """Parse 'label:path' into (label, Path)."""
    if ":" not in value:
        raise argparse.ArgumentTypeError(
            f"--include must be 'label:path', got '{value}'"
        )
    label, path_str = value.split(":", 1)
    return label.strip(), Path(path_str.strip())


def main():
    parser = argparse.ArgumentParser(
        description="Compose training COCO dataset from annotation sources",
    )
    parser.add_argument(
        "--include", type=_parse_include, action="append", required=True,
        dest="sources", metavar="LABEL:PATH",
        help="Source annotation file as 'label:path' (repeatable)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output path for composed annotations",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Overwrite existing output (symlink or file)",
    )
    parser.add_argument(
        "--allow-duplicate-filenames", action="store_true",
        help="Allow identical duplicate file_name entries across sources",
    )
    args = parser.parse_args()

    manifest = compose_training_coco(
        sources=args.sources,
        output_path=args.output,
        allow_duplicate_filenames=args.allow_duplicate_filenames,
        force=args.force,
    )

    mode = manifest["mode"]
    n_sources = len(manifest["sources"])
    print(f"Mode:    {mode} ({n_sources} source{'s' if n_sources > 1 else ''})")
    print(f"Images:  {manifest['n_images_total']}")
    print(f"Annotations: {manifest['n_annotations_total']}")
    print(f"Output:  {args.output}")


if __name__ == "__main__":
    main()
