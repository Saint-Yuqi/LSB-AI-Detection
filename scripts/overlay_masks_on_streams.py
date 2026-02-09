#!/usr/bin/env python3
"""
Overlay masks from gt_folder (e.g., satellites) onto img_folder images (e.g., streams).

Default behavior:
  - Scan img_root for folders matching "*_streams"
  - For each folder, find the corresponding mask folder with the same galaxy_id,
    orientation, and SB threshold but with feature type "satellites"
  - Overlay the mask on the image and save to out_root preserving folder structure
"""

import argparse
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image


INSTANCE_COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 255, 128),    # Spring green
    (255, 0, 128),    # Rose
    (128, 255, 0),    # Lime
    (0, 128, 255),    # Sky blue
]

try:
    RESAMPLE_NEAREST = Image.Resampling.NEAREST
except AttributeError:
    RESAMPLE_NEAREST = Image.NEAREST


def parse_folder_name(folder_name: str):
    pattern = r"^(\d+)_([a-z]+)_SB([\d.]+)_(\w+)$"
    match = re.match(pattern, folder_name)
    if not match:
        return None
    return {
        "galaxy_id": int(match.group(1)),
        "orientation": match.group(2),
        "sb_threshold": match.group(3),
        "feature_type": match.group(4),
    }


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    if img.dtype == np.uint8:
        return img
    img = img.astype(np.float32)
    img_min = float(np.min(img))
    img_max = float(np.max(img))
    if img_max <= img_min:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - img_min) / (img_max - img_min)
    img = (img * 255.0).clip(0, 255).astype(np.uint8)
    return img


def load_image(path: Path) -> np.ndarray:
    img = np.array(Image.open(path))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[2] == 4:
        img = img[:, :, :3]
    img = normalize_to_uint8(img)
    return img


def load_mask(path: Path) -> np.ndarray:
    mask = np.array(Image.open(path))
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    alpha: float = 0.6,
    single_color=(255, 0, 0),
    instance_colors=None,
) -> np.ndarray:
    overlay = image.copy()
    if mask.max() <= 0:
        return overlay

    instance_ids = np.unique(mask)
    instance_ids = instance_ids[instance_ids > 0]

    if instance_colors is None or len(instance_ids) <= 1:
        mask_region = mask > 0
        color_arr = np.array(single_color, dtype=np.float32)
        overlay[mask_region] = (
            overlay[mask_region].astype(np.float32) * (1.0 - alpha)
            + color_arr * alpha
        ).astype(np.uint8)
        return overlay

    mask_layer = np.zeros_like(overlay, dtype=np.uint8)
    for idx, inst_id in enumerate(instance_ids):
        color = instance_colors[idx % len(instance_colors)]
        inst_mask = mask == inst_id
        mask_layer[inst_mask] = color

    mask_region = np.any(mask_layer > 0, axis=-1)
    overlay[mask_region] = (
        overlay[mask_region].astype(np.float32) * (1.0 - alpha)
        + mask_layer[mask_region].astype(np.float32) * alpha
    ).astype(np.uint8)
    return overlay


def build_mask_folder_name(parsed, mask_type: str, mask_sb: Optional[str]) -> str:
    sb = mask_sb if mask_sb is not None else parsed["sb_threshold"]
    return f"{parsed['galaxy_id']:05d}_{parsed['orientation']}_SB{sb}_{mask_type}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Overlay gt masks (e.g., satellites) onto img images (e.g., streams)."
    )
    parser.add_argument(
        "--img-root",
        type=Path,
        default=Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam2_prepared/img_folder"),
        help="Root of img_folder",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam2_prepared/gt_folder"),
        help="Root of gt_folder",
    )
    parser.add_argument(
        "--out-root",
        type=Path,
        default=Path("/home/yuqyan/Yuqi/LSB-AI-Detection/data/02_processed/sam2_prepared/overlays_satellites_on_streams"),
        help="Output directory",
    )
    parser.add_argument("--img-type", default="streams", help="Type in img folders (default: streams)")
    parser.add_argument("--mask-type", default="satellites", help="Type in gt folders (default: satellites)")
    parser.add_argument(
        "--mask-sb",
        type=str,
        default=None,
        help="Override SB threshold for masks (string, e.g., '31.5').",
    )
    parser.add_argument("--frame", default="0000.png", help="Frame filename inside each folder")
    parser.add_argument("--alpha", type=float, default=0.6, help="Overlay alpha")
    parser.add_argument(
        "--color",
        type=str,
        default="255,0,0",
        help="Single mask color as R,G,B (used for binary or single-instance masks)",
    )
    parser.add_argument(
        "--no-instance-colors",
        action="store_true",
        help="Disable per-instance colors even if mask has multiple labels",
    )
    parser.add_argument(
        "--resize-mask",
        action="store_true",
        help="If mask and image sizes mismatch, resize mask to image size (nearest)",
    )
    parser.add_argument("--filter", type=str, default=None, help="Only process folder names containing this substring")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of folders processed")
    parser.add_argument("--dry-run", action="store_true", help="Print actions without writing files")
    args = parser.parse_args()

    img_root = args.img_root
    mask_root = args.mask_root
    out_root = args.out_root

    if not img_root.exists():
        print(f"Image root not found: {img_root}", file=sys.stderr)
        return 1
    if not mask_root.exists():
        print(f"Mask root not found: {mask_root}", file=sys.stderr)
        return 1

    try:
        color_parts = [int(x) for x in args.color.split(",")]
        if len(color_parts) != 3:
            raise ValueError
        single_color = tuple(color_parts)
    except ValueError:
        print("Invalid --color. Use format R,G,B (e.g., 255,0,0).", file=sys.stderr)
        return 1

    instance_colors = None if args.no_instance_colors else INSTANCE_COLORS

    folders = []
    for folder in sorted(img_root.iterdir()):
        if not folder.is_dir():
            continue
        if args.filter and args.filter not in folder.name:
            continue
        parsed = parse_folder_name(folder.name)
        if not parsed:
            continue
        if parsed["feature_type"] != args.img_type:
            continue
        folders.append((folder, parsed))

    if args.limit is not None:
        folders = folders[: args.limit]

    if not folders:
        print("No matching image folders found.")
        return 0

    out_root.mkdir(parents=True, exist_ok=True)

    processed = 0
    skipped_missing = 0
    skipped_mismatch = 0

    for folder, parsed in folders:
        img_path = folder / args.frame
        mask_folder_name = build_mask_folder_name(parsed, args.mask_type, args.mask_sb)
        mask_path = mask_root / mask_folder_name / args.frame

        if not img_path.exists():
            skipped_missing += 1
            print(f"[skip] image not found: {img_path}")
            continue
        if not mask_path.exists():
            skipped_missing += 1
            print(f"[skip] mask not found: {mask_path}")
            continue

        img = load_image(img_path)
        mask = load_mask(mask_path)

        if img.shape[:2] != mask.shape[:2]:
            if args.resize_mask:
                mask_img = Image.fromarray(mask)
                mask_img = mask_img.resize((img.shape[1], img.shape[0]), RESAMPLE_NEAREST)
                mask = np.array(mask_img)
            else:
                skipped_mismatch += 1
                print(
                    f"[skip] size mismatch img={img.shape[:2]} mask={mask.shape[:2]} for {folder.name}"
                )
                continue

        overlay = overlay_mask(
            img,
            mask,
            alpha=args.alpha,
            single_color=single_color,
            instance_colors=instance_colors,
        )

        out_dir = out_root / folder.name
        out_path = out_dir / args.frame

        if args.dry_run:
            print(f"[dry-run] {img_path} + {mask_path} -> {out_path}")
        else:
            out_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(overlay).save(out_path)

        processed += 1

    print(
        f"Done. processed={processed} skipped_missing={skipped_missing} skipped_mismatch={skipped_mismatch}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
