"""
Loader helpers for the MATLAS_Sola2025_eval dataset.

Provides a Dataset class and a quick CLI for inspecting the per-image GT.
Designed so the eval pipeline can iterate (image, gt_masks, metadata) tuples
in a way compatible with the firebox_dr1_streams benchmark in
configs/eval_checkpoint.yaml.

Default class binding (override via --bind):
    positive = {Stream, Tail, Halo, Shells}    -> "stellar stream" prompt
    negative = {Inner}                          -> dropped from positive mask
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
from PIL import Image


DEFAULT_ROOT = Path(__file__).resolve().parents[3] / "data/01_raw/MATLAS_Sola2025_eval"

DEFAULT_BIND = {
    "Stream": "positive",
    "Tail":   "positive",
    "Halo":   "positive",
    "Shells": "positive",
    "Inner":  "ignore",
}


@dataclass
class MatlasSample:
    galaxy: str
    image_path: Path
    rgb: np.ndarray              # (H, W, 3) uint8
    gt_mask_positive: np.ndarray # (H, W) uint8, instance IDs (0 = background)
    gt_mask_class: dict[str, np.ndarray]  # per-class binary masks (uint8)
    metadata: dict


class MatlasSolaDataset:
    def __init__(self, root: Path = DEFAULT_ROOT,
                 image_variant: str = "linear_magnitude",
                 bind: dict[str, str] | None = None,
                 shell_eval_thickness_px: int = 3) -> None:
        self.root = Path(root)
        self.image_dir = self.root / f"png_{image_variant}"
        self.per_image_dir = self.root / "annotations" / "per_image"
        self.bind = bind or DEFAULT_BIND
        self.shell_eval_thickness_px = shell_eval_thickness_px
        self.taxonomy = json.loads((self.root / "annotations" / "taxonomy.json").read_text())
        manifest = self.root / "manifest.csv"
        import csv
        with manifest.open() as fh:
            reader = csv.DictReader(fh)
            self.entries = [row for row in reader]
        self.galaxies = [r["galaxy"] for r in self.entries]

    def __len__(self) -> int:
        return len(self.entries)

    def __iter__(self) -> Iterator[MatlasSample]:
        for row in self.entries:
            yield self.load(row["galaxy"])

    def load(self, galaxy: str) -> MatlasSample:
        img_path = self.image_dir / f"{galaxy}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"missing rendered PNG: {img_path}")
        rgb = np.array(Image.open(img_path).convert("RGB"))

        per_img_path = self.per_image_dir / f"{galaxy}.json"
        meta = json.loads(per_img_path.read_text())

        H, W = rgb.shape[:2]
        gt_pos = np.zeros((H, W), dtype=np.uint16)
        per_class: dict[str, np.ndarray] = {}

        import cv2
        next_inst = 1
        for poly in meta["polygons"]:
            cls = poly["type"]
            shape = poly.get("shape", "polygon")
            pts = np.array(poly["polygon_xy"], dtype=np.int32).reshape(-1, 1, 2)
            mask = np.zeros((H, W), dtype=np.uint8)
            if shape == "polygon":
                cv2.fillPoly(mask, [pts], color=1)
            else:
                cv2.polylines(mask, [pts], isClosed=False, color=1,
                              thickness=self.shell_eval_thickness_px)
            if mask.sum() == 0:
                continue
            per_class.setdefault(cls, np.zeros((H, W), dtype=np.uint8))
            per_class[cls] = np.maximum(per_class[cls], mask)
            if self.bind.get(cls) == "positive":
                gt_pos[(mask > 0) & (gt_pos == 0)] = next_inst
                next_inst += 1

        return MatlasSample(
            galaxy=galaxy, image_path=img_path, rgb=rgb,
            gt_mask_positive=gt_pos.astype(np.uint8),
            gt_mask_class=per_class,
            metadata=meta,
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--variant", default="linear_magnitude",
                    choices=["linear_magnitude", "asinh_stretch"])
    ap.add_argument("--galaxy", default=None,
                    help="Inspect a single galaxy and print its mask coverage.")
    ap.add_argument("--limit", type=int, default=5,
                    help="When iterating, stop after this many samples (debug).")
    args = ap.parse_args()

    ds = MatlasSolaDataset(args.root, image_variant=args.variant)
    print(f"loaded MATLAS_Sola2025_eval: {len(ds)} galaxies, variant={args.variant}")

    targets = [args.galaxy] if args.galaxy else ds.galaxies[:args.limit]
    for g in targets:
        sample = ds.load(g)
        cls_summary = {k: int(v.sum()) for k, v in sample.gt_mask_class.items()}
        pos_pixels = int((sample.gt_mask_positive > 0).sum())
        n_inst = int(sample.gt_mask_positive.max())
        print(f"  {g:<10s} pos_pixels={pos_pixels:>9d}  instances={n_inst:>3d}  "
              f"per_class={cls_summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
