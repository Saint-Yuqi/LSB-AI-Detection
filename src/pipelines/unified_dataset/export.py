"""
Phase 4: Generate SAM2 symlinks and SAM3 annotations.json.

Output:
    sam2_prepared/img_folder/... + gt_folder/... (symlinks)
    sam3_prepared/images/... + annotations.json (COCO format)
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

import numpy as np
from PIL import Image

from src.utils.coco_utils import mask_to_rle, get_bbox_from_mask
from .keys import BaseKey, VariantKey
from .paths import PathResolver


def run_export_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Generate SAM2 symlinks and SAM3 annotations.json."""
    logger.info("=" * 60)
    logger.info("PHASE 4: EXPORT (SAM2 + SAM3)")
    logger.info("=" * 60)

    resolver = PathResolver(config)
    variants = config["preprocessing_variants"]
    target_size = tuple(config["processing"]["target_size"])

    sam2_dir = resolver.get_sam2_dir()
    sam3_dir = resolver.get_sam3_dir()

    (sam2_dir / "img_folder").mkdir(parents=True, exist_ok=True)
    (sam2_dir / "gt_folder").mkdir(parents=True, exist_ok=True)
    (sam3_dir / "images").mkdir(parents=True, exist_ok=True)

    sam3_images: list[dict] = []
    sam3_annotations: list[dict] = []
    sam3_categories = [
        {"id": 1, "name": "stellar stream", "supercategory": "lsb"},
        {"id": 2, "name": "satellite galaxy", "supercategory": "lsb"},
    ]
    image_id = 0
    ann_id = 0

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        instance_map_path = gt_dir / "instance_map_uint8.png"
        instances_path = gt_dir / "instances.json"

        if not instance_map_path.exists():
            continue

        instances = json.loads(instances_path.read_text()) if instances_path.exists() else []
        instance_map = np.array(Image.open(instance_map_path))
        H, W = instance_map.shape

        for variant in variants:
            vname = variant["name"]
            variant_key = VariantKey(key, vname)

            render_path = resolver.get_render_dir(vname, key) / "0000.png"
            if not render_path.exists():
                continue

            # SAM2: Create symlinks
            sam2_img_dir = sam2_dir / "img_folder" / str(variant_key)
            sam2_gt_dir = sam2_dir / "gt_folder" / str(variant_key)
            sam2_img_dir.mkdir(parents=True, exist_ok=True)
            sam2_gt_dir.mkdir(parents=True, exist_ok=True)

            sam2_img_link = sam2_img_dir / "0000.png"
            sam2_gt_link = sam2_gt_dir / "0000.png"

            if force_variants and vname in force_variants:
                if sam2_img_link.exists() or sam2_img_link.is_symlink():
                    sam2_img_link.unlink()
                if sam2_gt_link.exists() or sam2_gt_link.is_symlink():
                    sam2_gt_link.unlink()

            if not sam2_img_link.exists():
                sam2_img_link.symlink_to(render_path.resolve())
            if not sam2_gt_link.exists():
                sam2_gt_link.symlink_to(instance_map_path.resolve())

            # SAM3: Flat image copy + annotations
            sam3_img_name = f"{variant_key}.png"
            sam3_img_path = sam3_dir / "images" / sam3_img_name

            if force_variants and vname in force_variants:
                if sam3_img_path.exists() or sam3_img_path.is_symlink():
                    sam3_img_path.unlink()

            if not sam3_img_path.exists():
                sam3_img_path.symlink_to(render_path.resolve())

            image_id += 1
            sam3_images.append({
                "id": image_id,
                "file_name": f"images/{sam3_img_name}",
                "width": W,
                "height": H,
                "galaxy_id": key.galaxy_id,
                "view_id": key.view_id,
                "orientation": key.view_id,
                "variant": vname,
                "base_key": str(key),
            })

            for inst in instances:
                inst_id = inst["id"]
                inst_type = inst["type"]
                category_id = 1 if inst_type == "streams" else 2

                binary_mask = (instance_map == inst_id).astype(np.uint8)
                if binary_mask.sum() == 0:
                    continue

                rle = mask_to_rle(binary_mask)
                bbox = get_bbox_from_mask(binary_mask)
                area = int(binary_mask.sum())

                ann_id += 1
                sam3_annotations.append({
                    "id": ann_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": rle,
                    "bbox": bbox,
                    "area": area,
                    "iscrowd": 0,
                })

    coco = {
        "info": {
            "description": "LSB-AI-Detection Unified Dataset",
            "version": "1.0",
            "date_created": datetime.now().isoformat(),
        },
        "images": sam3_images,
        "annotations": sam3_annotations,
        "categories": sam3_categories,
    }
    ann_filename = config.get("export_phase", {}).get("annotations_filename", "annotations.json")
    (sam3_dir / ann_filename).write_text(json.dumps(coco, indent=2))

    logger.info(f"SAM2: {len(base_keys) * len(variants)} symlinks")
    logger.info(f"SAM3: {len(sam3_images)} images, {len(sam3_annotations)} annotations")
