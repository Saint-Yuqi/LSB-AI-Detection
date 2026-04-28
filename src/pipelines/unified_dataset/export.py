"""
Phase 4: Generate SAM3 annotations.json (COCO-format).

Output:
    sam3_prepared/images/... + annotations.json
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
from .taxonomy import CATEGORIES, CATEGORY_ID_BY_TYPE, normalize_type_label
from src.review.authoritative_gt import extract_annotation_provenance


def run_export_phase(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force_variants: set[str] | None = None,
) -> None:
    """Generate SAM3 annotations.json (COCO) and image symlinks."""
    logger.info("=" * 60)
    logger.info("PHASE 4: EXPORT (SAM3)")
    logger.info("=" * 60)

    resolver = PathResolver(config)
    variants = config["preprocessing_variants"]
    dataset_name = resolver.dataset_name
    authoritative_conditions = [
        condition
        for condition in resolver.get_active_conditions()
        if resolver.get_label_mode(condition) == "authoritative"
    ] or ["clean"]

    sam3_dir = resolver.get_sam3_dir(dataset_name)
    (sam3_dir / "images").mkdir(parents=True, exist_ok=True)

    sam3_images: list[dict] = []
    sam3_annotations: list[dict] = []
    # Three-class taxonomy on every export. Legacy 2-class GT folds in via
    # normalize_type_label (streams -> tidal_features), so a mixed cohort
    # rolls up cleanly. inner_galaxy is reported as id 3 even when no rows
    # of that class exist in a particular GT.
    sam3_categories = list(CATEGORIES)
    image_id = 0
    ann_id = 0

    for condition in authoritative_conditions:
        for key in base_keys:
            gt_dir = (
                resolver.get_gt_dir(key)
                if dataset_name == "dr1"
                else resolver.get_pseudo_gt_dir(key, dataset=dataset_name, condition=condition)
            )
            instances_path = gt_dir / "instances.json"
            if not instances_path.exists():
                continue
            instances = json.loads(instances_path.read_text())

            # Detect path version by file presence. New (tidal_v1) GT carries
            # tidal_features_instance_map.npy; legacy GT carries a single
            # instance_map_uint8.png.
            tidal_npy = gt_dir / "tidal_features_instance_map.npy"
            on_new_path = tidal_npy.exists()
            legacy_uint8_map_cache: list[np.ndarray] = []  # lazy single-element cache

            def _legacy_uint8_map() -> np.ndarray:
                if not legacy_uint8_map_cache:
                    legacy_uint8_map_cache.append(
                        np.array(Image.open(gt_dir / "instance_map_uint8.png")).astype(np.int32)
                    )
                return legacy_uint8_map_cache[0]

            # Build prediction RLE index keyed by raw_index for SAM-derived
            # rows (F2: PNbody tidal needs JSON RLEs to preserve within-class
            # overlap; F10: the JSON key is "rle", not "segmentation").
            predictions_index: dict[int, dict[str, Any]] = {}
            pred_path = gt_dir / "sam3_predictions_post.json"
            if pred_path.exists():
                pred_doc = json.loads(pred_path.read_text())
                for p in pred_doc.get("predictions", []):
                    if "raw_index" in p and "rle" in p:
                        predictions_index[int(p["raw_index"])] = p["rle"]

            # Per-class map cache: {map_filename: np.ndarray int32}
            per_class_map_cache: dict[str, np.ndarray] = {}

            def _per_class_map(filename: str) -> np.ndarray:
                if filename not in per_class_map_cache:
                    per_class_map_cache[filename] = np.load(gt_dir / filename).astype(np.int32)
                return per_class_map_cache[filename]

            # Resolve image dimensions from whichever map is available.
            if on_new_path:
                _arr = _per_class_map("tidal_features_instance_map.npy")
                H, W = _arr.shape
            else:
                _arr = _legacy_uint8_map()
                H, W = _arr.shape

            for variant in variants:
                vname = variant["name"]
                variant_key = VariantKey(key, vname)

                render_path = resolver.get_render_dir(
                    vname,
                    key,
                    dataset=dataset_name,
                    condition=condition,
                ) / "0000.png"
                if not render_path.exists():
                    continue

                variant_key_str = str(variant_key)
                if dataset_name != "dr1" or condition != "clean":
                    variant_key_str = f"{variant_key_str}__{condition}"

                sam3_img_name = f"{variant_key_str}.png"
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
                    "dataset": dataset_name,
                    "condition": condition,
                    "label_mode": resolver.get_label_mode(condition),
                })

                for inst in instances:
                    type_label = normalize_type_label(inst.get("type_label") or inst["type"])
                    category_id = CATEGORY_ID_BY_TYPE[type_label]

                    raw_idx = inst.get("raw_index")
                    source = inst.get("source")
                    rle: dict[str, Any] | None = None

                    if raw_idx is not None and int(raw_idx) in predictions_index:
                        # SAM-derived: prefer the RLE from sam3_predictions_post.json
                        # so within-class overlap is preserved (F2).
                        rle = predictions_index[int(raw_idx)]
                    elif source == "sam3_post":
                        # F20: fail closed. A row that declares it came from
                        # the SAM3 pipeline must have a matching prediction
                        # RLE; falling back to a per-class map decode would
                        # silently collapse within-class overlap.
                        raise KeyError(
                            f"sam3_post row {inst.get('candidate_id')!r} (raw_index={raw_idx}) "
                            f"missing from {pred_path}; refusing to fall back to map decode."
                        )
                    elif "map_file" in inst and "local_id" in inst:
                        # New-path FITS-derived (DR1 tidal_features): decode
                        # from the per-class instance map.
                        m_map = _per_class_map(inst["map_file"])
                        binary_mask = (m_map == int(inst["local_id"])).astype(np.uint8)
                        if binary_mask.sum() == 0:
                            continue
                        rle = mask_to_rle(binary_mask)
                    else:
                        # Legacy row (F14): only id + type, mask comes from
                        # the legacy instance_map_uint8.png.
                        binary_mask = (_legacy_uint8_map() == int(inst["id"])).astype(np.uint8)
                        if binary_mask.sum() == 0:
                            continue
                        rle = mask_to_rle(binary_mask)

                    if rle is None:
                        continue

                    # Decode once for bbox + area (cheap relative to the JSON write).
                    from src.utils.coco_utils import decode_rle as _decode_rle
                    binary_mask = _decode_rle(rle).astype(np.uint8)
                    if binary_mask.sum() == 0:
                        continue
                    bbox = get_bbox_from_mask(binary_mask)
                    area = int(binary_mask.sum())

                    ann_id += 1
                    annotation = {
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "segmentation": rle,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                    annotation.update(extract_annotation_provenance(inst))
                    sam3_annotations.append(annotation)

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

    logger.info(f"SAM3 root: {sam3_dir}")
    logger.info(f"SAM3: {len(sam3_images)} images, {len(sam3_annotations)} annotations")
