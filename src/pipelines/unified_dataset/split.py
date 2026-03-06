"""
Galaxy-level COCO train/val split.

Splits annotations.json so that ALL images of a given galaxy end up in
the same partition (train or val), preventing data leakage across
orientations, variants, or noise profiles of the same galaxy.

Split stability:
    Uses hash(f"{seed}:{galaxy_id}") to assign each galaxy deterministically.
    Adding new galaxies never reassigns existing ones.

    When reuse_manifest is provided, previously assigned galaxies keep their
    prior split; only new galaxies are hash-assigned.
"""
from __future__ import annotations

import hashlib
import logging
import re
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

_GALAXY_ID_RE = re.compile(r"(\d{5})_")


def _extract_galaxy_id(img: dict[str, Any]) -> int:
    """Extract galaxy_id from COCO image entry.

    Primary: structured 'galaxy_id' field (added by export.py).
    Fallback: parse 5-digit prefix from file_name.
    """
    gid = img.get("galaxy_id")
    if gid is not None:
        return int(gid)
    m = _GALAXY_ID_RE.search(img.get("file_name", ""))
    if m:
        logger.warning("galaxy_id field missing; falling back to filename parsing")
        return int(m.group(1))
    raise ValueError(f"Cannot extract galaxy_id from image entry: {img}")


def _hash_assign(seed: int, galaxy_id: int, train_ratio: float) -> str:
    """Deterministic hash-based train/val assignment."""
    h = hashlib.sha256(f"{seed}:{galaxy_id}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if frac < train_ratio else "val"


def galaxy_split_coco(
    coco: dict[str, Any],
    train_ratio: float = 0.8,
    seed: int = 42,
    reuse_manifest: dict[str, Any] | None = None,
) -> tuple[dict, dict, dict]:
    """Split a COCO dataset at the galaxy level.

    Returns (coco_train, coco_val, split_manifest).
    """
    images = coco["images"]
    annotations = coco["annotations"]

    # Group images by galaxy_id
    galaxy_to_imgs: dict[int, list[dict]] = {}
    for img in images:
        gid = _extract_galaxy_id(img)
        galaxy_to_imgs.setdefault(gid, []).append(img)

    all_galaxies = sorted(galaxy_to_imgs.keys())

    # Build assignment map
    prior_train = set()
    prior_val = set()
    if reuse_manifest:
        prior_train = set(reuse_manifest.get("train_galaxy_ids", []))
        prior_val = set(reuse_manifest.get("val_galaxy_ids", []))

    train_galaxies: list[int] = []
    val_galaxies: list[int] = []

    for gid in all_galaxies:
        if gid in prior_train:
            train_galaxies.append(gid)
        elif gid in prior_val:
            val_galaxies.append(gid)
        else:
            assignment = _hash_assign(seed, gid, train_ratio)
            if assignment == "train":
                train_galaxies.append(gid)
            else:
                val_galaxies.append(gid)

    train_image_ids = set()
    val_image_ids = set()

    for gid in train_galaxies:
        for img in galaxy_to_imgs[gid]:
            train_image_ids.add(img["id"])
    for gid in val_galaxies:
        for img in galaxy_to_imgs[gid]:
            val_image_ids.add(img["id"])

    # Partition
    train_images_raw = [img for img in images if img["id"] in train_image_ids]
    val_images_raw = [img for img in images if img["id"] in val_image_ids]
    train_anns_raw = [a for a in annotations if a["image_id"] in train_image_ids]
    val_anns_raw = [a for a in annotations if a["image_id"] in val_image_ids]

    # Renumber to consecutive 1-based IDs
    train_images, train_anns = _renumber(train_images_raw, train_anns_raw)
    val_images, val_anns = _renumber(val_images_raw, val_anns_raw)

    shared = {"categories": coco.get("categories", []), "info": coco.get("info", {})}

    coco_train = {**shared, "images": train_images, "annotations": train_anns}
    coco_val = {**shared, "images": val_images, "annotations": val_anns}

    manifest = {
        "source_annotations": coco.get("info", {}).get("description", ""),
        "train_ratio": train_ratio,
        "seed": seed,
        "train_galaxy_ids": sorted(train_galaxies),
        "val_galaxy_ids": sorted(val_galaxies),
        "n_train_images": len(train_images),
        "n_val_images": len(val_images),
        "n_train_annotations": len(train_anns),
        "n_val_annotations": len(val_anns),
        "created_at": datetime.now().isoformat(),
    }

    return coco_train, coco_val, manifest


def _renumber(
    images: list[dict], annotations: list[dict]
) -> tuple[list[dict], list[dict]]:
    """Renumber image_id and annotation id to consecutive 1-based integers."""
    old_to_new_img: dict[int, int] = {}
    new_images = []
    for i, img in enumerate(images, 1):
        old_to_new_img[img["id"]] = i
        new_img = {**img, "id": i}
        new_images.append(new_img)

    new_annotations = []
    for i, ann in enumerate(annotations, 1):
        new_ann = {**ann, "id": i, "image_id": old_to_new_img[ann["image_id"]]}
        new_annotations.append(new_ann)

    return new_images, new_annotations
