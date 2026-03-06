"""
Noise augmentation for train-only COCO annotations.

Takes a clean annotations_train.json and produces an augmented COCO
dataset containing the original clean images plus noisy variants
(symlinked from pre-rendered noisy PNGs).

Guardrails:
    - Two-way leakage check against split_manifest
    - Re-augmentation rejection (refuses input that already contains noisy images)
    - Per-image dimension validation (noisy PNG must match clean resolution)
    - Idempotent symlink creation

The core function returns statistics only; path-based provenance is
injected by the CLI wrapper before writing the manifest to disk.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

from PIL import Image

logger = logging.getLogger(__name__)

_SNR_SENTINEL = "__snr"


def build_noise_augmented_coco(
    coco_train: dict[str, Any],
    noisy_root: Path,
    snr_tags: list[str],
    dataset_root: Path,
    split_manifest: dict[str, Any],
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a noise-augmented COCO dataset from a clean train split.

    Args:
        coco_train: COCO dict loaded from annotations_train.json (clean only).
        noisy_root: Root of pre-rendered noisy PNGs
                    (layout: {variant}/{snr_tag}/{base_key}/0000.png).
        snr_tags: SNR profiles to include, e.g. ["snr20", "snr50"].
        dataset_root: SAM3 dataset root where image symlinks are created.
        split_manifest: Galaxy split manifest (must contain train_galaxy_ids
                        and val_galaxy_ids).
        force: If True, overwrite divergent symlinks instead of raising.

    Returns:
        (augmented_coco, noise_aug_stats) where noise_aug_stats contains
        computed counts only (no file paths).

    Raises:
        ValueError: Leakage detected, re-augmentation attempt, or
                    dimension mismatch.
        FileNotFoundError: A required noisy PNG is missing.
        FileExistsError: Symlink target conflict (when force=False).
    """
    images = coco_train["images"]
    annotations = coco_train["annotations"]

    _validate_no_leakage(images, split_manifest)
    _validate_not_augmented(images)

    img_id_to_anns: dict[int, list[dict]] = {}
    for ann in annotations:
        img_id_to_anns.setdefault(ann["image_id"], []).append(ann)

    all_images: list[dict] = []
    all_annotations: list[dict] = []

    for img in images:
        orig_id = img["id"]
        clean_img = {
            **img,
            "snr_tag": "clean",
            "source_image_id_in_base": orig_id,
            "source_file_name": img["file_name"],
        }
        all_images.append(clean_img)

        for ann in img_id_to_anns.get(orig_id, []):
            all_annotations.append({
                **ann,
                "source_annotation_id_in_base": ann["id"],
            })

    n_clean_images = len(all_images)
    n_clean_anns = len(all_annotations)

    images_dir = dataset_root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        orig_id = img["id"]
        variant = img["variant"]
        base_key = img["base_key"]

        for snr in snr_tags:
            noisy_path = noisy_root / variant / snr / base_key / "0000.png"
            if not noisy_path.exists():
                raise FileNotFoundError(
                    f"Noisy render missing: {noisy_path}"
                )

            noisy_size = Image.open(noisy_path).size  # (W, H)
            if noisy_size != (img["width"], img["height"]):
                raise ValueError(
                    f"Dimension mismatch for {noisy_path}: "
                    f"expected ({img['width']}, {img['height']}), "
                    f"got {noisy_size}"
                )

            stem = f"{base_key}_{variant}{_SNR_SENTINEL}{snr[3:]}"
            sym_name = f"{stem}.png"
            sym_path = images_dir / sym_name

            _create_symlink(sym_path, noisy_path.resolve(), force)

            noisy_img = {
                **img,
                "file_name": f"images/{sym_name}",
                "snr_tag": snr,
                "source_image_id_in_base": orig_id,
                "source_file_name": img["file_name"],
            }
            all_images.append(noisy_img)

            for ann in img_id_to_anns.get(orig_id, []):
                all_annotations.append({
                    **ann,
                    "source_annotation_id_in_base": ann["id"],
                })

    new_images, new_anns = _renumber(all_images, all_annotations)

    augmented_coco = {
        "info": coco_train.get("info", {}),
        "categories": coco_train.get("categories", []),
        "images": new_images,
        "annotations": new_anns,
    }

    n_noisy_images = len(new_images) - n_clean_images
    n_noisy_anns = len(new_anns) - n_clean_anns

    stats = {
        "dataset_variant": "noise_aug",
        "snr_tags": list(snr_tags),
        "n_images_clean": n_clean_images,
        "n_images_noisy": n_noisy_images,
        "n_images_total": len(new_images),
        "n_annotations_clean": n_clean_anns,
        "n_annotations_noisy": n_noisy_anns,
        "n_annotations_total": len(new_anns),
        "created_at": datetime.now().isoformat(),
    }

    logger.info(
        "Noise augmentation: %d clean + %d noisy = %d images, %d annotations",
        n_clean_images, n_noisy_images, len(new_images), len(new_anns),
    )

    return augmented_coco, stats


def _validate_no_leakage(
    images: list[dict[str, Any]],
    split_manifest: dict[str, Any],
) -> None:
    """Two-way galaxy leakage check."""
    input_gids = {img["galaxy_id"] for img in images}
    val_gids = set(split_manifest["val_galaxy_ids"])
    train_gids = set(split_manifest["train_galaxy_ids"])

    overlap = input_gids & val_gids
    if overlap:
        raise ValueError(
            f"Leakage: galaxies {sorted(overlap)} appear in both "
            f"input and val_galaxy_ids"
        )

    outside = input_gids - train_gids
    if outside:
        raise ValueError(
            f"Input galaxies {sorted(outside)} are not in "
            f"train_galaxy_ids from split manifest"
        )


def _validate_not_augmented(images: list[dict[str, Any]]) -> None:
    """Reject input that already contains augmented images."""
    for img in images:
        snr_tag = img.get("snr_tag")
        if snr_tag is not None and snr_tag != "clean":
            raise ValueError(
                f"Input already contains augmented images "
                f"(image {img['id']} has snr_tag={snr_tag!r}); "
                f"refusing to re-augment"
            )
        if _SNR_SENTINEL in img.get("file_name", ""):
            raise ValueError(
                f"Input already contains augmented images "
                f"(image {img['id']} file_name contains '{_SNR_SENTINEL}'); "
                f"refusing to re-augment"
            )


def _create_symlink(link: Path, target: Path, force: bool) -> None:
    """Create symlink idempotently."""
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == target:
            return
        if not force:
            raise FileExistsError(
                f"Symlink {link} exists but points to {link.resolve()}, "
                f"expected {target}. Use force=True to overwrite."
            )
        link.unlink()
    link.symlink_to(target)


def _renumber(
    images: list[dict], annotations: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Renumber image_id and annotation id to consecutive 1-based integers."""
    old_to_new_img: dict[int, int] = {}
    new_images = []
    for i, img in enumerate(images, 1):
        old_to_new_img[img["id"]] = i
        new_images.append({**img, "id": i})

    new_anns = []
    for i, ann in enumerate(annotations, 1):
        new_anns.append({
            **ann,
            "id": i,
            "image_id": old_to_new_img[ann["image_id"]],
        })

    return new_images, new_anns
