"""
Noise augmentation for split-aware COCO annotations.

Takes a clean split JSON (train or val) and produces an augmented COCO
dataset containing the original clean images plus noisy variants
(symlinked from pre-rendered noisy PNGs).

Guardrails:
    - Split-aware leakage check against split_manifest
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
from typing import Any, Literal

from PIL import Image

logger = logging.getLogger(__name__)

_NOISE_SENTINEL = "__noise_"
_LEGACY_SNR_SENTINEL = "__snr"
_DEFAULT_NOISY_VARIANTS = {"linear_magnitude"}


def _sanitize_tag(tag: str) -> str:
    """Make a profile tag safe for filenames while preserving readability."""
    return tag.replace(".", "p")


def build_noise_augmented_coco(
    coco_split: dict[str, Any],
    noisy_root: Path,
    noise_tags: list[str],
    dataset_root: Path,
    split_manifest: dict[str, Any],
    target_split: Literal["train", "val"],
    noisy_variants: set[str] | None = None,
    force: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build a split-aware noise-augmented COCO dataset from a clean split.

    Args:
        coco_split: COCO dict loaded from annotations_{split}.json (clean only).
        noisy_root: Root of pre-rendered noisy PNGs
                    (layout: {variant}/{noise_tag}/{base_key}/0000.png).
        noise_tags: Noise profiles to include, e.g. ["sb30", "sb31.5"].
        dataset_root: SAM3 dataset root where image symlinks are created.
        split_manifest: Galaxy split manifest (must contain train_galaxy_ids
                        and val_galaxy_ids).
        target_split: Which split is being augmented ("train" or "val").
        noisy_variants: Variants that should receive noisy clones.
        force: If True, overwrite divergent symlinks instead of raising.

    Returns:
        (augmented_coco, noise_aug_stats) where noise_aug_stats contains
        computed counts only (no file paths).

    Raises:
        ValueError: Leakage detected, re-augmentation attempt, invalid split,
                    or dimension mismatch.
        FileNotFoundError: A required noisy PNG is missing.
        FileExistsError: Symlink target conflict (when force=False).
    """
    if target_split not in {"train", "val"}:
        raise ValueError(
            f"target_split must be 'train' or 'val', got {target_split!r}"
        )

    noisy_variants = set(noisy_variants or _DEFAULT_NOISY_VARIANTS)

    images = coco_split["images"]
    annotations = coco_split["annotations"]

    _validate_no_leakage(images, split_manifest, target_split)
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
            "noise_tag": "clean",
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
        if variant not in noisy_variants:
            continue

        base_key = img["base_key"]

        for noise_tag in noise_tags:
            noisy_path = noisy_root / variant / noise_tag / base_key / "0000.png"
            if not noisy_path.exists():
                raise FileNotFoundError(
                    f"Noisy render missing: {noisy_path}"
                )

            with Image.open(noisy_path) as noisy_image:
                noisy_size = noisy_image.size
            if noisy_size != (img["width"], img["height"]):
                raise ValueError(
                    f"Dimension mismatch for {noisy_path}: "
                    f"expected ({img['width']}, {img['height']}), "
                    f"got {noisy_size}"
                )

            stem = (
                f"{base_key}_{variant}"
                f"{_NOISE_SENTINEL}{_sanitize_tag(noise_tag)}"
            )
            sym_name = f"{stem}.png"
            sym_path = images_dir / sym_name

            _create_symlink(sym_path, noisy_path.resolve(), force)

            noisy_img = {
                **img,
                "file_name": f"images/{sym_name}",
                "snr_tag": noise_tag,
                "noise_tag": noise_tag,
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
        "info": coco_split.get("info", {}),
        "categories": coco_split.get("categories", []),
        "images": new_images,
        "annotations": new_anns,
    }

    n_noisy_images = len(new_images) - n_clean_images
    n_noisy_anns = len(new_anns) - n_clean_anns

    stats = {
        "dataset_variant": "noise_aug",
        "target_split": target_split,
        "noise_tags": list(noise_tags),
        "noisy_variants": sorted(noisy_variants),
        "n_images_clean": n_clean_images,
        "n_images_noisy": n_noisy_images,
        "n_images_total": len(new_images),
        "n_annotations_clean": n_clean_anns,
        "n_annotations_noisy": n_noisy_anns,
        "n_annotations_total": len(new_anns),
        "created_at": datetime.now().isoformat(),
    }

    logger.info(
        "Noise augmentation (%s): %d clean + %d noisy = %d images, %d annotations",
        target_split,
        n_clean_images,
        n_noisy_images,
        len(new_images),
        len(new_anns),
    )

    return augmented_coco, stats


def _validate_no_leakage(
    images: list[dict[str, Any]],
    split_manifest: dict[str, Any],
    target_split: Literal["train", "val"],
) -> None:
    """Split-aware galaxy leakage check."""
    input_gids = {img["galaxy_id"] for img in images}
    val_gids = set(split_manifest["val_galaxy_ids"])
    train_gids = set(split_manifest["train_galaxy_ids"])

    if target_split == "train":
        allowed_gids = train_gids
        forbidden_gids = val_gids
        allowed_label = "train_galaxy_ids"
        forbidden_label = "val_galaxy_ids"
    else:
        allowed_gids = val_gids
        forbidden_gids = train_gids
        allowed_label = "val_galaxy_ids"
        forbidden_label = "train_galaxy_ids"

    overlap = input_gids & forbidden_gids
    if overlap:
        raise ValueError(
            f"Leakage: galaxies {sorted(overlap)} appear in both "
            f"input and {forbidden_label}"
        )

    outside = input_gids - allowed_gids
    if outside:
        raise ValueError(
            f"Input galaxies {sorted(outside)} are not in "
            f"{allowed_label} from split manifest"
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

        noise_tag = img.get("noise_tag")
        if noise_tag is not None and noise_tag != "clean":
            raise ValueError(
                f"Input already contains augmented images "
                f"(image {img['id']} has noise_tag={noise_tag!r}); "
                f"refusing to re-augment"
            )

        file_name = img.get("file_name", "")
        if _LEGACY_SNR_SENTINEL in file_name or _NOISE_SENTINEL in file_name:
            raise ValueError(
                f"Input already contains augmented images "
                f"(image {img['id']} file_name contains a noise sentinel); "
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
