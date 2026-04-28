"""
Compose a training COCO dataset from one or more source annotation files.

Single source  -> symlink mode (O(1) switch, no modification).
Multiple sources -> merge mode (concatenate, renumber, tag dataset_source).

Precondition (shared dataset root contract):
    All source annotation files must use file_name paths relative to the
    same dataset root.  The compose layer does NOT copy, move, or
    re-symlink any images -- it only merges annotation JSON.  Image
    symlinks are the responsibility of upstream producers (export.py,
    noise_aug.py, etc.).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_NON_ID_IMAGE_FIELDS = (
    "width", "height", "galaxy_id", "view_id", "orientation",
    "variant", "base_key", "snr_tag", "noise_tag",
)

_NON_ID_ANN_FIELDS = (
    "category_id", "segmentation", "bbox", "area", "iscrowd",
)


def compose_training_coco(
    sources: list[tuple[str, Path]],
    output_path: Path,
    allow_duplicate_filenames: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Compose a training COCO dataset from one or more sources.

    Args:
        sources: list of (label, path) pairs pointing to COCO JSON files.
        output_path: where to write annotations_train_active.json.
        allow_duplicate_filenames: if False (default), raise ValueError on
            duplicate file_name across sources.  If True, verify that
            duplicate images AND their annotations have identical non-ID
            fields before deduplicating; raise ValueError otherwise.
        force: if True, overwrite existing output_path.

    Returns:
        compose_manifest dict.

    Raises:
        ValueError: duplicate file_name, category mismatch, or field
            mismatch when allow_duplicate_filenames=True.
        FileExistsError: output_path already exists with different content
            (when force=False).
    """
    if not sources:
        raise ValueError("At least one source is required")

    if len(sources) == 1:
        return _symlink_mode(sources[0], output_path, force)

    return _merge_mode(sources, output_path, allow_duplicate_filenames, force)


def _symlink_mode(
    source: tuple[str, Path],
    output_path: Path,
    force: bool,
) -> dict[str, Any]:
    """Single source: create output as symlink, no dataset_source tag."""
    label, src_path = source
    target = src_path.resolve()

    if output_path.is_symlink() or output_path.exists():
        if output_path.is_symlink() and output_path.resolve() == target:
            logger.info("Symlink already correct: %s -> %s", output_path, target)
        elif force:
            output_path.unlink()
            output_path.symlink_to(target)
            logger.info("Replaced symlink: %s -> %s", output_path, target)
        else:
            raise FileExistsError(
                f"{output_path} already exists (points to "
                f"{output_path.resolve() if output_path.is_symlink() else 'regular file'}). "
                f"Use force=True to overwrite."
            )
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.symlink_to(target)
        logger.info("Created symlink: %s -> %s", output_path, target)

    coco = json.loads(src_path.read_text())

    manifest = {
        "dataset_variant": "composed",
        "mode": "symlink",
        "sources": [{
            "label": label,
            "path": str(src_path),
            "n_images": len(coco.get("images", [])),
            "n_annotations": len(coco.get("annotations", [])),
        }],
        "n_images_total": len(coco.get("images", [])),
        "n_annotations_total": len(coco.get("annotations", [])),
        "created_at": datetime.now().isoformat(),
    }

    _write_manifest(output_path.parent, manifest)
    return manifest


def _merge_mode(
    sources: list[tuple[str, Path]],
    output_path: Path,
    allow_duplicate_filenames: bool,
    force: bool,
) -> dict[str, Any]:
    """Multiple sources: merge, renumber, tag dataset_source."""
    loaded: list[tuple[str, dict]] = []
    for label, path in sources:
        loaded.append((label, json.loads(path.read_text())))

    _verify_categories([coco for _, coco in loaded])

    seen_filenames: dict[str, tuple[str, dict, list[dict]]] = {}
    all_images: list[dict] = []
    all_annotations: list[dict] = []

    for label, coco in loaded:
        ann_by_img: dict[int, list[dict]] = {}
        for ann in coco.get("annotations", []):
            ann_by_img.setdefault(ann["image_id"], []).append(ann)

        for img in coco.get("images", []):
            fname = img["file_name"]
            img_anns = ann_by_img.get(img["id"], [])

            if fname in seen_filenames:
                prev_label, prev_img, prev_anns = seen_filenames[fname]
                if not allow_duplicate_filenames:
                    raise ValueError(
                        f"Duplicate file_name '{fname}' in sources "
                        f"'{prev_label}' and '{label}'"
                    )
                _verify_duplicate_identical(
                    fname, prev_label, prev_img, prev_anns,
                    label, img, img_anns,
                )
                continue

            tagged_img = {**img, "dataset_source": label}
            all_images.append(tagged_img)
            all_annotations.extend(img_anns)
            seen_filenames[fname] = (label, img, img_anns)

    new_images, new_anns = _renumber(all_images, all_annotations)
    categories = loaded[0][1].get("categories", [])

    merged_coco = {
        "info": loaded[0][1].get("info", {}),
        "categories": categories,
        "images": new_images,
        "annotations": new_anns,
    }

    if output_path.is_symlink() or output_path.exists():
        if not force:
            raise FileExistsError(
                f"{output_path} already exists. Use force=True to overwrite."
            )
        output_path.unlink()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(merged_coco, indent=2))
    logger.info("Wrote merged dataset: %s", output_path)

    source_info = []
    for label, coco in loaded:
        path = dict(sources)[label]
        source_info.append({
            "label": label,
            "path": str(path),
            "n_images": len(coco.get("images", [])),
            "n_annotations": len(coco.get("annotations", [])),
        })

    manifest = {
        "dataset_variant": "composed",
        "mode": "merged",
        "sources": source_info,
        "n_images_total": len(new_images),
        "n_annotations_total": len(new_anns),
        "created_at": datetime.now().isoformat(),
    }

    _write_manifest(output_path.parent, manifest)
    return manifest


def _verify_categories(cocos: list[dict]) -> None:
    """All sources must share identical categories."""
    ref = cocos[0].get("categories", [])
    for i, coco in enumerate(cocos[1:], 1):
        cats = coco.get("categories", [])
        if cats != ref:
            raise ValueError(
                f"Category mismatch: source 0 has {ref}, "
                f"source {i} has {cats}"
            )


def _verify_duplicate_identical(
    fname: str,
    label_a: str, img_a: dict, anns_a: list[dict],
    label_b: str, img_b: dict, anns_b: list[dict],
) -> None:
    """When allow_duplicate_filenames=True, verify duplicates are identical."""
    for field in _NON_ID_IMAGE_FIELDS:
        va, vb = img_a.get(field), img_b.get(field)
        if va != vb:
            raise ValueError(
                f"Duplicate file_name '{fname}' has mismatched "
                f"image field '{field}': '{label_a}' has {va!r}, "
                f"'{label_b}' has {vb!r}"
            )

    if len(anns_a) != len(anns_b):
        raise ValueError(
            f"Duplicate file_name '{fname}' has different annotation "
            f"counts: '{label_a}' has {len(anns_a)}, "
            f"'{label_b}' has {len(anns_b)}"
        )

    sorted_a = sorted(anns_a, key=lambda a: tuple(a.get(f, "") for f in _NON_ID_ANN_FIELDS))
    sorted_b = sorted(anns_b, key=lambda a: tuple(a.get(f, "") for f in _NON_ID_ANN_FIELDS))

    for aa, bb in zip(sorted_a, sorted_b):
        for field in _NON_ID_ANN_FIELDS:
            va, vb = aa.get(field), bb.get(field)
            if va != vb:
                raise ValueError(
                    f"Duplicate file_name '{fname}' has mismatched "
                    f"annotation field '{field}': '{label_a}' has {va!r}, "
                    f"'{label_b}' has {vb!r}"
                )


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


def _write_manifest(directory: Path, manifest: dict) -> None:
    path = directory / "compose_manifest.json"
    path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote compose manifest: %s", path)
