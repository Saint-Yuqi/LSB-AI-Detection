#!/usr/bin/env python3
"""
Rebuild sam3_prepared in place from fresh GT + locked split + sb noise tiers.

Usage:
    conda run -n sam3 python scripts/data/rebuild_sam3_dataset.py
    conda run -n sam3 python scripts/data/rebuild_sam3_dataset.py \
        --config configs/unified_data_prep.yaml \
        --previous-root data/02_processed/sam3_prepared_previous
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.compose import compose_training_coco
from src.pipelines.unified_dataset.config import generate_base_keys, load_config
from src.pipelines.unified_dataset.export import run_export_phase
from src.pipelines.unified_dataset.noise_aug import build_noise_augmented_coco
from src.pipelines.unified_dataset.paths import PathResolver
from src.pipelines.unified_dataset.split import galaxy_split_coco

EXPECTED_GT_BASE_KEYS = 70
EXPECTED_SPLIT_GALAXY_COUNTS = {"train": 27, "val": 8}
DEFAULT_NOISE_TAGS = ["sb31.5"]

EXPECTED_COUNTS = {
    "annotations.json": {"images": 140, "annotations": 1654},
    "annotations_train.json": {"images": 108, "annotations": 1174},
    "annotations_val.json": {"images": 32, "annotations": 480},
}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _ensure_missing_or_empty(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not any(path.iterdir()):
        return
    raise FileExistsError(
        f"{path} already exists and is not empty. Move it aside before rebuilding."
    )


def _make_base_key_re(views: list[str]) -> re.Pattern[str]:
    view_pattern = "|".join(re.escape(view) for view in sorted(views))
    return re.compile(rf"^\d{{5}}_({view_pattern})$")


def _collect_gt_keys(
    gt_root: Path,
    *,
    base_key_re: re.Pattern[str],
    expected_gt_base_keys: int | None = None,
) -> list[str]:
    if not gt_root.exists():
        raise FileNotFoundError(f"GT root not found: {gt_root}")

    keys: list[str] = []
    for subdir in sorted(p for p in gt_root.iterdir() if p.is_dir()):
        if not base_key_re.match(subdir.name):
            raise ValueError(
                f"Unexpected GT key {subdir.name!r}; does not match configured views."
            )
        for required in ("instance_map_uint8.png", "instances.json"):
            required_path = subdir / required
            if not required_path.exists():
                raise FileNotFoundError(f"Missing GT artifact: {required_path}")
        keys.append(subdir.name)

    if expected_gt_base_keys is not None and len(keys) != expected_gt_base_keys:
        raise ValueError(
            f"Expected {expected_gt_base_keys} GT base keys under {gt_root}, "
            f"found {len(keys)}"
        )
    return keys


def _validate_noise_coverage(
    noisy_root: Path,
    gt_keys: list[str],
    variant: str,
    noise_tag: str,
    *,
    base_key_re: re.Pattern[str],
) -> None:
    profile_dir = noisy_root / variant / noise_tag
    if not profile_dir.exists():
        raise FileNotFoundError(f"Noise profile directory not found: {profile_dir}")

    available = {
        path.name
        for path in profile_dir.iterdir()
        if path.is_dir() and base_key_re.match(path.name)
    }
    missing = sorted(set(gt_keys) - available)
    if missing:
        raise ValueError(
            f"{noise_tag} is missing {len(missing)} GT-matched {variant} keys, "
            f"e.g. {missing[:5]}"
        )


def _validate_counts(path: Path, expected: dict[str, int]) -> None:
    coco = _load_json(path)
    n_images = len(coco.get("images", []))
    n_annotations = len(coco.get("annotations", []))
    if n_images != expected["images"] or n_annotations != expected["annotations"]:
        raise ValueError(
            f"Unexpected counts for {path.name}: "
            f"images={n_images} annotations={n_annotations}, "
            f"expected {expected}"
        )


def _expected_augmented_counts(
    coco_split: dict[str, Any],
    *,
    noise_tags: list[str],
    noisy_variants: set[str],
) -> dict[str, int]:
    """Compute expected counts after cloning eligible images across noise tags."""
    eligible_image_ids = {
        img["id"]
        for img in coco_split.get("images", [])
        if img.get("variant") in noisy_variants
    }
    eligible_annotations = sum(
        1
        for ann in coco_split.get("annotations", [])
        if ann.get("image_id") in eligible_image_ids
    )

    n_clean_images = len(coco_split.get("images", []))
    n_clean_annotations = len(coco_split.get("annotations", []))
    n_noise_tags = len(noise_tags)

    return {
        "images": n_clean_images + (len(eligible_image_ids) * n_noise_tags),
        "annotations": n_clean_annotations + (eligible_annotations * n_noise_tags),
    }


def _validate_views(
    coco: dict[str, Any],
    *,
    allowed_views: set[str],
    base_key_re: re.Pattern[str],
) -> None:
    bad = [
        img["file_name"]
        for img in coco.get("images", [])
        if img.get("view_id") not in allowed_views
        or not base_key_re.match(img.get("base_key", ""))
    ]
    if bad:
        raise ValueError(
            f"Unexpected view/base_key entries found, e.g. {bad[:5]}"
        )


def _validate_noisy_asinh(coco: dict[str, Any]) -> None:
    bad = [
        img["file_name"]
        for img in coco.get("images", [])
        if img.get("variant") != "linear_magnitude"
        and (
            img.get("noise_tag") not in (None, "clean")
            or "__noise_" in img.get("file_name", "")
        )
    ]
    if bad:
        raise ValueError(
            f"Found noisy images outside linear_magnitude, e.g. {bad[:5]}"
        )


def _validate_split_manifest(
    manifest: dict[str, Any],
    *,
    expected_galaxy_ids: set[int] | None = None,
    expected_split_counts: dict[str, int] | None = None,
) -> None:
    train = set(manifest.get("train_galaxy_ids", []))
    val = set(manifest.get("val_galaxy_ids", []))
    overlap = train & val
    if overlap:
        raise ValueError(f"Split manifest leakage detected: galaxies {sorted(overlap)}")

    if expected_galaxy_ids is not None and train | val != expected_galaxy_ids:
        missing = sorted(expected_galaxy_ids - (train | val))
        extra = sorted((train | val) - expected_galaxy_ids)
        raise ValueError(
            f"Split manifest galaxy mismatch; missing={missing[:5]} extra={extra[:5]}"
        )

    if expected_split_counts:
        for split_name, expected_count in expected_split_counts.items():
            field = f"{split_name}_galaxy_ids"
            actual_count = len(manifest.get(field, []))
            if actual_count != expected_count:
                raise ValueError(
                    f"Expected {expected_count} {split_name} galaxies, got {actual_count}"
                )


def rebuild_sam3_dataset(
    config_path: Path,
    previous_root: Path,
    noise_tags: list[str] | None = None,
    noisy_variants: set[str] | None = None,
) -> dict[str, Any]:
    """Rebuild sam3_prepared using fresh export + locked split + noise aug."""
    noise_tags = list(DEFAULT_NOISE_TAGS if noise_tags is None else noise_tags)
    noisy_variants = set({"linear_magnitude"} if noisy_variants is None else noisy_variants)

    logger = logging.getLogger(__name__)
    config = load_config(config_path)
    resolver = PathResolver(config)
    dataset_name = config.get("dataset_name", "dr1")
    allowed_views = list(config["data_selection"].get("views") or config["data_selection"]["orientations"])
    base_key_re = _make_base_key_re(allowed_views)
    expected_gt_base_keys = len(generate_base_keys(config))
    expected_galaxy_ids = set(config["data_selection"]["galaxy_ids"])
    expected_split_counts = (
        EXPECTED_SPLIT_GALAXY_COUNTS
        if dataset_name == "dr1" and set(allowed_views) == {"eo", "fo"}
        else None
    )

    sam3_dir = resolver.get_sam3_dir()
    if dataset_name == "dr1":
        gt_root = Path(config["paths"]["output_root"]) / "gt_canonical" / "current"
    else:
        gt_root = resolver.get_pseudo_gt_condition_root(dataset=dataset_name, condition="clean")
    noisy_root = Path(config["paths"]["output_root"]) / "renders" / "noisy"
    effective_previous_root = previous_root
    if dataset_name != "dr1" and not effective_previous_root.exists():
        dataset_scoped = previous_root / dataset_name
        if dataset_scoped.exists():
            effective_previous_root = dataset_scoped
    previous_manifest_path = effective_previous_root / "split_manifest.json"
    ann_filename = config.get("export_phase", {}).get("annotations_filename", "annotations.json")

    logger.info("Verifying rebuild preconditions")
    _ensure_missing_or_empty(sam3_dir)
    if not previous_manifest_path.exists():
        raise FileNotFoundError(
            f"Previous split manifest not found: {previous_manifest_path}"
        )
    gt_keys = _collect_gt_keys(
        gt_root,
        base_key_re=base_key_re,
        expected_gt_base_keys=expected_gt_base_keys,
    )
    for noise_tag in noise_tags:
        for variant in noisy_variants:
            _validate_noise_coverage(
                noisy_root,
                gt_keys,
                variant,
                noise_tag,
                base_key_re=base_key_re,
            )

    logger.info("Running fresh export into %s", sam3_dir)
    base_keys = generate_base_keys(config)
    run_export_phase(config, base_keys, logger)

    annotations_path = sam3_dir / ann_filename
    if not annotations_path.exists():
        raise FileNotFoundError(f"Export did not create {annotations_path}")

    coco = _load_json(annotations_path)
    reuse_manifest = _load_json(previous_manifest_path)
    coco_train, coco_val, split_manifest = galaxy_split_coco(
        coco,
        train_ratio=reuse_manifest.get("train_ratio", 0.8),
        seed=reuse_manifest.get("seed", 42),
        reuse_manifest=reuse_manifest,
    )
    _validate_split_manifest(
        split_manifest,
        expected_galaxy_ids=expected_galaxy_ids,
        expected_split_counts=expected_split_counts,
    )

    train_path = sam3_dir / "annotations_train.json"
    val_path = sam3_dir / "annotations_val.json"
    split_manifest_path = sam3_dir / "split_manifest.json"
    _write_json(train_path, coco_train)
    _write_json(val_path, coco_val)
    _write_json(split_manifest_path, split_manifest)

    logger.info("Building train noise augmentation")
    train_noise_coco, train_noise_stats = build_noise_augmented_coco(
        coco_split=coco_train,
        noisy_root=noisy_root,
        noise_tags=noise_tags,
        dataset_root=sam3_dir,
        split_manifest=split_manifest,
        target_split="train",
        noisy_variants=noisy_variants,
        force=False,
    )
    train_noise_path = sam3_dir / "annotations_train_noise_augmented.json"
    train_manifest_path = sam3_dir / "noise_aug_manifest_train.json"
    _write_json(train_noise_path, train_noise_coco)
    _write_json(
        train_manifest_path,
        {
            **train_noise_stats,
            "source_annotations": str(train_path),
            "split_manifest": str(split_manifest_path),
        },
    )

    logger.info("Building val noise augmentation")
    val_noise_coco, val_noise_stats = build_noise_augmented_coco(
        coco_split=coco_val,
        noisy_root=noisy_root,
        noise_tags=noise_tags,
        dataset_root=sam3_dir,
        split_manifest=split_manifest,
        target_split="val",
        noisy_variants=noisy_variants,
        force=False,
    )
    val_noise_path = sam3_dir / "annotations_val_noise_augmented.json"
    val_manifest_path = sam3_dir / "noise_aug_manifest_val.json"
    _write_json(val_noise_path, val_noise_coco)
    _write_json(
        val_manifest_path,
        {
            **val_noise_stats,
            "source_annotations": str(val_path),
            "split_manifest": str(split_manifest_path),
        },
    )

    logger.info("Switching annotations_train_active.json to the rebuilt noisy train set")
    active_path = sam3_dir / "annotations_train_active.json"
    compose_training_coco(
        sources=[("noise_aug", train_noise_path)],
        output_path=active_path,
        force=True,
    )

    logger.info("Validating rebuild outputs")
    if dataset_name == "dr1" and set(allowed_views) == {"eo", "fo"}:
        for filename, expected in EXPECTED_COUNTS.items():
            _validate_counts(sam3_dir / filename, expected)
        _validate_counts(
            train_noise_path,
            _expected_augmented_counts(
                coco_train,
                noise_tags=noise_tags,
                noisy_variants=noisy_variants,
            ),
        )
        _validate_counts(
            val_noise_path,
            _expected_augmented_counts(
                coco_val,
                noise_tags=noise_tags,
                noisy_variants=noisy_variants,
            ),
        )

    for path in (
        annotations_path,
        train_path,
        val_path,
        train_noise_path,
        val_noise_path,
    ):
        coco_payload = _load_json(path)
        _validate_views(
            coco_payload,
            allowed_views=set(allowed_views),
            base_key_re=base_key_re,
        )
        if "noise_augmented" in path.name:
            _validate_noisy_asinh(coco_payload)

    if not active_path.is_symlink():
        raise ValueError(f"{active_path} is not a symlink")
    if active_path.resolve() != train_noise_path.resolve():
        raise ValueError(
            f"{active_path} points to {active_path.resolve()}, expected {train_noise_path.resolve()}"
        )

    return {
        "sam3_dir": str(sam3_dir),
        "noise_tags": noise_tags,
        "noisy_variants": sorted(noisy_variants),
        "split_manifest": split_manifest,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/unified_data_prep.yaml"),
        help="Unified dataset config used for export",
    )
    parser.add_argument(
        "--previous-root",
        type=Path,
        default=Path("data/02_processed/sam3_prepared_previous"),
        help="Backup sam3_prepared directory containing the split manifest lock",
    )
    parser.add_argument(
        "--noise-tags",
        nargs="+",
        default=DEFAULT_NOISE_TAGS,
        help="Noise profiles to include (default: sb31.5)",
    )
    parser.add_argument(
        "--noisy-variants",
        nargs="+",
        default=["linear_magnitude"],
        help="Variants that receive noisy clones (default: linear_magnitude)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    rebuild_sam3_dataset(
        config_path=args.config,
        previous_root=args.previous_root,
        noise_tags=args.noise_tags,
        noisy_variants=set(args.noisy_variants),
    )


if __name__ == "__main__":
    main()
