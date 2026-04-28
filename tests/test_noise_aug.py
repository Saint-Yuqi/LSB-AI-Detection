"""
Tests for split-aware noise augmentation guardrails.

Covers: missing noisy file, dimension mismatch, re-augmentation rejection,
split-aware leakage, linear-only noisy variants, filename encoding, and
idempotent symlink creation.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.noise_aug import build_noise_augmented_coco


def _make_clean_coco(
    galaxy_ids=(11, 13),
    orientations=("eo",),
    variants=("asinh_stretch",),
    anns_per_image=2,
    width=64,
    height=64,
):
    """Minimal clean COCO fixture."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0

    for gid in galaxy_ids:
        for ori in orientations:
            for var in variants:
                img_id += 1
                base_key = f"{gid:05d}_{ori}"
                images.append({
                    "id": img_id,
                    "file_name": f"images/{base_key}_{var}.png",
                    "width": width,
                    "height": height,
                    "galaxy_id": gid,
                    "view_id": ori,
                    "orientation": ori,
                    "variant": var,
                    "base_key": base_key,
                })
                for _ in range(anns_per_image):
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "segmentation": {"size": [height, width], "counts": "abc"},
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    })

    return {
        "info": {"description": "test"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "stellar stream", "supercategory": "lsb"},
        ],
    }


def _make_split_manifest(train_ids=(11, 13), val_ids=(19, 24)):
    return {
        "train_galaxy_ids": list(train_ids),
        "val_galaxy_ids": list(val_ids),
    }


def _create_noisy_pngs(
    noisy_root,
    coco,
    noise_tags,
    variants=None,
    width=64,
    height=64,
):
    """Create placeholder noisy PNGs at expected paths."""
    variants = set(variants or {img["variant"] for img in coco["images"]})
    for img in coco["images"]:
        if img["variant"] not in variants:
            continue
        for noise_tag in noise_tags:
            d = noisy_root / img["variant"] / noise_tag / img["base_key"]
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.zeros((height, width, 3), dtype=np.uint8)
            ).save(d / "0000.png")


class TestNoiseAug:
    def test_missing_noisy_file(self, tmp_path):
        coco = _make_clean_coco(variants=("linear_magnitude",))
        manifest = _make_split_manifest()
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Noisy render missing"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=tmp_path / "noisy",
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="train",
            )

    def test_dimension_mismatch(self, tmp_path):
        coco = _make_clean_coco(variants=("linear_magnitude",), width=64, height=64)
        manifest = _make_split_manifest()
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        _create_noisy_pngs(
            noisy_root,
            coco,
            ["sb30"],
            variants={"linear_magnitude"},
            width=128,
            height=128,
        )

        with pytest.raises(ValueError, match="Dimension mismatch"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=noisy_root,
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="train",
            )

    def test_reject_re_augmentation_legacy(self, tmp_path):
        coco = _make_clean_coco(variants=("linear_magnitude",))
        coco["images"][0]["file_name"] = "images/00011_eo_linear_magnitude__snr20.png"
        manifest = _make_split_manifest()
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="already contains augmented"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=tmp_path / "noisy",
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="train",
            )

    def test_reject_re_augmentation_new_sentinel(self, tmp_path):
        coco = _make_clean_coco(variants=("linear_magnitude",))
        coco["images"][0]["file_name"] = (
            "images/00011_eo_linear_magnitude__noise_sb30.png"
        )
        manifest = _make_split_manifest()
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="already contains augmented"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=tmp_path / "noisy",
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="train",
            )

    def test_galaxy_not_in_target_split_manifest_train(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11, 99), variants=("linear_magnitude",))
        manifest = _make_split_manifest(train_ids=(11, 13), val_ids=(19,))
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="train_galaxy_ids"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=tmp_path / "noisy",
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="train",
            )

    def test_target_split_val_rejects_train_overlap(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11,), variants=("linear_magnitude",))
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="train_galaxy_ids"):
            build_noise_augmented_coco(
                coco_split=coco,
                noisy_root=tmp_path / "noisy",
                noise_tags=["sb30"],
                dataset_root=dataset_root,
                split_manifest=manifest,
                target_split="val",
            )

    def test_target_split_val_succeeds(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(19,), variants=("linear_magnitude",))
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)
        _create_noisy_pngs(noisy_root, coco, ["sb30"], variants={"linear_magnitude"})

        result, stats = build_noise_augmented_coco(
            coco_split=coco,
            noisy_root=noisy_root,
            noise_tags=["sb30"],
            dataset_root=dataset_root,
            split_manifest=manifest,
            target_split="val",
        )

        assert stats["target_split"] == "val"
        assert len(result["images"]) == 2
        assert len(result["annotations"]) == 4

    def test_noisy_variants_filter_only_linear_magnitude(self, tmp_path):
        coco = _make_clean_coco(
            galaxy_ids=(11,),
            variants=("asinh_stretch", "linear_magnitude"),
        )
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)
        _create_noisy_pngs(
            noisy_root,
            coco,
            ["sb30", "sb31.5"],
            variants={"linear_magnitude"},
        )

        result, stats = build_noise_augmented_coco(
            coco_split=coco,
            noisy_root=noisy_root,
            noise_tags=["sb30", "sb31.5"],
            dataset_root=dataset_root,
            split_manifest=manifest,
            target_split="train",
            noisy_variants={"linear_magnitude"},
        )

        assert stats["n_images_clean"] == 2
        assert stats["n_images_noisy"] == 2
        assert stats["n_images_total"] == 4

        noisy_images = [
            img for img in result["images"] if img["noise_tag"] != "clean"
        ]
        assert len(noisy_images) == 2
        assert {img["variant"] for img in noisy_images} == {"linear_magnitude"}
        assert {
            img["noise_tag"] for img in noisy_images
        } == {"sb30", "sb31.5"}

    def test_filename_encoding_and_noise_fields(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11,), variants=("linear_magnitude",))
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)
        _create_noisy_pngs(
            noisy_root,
            coco,
            ["sb31.5"],
            variants={"linear_magnitude"},
        )

        result, _ = build_noise_augmented_coco(
            coco_split=coco,
            noisy_root=noisy_root,
            noise_tags=["sb31.5"],
            dataset_root=dataset_root,
            split_manifest=manifest,
            target_split="train",
        )

        noisy_images = [img for img in result["images"] if img["noise_tag"] != "clean"]
        assert len(noisy_images) == 1
        noisy_img = noisy_images[0]
        assert noisy_img["file_name"].endswith("__noise_sb31p5.png")
        assert noisy_img["snr_tag"] == "sb31.5"
        assert noisy_img["noise_tag"] == "sb31.5"

    def test_symlink_idempotent(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11,), variants=("linear_magnitude",))
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)
        _create_noisy_pngs(noisy_root, coco, ["sb30"], variants={"linear_magnitude"})

        result1, stats1 = build_noise_augmented_coco(
            coco_split=coco,
            noisy_root=noisy_root,
            noise_tags=["sb30"],
            dataset_root=dataset_root,
            split_manifest=manifest,
            target_split="train",
        )
        result2, stats2 = build_noise_augmented_coco(
            coco_split=coco,
            noisy_root=noisy_root,
            noise_tags=["sb30"],
            dataset_root=dataset_root,
            split_manifest=manifest,
            target_split="train",
        )

        assert len(result1["images"]) == len(result2["images"])
        assert len(result1["annotations"]) == len(result2["annotations"])
        assert stats1["n_images_total"] == stats2["n_images_total"]

        for img in result1["images"]:
            assert img["snr_tag"] in ("clean", "sb30")
            assert img["noise_tag"] in ("clean", "sb30")
            assert "source_image_id_in_base" in img

        img_ids = [img["id"] for img in result1["images"]]
        assert img_ids == list(range(1, len(img_ids) + 1))
