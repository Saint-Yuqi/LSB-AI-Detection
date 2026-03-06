"""
Tests for noise augmentation guardrails.

Covers: missing noisy file, dimension mismatch, re-augmentation rejection,
galaxy not in train manifest, val overlap rejection, idempotent symlink.

Usage:
    pytest tests/test_noise_aug.py -v
"""
from __future__ import annotations

import json
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


def _create_noisy_pngs(noisy_root, coco, snr_tags, width=64, height=64):
    """Create placeholder noisy PNGs at expected paths."""
    for img in coco["images"]:
        for snr in snr_tags:
            d = noisy_root / img["variant"] / snr / img["base_key"]
            d.mkdir(parents=True, exist_ok=True)
            Image.fromarray(
                np.zeros((height, width, 3), dtype=np.uint8)
            ).save(d / "0000.png")


class TestNoiseAug:
    def test_missing_noisy_file(self, tmp_path):
        coco = _make_clean_coco()
        manifest = _make_split_manifest()
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(FileNotFoundError, match="Noisy render missing"):
            build_noise_augmented_coco(
                coco, tmp_path / "noisy", ["snr20"],
                dataset_root, manifest,
            )

    def test_dimension_mismatch(self, tmp_path):
        coco = _make_clean_coco(width=64, height=64)
        manifest = _make_split_manifest()
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        _create_noisy_pngs(noisy_root, coco, ["snr20"], width=128, height=128)

        with pytest.raises(ValueError, match="Dimension mismatch"):
            build_noise_augmented_coco(
                coco, noisy_root, ["snr20"],
                dataset_root, manifest,
            )

    def test_reject_re_augmentation(self, tmp_path):
        coco = _make_clean_coco()
        coco["images"][0]["snr_tag"] = "snr20"
        manifest = _make_split_manifest()
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="already contains augmented"):
            build_noise_augmented_coco(
                coco, tmp_path / "noisy", ["snr20"],
                dataset_root, manifest,
            )

    def test_galaxy_not_in_train_manifest(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11, 99))
        manifest = _make_split_manifest(train_ids=(11, 13), val_ids=(19,))
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="not in train_galaxy_ids"):
            build_noise_augmented_coco(
                coco, tmp_path / "noisy", ["snr20"],
                dataset_root, manifest,
            )

    def test_val_overlap_rejected(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11, 19))
        manifest = _make_split_manifest(train_ids=(11, 13), val_ids=(19, 24))
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)

        with pytest.raises(ValueError, match="Leakage"):
            build_noise_augmented_coco(
                coco, tmp_path / "noisy", ["snr20"],
                dataset_root, manifest,
            )

    def test_symlink_idempotent(self, tmp_path):
        coco = _make_clean_coco(galaxy_ids=(11,))
        manifest = _make_split_manifest(train_ids=(11,), val_ids=(19,))
        noisy_root = tmp_path / "noisy"
        dataset_root = tmp_path / "sam3"
        (dataset_root / "images").mkdir(parents=True)
        _create_noisy_pngs(noisy_root, coco, ["snr20"])

        result1, stats1 = build_noise_augmented_coco(
            coco, noisy_root, ["snr20"], dataset_root, manifest,
        )
        result2, stats2 = build_noise_augmented_coco(
            coco, noisy_root, ["snr20"], dataset_root, manifest,
        )

        assert len(result1["images"]) == len(result2["images"])
        assert len(result1["annotations"]) == len(result2["annotations"])
        assert stats1["n_images_total"] == stats2["n_images_total"]

        for img in result1["images"]:
            assert img["snr_tag"] in ("clean", "snr20")
            assert "source_image_id_in_base" in img

        img_ids = [img["id"] for img in result1["images"]]
        assert img_ids == list(range(1, len(img_ids) + 1))
