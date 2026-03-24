"""
Tests for galaxy-level COCO train/val split.

Covers: no leakage, all-same-galaxy in one split, count preservation,
consecutive IDs, determinism, stability under growth, reuse-manifest,
and filename-parsing fallback.

Usage:
    pytest tests/test_galaxy_split.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.split import galaxy_split_coco


def _make_coco(n_galaxies=5, orientations=("eo", "fo"), variants=("a", "b"), anns_per_image=3):
    """Synthetic COCO: n_galaxies x orientations x variants images."""
    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    galaxy_ids = list(range(11, 11 + n_galaxies * 2, 2))  # [11,13,15,17,19]

    for gid in galaxy_ids:
        for ori in orientations:
            for var in variants:
                img_id += 1
                images.append({
                    "id": img_id,
                    "file_name": f"images/{gid:05d}_{ori}_{var}.png",
                    "width": 64,
                    "height": 64,
                    "galaxy_id": gid,
                    "view_id": ori,
                    "orientation": ori,
                    "variant": var,
                    "base_key": f"{gid:05d}_{ori}",
                })
                for _ in range(anns_per_image):
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [0, 0, 10, 10],
                        "area": 100,
                        "iscrowd": 0,
                    })

    return {
        "info": {"description": "test"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "stellar stream"}],
    }


class TestGalaxySplit:
    def test_no_leakage(self):
        coco = _make_coco()
        train, val, manifest = galaxy_split_coco(coco, train_ratio=0.6, seed=42)
        train_gids = set(manifest["train_galaxy_ids"])
        val_gids = set(manifest["val_galaxy_ids"])
        assert train_gids.isdisjoint(val_gids)

    def test_all_images_of_galaxy_in_same_split(self):
        coco = _make_coco()
        train, val, _ = galaxy_split_coco(coco, train_ratio=0.6, seed=42)
        train_gids = {img["galaxy_id"] for img in train["images"]}
        val_gids = {img["galaxy_id"] for img in val["images"]}
        assert train_gids.isdisjoint(val_gids)

    def test_count_preservation(self):
        coco = _make_coco()
        train, val, manifest = galaxy_split_coco(coco, train_ratio=0.8, seed=42)
        total_imgs = len(coco["images"])
        total_anns = len(coco["annotations"])
        assert len(train["images"]) + len(val["images"]) == total_imgs
        assert len(train["annotations"]) + len(val["annotations"]) == total_anns
        assert manifest["n_train_images"] + manifest["n_val_images"] == total_imgs
        assert manifest["n_train_annotations"] + manifest["n_val_annotations"] == total_anns

    def test_consecutive_ids(self):
        coco = _make_coco()
        train, val, _ = galaxy_split_coco(coco, train_ratio=0.8, seed=42)
        for split in [train, val]:
            if split["images"]:
                img_ids = [img["id"] for img in split["images"]]
                assert img_ids == list(range(1, len(img_ids) + 1))
            if split["annotations"]:
                ann_ids = [a["id"] for a in split["annotations"]]
                assert ann_ids == list(range(1, len(ann_ids) + 1))

    def test_deterministic(self):
        coco = _make_coco()
        _, _, m1 = galaxy_split_coco(coco, seed=42)
        _, _, m2 = galaxy_split_coco(coco, seed=42)
        assert m1["train_galaxy_ids"] == m2["train_galaxy_ids"]
        assert m1["val_galaxy_ids"] == m2["val_galaxy_ids"]

    def test_stable_under_growth(self):
        coco_5 = _make_coco(n_galaxies=5)
        _, _, m5 = galaxy_split_coco(coco_5, train_ratio=0.8, seed=42)

        coco_6 = _make_coco(n_galaxies=6)
        _, _, m6 = galaxy_split_coco(coco_6, train_ratio=0.8, seed=42)

        for gid in m5["train_galaxy_ids"]:
            assert gid in m6["train_galaxy_ids"], f"galaxy {gid} moved from train"
        for gid in m5["val_galaxy_ids"]:
            assert gid in m6["val_galaxy_ids"], f"galaxy {gid} moved from val"

    def test_reuse_manifest(self):
        coco_5 = _make_coco(n_galaxies=5)
        _, _, m5 = galaxy_split_coco(coco_5, train_ratio=0.8, seed=42)

        coco_6 = _make_coco(n_galaxies=6)
        _, _, m6 = galaxy_split_coco(coco_6, train_ratio=0.8, seed=99, reuse_manifest=m5)

        for gid in m5["train_galaxy_ids"]:
            assert gid in m6["train_galaxy_ids"]
        for gid in m5["val_galaxy_ids"]:
            assert gid in m6["val_galaxy_ids"]

    def test_manifest_schema(self):
        coco = _make_coco()
        _, _, manifest = galaxy_split_coco(coco, train_ratio=0.8, seed=42)
        required_keys = {
            "source_annotations", "train_ratio", "seed",
            "train_galaxy_ids", "val_galaxy_ids",
            "n_train_images", "n_val_images",
            "n_train_annotations", "n_val_annotations",
            "created_at",
        }
        assert required_keys.issubset(manifest.keys())

    def test_fallback_filename_parsing(self):
        """When galaxy_id field is missing, fall back to filename regex."""
        coco = _make_coco(n_galaxies=3)
        for img in coco["images"]:
            del img["galaxy_id"]

        train, val, manifest = galaxy_split_coco(coco, train_ratio=0.8, seed=42)
        assert len(train["images"]) + len(val["images"]) == len(coco["images"])
        assert set(manifest["train_galaxy_ids"]).isdisjoint(set(manifest["val_galaxy_ids"]))

    def test_categories_preserved(self):
        coco = _make_coco()
        train, val, _ = galaxy_split_coco(coco)
        assert train["categories"] == coco["categories"]
        assert val["categories"] == coco["categories"]

    def test_los_views_grouped_by_galaxy(self):
        """With losNN views, split still groups by galaxy_id."""
        coco = _make_coco(n_galaxies=5, orientations=("los00", "los01"))
        train, val, manifest = galaxy_split_coco(coco, train_ratio=0.6, seed=42)
        train_gids = set(manifest["train_galaxy_ids"])
        val_gids = set(manifest["val_galaxy_ids"])
        assert train_gids.isdisjoint(val_gids)

        train_img_gids = {img["galaxy_id"] for img in train["images"]}
        val_img_gids = {img["galaxy_id"] for img in val["images"]}
        assert train_img_gids.isdisjoint(val_img_gids)

    def test_anti_leakage_pnbody_views(self):
        """If galaxy X is val, ALL its losNN views must be excluded from train."""
        coco = _make_coco(n_galaxies=5, orientations=tuple(f"los{i:02d}" for i in range(24)))
        train, val, manifest = galaxy_split_coco(coco, train_ratio=0.6, seed=42)

        val_gids = set(manifest["val_galaxy_ids"])
        for img in train["images"]:
            assert img["galaxy_id"] not in val_gids, (
                f"Leakage: galaxy {img['galaxy_id']} is val but found in train"
            )
