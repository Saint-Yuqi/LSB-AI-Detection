"""
Smoke tests for the in-process SAM3 rebuild orchestrator.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.data import rebuild_sam3_dataset as rebuild_mod


def _write_png(path: Path, width: int = 64, height: int = 64) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((height, width, 3), dtype=np.uint8)).save(path)


def _write_gt_dir(root: Path, base_key: str) -> None:
    gt_dir = root / base_key
    gt_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.ones((64, 64), dtype=np.uint8)).save(
        gt_dir / "instance_map_uint8.png"
    )
    (gt_dir / "instances.json").write_text(json.dumps([{"id": 1, "type": "streams"}]))


def _fake_export(config, base_keys, logger, force_variants=None) -> None:
    output_root = Path(config["paths"]["output_root"])
    sam3_dir = output_root / "sam3_prepared"
    images_dir = sam3_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    for key in base_keys:
        for variant in ("asinh_stretch", "linear_magnitude"):
            img_id += 1
            images.append({
                "id": img_id,
                "file_name": f"images/{key}_{variant}.png",
                "width": 64,
                "height": 64,
                "galaxy_id": key.galaxy_id,
                "view_id": key.view_id,
                "orientation": key.view_id,
                "variant": variant,
                "base_key": str(key),
            })
            ann_id += 1
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": 1,
                "segmentation": {"size": [64, 64], "counts": "abc"},
                "bbox": [0, 0, 10, 10],
                "area": 100,
                "iscrowd": 0,
            })

    coco = {
        "info": {"description": "test export"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "stellar stream", "supercategory": "lsb"}],
    }
    (sam3_dir / "annotations.json").write_text(json.dumps(coco, indent=2))


def _fake_export_pnbody(config, base_keys, logger, force_variants=None) -> None:
    output_root = Path(config["paths"]["output_root"])
    sam3_dir = output_root / "sam3_prepared" / "pnbody"
    images_dir = sam3_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    images = []
    annotations = []
    img_id = 0
    ann_id = 0
    for key in base_keys:
        img_id += 1
        images.append({
            "id": img_id,
            "file_name": f"images/{key}_linear_magnitude__clean.png",
            "width": 64,
            "height": 64,
            "galaxy_id": key.galaxy_id,
            "view_id": key.view_id,
            "orientation": key.view_id,
            "variant": "linear_magnitude",
            "base_key": str(key),
            "dataset": "pnbody",
            "condition": "clean",
            "label_mode": "authoritative",
        })
        ann_id += 1
        annotations.append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "segmentation": {"size": [64, 64], "counts": "abc"},
            "bbox": [0, 0, 10, 10],
            "area": 100,
            "iscrowd": 0,
        })

    coco = {
        "info": {"description": "pnbody export"},
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "stellar stream", "supercategory": "lsb"}],
    }
    (sam3_dir / "annotations_pnbody_pseudo.json").write_text(json.dumps(coco, indent=2))


class TestRebuildSam3Dataset:
    def test_rebuild_pipeline_smoke(self, tmp_path, monkeypatch):
        output_root = tmp_path / "processed"
        gt_root = output_root / "gt_canonical" / "current"
        noisy_root = output_root / "renders" / "noisy" / "linear_magnitude"
        previous_root = output_root / "sam3_prepared_previous"

        for base_key in ("00011_eo", "00019_eo"):
            _write_gt_dir(gt_root, base_key)
            for noise_tag in ("sb30", "sb31.5"):
                _write_png(noisy_root / noise_tag / base_key / "0000.png")

        previous_root.mkdir(parents=True, exist_ok=True)
        (previous_root / "split_manifest.json").write_text(json.dumps({
            "train_ratio": 0.8,
            "seed": 42,
            "train_galaxy_ids": [11],
            "val_galaxy_ids": [19],
        }))

        config = {
            "paths": {
                "firebox_root": str(tmp_path / "raw"),
                "output_root": str(output_root),
            },
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "unused-{galaxy_id}-{orientation}.fits.gz",
                }
            },
            "data_selection": {"galaxy_ids": [11, 19], "views": ["eo"]},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [
                {"name": "asinh_stretch"},
                {"name": "linear_magnitude"},
            ],
        }

        monkeypatch.setattr(rebuild_mod, "load_config", lambda path: config)
        monkeypatch.setattr(rebuild_mod, "run_export_phase", _fake_export)
        monkeypatch.setattr(rebuild_mod, "EXPECTED_GT_BASE_KEYS", 2)
        monkeypatch.setattr(
            rebuild_mod,
            "EXPECTED_SPLIT_GALAXY_COUNTS",
            {"train": 1, "val": 1},
        )
        monkeypatch.setattr(
            rebuild_mod,
            "EXPECTED_COUNTS",
            {
                "annotations.json": {"images": 4, "annotations": 4},
                "annotations_train.json": {"images": 2, "annotations": 2},
                "annotations_val.json": {"images": 2, "annotations": 2},
            },
        )

        result = rebuild_mod.rebuild_sam3_dataset(
            config_path=Path("unused.yaml"),
            previous_root=previous_root,
            noise_tags=["sb30", "sb31.5"],
            noisy_variants={"linear_magnitude"},
        )

        sam3_dir = Path(result["sam3_dir"])
        assert sam3_dir == output_root / "sam3_prepared"

        expected_files = [
            "annotations.json",
            "annotations_train.json",
            "annotations_val.json",
            "annotations_train_noise_augmented.json",
            "annotations_val_noise_augmented.json",
            "split_manifest.json",
            "noise_aug_manifest_train.json",
            "noise_aug_manifest_val.json",
            "compose_manifest.json",
        ]
        for filename in expected_files:
            assert (sam3_dir / filename).exists()

        train_noisy = json.loads(
            (sam3_dir / "annotations_train_noise_augmented.json").read_text()
        )
        val_noisy = json.loads(
            (sam3_dir / "annotations_val_noise_augmented.json").read_text()
        )

        for payload in (train_noisy, val_noisy):
            for img in payload["images"]:
                assert img["view_id"] == "eo"
                assert img["base_key"].endswith("_eo")
                if img["noise_tag"] != "clean":
                    assert img["variant"] == "linear_magnitude"

        active_path = sam3_dir / "annotations_train_active.json"
        assert active_path.is_symlink()
        assert active_path.resolve() == (
            sam3_dir / "annotations_train_noise_augmented.json"
        ).resolve()

    def test_rebuild_defaults_to_sb31p5_only(self, tmp_path, monkeypatch):
        output_root = tmp_path / "processed"
        gt_root = output_root / "gt_canonical" / "current"
        noisy_root = output_root / "renders" / "noisy" / "linear_magnitude"
        previous_root = output_root / "sam3_prepared_previous"

        for base_key in ("00011_eo", "00019_eo"):
            _write_gt_dir(gt_root, base_key)
            _write_png(noisy_root / "sb31.5" / base_key / "0000.png")

        previous_root.mkdir(parents=True, exist_ok=True)
        (previous_root / "split_manifest.json").write_text(json.dumps({
            "train_ratio": 0.8,
            "seed": 42,
            "train_galaxy_ids": [11],
            "val_galaxy_ids": [19],
        }))

        config = {
            "paths": {
                "firebox_root": str(tmp_path / "raw"),
                "output_root": str(output_root),
            },
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "unused-{galaxy_id}-{orientation}.fits.gz",
                }
            },
            "data_selection": {"galaxy_ids": [11, 19], "views": ["eo"]},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [
                {"name": "asinh_stretch"},
                {"name": "linear_magnitude"},
            ],
        }

        monkeypatch.setattr(rebuild_mod, "load_config", lambda path: config)
        monkeypatch.setattr(rebuild_mod, "run_export_phase", _fake_export)
        monkeypatch.setattr(rebuild_mod, "EXPECTED_GT_BASE_KEYS", 2)
        monkeypatch.setattr(
            rebuild_mod,
            "EXPECTED_SPLIT_GALAXY_COUNTS",
            {"train": 1, "val": 1},
        )
        monkeypatch.setattr(
            rebuild_mod,
            "EXPECTED_COUNTS",
            {
                "annotations.json": {"images": 4, "annotations": 4},
                "annotations_train.json": {"images": 2, "annotations": 2},
                "annotations_val.json": {"images": 2, "annotations": 2},
            },
        )

        result = rebuild_mod.rebuild_sam3_dataset(
            config_path=Path("unused.yaml"),
            previous_root=previous_root,
            noisy_variants={"linear_magnitude"},
        )

        sam3_dir = Path(result["sam3_dir"])
        train_noisy = json.loads(
            (sam3_dir / "annotations_train_noise_augmented.json").read_text()
        )
        val_noisy = json.loads(
            (sam3_dir / "annotations_val_noise_augmented.json").read_text()
        )

        assert result["noise_tags"] == ["sb31.5"]
        for payload in (train_noisy, val_noisy):
            noisy_tags = {
                img["noise_tag"]
                for img in payload["images"]
                if img["noise_tag"] != "clean"
            }
            assert noisy_tags == {"sb31.5"}
            assert len(payload["images"]) == 3
            assert len(payload["annotations"]) == 3

    def test_rebuild_pipeline_pnbody_losxx_dataset_scoped(self, tmp_path, monkeypatch):
        output_root = tmp_path / "processed"
        gt_root = output_root / "pseudo_gt_canonical" / "pnbody" / "clean" / "current"
        noisy_root = output_root / "renders" / "noisy" / "linear_magnitude"
        previous_root = output_root / "sam3_prepared_previous" / "pnbody"

        for base_key in ("00011_los00", "00019_los00"):
            _write_gt_dir(gt_root, base_key)
            for noise_tag in ("sb30", "sb31.5"):
                _write_png(noisy_root / noise_tag / base_key / "0000.png")

        previous_root.mkdir(parents=True, exist_ok=True)
        (previous_root / "split_manifest.json").write_text(json.dumps({
            "train_ratio": 0.8,
            "seed": 42,
            "train_galaxy_ids": [11],
            "val_galaxy_ids": [19],
        }))

        config = {
            "dataset_name": "pnbody",
            "paths": {
                "firebox_root": str(tmp_path / "raw"),
                "output_root": str(output_root),
            },
            "data_conditions": {
                "clean": {"label_mode": "authoritative"},
                "sb30": {"label_mode": "clone_from_clean"},
                "sb31.5": {"label_mode": "clone_from_clean"},
            },
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "{galaxy_id:05d}/unused-{view_id}.fits.gz",
                }
            },
            "data_selection": {"galaxy_ids": [11, 19], "views": ["los00"]},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [{"name": "linear_magnitude"}],
            "export_phase": {"annotations_filename": "annotations_pnbody_pseudo.json"},
        }

        monkeypatch.setattr(rebuild_mod, "load_config", lambda path: config)
        monkeypatch.setattr(rebuild_mod, "run_export_phase", _fake_export_pnbody)

        result = rebuild_mod.rebuild_sam3_dataset(
            config_path=Path("unused.yaml"),
            previous_root=previous_root,
            noise_tags=["sb30", "sb31.5"],
            noisy_variants={"linear_magnitude"},
        )

        sam3_dir = Path(result["sam3_dir"])
        assert sam3_dir == output_root / "sam3_prepared" / "pnbody"

        train_noisy = json.loads(
            (sam3_dir / "annotations_train_noise_augmented.json").read_text()
        )
        assert {img["view_id"] for img in train_noisy["images"]} == {"los00"}
        assert all(img["base_key"].endswith("_los00") for img in train_noisy["images"])
        assert all(img.get("dataset") == "pnbody" for img in train_noisy["images"])
