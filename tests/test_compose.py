"""
Tests for COCO dataset composition.

Covers: duplicate file_name rejection, consecutive ID renumbering,
category preservation, dataset_source tagging, single-source symlink mode.

Usage:
    pytest tests/test_compose.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.compose import compose_training_coco


def _make_coco(prefix="a", n_images=3, n_anns_per_image=2, start_id=1):
    """Minimal COCO fixture with unique file_names per prefix."""
    images = []
    annotations = []
    img_id = start_id - 1
    ann_id = (start_id - 1) * n_anns_per_image

    for i in range(n_images):
        img_id += 1
        images.append({
            "id": img_id,
            "file_name": f"images/{prefix}_{i:04d}.png",
            "width": 64,
            "height": 64,
            "galaxy_id": 10 + i,
            "orientation": "eo",
            "variant": "asinh_stretch",
            "base_key": f"{10 + i:05d}_eo",
            "snr_tag": "clean",
        })
        for j in range(n_anns_per_image):
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

    return {
        "info": {"description": "test"},
        "images": images,
        "annotations": annotations,
        "categories": [
            {"id": 1, "name": "stellar stream", "supercategory": "lsb"},
            {"id": 2, "name": "satellite galaxy", "supercategory": "lsb"},
        ],
    }


def _write_coco(tmp_path, name, coco):
    p = tmp_path / name
    p.write_text(json.dumps(coco))
    return p


class TestCompose:
    def test_compose_reject_duplicate_file_name_by_default(self, tmp_path):
        coco_a = _make_coco(prefix="shared", n_images=2)
        coco_b = _make_coco(prefix="shared", n_images=2)
        path_a = _write_coco(tmp_path, "a.json", coco_a)
        path_b = _write_coco(tmp_path, "b.json", coco_b)
        output = tmp_path / "active.json"

        with pytest.raises(ValueError, match="Duplicate file_name"):
            compose_training_coco(
                sources=[("a", path_a), ("b", path_b)],
                output_path=output,
            )

    def test_compose_renumbers_ids_consecutively(self, tmp_path):
        coco_a = _make_coco(prefix="a", n_images=3, start_id=100)
        coco_b = _make_coco(prefix="b", n_images=2, start_id=200)
        path_a = _write_coco(tmp_path, "a.json", coco_a)
        path_b = _write_coco(tmp_path, "b.json", coco_b)
        output = tmp_path / "active.json"

        compose_training_coco(
            sources=[("a", path_a), ("b", path_b)],
            output_path=output,
            force=True,
        )

        merged = json.loads(output.read_text())
        img_ids = [img["id"] for img in merged["images"]]
        ann_ids = [a["id"] for a in merged["annotations"]]

        assert img_ids == list(range(1, len(img_ids) + 1))
        assert ann_ids == list(range(1, len(ann_ids) + 1))

        img_id_set = {img["id"] for img in merged["images"]}
        for ann in merged["annotations"]:
            assert ann["image_id"] in img_id_set

    def test_compose_preserves_categories_exactly(self, tmp_path):
        coco_a = _make_coco(prefix="a")
        coco_b = _make_coco(prefix="b")
        coco_b["categories"] = [{"id": 1, "name": "different"}]
        path_a = _write_coco(tmp_path, "a.json", coco_a)
        path_b = _write_coco(tmp_path, "b.json", coco_b)
        output = tmp_path / "active.json"

        with pytest.raises(ValueError, match="Category mismatch"):
            compose_training_coco(
                sources=[("a", path_a), ("b", path_b)],
                output_path=output,
            )

        coco_b["categories"] = coco_a["categories"]
        path_b.write_text(json.dumps(coco_b))

        compose_training_coco(
            sources=[("a", path_a), ("b", path_b)],
            output_path=output,
            force=True,
        )
        merged = json.loads(output.read_text())
        assert merged["categories"] == coco_a["categories"]

    def test_compose_tags_dataset_source_on_merged_images(self, tmp_path):
        coco_a = _make_coco(prefix="a", n_images=2)
        coco_b = _make_coco(prefix="b", n_images=3)
        path_a = _write_coco(tmp_path, "a.json", coco_a)
        path_b = _write_coco(tmp_path, "b.json", coco_b)
        output = tmp_path / "active.json"

        compose_training_coco(
            sources=[("clean", path_a), ("noisy", path_b)],
            output_path=output,
            force=True,
        )

        merged = json.loads(output.read_text())
        sources_seen = {img["dataset_source"] for img in merged["images"]}
        assert sources_seen == {"clean", "noisy"}

        for img in merged["images"]:
            assert "dataset_source" in img
            assert img["dataset_source"] in ("clean", "noisy")

    def test_compose_single_source_creates_symlink(self, tmp_path):
        coco = _make_coco(prefix="only", n_images=4)
        src_path = _write_coco(tmp_path, "source.json", coco)
        output = tmp_path / "active.json"

        manifest = compose_training_coco(
            sources=[("noise_aug", src_path)],
            output_path=output,
        )

        assert manifest["mode"] == "symlink"
        assert output.is_symlink()
        assert output.resolve() == src_path.resolve()

        through_symlink = json.loads(output.read_text())
        assert through_symlink == coco

        for img in through_symlink["images"]:
            assert "dataset_source" not in img
