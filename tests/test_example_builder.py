"""
Tests for example_builder: dual-source kept/rejected candidates,
candidate_bbox_xywh from get_bbox_from_mask, no index alignment with
id_map.json or predictions_post.json.

Usage:
    pytest tests/test_example_builder.py -v
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

from src.review.asset_manager import AssetManager
from src.review.example_builder import (
    _build_sat_mv_authoritative,
    _build_sat_mv_rejected,
    _example_to_dict,
)
from src.review.render_spec import RenderSpec
from src.review.schemas import TaskFamily
from src.review.silver_labeler import SilverLabel
from src.pipelines.unified_dataset.keys import BaseKey
from src.utils.coco_utils import mask_to_rle


@pytest.fixture
def spec():
    return RenderSpec(
        spec_id="sat_mv_v1", input_variant="asinh", crop_size=384,
        contour_color=(0, 255, 0), contour_alpha=0.6, contour_thickness=2,
        bbox_color=(255, 255, 255), bbox_thickness=1,
        image_order=("crop", "context"), fragment_hints_enabled=False,
    )


@pytest.fixture
def gt_dir(tmp_path):
    bk_dir = tmp_path / "gt" / "00011_eo"
    bk_dir.mkdir(parents=True)

    imap = np.zeros((100, 100), dtype=np.uint8)
    imap[20:40, 20:40] = 1
    imap[60:80, 60:80] = 2
    Image.fromarray(imap).save(bk_dir / "instance_map_uint8.png")

    instances = [
        {"id": 1, "type": "satellites"},
        {"id": 2, "type": "streams"},
    ]
    (bk_dir / "instances.json").write_text(json.dumps(instances))

    return tmp_path / "gt"


@pytest.fixture
def full_image():
    return np.ones((100, 100, 3), dtype=np.uint8) * 128


@pytest.fixture
def asset_mgr(tmp_path):
    return AssetManager(tmp_path / "assets")


@pytest.fixture
def silver_lookup():
    return {
        "inst_001": SilverLabel(
            sample_id="00011_eo", candidate_key="inst_001",
            decision_label="accept", confidence=0.9,
            signals={"best_iou": 0.9},
        ),
    }


class TestBuildAuthoritativeCandidates:
    def test_builds_from_instance_map(
        self, gt_dir, spec, full_image, asset_mgr, silver_lookup,
    ):
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, counter = _build_sat_mv_authoritative(
            bk, gt_dir, imap, instances, silver_lookup,
            asset_mgr, spec, "sat_mv_v1", full_image,
            "round_00", "initial", 0,
        )

        assert len(examples) == 1
        ex = examples[0]
        assert ex.asset_refs.candidate_source == "authoritative"
        assert ex.asset_refs.authoritative_instance_id == 1
        assert ex.decision_label == "accept"

    def test_bbox_from_mask_not_crop_spec(
        self, gt_dir, spec, full_image, asset_mgr, silver_lookup,
    ):
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_sat_mv_authoritative(
            bk, gt_dir, imap, instances, silver_lookup,
            asset_mgr, spec, "sat_mv_v1", full_image,
            "round_00", "initial", 0,
        )

        ex = examples[0]
        bbox = ex.asset_refs.candidate_bbox_xywh
        assert bbox[2] < 384  # bbox width < crop_size
        assert bbox[3] < 384  # bbox height < crop_size


class TestBuildRejectedCandidates:
    def test_builds_reject_examples(
        self, gt_dir, spec, full_image, asset_mgr,
    ):
        bk = BaseKey(galaxy_id=11, view_id="eo")

        reject_mask = np.zeros((100, 100), dtype=np.uint8)
        reject_mask[5:15, 5:15] = 1
        rejects = [{
            "segmentation": reject_mask,
            "rle": mask_to_rle(reject_mask),
            "bucket": "prior_rejected",
            "index": 0,
            "score": 0.3,
            "stability_score": 0.4,
        }]

        examples, counter = _build_sat_mv_rejected(
            bk, gt_dir, rejects,
            asset_mgr, spec, "sat_mv_v1", full_image,
            "round_00", "initial", 0,
        )

        assert len(examples) == 1
        ex = examples[0]
        assert ex.asset_refs.candidate_source == "pre_merge_rejected"
        assert ex.asset_refs.authoritative_instance_id is None
        assert ex.decision_label == "reject"
        assert "rej_" in ex.asset_refs.candidate_id


class TestExampleSerialization:
    def test_round_trip(
        self, gt_dir, spec, full_image, asset_mgr, silver_lookup,
    ):
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_sat_mv_authoritative(
            bk, gt_dir, imap, instances, silver_lookup,
            asset_mgr, spec, "sat_mv_v1", full_image,
            "round_00", "initial", 0,
        )

        d = _example_to_dict(examples[0])
        serialized = json.dumps(d, sort_keys=True)
        parsed = json.loads(serialized)

        assert parsed["task_family"] == "satellite_mv"
        assert parsed["asset_refs"]["candidate_source"] == "authoritative"
        assert parsed["asset_refs"]["authoritative_instance_id"] == 1


class TestEvExampleBuilder:
    """Multi-variant EV example builder tests."""

    @pytest.fixture
    def ev_spec(self):
        return RenderSpec(
            spec_id="sat_ev_v1", input_variant="asinh", crop_size=384,
            contour_color=(0, 255, 0), contour_alpha=0.6, contour_thickness=2,
            bbox_color=(255, 255, 255), bbox_thickness=1,
            image_order=("full_image",), fragment_hints_enabled=True,
        )

    @pytest.fixture
    def ev_gt_dir(self, tmp_path):
        bk_dir = tmp_path / "gt" / "00011_eo"
        bk_dir.mkdir(parents=True)

        imap = np.zeros((100, 100), dtype=np.uint8)
        imap[20:40, 20:40] = 1  # satellite 1 (area 400)
        imap[60:80, 60:80] = 2  # satellite 2 (area 400)
        Image.fromarray(imap).save(bk_dir / "instance_map_uint8.png")

        instances = [
            {"id": 1, "type": "satellites"},
            {"id": 2, "type": "satellites"},
        ]
        (bk_dir / "instances.json").write_text(json.dumps(instances))
        return tmp_path / "gt"

    @pytest.fixture
    def ev_silver_lookup(self):
        """Reduced-variant silver lookup for satellite_ev."""
        return {
            "image:gt_complete": SilverLabel(
                sample_id="00011_eo", candidate_key="image:gt_complete",
                decision_label="confirm_complete", confidence=1.0,
                signals={
                    "synthetic_variant_id": "gt_complete",
                    "visible_instance_ids": [1, 2],
                    "hidden_instance_ids": [],
                },
            ),
            "image:drop_top2": SilverLabel(
                sample_id="00011_eo", candidate_key="image:drop_top2",
                decision_label="add_missing", confidence=1.0,
                signals={
                    "synthetic_variant_id": "drop_top2",
                    "visible_instance_ids": [],
                    "hidden_instance_ids": [1, 2],
                },
            ),
        }

    def test_multi_variant_count(
        self, ev_gt_dir, ev_spec, full_image, asset_mgr, ev_silver_lookup,
    ):
        from src.review.example_builder import _build_ev_examples
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(ev_gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(ev_gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, counter = _build_ev_examples(
            bk, TaskFamily.SATELLITE_EV, ev_gt_dir, imap, instances,
            ev_silver_lookup, asset_mgr, ev_spec, "sat_ev_v1", full_image,
            "round_00", "initial", 0,
        )
        assert len(examples) == 2
        assert counter == 2

    def test_unique_review_image_paths(
        self, ev_gt_dir, ev_spec, full_image, asset_mgr, ev_silver_lookup,
    ):
        from src.review.example_builder import _build_ev_examples
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(ev_gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(ev_gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_ev_examples(
            bk, TaskFamily.SATELLITE_EV, ev_gt_dir, imap, instances,
            ev_silver_lookup, asset_mgr, ev_spec, "sat_ev_v1", full_image,
            "round_00", "initial", 0,
        )
        paths = [ex.asset_refs.review_image_path for ex in examples]
        assert len(set(paths)) == len(paths), "Paths must be unique"

    def test_unique_revision_hashes(
        self, ev_gt_dir, ev_spec, full_image, asset_mgr, ev_silver_lookup,
    ):
        from src.review.example_builder import _build_ev_examples
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(ev_gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(ev_gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_ev_examples(
            bk, TaskFamily.SATELLITE_EV, ev_gt_dir, imap, instances,
            ev_silver_lookup, asset_mgr, ev_spec, "sat_ev_v1", full_image,
            "round_00", "initial", 0,
        )
        hashes = [ex.revision_hash for ex in examples]
        assert len(set(hashes)) == len(hashes), "Revision hashes must be unique"

    def test_hidden_ids_not_in_fragment_hints(
        self, ev_gt_dir, ev_spec, full_image, asset_mgr, ev_silver_lookup,
    ):
        from src.review.example_builder import _build_ev_examples
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(ev_gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(ev_gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_ev_examples(
            bk, TaskFamily.SATELLITE_EV, ev_gt_dir, imap, instances,
            ev_silver_lookup, asset_mgr, ev_spec, "sat_ev_v1", full_image,
            "round_00", "initial", 0,
        )

        for ex in examples:
            hidden = set(ex.asset_refs.hidden_instance_ids or ())
            if ex.asset_refs.fragment_hints is not None:
                hint_ids = {h["instance_id"] for h in ex.asset_refs.fragment_hints}
                assert hidden & hint_ids == set(), \
                    f"Hidden IDs rendered in hints for {ex.asset_refs.synthetic_variant_id}"

    def test_serializes_new_fields(
        self, ev_gt_dir, ev_spec, full_image, asset_mgr, ev_silver_lookup,
    ):
        from src.review.example_builder import _build_ev_examples
        bk = BaseKey(galaxy_id=11, view_id="eo")
        imap = np.array(Image.open(ev_gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(ev_gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)

        examples, _ = _build_ev_examples(
            bk, TaskFamily.SATELLITE_EV, ev_gt_dir, imap, instances,
            ev_silver_lookup, asset_mgr, ev_spec, "sat_ev_v1", full_image,
            "round_00", "initial", 0,
        )
        drop_example = next(
            ex for ex in examples
            if ex.asset_refs.synthetic_variant_id != "gt_complete"
        )
        d = _example_to_dict(drop_example)
        arefs = d["asset_refs"]
        assert "synthetic_variant_id" in arefs
        assert "hidden_instance_ids" in arefs
        assert "visible_instance_ids" in arefs
