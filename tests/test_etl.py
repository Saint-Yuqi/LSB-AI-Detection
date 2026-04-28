"""
Tests for ETL: business JSONL → chat JSONL round-trip, hash stability,
stamped-context contract, and family isolation.

Usage:
    pytest tests/test_etl.py -v
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

from src.review.etl import run_etl, transform_example
from src.review.prompt_registry import PromptRegistry
from src.review.render_spec import RenderSpecRegistry
from src.review.schemas import CropSpec, TaskFamily
from src.utils.coco_utils import mask_to_rle


@pytest.fixture
def registries(tmp_path):
    spec_yaml = tmp_path / "spec.yaml"
    spec_yaml.write_text("""
render_specs:
  - spec_id: sat_mv_v1
    input_variant: asinh
    crop_size: 384
    contour_color: [0, 255, 0]
    contour_alpha: 0.6
    contour_thickness: 2
    bbox_color: [255, 255, 255]
    bbox_thickness: 1
    image_order: [crop, context]
    fragment_hints_enabled: false
  - spec_id: sat_ev_v1
    input_variant: asinh
    crop_size: 384
    contour_color: [0, 255, 0]
    contour_alpha: 0.6
    contour_thickness: 2
    bbox_color: [255, 255, 255]
    bbox_thickness: 1
    image_order: [full_image]
    fragment_hints_enabled: false
""")

    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("""
prompts:
  - fixed_prompt_id: sat_mv_v1
    isolation_token: "<SAT_MV>"
    text_template: "Decide: accept | reject"
    allowed_labels: [accept, reject]
  - fixed_prompt_id: sat_ev_v1
    isolation_token: "<SAT_EV>"
    text_template: "Decide: confirm_complete | add_missing | confirm_empty"
    allowed_labels: [confirm_complete, add_missing, confirm_empty]
""")

    return (
        RenderSpecRegistry.from_yaml(spec_yaml),
        PromptRegistry.from_yaml(prompt_yaml),
    )


@pytest.fixture
def asset_root(tmp_path):
    ar = tmp_path / "assets"
    ar.mkdir()

    ctx_dir = ar / "review_context"
    ctx_dir.mkdir()
    img = np.ones((512, 512, 3), dtype=np.uint8) * 128
    Image.fromarray(img).save(ctx_dir / "00011_eo_asinh.png")

    crop_dir = ar / "review_crops" / "00011_eo"
    crop_dir.mkdir(parents=True)
    crop = np.ones((384, 384, 3), dtype=np.uint8) * 100
    Image.fromarray(crop).save(crop_dir / "225_225_384_zero_sat_mv_v1.png")

    ev_dir = ar / "review_ev"
    ev_dir.mkdir()
    Image.fromarray(img).save(ev_dir / "00011_eo_asinh_sat_ev_v1.png")

    return ar


def _make_sat_mv_record():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:250, 200:250] = 1
    rle = mask_to_rle(mask)
    return {
        "example_id": "satell_000001",
        "task_family": "satellite_mv",
        "sample_id": "00011_eo",
        "halo_id": 11,
        "view_id": "eo",
        "task_revision_id": "rev_00",
        "render_spec_id": "sat_mv_v1",
        "fixed_prompt_id": "sat_mv_v1",
        "decision_label": "accept",
        "label_source": "silver",
        "source_round": "round_00",
        "source_checkpoint": "initial",
        "revision_hash": "abc123",
        "asset_refs": {
            "bare_context_path": "review_context/00011_eo_asinh.png",
            "bare_context_sha1": "sha1hex",
            "crop_spec": {"center_x": 225, "center_y": 225, "size": 384, "pad_mode": "zero"},
            "candidate_bbox_xywh": [200, 200, 50, 50],
            "candidate_rle": rle,
            "candidate_id": "inst_001",
            "authoritative_instance_id": 1,
            "candidate_source": "authoritative",
        },
    }


def _make_ev_record():
    return {
        "example_id": "satell_000002",
        "task_family": "satellite_ev",
        "sample_id": "00011_eo",
        "halo_id": 11,
        "view_id": "eo",
        "task_revision_id": "rev_00",
        "render_spec_id": "sat_ev_v1",
        "fixed_prompt_id": "sat_ev_v1",
        "decision_label": "confirm_complete",
        "label_source": "silver",
        "source_round": "round_00",
        "source_checkpoint": "initial",
        "revision_hash": "def456",
        "asset_refs": {
            "review_image_path": "review_ev/00011_eo_asinh_sat_ev_v1.png",
            "review_image_sha1": "sha1",
            "annotation_state_hash": "annhash",
        },
    }


class TestTransformExample:
    def test_sat_mv_two_turn(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_sat_mv_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        assert len(chat["messages"]) == 2
        assert chat["messages"][0]["role"] == "user"
        assert chat["messages"][1]["role"] == "assistant"
        assert chat["messages"][1]["content"] == "accept"

    def test_no_system_role(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_sat_mv_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        roles = [m["role"] for m in chat["messages"]]
        assert "system" not in roles

    def test_sat_mv_has_two_images(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_sat_mv_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        user_content = chat["messages"][0]["content"]
        image_items = [c for c in user_content if c["type"] == "image"]
        assert len(image_items) == 2

    def test_isolation_token_present(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_sat_mv_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        text = chat["messages"][0]["content"][0]["text"]
        assert "<SAT_MV>" in text

    def test_metadata_present(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_sat_mv_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        assert "metadata" in chat
        assert chat["metadata"]["example_id"] == rec["example_id"]
        assert chat["metadata"]["revision_hash"] == rec["revision_hash"]

    def test_ev_single_image(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries
        rec = _make_ev_record()
        chat = transform_example(rec, prompt_reg, spec_reg, asset_root, tmp_path)
        user_content = chat["messages"][0]["content"]
        image_items = [c for c in user_content if c["type"] == "image"]
        assert len(image_items) == 1


class TestRunEtl:
    def test_round_trip(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries

        examples_path = tmp_path / "examples.jsonl"
        with open(examples_path, "w") as f:
            f.write(json.dumps(_make_sat_mv_record()) + "\n")

        output_path = tmp_path / "chat.jsonl"
        manifest = run_etl(
            examples_path, prompt_reg, spec_reg,
            asset_root, tmp_path / "renders", output_path,
        )

        assert output_path.exists()
        assert manifest["num_records"] == 1
        assert "output_sha256" in manifest

        with open(output_path) as f:
            chat = json.loads(f.readline())
        assert chat["messages"][1]["content"] == "accept"

    def test_hash_stability(self, registries, asset_root, tmp_path):
        spec_reg, prompt_reg = registries

        examples_path = tmp_path / "examples.jsonl"
        with open(examples_path, "w") as f:
            f.write(json.dumps(_make_sat_mv_record()) + "\n")

        out1 = tmp_path / "chat1.jsonl"
        out2 = tmp_path / "chat2.jsonl"

        m1 = run_etl(examples_path, prompt_reg, spec_reg,
                      asset_root, tmp_path / "r1", out1)
        m2 = run_etl(examples_path, prompt_reg, spec_reg,
                      asset_root, tmp_path / "r2", out2)

        assert m1["output_sha256"] == m2["output_sha256"]
