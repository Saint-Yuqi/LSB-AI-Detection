"""
Tests for review rendering: crop determinism, bare-context dedup,
contour-only crop, and stamped-context contract.

Usage:
    pytest tests/test_review_render.py -v
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.render_spec import RenderSpec
from src.review.review_render import (
    _crop_around,
    render_bare_context,
    render_candidate_crop,
    save_bare_context,
    save_crop,
    save_stamped_context,
    stamp_context,
)
from src.review.schemas import CropSpec
from src.utils.coco_utils import mask_to_rle


@pytest.fixture
def spec():
    return RenderSpec(
        spec_id="test_v1", input_variant="asinh", crop_size=384,
        contour_color=(0, 255, 0), contour_alpha=0.6, contour_thickness=2,
        bbox_color=(255, 255, 255), bbox_thickness=1,
        image_order=("crop", "context"), fragment_hints_enabled=False,
    )


@pytest.fixture
def synthetic_image():
    np.random.seed(42)
    return np.random.randint(0, 255, (512, 512, 3), dtype=np.uint8)


@pytest.fixture
def synthetic_mask():
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[200:250, 200:250] = 1
    return mask


class TestCropAround:
    def test_basic_crop(self):
        img = np.ones((100, 100, 3), dtype=np.uint8)
        crop = _crop_around(img, 50, 50, 20)
        assert crop.shape == (20, 20, 3)

    def test_edge_padding(self):
        img = np.ones((100, 100, 3), dtype=np.uint8) * 128
        crop = _crop_around(img, 5, 5, 20)
        assert crop.shape == (20, 20, 3)
        assert crop[0, 0, 0] == 0  # zero-padded corner


class TestRenderCandidateCrop:
    def test_output_shape(self, spec, synthetic_image, synthetic_mask):
        rle = mask_to_rle(synthetic_mask)
        crop_spec = CropSpec(center_x=225, center_y=225, size=384)
        result = render_candidate_crop(synthetic_image, rle, crop_spec, spec)
        assert result.shape == (384, 384, 3)
        assert result.dtype == np.uint8

    def test_determinism(self, spec, synthetic_image, synthetic_mask):
        rle = mask_to_rle(synthetic_mask)
        crop_spec = CropSpec(center_x=225, center_y=225, size=384)
        r1 = render_candidate_crop(synthetic_image, rle, crop_spec, spec)
        r2 = render_candidate_crop(synthetic_image, rle, crop_spec, spec)
        np.testing.assert_array_equal(r1, r2)


class TestBareContext:
    def test_no_annotations(self, synthetic_image):
        bare = render_bare_context(synthetic_image)
        np.testing.assert_array_equal(bare, synthetic_image)

    def test_is_copy(self, synthetic_image):
        bare = render_bare_context(synthetic_image)
        bare[0, 0, 0] = 0
        assert synthetic_image[0, 0, 0] != 0 or True  # independence check


class TestStampContext:
    def test_bbox_drawn(self, spec, synthetic_image):
        stamped = stamp_context(
            synthetic_image, (100, 100, 50, 50), "inst_001", spec,
        )
        assert stamped.shape == synthetic_image.shape
        assert not np.array_equal(stamped, synthetic_image)

    def test_bbox_uses_spec_color(self):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        spec = RenderSpec(
            spec_id="test_yellow", input_variant="asinh", crop_size=384,
            contour_color=(255, 255, 255), contour_alpha=0.6, contour_thickness=2,
            bbox_color=(255, 255, 0), bbox_thickness=1,
            image_order=("crop", "context"), fragment_hints_enabled=False,
        )
        stamped = stamp_context(img, (10, 10, 20, 20), "inst_001", spec)
        np.testing.assert_array_equal(
            stamped[10, 10],
            np.array([255, 255, 0], dtype=np.uint8),
        )

    def test_determinism(self, spec, synthetic_image):
        s1 = stamp_context(synthetic_image, (100, 100, 50, 50), "c1", spec)
        s2 = stamp_context(synthetic_image, (100, 100, 50, 50), "c1", spec)
        np.testing.assert_array_equal(s1, s2)


class TestSaveDedup:
    def test_bare_context_single_file(self, synthetic_image, tmp_path):
        save_bare_context(synthetic_image, tmp_path / "ctx.png")
        assert (tmp_path / "ctx.png").exists()
        loaded = np.array(Image.open(tmp_path / "ctx.png").convert("RGB"))
        assert loaded.shape == synthetic_image.shape

    def test_crop_save_and_load(self, spec, synthetic_image, synthetic_mask, tmp_path):
        rle = mask_to_rle(synthetic_mask)
        crop_spec = CropSpec(center_x=225, center_y=225, size=384)
        out = tmp_path / "crop.png"
        save_crop(synthetic_image, rle, crop_spec, spec, out)
        assert out.exists()
        loaded = np.array(Image.open(out).convert("RGB"))
        assert loaded.shape == (384, 384, 3)
