"""
Behavior-locking tests for BaseKey, VariantKey, and generate_base_keys.

Pins zero-padding format, composite string repr, and galaxy filter logic
before any refactoring begins.

Usage:
    pytest tests/test_dataset_keys.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from prepare_unified_dataset import BaseKey, VariantKey, generate_base_keys


class TestBaseKey:
    def test_str_zero_pads_galaxy_id(self):
        k = BaseKey(galaxy_id=11, view_id="eo")
        assert str(k) == "00011_eo"

    def test_str_large_galaxy_id(self):
        k = BaseKey(galaxy_id=99999, view_id="fo")
        assert str(k) == "99999_fo"

    def test_str_single_digit(self):
        k = BaseKey(galaxy_id=3, view_id="eo")
        assert str(k) == "00003_eo"

    def test_str_los_view(self):
        k = BaseKey(galaxy_id=11, view_id="los00")
        assert str(k) == "00011_los00"

    def test_frozen(self):
        k = BaseKey(galaxy_id=11, view_id="eo")
        with pytest.raises(AttributeError):
            k.galaxy_id = 12

    def test_equality_and_hash(self):
        a = BaseKey(11, "eo")
        b = BaseKey(11, "eo")
        assert a == b
        assert hash(a) == hash(b)
        assert a != BaseKey(11, "fo")
        assert a != BaseKey(11, "los00")


class TestVariantKey:
    def test_str_format(self):
        bk = BaseKey(galaxy_id=13, view_id="fo")
        vk = VariantKey(base_key=bk, preprocessing="asinh_stretch")
        assert str(vk) == "00013_fo_asinh_stretch"

    def test_str_multi_exposure(self):
        bk = BaseKey(galaxy_id=7, view_id="eo")
        vk = VariantKey(base_key=bk, preprocessing="multi_exposure")
        assert str(vk) == "00007_eo_multi_exposure"

    def test_str_los_view(self):
        bk = BaseKey(galaxy_id=11, view_id="los05")
        vk = VariantKey(base_key=bk, preprocessing="linear_magnitude")
        assert str(vk) == "00011_los05_linear_magnitude"


class TestGenerateBaseKeys:
    @pytest.fixture
    def config(self):
        return {
            "data_selection": {
                "galaxy_ids": [11, 13, 15, 17],
                "views": ["eo", "fo"],
            }
        }

    def test_all_keys(self, config):
        keys = generate_base_keys(config)
        assert len(keys) == 8  # 4 galaxies x 2 views
        assert all(isinstance(k, BaseKey) for k in keys)

    def test_galaxy_filter(self, config):
        keys = generate_base_keys(config, galaxy_filter=[11, 17])
        assert len(keys) == 4  # 2 galaxies x 2 views
        galaxy_ids = {k.galaxy_id for k in keys}
        assert galaxy_ids == {11, 17}

    def test_galaxy_filter_empty(self, config):
        keys = generate_base_keys(config, galaxy_filter=[999])
        assert len(keys) == 0

    def test_no_filter(self, config):
        keys = generate_base_keys(config, galaxy_filter=None)
        assert len(keys) == 8

    def test_order_galaxy_first_then_view(self, config):
        keys = generate_base_keys(config)
        assert str(keys[0]) == "00011_eo"
        assert str(keys[1]) == "00011_fo"
        assert str(keys[2]) == "00013_eo"

    def test_legacy_orientations_compat(self):
        """data_selection.orientations still works with deprecation warning."""
        import warnings
        config = {
            "data_selection": {
                "galaxy_ids": [11],
                "orientations": ["eo", "fo"],
            }
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            keys = generate_base_keys(config)
            assert len(keys) == 2
            assert any("deprecated" in str(warning.message).lower() for warning in w)

    def test_24_view_keys(self):
        views = [f"los{i:02d}" for i in range(24)]
        config = {"data_selection": {"galaxy_ids": [11], "views": views}}
        keys = generate_base_keys(config)
        assert len(keys) == 24
        assert str(keys[0]) == "00011_los00"
        assert str(keys[23]) == "00011_los23"
