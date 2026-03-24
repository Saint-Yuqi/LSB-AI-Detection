"""
Behavior-locking tests for CLI argument parsing and phase dispatch.

Pins: --phase satellites alias, --force populates force_variants from config,
--force-variants only sets the named set.

Usage:
    pytest tests/test_cli_compat.py -v
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "data"))


class TestPhaseChoices:
    def test_satellites_is_valid_choice(self):
        """--phase satellites must be accepted by argparse."""
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--phase",
            type=str,
            choices=["render", "gt", "inference", "satellites", "export", "all"],
            default="all",
        )
        args = parser.parse_args(["--phase", "satellites"])
        assert args.phase == "satellites"

    def test_satellites_dispatches_to_inference(self):
        """In main(), 'satellites' phase dispatches to run_inference_phase."""
        from prepare_unified_dataset import main

        phases_called = []

        def fake_render(*a, **kw):
            phases_called.append("render")

        def fake_gt(*a, **kw):
            phases_called.append("gt")

        def fake_inference(*a, **kw):
            phases_called.append("inference")

        def fake_export(*a, **kw):
            phases_called.append("export")

        minimal_config = {
            "paths": {"firebox_root": "/dummy", "output_root": "/dummy"},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [{"name": "asinh_stretch"}],
            "data_selection": {
                "galaxy_ids": [11],
                "views": ["eo"],
                "canonical_sb_threshold": 32,
            },
        }

        with (
            patch("sys.argv", ["prog", "--config", "dummy.yaml", "--phase", "satellites"]),
            patch("prepare_unified_dataset.load_config", return_value=minimal_config),
            patch("prepare_unified_dataset.run_render_phase", fake_render),
            patch("prepare_unified_dataset.run_gt_phase", fake_gt),
            patch("prepare_unified_dataset.run_inference_phase", fake_inference),
            patch("prepare_unified_dataset.run_export_phase", fake_export),
        ):
            main()

        assert phases_called == ["inference"]


class TestForceVariants:
    @pytest.fixture
    def minimal_config(self):
        return {
            "paths": {"firebox_root": "/dummy", "output_root": "/dummy"},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [
                {"name": "asinh_stretch"},
                {"name": "linear_magnitude"},
            ],
            "data_selection": {
                "galaxy_ids": [11],
                "views": ["eo"],
                "canonical_sb_threshold": 32,
            },
        }

    def test_force_flag_populates_all_variants(self, minimal_config):
        """--force should set force_variants to all config variant names."""
        captured_fv = {}

        def capture_render(config, keys, logger, force_variants=None):
            captured_fv["render"] = force_variants

        with (
            patch("sys.argv", ["prog", "--config", "d.yaml", "--phase", "render", "--force"]),
            patch("prepare_unified_dataset.load_config", return_value=minimal_config),
            patch("prepare_unified_dataset.run_render_phase", capture_render),
        ):
            from prepare_unified_dataset import main
            main()

        fv = captured_fv["render"]
        assert fv is not None
        assert fv == {"asinh_stretch", "linear_magnitude"}

    def test_force_variants_flag_named_only(self, minimal_config):
        """--force-variants asinh_stretch should only include that variant."""
        captured_fv = {}

        def capture_render(config, keys, logger, force_variants=None):
            captured_fv["render"] = force_variants

        with (
            patch(
                "sys.argv",
                ["prog", "--config", "d.yaml", "--phase", "render", "--force-variants", "asinh_stretch"],
            ),
            patch("prepare_unified_dataset.load_config", return_value=minimal_config),
            patch("prepare_unified_dataset.run_render_phase", capture_render),
        ):
            from prepare_unified_dataset import main
            main()

        fv = captured_fv["render"]
        assert fv == {"asinh_stretch"}

    def test_no_force_gives_none(self, minimal_config):
        """Without --force or --force-variants, force_variants should be None."""
        captured_fv = {}

        def capture_render(config, keys, logger, force_variants=None):
            captured_fv["render"] = force_variants

        with (
            patch("sys.argv", ["prog", "--config", "d.yaml", "--phase", "render"]),
            patch("prepare_unified_dataset.load_config", return_value=minimal_config),
            patch("prepare_unified_dataset.run_render_phase", capture_render),
        ):
            from prepare_unified_dataset import main
            main()

        assert captured_fv["render"] is None


class TestGtPhaseSkip:
    def test_phase_all_skips_gt_when_disabled(self):
        """--phase all with gt_phase.enabled=false must NOT call run_gt_phase."""
        from prepare_unified_dataset import main

        phases_called = []

        def fake_render(*a, **kw):
            phases_called.append("render")

        def fake_gt(*a, **kw):
            phases_called.append("gt")

        def fake_inference(*a, **kw):
            phases_called.append("inference")

        def fake_export(*a, **kw):
            phases_called.append("export")

        config_gt_disabled = {
            "paths": {"firebox_root": "/dummy", "output_root": "/dummy"},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [64, 64]},
            "preprocessing_variants": [{"name": "linear_magnitude"}],
            "data_selection": {
                "galaxy_ids": [11],
                "views": ["los00"],
                "canonical_sb_threshold": 32,
            },
            "gt_phase": {"enabled": False},
        }

        with (
            patch("sys.argv", ["prog", "--config", "dummy.yaml", "--phase", "all"]),
            patch("prepare_unified_dataset.load_config", return_value=config_gt_disabled),
            patch("prepare_unified_dataset.run_render_phase", fake_render),
            patch("prepare_unified_dataset.run_gt_phase", fake_gt),
            patch("prepare_unified_dataset.run_inference_phase", fake_inference),
            patch("prepare_unified_dataset.run_export_phase", fake_export),
        ):
            main()

        assert "gt" not in phases_called
        assert "render" in phases_called
        assert "inference" in phases_called
        assert "export" in phases_called
