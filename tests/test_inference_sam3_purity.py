"""Purity guardrails for the standard DR1 SAM3 inference path.

These tests do not exercise the full pipeline; they verify the narrow contract
that the runtime never consumes legacy satellite override configuration and
never injects external raw masks.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset import inference_sam3 as INF


class TestSatelliteOverridesConfigGuard:
    def test_fail_fast_when_overrides_path_present(self) -> None:
        cfg = {
            "inference_phase": {
                "sam3": {
                    "satellite_overrides_path": "configs/archive/sam3_satellite_overrides_legacy.yaml",
                }
            }
        }
        with pytest.raises(ValueError, match="satellite_overrides_path"):
            INF._assert_no_runtime_overrides(cfg)

    def test_passes_when_overrides_path_absent(self) -> None:
        cfg = {"inference_phase": {"sam3": {}}}
        INF._assert_no_runtime_overrides(cfg)

    def test_passes_when_inference_phase_missing(self) -> None:
        INF._assert_no_runtime_overrides({})


class TestLegacyOverrideHelpersAreGone:
    @pytest.mark.parametrize(
        "attr",
        [
            "_load_satellite_overrides",
            "_inject_external_satellite_masks",
            "_mask_rle_sha1",
        ],
    )
    def test_helper_is_removed(self, attr: str) -> None:
        assert not hasattr(INF, attr), (
            f"{attr} must not exist on inference_sam3 after runtime cleanup; "
            "reviewed exceptions are migrated through the Shadow GT flow."
        )
