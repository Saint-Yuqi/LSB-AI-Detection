"""
Path resolution for the unified dataset pipeline.

All input/output paths are derived from config — no hardcoded paths.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .keys import BaseKey


class PathResolver:
    """Resolves input/output paths based on config."""

    def __init__(self, config: dict[str, Any]):
        self.firebox_root = Path(config["paths"]["firebox_root"])
        self.output_root = Path(config["paths"]["output_root"])
        self.data_sources = config["data_sources"]
        self.target_size = tuple(config["processing"]["target_size"])

    # Input paths
    def get_fits_path(self, key: BaseKey) -> Path:
        src = self.data_sources["streams"]
        pattern = src["image_pattern"].format(
            galaxy_id=key.galaxy_id, orientation=key.orientation
        )
        return self.firebox_root / src["image_subdir"] / pattern

    def get_mask_path(self, key: BaseKey, sb_threshold: float) -> Path:
        src = self.data_sources["streams"]
        subdir = src["mask_subdir_eo"] if key.orientation == "eo" else src["mask_subdir_fo"]
        pattern = src["mask_pattern"].format(
            galaxy_id=key.galaxy_id,
            orientation=key.orientation,
            threshold=int(sb_threshold) if sb_threshold == int(sb_threshold) else sb_threshold,
        )
        return self.firebox_root / subdir / pattern

    # Output paths
    def get_render_dir(self, preprocessing: str, key: BaseKey) -> Path:
        return self.output_root / "renders" / "current" / preprocessing / str(key)

    def get_gt_dir(self, key: BaseKey) -> Path:
        return self.output_root / "gt_canonical" / "current" / str(key)

    def get_sam2_dir(self) -> Path:
        return self.output_root / "sam2_prepared"

    def get_sam3_dir(self) -> Path:
        return self.output_root / "sam3_prepared"
