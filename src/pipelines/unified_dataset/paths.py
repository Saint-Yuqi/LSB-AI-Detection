"""
Path resolution for the unified dataset pipeline.

All input/output paths are derived from config — no hardcoded paths.
Supports both legacy {orientation} and new {view_id} format keys in patterns.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .keys import BaseKey


def _format_pattern(pattern: str, key: BaseKey, **extra) -> str:
    """Format a pattern string, supporting both {view_id} and legacy {orientation}."""
    fmt_kwargs = {"galaxy_id": key.galaxy_id, **extra}
    if "{view_id}" in pattern:
        fmt_kwargs["view_id"] = key.view_id
    else:
        fmt_kwargs["orientation"] = key.view_id
    return pattern.format(**fmt_kwargs)


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
        pattern = _format_pattern(src["image_pattern"], key)
        return self.firebox_root / src["image_subdir"] / pattern

    def get_mask_path(self, key: BaseKey, sb_threshold: float) -> Optional[Path]:
        src = self.data_sources["streams"]

        mask_subdir_map = src.get("mask_subdir_map")
        if mask_subdir_map is not None:
            subdir = mask_subdir_map.get(key.view_id)
            if subdir is None:
                return None
        else:
            subdir_eo = src.get("mask_subdir_eo")
            subdir_fo = src.get("mask_subdir_fo")
            if subdir_eo is None and subdir_fo is None:
                return None
            subdir = subdir_eo if key.view_id == "eo" else subdir_fo

        mask_pattern = src.get("mask_pattern")
        if mask_pattern is None:
            return None

        threshold = int(sb_threshold) if sb_threshold == int(sb_threshold) else sb_threshold
        pattern = _format_pattern(mask_pattern, key, threshold=threshold)
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
