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
        paths_cfg = config["paths"]
        self.firebox_root = Path(paths_cfg.get("firebox_root", "."))
        self.output_root = Path(paths_cfg["output_root"])
        # Subdirectory names under output_root. New (tidal_v1) configs set these
        # to a versioned name (e.g. "gt_canonical_tidal_v1"); old configs leave
        # them at the legacy defaults so existing GT trees remain untouched.
        self.gt_subdir = paths_cfg.get("gt_subdir", "gt_canonical")
        self.pseudo_gt_subdir = paths_cfg.get("pseudo_gt_subdir")  # None -> use legacy pseudo_gt_root
        if self.pseudo_gt_subdir is not None:
            self.pseudo_gt_root = self.output_root / self.pseudo_gt_subdir
        else:
            self.pseudo_gt_root = Path(
                paths_cfg.get("pseudo_gt_root", self.output_root / "pseudo_gt_canonical")
            )
        self.diagnostics_root = Path(paths_cfg.get("diagnostics_root", self.output_root / "sam3_diagnostics"))
        self.sam3_root = Path(paths_cfg.get("sam3_prepared_root", self.output_root / "sam3_prepared"))
        self.data_sources = config.get("data_sources", {})
        self.target_size = tuple(config["processing"]["target_size"])
        self.dataset_name = config.get("dataset_name", "dr1")
        self.data_conditions = config.get("data_conditions", {})
        self.active_conditions = list(
            config.get("_active_conditions")
            or self.data_conditions.keys()
            or ["clean"]
        )
        self.default_condition = config.get("_default_condition", self.active_conditions[0])

    def get_active_conditions(self) -> list[str]:
        return list(self.active_conditions)

    def get_label_mode(self, condition: str | None = None) -> str:
        cfg = self._get_condition_cfg(condition)
        return cfg.get("label_mode", "authoritative")

    def get_dataset_name(self, dataset: str | None = None) -> str:
        return dataset or self.dataset_name

    def _get_condition_cfg(self, condition: str | None = None) -> dict[str, Any]:
        if not self.data_conditions:
            return {}
        cond = condition or self.default_condition
        try:
            return self.data_conditions[cond]
        except KeyError as exc:
            raise KeyError(f"Unknown condition {cond!r}; expected one of {sorted(self.data_conditions)}") from exc

    def _get_condition_fits_root(self, condition: str | None = None) -> Path:
        cfg = self._get_condition_cfg(condition)
        if "fits_root" in cfg:
            return Path(cfg["fits_root"])
        if "image_subdir" in cfg:
            return self.firebox_root / cfg["image_subdir"]
        src = self.data_sources.get("streams", {})
        return self.firebox_root / src.get("image_subdir", "")

    def _get_condition_image_pattern(self, condition: str | None = None) -> str:
        cfg = self._get_condition_cfg(condition)
        if "image_pattern" in cfg:
            return cfg["image_pattern"]
        src = self.data_sources.get("streams", {})
        pattern = src.get("image_pattern")
        if pattern is None:
            raise KeyError("Config must define data_sources.streams.image_pattern")
        return pattern

    def _scoped_root(self, base_root: Path, dataset: str | None = None) -> Path:
        dataset_name = self.get_dataset_name(dataset)
        return base_root if dataset_name == "dr1" else base_root / dataset_name

    # Input paths
    def get_fits_path(self, key: BaseKey) -> Path:
        return self.get_condition_fits_path(key, condition=self.default_condition)

    def get_condition_fits_path(
        self,
        key: BaseKey,
        dataset: str | None = None,
        condition: str | None = None,
    ) -> Path:
        del dataset  # reserved for future dataset-specific FITS handling
        root = self._get_condition_fits_root(condition)
        pattern = _format_pattern(self._get_condition_image_pattern(condition), key)
        return root / pattern

    def get_mask_path(self, key: BaseKey, sb_threshold: float) -> Optional[Path]:
        src = self.data_sources.get("streams", {})

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
    def get_render_dir(
        self,
        preprocessing: str,
        key: BaseKey,
        dataset: str | None = None,
        condition: str | None = None,
    ) -> Path:
        del dataset  # renders are currently shared under output_root
        cond = condition or self.default_condition
        if cond == "clean":
            return self.output_root / "renders" / "current" / preprocessing / str(key)
        return self.output_root / "renders" / "noisy" / preprocessing / cond / str(key)

    def get_gt_dir(self, key: BaseKey) -> Path:
        # Pure getter (F15): return the path; writers create the directory.
        return self.output_root / self.gt_subdir / "current" / str(key)

    def is_new_path(self) -> bool:
        """True when either GT subdir is non-default (tidal_v1 path active).

        Used by ``_prepare_sam3_context`` to gate the strict
        ``inference_phase.sam3.prior_filter`` requirement and the
        per-path runner construction (F4, F18).
        """
        return (
            self.gt_subdir != "gt_canonical"
            or (self.pseudo_gt_subdir is not None and self.pseudo_gt_subdir != "pseudo_gt_canonical")
        )

    def get_pseudo_gt_dir(
        self,
        key: BaseKey,
        dataset: str | None = None,
        condition: str | None = None,
    ) -> Path:
        cond = condition or self.default_condition
        return self._scoped_root(self.pseudo_gt_root, dataset) / cond / "current" / str(key)

    def get_pseudo_gt_condition_root(
        self,
        dataset: str | None = None,
        condition: str | None = None,
    ) -> Path:
        cond = condition or self.default_condition
        return self._scoped_root(self.pseudo_gt_root, dataset) / cond / "current"

    def get_diagnostics_dir(
        self,
        key: BaseKey,
        dataset: str | None = None,
        condition: str | None = None,
    ) -> Path:
        cond = condition or self.default_condition
        return self._scoped_root(self.diagnostics_root, dataset) / cond / "current" / str(key)

    def get_sam3_dir(self, dataset: str | None = None) -> Path:
        return self._scoped_root(self.sam3_root, dataset)
