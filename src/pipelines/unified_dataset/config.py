"""
Config loading and base key generation for the unified dataset pipeline.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

from .keys import BaseKey

logger = logging.getLogger(__name__)


def load_config(path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _resolve_views(config: dict) -> list[str]:
    """Read data_selection.views; fall back to orientations with deprecation warning."""
    ds = config["data_selection"]
    views = ds.get("views")
    if views is not None:
        return views
    orientations = ds.get("orientations")
    if orientations is not None:
        warnings.warn(
            "data_selection.orientations is deprecated; use data_selection.views instead",
            DeprecationWarning,
            stacklevel=3,
        )
        return orientations
    raise KeyError("Config must contain data_selection.views (or legacy data_selection.orientations)")


def generate_base_keys(
    config: dict, galaxy_filter: list[int] | None = None
) -> list[BaseKey]:
    """Generate BaseKeys from config."""
    galaxy_ids = config["data_selection"]["galaxy_ids"]
    if galaxy_filter:
        galaxy_ids = [g for g in galaxy_ids if g in galaxy_filter]
    views = _resolve_views(config)

    return [
        BaseKey(gid, view)
        for gid in galaxy_ids
        for view in views
    ]
