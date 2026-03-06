"""
Config loading and base key generation for the unified dataset pipeline.
"""
from __future__ import annotations

from pathlib import Path

from .keys import BaseKey


def load_config(path: Path) -> dict:
    """Load YAML config."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def generate_base_keys(
    config: dict, galaxy_filter: list[int] | None = None
) -> list[BaseKey]:
    """Generate BaseKeys from config."""
    galaxy_ids = config["data_selection"]["galaxy_ids"]
    if galaxy_filter:
        galaxy_ids = [g for g in galaxy_ids if g in galaxy_filter]
    orientations = config["data_selection"]["orientations"]

    return [
        BaseKey(gid, ori)
        for gid in galaxy_ids
        for ori in orientations
    ]
