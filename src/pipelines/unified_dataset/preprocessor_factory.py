"""
Single source of truth for preprocessor instantiation.

Maps variant name strings to src.data.preprocessing classes.
"""
from __future__ import annotations

from src.data.preprocessing import (
    LSBPreprocessor,
    LinearMagnitudePreprocessor,
    MultiExposurePreprocessor,
)


def create_preprocessor(name: str, params: dict, target_size: tuple[int, int]):
    """Factory for preprocessors matching src/data/preprocessing.py."""
    if name == "asinh_stretch":
        return LSBPreprocessor(
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 50.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            target_size=target_size,
        )
    elif name == "linear_magnitude":
        return LinearMagnitudePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            target_size=target_size,
        )
    elif name == "multi_exposure":
        return MultiExposurePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 300.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            gamma=params.get("gamma", 0.5),
            b_mode=params.get("b_mode", "gamma"),
            target_size=target_size,
        )
    else:
        raise ValueError(f"Unknown preprocessor: {name}")
