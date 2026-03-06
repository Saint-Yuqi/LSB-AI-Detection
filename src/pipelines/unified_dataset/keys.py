"""
Immutable data keys for the unified dataset pipeline.

BaseKey identifies a unique galaxy+orientation pair.
VariantKey extends BaseKey with a preprocessing variant name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseKey:
    """Immutable key for a unique galaxy+orientation."""
    galaxy_id: int
    orientation: str

    def __str__(self) -> str:
        return f"{self.galaxy_id:05d}_{self.orientation}"


@dataclass(frozen=True)
class VariantKey:
    """Extends BaseKey with preprocessing variant."""
    base_key: BaseKey
    preprocessing: str

    def __str__(self) -> str:
        return f"{self.base_key}_{self.preprocessing}"
