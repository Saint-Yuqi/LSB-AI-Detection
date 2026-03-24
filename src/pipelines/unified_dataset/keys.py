"""
Immutable data keys for the unified dataset pipeline.

BaseKey identifies a unique galaxy+view pair (e.g. galaxy 11 viewed from los00).
VariantKey extends BaseKey with a preprocessing variant name.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BaseKey:
    """Immutable key for a unique galaxy+view (e.g. eo, fo, los00..los23)."""
    galaxy_id: int
    view_id: str

    def __str__(self) -> str:
        return f"{self.galaxy_id:05d}_{self.view_id}"


@dataclass(frozen=True)
class VariantKey:
    """Extends BaseKey with preprocessing variant."""
    base_key: BaseKey
    preprocessing: str

    def __str__(self) -> str:
        return f"{self.base_key}_{self.preprocessing}"
