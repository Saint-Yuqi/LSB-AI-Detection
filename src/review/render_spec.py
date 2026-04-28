"""
Versioned render-specification registry.

Each ``RenderSpec`` is a frozen, versioned bundle of visual parameters
(crop size, contour colour, bbox style, image order, …).  Changing any
field requires a new ``spec_id``; the ``spec_id`` enters the revision
hash so that render changes force new revisions.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RenderSpec:
    """Immutable visual specification for one task family."""
    spec_id: str
    input_variant: str
    crop_size: int
    contour_color: tuple[int, int, int]
    contour_alpha: float
    contour_thickness: int
    bbox_color: tuple[int, int, int]
    bbox_thickness: int
    image_order: tuple[str, ...]
    fragment_hints_enabled: bool


def _parse_spec(raw: dict[str, Any]) -> RenderSpec:
    return RenderSpec(
        spec_id=raw["spec_id"],
        input_variant=raw["input_variant"],
        crop_size=int(raw["crop_size"]),
        contour_color=tuple(raw["contour_color"]),
        contour_alpha=float(raw["contour_alpha"]),
        contour_thickness=int(raw["contour_thickness"]),
        bbox_color=tuple(raw["bbox_color"]),
        bbox_thickness=int(raw["bbox_thickness"]),
        image_order=tuple(raw["image_order"]),
        fragment_hints_enabled=bool(raw.get("fragment_hints_enabled", False)),
    )


class RenderSpecRegistry:
    """Load and look up ``RenderSpec`` objects from a YAML config."""

    def __init__(self, specs: dict[str, RenderSpec]) -> None:
        self._specs = specs

    @classmethod
    def from_yaml(cls, path: Path | str) -> RenderSpecRegistry:
        path = Path(path)
        with open(path) as fh:
            raw = yaml.safe_load(fh)
        specs: dict[str, RenderSpec] = {}
        for entry in raw.get("render_specs", []):
            rs = _parse_spec(entry)
            specs[rs.spec_id] = rs
        return cls(specs)

    def get(self, spec_id: str) -> RenderSpec:
        try:
            return self._specs[spec_id]
        except KeyError:
            raise KeyError(
                f"Unknown render_spec_id {spec_id!r}; "
                f"available: {sorted(self._specs)}"
            )

    def __contains__(self, spec_id: str) -> bool:
        return spec_id in self._specs
