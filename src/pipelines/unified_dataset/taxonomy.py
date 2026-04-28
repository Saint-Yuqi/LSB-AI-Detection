"""
Canonical 3-class taxonomy for the unified dataset (tidal_v1).

Replaces the legacy 2-class system (`streams`, `satellites`) with three classes:
- `tidal_features`: FIREbox SB31.5 FITS-derived (DR1) or SAM3 stream-prompt-derived (PNbody)
- `satellites`:     SAM3 satellite-galaxy-prompt-derived; passed through prior filter
- `inner_galaxy`:   satellite candidates relabeled by the prior filter when
                    `dist_frac < hard_center_radius_frac` (new path only)

Legacy values (`"streams"`, `"stellar stream"`, `"satellite galaxy"`) are mapped
through `normalize_type_label`. Unknown values raise ValueError so we fail
closed at category resolution time.
"""
from __future__ import annotations

from typing import Any

TIDAL_FEATURES = "tidal_features"
SATELLITES = "satellites"
INNER_GALAXY = "inner_galaxy"

CATEGORIES: list[dict[str, Any]] = [
    {"id": 1, "name": "tidal features", "supercategory": "lsb"},
    {"id": 2, "name": "satellite galaxies", "supercategory": "lsb"},
    {"id": 3, "name": "inner galaxy", "supercategory": "lsb"},
]

CATEGORY_ID_BY_TYPE: dict[str, int] = {
    TIDAL_FEATURES: 1,
    SATELLITES: 2,
    INNER_GALAXY: 3,
}

_LABEL_ALIASES: dict[str, str] = {
    "streams": TIDAL_FEATURES,
    "stellar stream": TIDAL_FEATURES,
    "stellar streams": TIDAL_FEATURES,
    "satellite galaxy": SATELLITES,
    "satellite galaxies": SATELLITES,
    TIDAL_FEATURES: TIDAL_FEATURES,
    SATELLITES: SATELLITES,
    INNER_GALAXY: INNER_GALAXY,
}


def normalize_type_label(label: str) -> str:
    """Map a label to the canonical taxonomy value.

    Idempotent: a canonical value is returned unchanged. Aliases (e.g.
    ``"streams"``, ``"stellar stream"``) are mapped to the canonical class.
    Unknown values raise ``ValueError`` — there is no silent passthrough.
    """
    if label in _LABEL_ALIASES:
        return _LABEL_ALIASES[label]
    raise ValueError(
        f"Unknown type_label {label!r}; expected one of "
        f"{sorted(set(_LABEL_ALIASES.values()))} or a known alias "
        f"({sorted(k for k in _LABEL_ALIASES if k not in _LABEL_ALIASES.values())})"
    )
