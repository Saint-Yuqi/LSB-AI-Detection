from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocess.satellite_prior_filter import SatellitePriorFilter


def _mask(
    *,
    shape: tuple[int, int] = (1024, 1024),
    box: tuple[int, int, int, int] = (100, 100, 130, 130),
    area_clean: int = 900,
    solidity: float = 0.95,
    aspect_sym_moment: float = 1.2,
) -> dict:
    seg = np.zeros(shape, dtype=np.uint8)
    y0, x0, y1, x1 = box
    seg[y0:y1, x0:x1] = 1
    return {
        "segmentation": seg.astype(bool),
        "type_label": "satellites",
        "area_clean": area_clean,
        "solidity": solidity,
        "aspect_sym_moment": aspect_sym_moment,
    }


def test_hard_center_candidate_is_rejected_by_prior_filter() -> None:
    flt = SatellitePriorFilter(
        {"area_min": 30, "solidity_min": 0.83, "aspect_sym_max": 1.75, "hard_center_radius_frac": 0.03}
    )
    m = _mask(box=(497, 497, 527, 527))
    kept, rejected, ambiguous = flt.filter([m])
    assert kept == []
    assert ambiguous == []
    assert len(rejected) == 1
    assert rejected[0]["reject_reason"] == "prior_hard_center"


def test_non_center_candidate_still_uses_morphology_rules() -> None:
    flt = SatellitePriorFilter(
        {"area_min": 30, "solidity_min": 0.83, "aspect_sym_max": 1.75, "hard_center_radius_frac": 0.03}
    )
    m = _mask(solidity=0.5, box=(100, 100, 130, 130))
    kept, rejected, _ambiguous = flt.filter([m])
    assert kept == []
    assert len(rejected) == 1
    assert rejected[0]["reject_reason"] == "prior_solidity"
