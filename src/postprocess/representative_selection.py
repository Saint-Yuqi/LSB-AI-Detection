"""
Representative selection – pick one best mask per group.

Usage:
    from src.postprocess.representative_selection import select_representatives, load_area_target
    area_target = load_area_target(stats_json)
    reps, dups = select_representatives(masks, cfg={"area_target": area_target})

Algorithm:
    1. Group masks by group_id (must be set by candidate_grouping)
    2. Score each mask: stability + iou - aspect_penalty - area_distance
    3. Keep highest-scoring mask per group as representative
    4. Mark others as duplicates with reject_reason="duplicate"

Note: Solidity is NOT used in scoring (computed later for prior filter).
"""
from __future__ import annotations

import json
import warnings
from collections import defaultdict
from math import log
from pathlib import Path
from typing import Any

_DEFAULT_AREA_TARGET = 144.0


def load_area_target(
    stats_json: Path | str,
    key: str = "satellites_global",
) -> float:
    """Load quantiles.area.p50 from mask_stats_summary.json with 3-tier guards.

    Tier 1: file existence. Tier 2: JSON parse. Tier 3: key/field presence.
    Each tier emits warnings.warn and falls back to _DEFAULT_AREA_TARGET.
    """
    stats_json = Path(stats_json)

    if not stats_json.exists():
        warnings.warn(
            f"Stats not found: {stats_json}; using default area_target={_DEFAULT_AREA_TARGET}. "
            "Run: python scripts/analysis/analyze_mask_stats.py",
            stacklevel=2,
        )
        return _DEFAULT_AREA_TARGET

    try:
        with open(stats_json) as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        warnings.warn(f"Failed to parse {stats_json}: {e}; using default area_target", stacklevel=2)
        return _DEFAULT_AREA_TARGET

    section = data.get(key)
    if section is None:
        warnings.warn(f"Key '{key}' missing in {stats_json}; using default area_target", stacklevel=2)
        return _DEFAULT_AREA_TARGET

    p50 = section.get("quantiles", {}).get("area", {}).get("p50")
    if p50 is None:
        warnings.warn(
            f"'quantiles.area.p50' missing under '{key}' in {stats_json}; "
            f"using default area_target={_DEFAULT_AREA_TARGET}",
            stacklevel=2,
        )
        return _DEFAULT_AREA_TARGET

    return float(p50)


def compute_rep_score(m: dict[str, Any], cfg: dict[str, Any]) -> float:
    """
    Compute representative score (higher = better).

    Formula (no solidity):
        score = w_stab * stability_score
              + w_iou * predicted_iou
              - w_aspect * |aspect_sym_moment - 1|
              - w_area * |log(area_clean) - log(area_target)|

    aspect_sym_moment: covariance-eigenvalue axis ratio (rotation-invariant).
    Reads aspect_sym_moment first, falls back to aspect_sym (legacy alias).

    Defaults: w_stab=1.0, w_iou=1.0, w_aspect=0.3, w_area=0.2
    """
    w_stab = cfg.get("w_stab", 1.0)
    w_iou = cfg.get("w_iou", 1.0)
    w_aspect = cfg.get("w_aspect", 0.3)
    w_area = cfg.get("w_area", 0.2)
    area_target = cfg.get("area_target", _DEFAULT_AREA_TARGET)

    # Safe log: area_clean >= 1
    area = max(m.get("area_clean", 1), 1)
    target = max(area_target, 1)

    # Prefer aspect_sym_moment (new), fall back to aspect_sym (legacy alias)
    aspect = m.get("aspect_sym_moment") or m.get("aspect_sym", 1.0)

    score = (
        w_stab * m.get("stability_score", 0.0)
        + w_iou * m.get("predicted_iou", 0.0)
        - w_aspect * abs(aspect - 1.0)
        - w_area * abs(log(area) - log(target))
    )
    return score


def select_representatives(
    masks: list[dict[str, Any]],
    cfg: dict[str, Any] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Select one representative per group; mark others as duplicates.

    Args:
        masks: list of mask dicts with 'group_id' (from candidate_grouping).
        cfg: scoring config with w_stab, w_iou, w_aspect, w_area, area_target.

    Returns:
        (representatives, duplicates)
        - representatives: best mask per group, with 'rep_score' added.
        - duplicates: other masks, with 'reject_reason'="duplicate" and 'group_id' preserved.
    """
    if cfg is None:
        cfg = {}

    if not masks:
        return [], []

    # Group by group_id
    groups: dict[int, list[dict]] = defaultdict(list)
    for m in masks:
        gid = m.get("group_id", 0)
        groups[gid].append(m)

    representatives = []
    duplicates = []

    for gid, group_masks in groups.items():
        if len(group_masks) == 1:
            m = group_masks[0]
            m["rep_score"] = compute_rep_score(m, cfg)
            representatives.append(m)
        else:
            # Score all masks in group
            scored = [(m, compute_rep_score(m, cfg)) for m in group_masks]
            scored.sort(key=lambda x: x[1], reverse=True)

            # Best one is representative
            best, best_score = scored[0]
            best["rep_score"] = best_score
            representatives.append(best)

            # Rest are duplicates
            for m, _ in scored[1:]:
                m["reject_reason"] = "duplicate"
                # group_id already present
                duplicates.append(m)

    return representatives, duplicates
