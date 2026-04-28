#!/usr/bin/env python3
"""Stage A: derive silver labels from GT-vs-prediction matching.

Thin CLI wrapper around ``src.review.silver_labeler``.
Outputs ``silver_labels_{family}.jsonl`` (intermediate format).
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from PIL import Image

from src.review.candidate_matcher import (
    compute_image_exhaustivity,
    match_candidates_to_gt,
)
from src.review.key_adapter import base_key_to_sample_id
from src.review.schemas import TaskFamily
from src.review.silver_labeler import (
    SilverLabel,
    label_satellite_ev_gt,
    label_satellite_mv_authoritative_candidates,
    label_satellite_mv_candidates,
    label_stream_ev_gt,
    load_silver_policy,
    write_silver_labels,
)
from src.pipelines.unified_dataset.keys import BaseKey

log = logging.getLogger(__name__)


def _load_gt(gt_dir: Path, bk: BaseKey) -> np.ndarray | None:
    path = gt_dir / str(bk) / "instance_map_uint8.png"
    if not path.exists():
        return None
    return np.array(Image.open(path))


def _load_instances(gt_dir: Path, bk: BaseKey) -> list[dict] | None:
    path = gt_dir / str(bk) / "instances.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _scope_gt_map(
    gt_map: np.ndarray,
    instances: list[dict],
    type_label: str,
) -> np.ndarray:
    """Return a copy of *gt_map* with IDs outside *type_label* zeroed out."""
    keep_ids = np.array(
        [inst["id"] for inst in instances if inst["type"] == type_label],
        dtype=gt_map.dtype,
    )
    if len(keep_ids) == 0:
        return np.zeros_like(gt_map)
    scoped = gt_map.copy()
    scoped[~np.isin(gt_map, keep_ids)] = 0
    return scoped


def _warn_zero_pixel_instances(
    gt_map: np.ndarray, instances: list[dict], bk: BaseKey,
) -> None:
    """Log a warning for instance IDs declared in instances.json but absent
    from the raster."""
    present_ids = set(np.unique(gt_map).tolist())
    for inst in instances:
        if inst["id"] not in present_ids:
            log.warning(
                "%s: instance id=%d (type=%s) has zero pixels in "
                "instance_map_uint8.png",
                bk, inst["id"], inst["type"],
            )


def _load_preds(pred_dir: Path, bk: BaseKey) -> list[dict] | None:
    path = pred_dir / str(bk) / "sam3_predictions_post.json"
    if not path.exists():
        path = pred_dir / str(bk) / "sam3_predictions_raw.json"
    if not path.exists():
        return None
    with open(path) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw["predictions"]
    from src.utils.coco_utils import decode_rle
    for m in raw:
        if "segmentation" not in m and "rle" in m:
            m["segmentation"] = decode_rle(m["rle"])
        elif "segmentation" in m and isinstance(m["segmentation"], dict):
            m["segmentation"] = decode_rle(m["segmentation"])
    return raw


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=[f.value for f in TaskFamily])
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to silver_policy_v1.yaml")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--pred-dir", type=Path, default=None,
                        help="Predictions dir (required for satellite_mv only)")
    parser.add_argument("--keys-file", type=Path, required=True,
                        help="JSON list of [galaxy_id, view_id] pairs")
    parser.add_argument("--output", "--out", dest="output", type=Path, required=True)
    args = parser.parse_args(argv)

    family = TaskFamily(args.family)
    if family == TaskFamily.SATELLITE_MV and args.pred_dir is None:
        parser.error("--pred-dir is required for satellite_mv")
    policy = load_silver_policy(args.config, family.value)

    with open(args.keys_file) as f:
        keys_raw = json.load(f)
    base_keys = [BaseKey(galaxy_id=k[0], view_id=k[1]) for k in keys_raw]

    all_labels: list[SilverLabel] = []

    for bk in base_keys:
        sample_id = base_key_to_sample_id(bk)
        gt_map = _load_gt(args.gt_dir, bk)
        if gt_map is None:
            continue

        instances = _load_instances(args.gt_dir, bk)
        if instances is None:
            log.warning("%s: instances.json missing, skipping", bk)
            continue

        _warn_zero_pixel_instances(gt_map, instances, bk)

        if family == TaskFamily.SATELLITE_MV:
            preds = _load_preds(args.pred_dir, bk)
            if preds is None:
                preds = []
            sat_preds = [m for m in preds if m.get("type_label") == "satellites"]
            sat_gt = _scope_gt_map(gt_map, instances, "satellites")
            cr = match_candidates_to_gt(sat_preds, sat_gt)
            exh = compute_image_exhaustivity(
                sat_preds,
                sat_gt,
                cr,
                match_iou_thresh=policy.ev_match_iou_thresh,
                confident_recall_thresh=policy.ev_recall_thresh,
                confident_precision_thresh=policy.ev_precision_thresh,
            )
            labels = label_satellite_mv_authoritative_candidates(
                sample_id, cr, exh, policy,
            )
            all_labels.extend(labels)

        elif family == TaskFamily.SATELLITE_EV:
            # GT-driven: no predictions needed
            labels = label_satellite_ev_gt(sample_id, gt_map, instances)
            all_labels.extend(labels)

        elif family == TaskFamily.STREAM_EV:
            # GT-driven: no predictions needed
            labels = label_stream_ev_gt(sample_id, gt_map, instances)
            all_labels.extend(labels)

    write_silver_labels(all_labels, args.output)
    print(f"Wrote {len(all_labels)} silver labels to {args.output}")


if __name__ == "__main__":
    main()
