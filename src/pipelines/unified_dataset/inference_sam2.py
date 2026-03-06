"""
Phase 3 – SAM2 engine: AutoMask -> filter -> merge -> cache.

Output: gt_canonical/current/{BaseKey}/instance_map_uint8.png + satellites_cache.npz
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from .keys import BaseKey
from .paths import PathResolver
from .fs_utils import sha1_file
from .artifacts import save_satellites_cache, merge_instances
from src.visualization.overlay import save_instance_overlay


def _sort_masks(masks: list[dict], policy: list[str]) -> list[dict]:
    """Sort masks deterministically by policy."""
    def sort_key(m):
        keys = []
        for p in policy:
            if p == "area_desc":
                keys.append(-m.get("area", 0))
            elif p == "area_asc":
                keys.append(m.get("area", 0))
            elif p == "centroid_x_asc":
                keys.append(m.get("centroid_x", 0))
            elif p == "centroid_y_asc":
                keys.append(m.get("centroid_y", 0))
        return tuple(keys)
    return sorted(masks, key=sort_key)


def run_inference_sam2(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """SAM2 AutoMask -> filter -> cache -> merge."""
    sat_cfg = config.get("satellites", {})
    if not sat_cfg.get("enabled", True):
        logger.info("Satellites disabled in config, skipping.")
        return

    inf_cfg = config.get("inference_phase", {})
    input_variant = inf_cfg.get(
        "input_image_variant",
        sat_cfg.get("input_image_variant", "linear_magnitude"),
    )
    target_size = tuple(config["processing"]["target_size"])
    resolver = PathResolver(config)

    from src.inference.sam2_automask_runner import AutoMaskRunner, DEFAULT_CHECKPOINT, DEFAULT_MODEL_CFG
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    from src.postprocess.core_exclusion_filter import CoreExclusionFilter
    from src.postprocess.candidate_grouping import group_by_centroid
    from src.postprocess.representative_selection import select_representatives, load_area_target
    from src.analysis.mask_metrics import append_metrics_to_masks

    checkpoint = sat_cfg.get("checkpoint", DEFAULT_CHECKPOINT)
    model_cfg = sat_cfg.get("model_cfg", DEFAULT_MODEL_CFG)
    runner = AutoMaskRunner(checkpoint=checkpoint, model_cfg=model_cfg)

    raw_gen_cfg = sat_cfg.get("generator", {})
    gen_cfg: dict[str, Any] = {}
    int_keys = {"points_per_side", "points_per_batch", "min_mask_region_area", "crop_n_layers", "crop_n_points_downscale_factor"}
    float_keys = {"pred_iou_thresh", "stability_score_thresh", "stability_score_offset", "box_nms_thresh", "crop_nms_thresh"}
    for k, v in raw_gen_cfg.items():
        if k in int_keys:
            gen_cfg[k] = int(v)
        elif k in float_keys:
            gen_cfg[k] = float(v)
        else:
            gen_cfg[k] = v
    grouping_cfg = sat_cfg.get("grouping", {})
    prior_cfg_entry = sat_cfg.get("prior", {})
    core_cfg = sat_cfg.get("core_exclusion", {})
    sort_policy = sat_cfg.get("satellite_sort_policy", ["area_desc"])
    overlap_policy = sat_cfg.get("overlap_policy", "keep_streams")

    stats_json = Path(prior_cfg_entry.get("stats_json", "outputs/mask_stats/mask_stats_summary.json"))
    if not stats_json.is_absolute():
        stats_json = PROJECT_ROOT / stats_json
    prior_cfg = load_filter_cfg(stats_json)
    prior_cfg["ambiguous_factor"] = prior_cfg_entry.get("ambiguous_factor", 0.25)
    prior_cfg["core_radius_frac"] = core_cfg.get("radius_frac", 0.08)

    area_target = load_area_target(stats_json)
    selection_cfg = sat_cfg.get("selection", {})
    selection_cfg.setdefault("area_target", area_target)
    logger.info(f"area_target={area_target:.1f} from {stats_json}")

    prior_flt = SatellitePriorFilter(prior_cfg)
    core_flt = CoreExclusionFilter(radius_frac=core_cfg.get("radius_frac", 0.08))

    first_key = base_keys[0] if base_keys else None
    if first_key:
        warmup_path = resolver.get_render_dir(input_variant, first_key) / "0000.png"
        if warmup_path.exists():
            warmup_img = np.array(Image.open(warmup_path).convert("RGB"))
            runner.warmup(warmup_img, n=2)

    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_render": 0, "skipped_no_streams": 0}

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)
        cache_path = gt_dir / "satellites_cache.npz"
        final_map_path = gt_dir / "instance_map_uint8.png"

        should_rebuild = force or (force_variants is not None)
        if final_map_path.exists() and cache_path.exists():
            if should_rebuild:
                final_map_path.unlink(missing_ok=True)
                cache_path.unlink(missing_ok=True)
                (gt_dir / "instances.json").unlink(missing_ok=True)
                (gt_dir / "id_map.json").unlink(missing_ok=True)
                (gt_dir / "overlay.png").unlink(missing_ok=True)
                logger.info(f"Force rebuild satellites: {key}")
            else:
                stats["skipped_exists"] += 1
                continue

        render_path = resolver.get_render_dir(input_variant, key) / "0000.png"
        if not render_path.exists():
            logger.warning(f"Render not found: {render_path}")
            stats["skipped_no_render"] += 1
            continue

        streams_npy = gt_dir / "streams_instance_map.npy"
        if not streams_npy.exists():
            logger.warning(f"Streams map not found: {streams_npy}")
            stats["skipped_no_streams"] += 1
            continue

        streams_map = np.load(streams_npy)
        max_stream_id = int(streams_map.max())

        image = np.array(Image.open(render_path).convert("RGB"))
        H, W = image.shape[:2]

        masks, time_ms = runner.run(image, gen_cfg)
        for m in (masks or []):
            m["type_label"] = "satellites"

        if masks:
            append_metrics_to_masks(masks, H, W, compute_hull=False)
            group_by_centroid(masks, dist_px=grouping_cfg.get("centroid_dist_px", 10.0))
            reps, dups = select_representatives(masks, selection_cfg)
            append_metrics_to_masks(reps, H, W, compute_hull=True)
            kept_prior, rej_prior, ambig = prior_flt.filter(reps)
            kept_final, core_hits, _ = core_flt.filter(kept_prior, H, W)
        else:
            kept_final, ambig, dups, rej_prior, core_hits = [], [], [], [], []

        kept_final = _sort_masks(kept_final, sort_policy)

        save_satellites_cache(
            cache_path, kept=kept_final, ambiguous=ambig,
            dup_rejected=dups, prior_rejected=rej_prior,
            core_rejected=core_hits,
            input_image_sha1=sha1_file(render_path), H=H, W=W,
        )

        instance_map, instances_list, id_map, overlap_stats = merge_instances(
            streams_map, kept_final, max_stream_id, overlap_policy,
        )

        max_final_id = int(instance_map.max())
        assert 0 <= max_final_id <= 255, f"max_id={max_final_id} exceeds uint8"
        assert instance_map.dtype == np.uint8, f"dtype={instance_map.dtype}"

        cv2.imwrite(str(final_map_path), instance_map)
        (gt_dir / "instances.json").write_text(json.dumps(instances_list, indent=2))
        (gt_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))

        manifest_path = gt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update({
            "engine": "sam2",
            "automask_config_name": sat_cfg.get("automask_config_name", "unknown"),
            "input_image_variant": input_variant,
            "satellite_sort_policy": sort_policy,
            "overlap_policy": overlap_policy,
            "n_raw_masks": len(masks) if masks else 0,
            "n_satellites_kept": len(kept_final),
            "n_ambiguous": len(ambig),
            "max_stream_id": max_stream_id,
            "max_final_id": max_final_id,
            "overlap_px": overlap_stats.get("overlap_px", 0),
            "overlap_rate": overlap_stats.get("overlap_rate", 0.0),
            "inference_time_ms": round(time_ms, 2),
            "updated_at": datetime.now().isoformat(),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))

        save_instance_overlay(gt_dir / "overlay.png", image, instance_map)

        stats["processed"] += 1
        if stats["processed"] % 10 == 0:
            logger.info(f"Progress: {stats['processed']}/{len(base_keys)}")

    logger.info(f"Processed: {stats['processed']}, Skipped (exists): {stats['skipped_exists']}")
