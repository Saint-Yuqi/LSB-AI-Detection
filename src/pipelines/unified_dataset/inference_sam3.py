"""
Phase 3 – SAM3 engine: text-prompt -> type-aware filter -> evaluate / pseudo_label.

run_mode='evaluate'      (default) – requires GT streams_instance_map.npy,
                          produces sam3_predictions_{raw,post}.json + sam3_eval_overlay.png
run_mode='pseudo_label'  – no GT required; rasterises post-filtered masks into
                          instance_map_uint8.png + instances.json and writes a
                          pred-only QA overlay (sam3_pseudo_label_overlay.png).

Output dir: gt_canonical/current/{BaseKey}/
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from .keys import BaseKey
from .paths import PathResolver
from .artifacts import save_predictions_json, save_pseudo_gt
from src.visualization.overlay import save_evaluation_overlay, save_pseudo_label_overlay

_PSEUDO_LABEL_ARTIFACTS = [
    "instance_map_uint8.png",
    "instances.json",
    "manifest.json",
    "sam3_predictions_raw.json",
    "sam3_predictions_post.json",
    "sam3_pseudo_label_overlay.png",
]


def _pseudo_label_complete(gt_dir: Path) -> bool:
    return all((gt_dir / f).exists() for f in _PSEUDO_LABEL_ARTIFACTS)


def _evaluate_complete(gt_dir: Path) -> bool:
    return (gt_dir / "sam3_predictions_raw.json").exists() and (gt_dir / "sam3_eval_overlay.png").exists()


def run_inference_sam3(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """SAM3 text-prompt -> type-aware filter -> evaluate or pseudo_label."""
    inf_cfg = config.get("inference_phase", {})
    input_variant = inf_cfg.get(
        "input_image_variant",
        config.get("satellites", {}).get("input_image_variant", "linear_magnitude"),
    )
    target_size = tuple(config["processing"]["target_size"])
    run_mode = inf_cfg.get("run_mode", "evaluate")
    resolver = PathResolver(config)
    sam3_cfg = inf_cfg.get("sam3", {})
    prompts = sam3_cfg.get("prompts", [])
    H_work, W_work = target_size

    from src.inference.sam3_prompt_runner import SAM3PromptRunner
    from src.postprocess.streams_sanity_filter import StreamsSanityFilter, load_streams_cfg
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    from src.postprocess.core_exclusion_filter import CoreExclusionFilter
    from src.analysis.mask_metrics import append_metrics_to_masks

    runner = SAM3PromptRunner(
        checkpoint=sam3_cfg.get("checkpoint"),
        bpe_path=sam3_cfg.get("bpe_path"),
        confidence_threshold=sam3_cfg.get("confidence_threshold", 0.55),
        resolution=sam3_cfg.get("resolution", 1008),
        target_size=target_size,
    )

    sanity_cfg = sam3_cfg.get("streams_sanity", {})
    streams_stats_json = Path(sanity_cfg.get("stats_json", "outputs/mask_stats/mask_stats_summary.json"))
    if not streams_stats_json.is_absolute():
        streams_stats_json = PROJECT_ROOT / streams_stats_json
    streams_gt_cfg = load_streams_cfg(streams_stats_json)
    streams_flt = StreamsSanityFilter(
        min_area=sanity_cfg.get("min_area", streams_gt_cfg["min_area"]),
        max_area_frac=sanity_cfg.get("max_area_frac", 0.5),
        edge_touch_frac=sanity_cfg.get("edge_touch_frac", streams_gt_cfg["edge_touch_frac"]),
        max_area_px=sanity_cfg.get("max_area_px", streams_gt_cfg["max_area_px"]),
    )
    sat_cfg = config.get("satellites", {})
    prior_cfg_entry = sat_cfg.get("prior", {})
    core_cfg = sat_cfg.get("core_exclusion", {})
    stats_json = Path(prior_cfg_entry.get("stats_json", "outputs/mask_stats/mask_stats_summary.json"))
    if not stats_json.is_absolute():
        stats_json = PROJECT_ROOT / stats_json
    prior_cfg = load_filter_cfg(stats_json)
    prior_cfg["ambiguous_factor"] = prior_cfg_entry.get("ambiguous_factor", 0.25)
    prior_cfg["core_radius_frac"] = core_cfg.get("radius_frac", 0.08)
    sat_prior_flt = SatellitePriorFilter(prior_cfg)
    sat_core_flt = CoreExclusionFilter(radius_frac=core_cfg.get("radius_frac", 0.08))

    is_pseudo = run_mode == "pseudo_label"
    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_render": 0, "skipped_no_streams": 0}

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)

        should_rebuild = force or (force_variants is not None)
        already_done = _pseudo_label_complete(gt_dir) if is_pseudo else _evaluate_complete(gt_dir)

        if already_done:
            if should_rebuild:
                for art in (_PSEUDO_LABEL_ARTIFACTS if is_pseudo else ["sam3_predictions_raw.json", "sam3_predictions_post.json", "sam3_eval_overlay.png"]):
                    (gt_dir / art).unlink(missing_ok=True)
                logger.info(f"Force rebuild SAM3 {run_mode}: {key}")
            else:
                stats["skipped_exists"] += 1
                continue

        render_path = resolver.get_render_dir(input_variant, key) / "0000.png"
        if not render_path.exists():
            logger.warning(f"Render not found: {render_path}")
            stats["skipped_no_render"] += 1
            continue

        streams_map = None
        if not is_pseudo:
            streams_npy = gt_dir / "streams_instance_map.npy"
            if not streams_npy.exists():
                logger.warning(f"Streams map not found: {streams_npy}")
                stats["skipped_no_streams"] += 1
                continue
            streams_map = np.load(streams_npy)

        gt_dir.mkdir(parents=True, exist_ok=True)

        image_pil = Image.open(render_path).convert("RGB")
        image_np = np.array(image_pil)

        masks, time_ms = runner.run(image_pil, prompts)
        logger.info(f"{key}: SAM3 returned {len(masks)} masks in {time_ms:.0f}ms")

        if masks:
            append_metrics_to_masks(masks, H_work, W_work, compute_hull=True)

        save_predictions_json(gt_dir / "sam3_predictions_raw.json", masks, H_work, W_work,
                              engine="sam3", layer="raw")

        stream_masks = [m for m in masks if m.get("type_label") == "streams"]
        sat_masks = [m for m in masks if m.get("type_label") == "satellites"]

        kept_streams, rej_streams = streams_flt.filter(stream_masks, H_work, W_work)

        if sat_masks:
            kept_sat_prior, rej_sat_prior, ambig_sat = sat_prior_flt.filter(sat_masks)
            kept_sat_final, core_hits, _ = sat_core_flt.filter(kept_sat_prior, H_work, W_work)
        else:
            kept_sat_final = []

        post_masks = kept_streams + kept_sat_final

        save_predictions_json(gt_dir / "sam3_predictions_post.json", post_masks, H_work, W_work,
                              engine="sam3", layer="post")

        if is_pseudo:
            save_pseudo_gt(gt_dir, post_masks, H_work, W_work)
            save_pseudo_label_overlay(gt_dir / "sam3_pseudo_label_overlay.png", image_np, post_masks)
        else:
            save_evaluation_overlay(gt_dir / "sam3_eval_overlay.png", image_np, streams_map, post_masks)

        manifest_path = gt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        manifest.update({
            "engine": "sam3",
            "run_mode": run_mode,
            "gt_source": "none" if is_pseudo else "streams_instance_map",
            "sam3_n_raw": len(masks),
            "sam3_n_post": len(post_masks),
            "sam3_n_streams_kept": len(kept_streams),
            "sam3_n_satellites_kept": len(kept_sat_final),
            "sam3_inference_time_ms": round(time_ms, 2),
            "sam3_updated_at": datetime.now().isoformat(),
        })
        manifest_path.write_text(json.dumps(manifest, indent=2))

        stats["processed"] += 1
        if stats["processed"] % 10 == 0:
            logger.info(f"Progress: {stats['processed']}/{len(base_keys)}")

    logger.info(f"SAM3 {run_mode}: {stats['processed']} processed, {stats['skipped_exists']} skipped")
