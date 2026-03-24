#!/usr/bin/env python3
"""
SAM3 Folder-Based Evaluation.

Discovers render PNG / GT npy pairs, runs SAM3 inference, computes
pixel-level (Dice, Precision, Recall, capped_hausdorff95) and
instance-level (optimal 1:1 matched IoU, instance recall) metrics.
Outputs both raw and post-filtered results in a single run.

Usage:
    python scripts/eval/evaluate_sam3.py --config configs/eval_sam3.yaml
    python scripts/eval/evaluate_sam3.py --config configs/eval_sam3.yaml \\
        --render-dir data/02_processed/renders/current/asinh_stretch \\
        --gt-dir data/02_processed/gt_canonical/current \\
        --max-samples 5 --save-overlays

Args:
    --config:      YAML config (checkpoint, prompts, filter thresholds)
    --render-dir:  Root of rendered PNGs ({base_key}/0000.png)
    --gt-dir:      Root of GT masks ({base_key}/streams_instance_map.npy)
    --output-dir:  JSON output directory
    --max-samples: Limit number of images (for debugging)
    --per-galaxy:  Aggregate by galaxy_id (merge eo+fo)
    --snr-tag:     Metadata label for SNR tier (e.g. "snr10")
    --save-overlays: Write QA overlay PNGs

Env:
    CUDA, bf16 Ampere+. SAM3 package must be importable.
    conda run -n lsb --no-capture-output python scripts/eval/evaluate_model.py --help
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.sam3_eval import (
    discover_pairs,
    run_and_evaluate,
    aggregate_results,
    save_results,
    save_eval_overlay,
)
from src.utils.logger import setup_logger


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", "-c", default="configs/eval_sam3.yaml",
                    help="YAML config file")
    ap.add_argument("--render-dir", default=None,
                    help="Override render directory (overrides config)")
    ap.add_argument("--gt-dir", default=None,
                    help="Override GT directory (overrides config)")
    ap.add_argument("--output-dir", default=None,
                    help="Override output directory (overrides config)")
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Limit images for debugging")
    ap.add_argument("--per-galaxy", action="store_true",
                    help="Aggregate by galaxy_id (merge eo+fo)")
    ap.add_argument("--snr-tag", default=None,
                    help="Metadata SNR label (e.g. snr10)")
    ap.add_argument("--save-overlays", action="store_true",
                    help="Write QA overlay PNGs")
    ap.add_argument("--checkpoint", default=None,
                    help="Override SAM3 checkpoint path (overrides config)")
    args = ap.parse_args()

    # --- Load config ---
    cfg_path = PROJECT_ROOT / args.config
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    render_dir = Path(args.render_dir or cfg["paths"]["render_dir"])
    gt_dir = Path(args.gt_dir or cfg["paths"]["gt_dir"])
    output_dir = Path(args.output_dir or cfg["paths"]["output_dir"])
    prompts = cfg.get("prompts", [{"text": "stellar stream", "type_label": "streams"}])
    match_thresh = cfg.get("match_iou_thresh", 0.5)
    target_size = tuple(cfg.get("target_size", [1024, 1024]))
    H_work, W_work = target_size

    log_dir = PROJECT_ROOT / "logs"
    logger = setup_logger("evaluate_model", log_dir)
    logger.info(f"Config: {cfg_path}")
    logger.info(f"Render dir: {render_dir}")
    logger.info(f"GT dir:     {gt_dir}")

    # --- Discover pairs ---
    pairs = discover_pairs(render_dir, gt_dir, max_samples=args.max_samples)
    logger.info(f"Discovered {len(pairs)} render/GT pairs")
    if not pairs:
        logger.error("No pairs found. Check --render-dir and --gt-dir.")
        sys.exit(1)

    # --- Load SAM3 runner ---
    # Processor gets the minimum across all per-prompt thresholds as floor
    per_prompt_thresholds = [p.get("confidence_threshold", 0.55) for p in prompts]
    floor_threshold = min(per_prompt_thresholds) if per_prompt_thresholds else 0.55

    ckpt = args.checkpoint or cfg["sam3"]["checkpoint"]
    from src.inference.sam3_prompt_runner import SAM3PromptRunner
    runner = SAM3PromptRunner(
        checkpoint=ckpt,
        bpe_path=cfg["sam3"]["bpe_path"],
        confidence_threshold=floor_threshold,
        resolution=cfg["sam3"].get("resolution", 1008),
        target_size=target_size,
    )
    logger.info(f"Checkpoint: {ckpt}")

    # --- Build post-filter (GT-driven via load_streams_cfg) ---
    from src.postprocess.streams_sanity_filter import StreamsSanityFilter, load_streams_cfg
    filt_cfg = cfg.get("post_filter", {})
    stats_json_raw = filt_cfg.get("stats_json", "outputs/mask_stats/mask_stats_summary.json")
    stats_json_path = Path(stats_json_raw)
    if not stats_json_path.is_absolute():
        stats_json_path = PROJECT_ROOT / stats_json_path
    streams_cfg = load_streams_cfg(stats_json_path)
    # Explicit config overrides take precedence over GT-derived values
    filter_min_area = filt_cfg.get("min_area", streams_cfg["min_area"])
    filter_max_area_px = filt_cfg.get("max_area_px", streams_cfg["max_area_px"])
    filter_edge_touch_frac = filt_cfg.get("edge_touch_frac", streams_cfg["edge_touch_frac"])
    streams_filter = StreamsSanityFilter(
        min_area=filter_min_area,
        max_area_frac=filt_cfg.get("max_area_frac", 0.5),
        edge_touch_frac=filter_edge_touch_frac,
        max_area_px=filter_max_area_px,
    )

    # --- Evaluate ---
    per_image: list[dict] = []
    for i, pair in enumerate(pairs):
        logger.info(f"[{i+1}/{len(pairs)}] {pair['base_key']} (gt_mode={pair['gt_mode']})")
        result = run_and_evaluate(
            runner, pair, prompts, H_work, W_work,
            match_iou_thresh=match_thresh,
            streams_filter=streams_filter,
        )
        per_image.append(result)

        # Per-type logging
        for tkey in ("streams", "satellites", "combined"):
            raw = result.get(tkey, {}).get("raw", {})
            if raw:
                logger.info(
                    f"  {tkey:>10s}.raw: dice={_fmt(raw.get('dice'))} "
                    f"matched_iou={_fmt(raw.get('matched_iou'))} "
                    f"inst_recall={_fmt(raw.get('instance_recall'))} "
                    f"n_pred={raw.get('num_pred', 0)}"
                )

        if args.save_overlays:
            from src.evaluation.sam3_eval import _load_gt_by_type
            vis_dir = output_dir / "overlays"
            vis_dir.mkdir(parents=True, exist_ok=True)
            gt_streams, gt_satellites, _ = _load_gt_by_type(pair)
            save_eval_overlay(
                vis_dir / f"{pair['base_key']}_overlay.png",
                pair["render_path"],
                gt_streams,
                gt_satellites,
                result.get("_stream_masks_raw", []),
                result.get("_satellite_masks_raw", []),
            )

    # --- Aggregate ---
    summary_overall = aggregate_results(per_image, group_by="overall")
    summary: dict = {"overall": summary_overall}
    if args.per_galaxy:
        summary["per_galaxy"] = aggregate_results(per_image, group_by="galaxy")

    # --- Build config for reproducibility ---
    config_record = {
        "checkpoint": ckpt,
        "bpe_path": cfg["sam3"]["bpe_path"],
        "render_dir": str(render_dir),
        "gt_dir": str(gt_dir),
        "prompts": prompts,
        "match_iou_thresh": match_thresh,
        "target_size": list(target_size),
        "post_filter": {
            "min_area": filter_min_area,
            "max_area_frac": filt_cfg.get("max_area_frac", 0.5),
            "max_area_px": filter_max_area_px,
            "edge_touch_frac": filter_edge_touch_frac,
            "stats_json": str(stats_json_path),
        },
        "snr_tag": args.snr_tag,
        "capped_hausdorff95_empty_penalty": "diagonal",
    }

    out_path = save_results(output_dir, config_record, summary, per_image)
    logger.info(f"Results saved to {out_path}")

    # --- Print summary ---
    logger.info("=" * 60)
    logger.info("SUMMARY (macro-averaged, null-skipping)")
    logger.info("=" * 60)
    for type_key in ("streams", "satellites", "combined"):
        type_data = summary_overall.get(type_key, {})
        if not type_data:
            continue
        for layer_name in ("RAW", "POST"):
            s = type_data.get(layer_name.lower(), {})
            if not s:
                continue
            logger.info(f"  {type_key.upper()} / {layer_name}:")
            logger.info(f"    Dice:      {_fmt(s.get('macro_mean_dice'))} ± {_fmt(s.get('macro_std_dice'))}")
            logger.info(f"    Precision: {_fmt(s.get('macro_mean_precision'))} ± {_fmt(s.get('macro_std_precision'))}")
            logger.info(f"    Recall:    {_fmt(s.get('macro_mean_recall_pixel'))} ± {_fmt(s.get('macro_std_recall_pixel'))}")
            logger.info(f"    HD95:      {_fmt(s.get('macro_mean_capped_hausdorff95'))} ± {_fmt(s.get('macro_std_capped_hausdorff95'))}")
            logger.info(f"    Match IoU: {_fmt(s.get('macro_mean_matched_iou'))} ± {_fmt(s.get('macro_std_matched_iou'))}")
            logger.info(f"    Inst Rec:  {_fmt(s.get('macro_mean_instance_recall'))} ± {_fmt(s.get('macro_std_instance_recall'))}")
            logger.info(f"    Micro D/P/R: {_fmt(s.get('micro_dice'))}/{_fmt(s.get('micro_precision'))}/{_fmt(s.get('micro_recall'))}")
            logger.info(f"    Empty: pred={s.get('n_empty_pred',0)} gt={s.get('n_empty_gt',0)} both={s.get('n_both_empty',0)}")


def _fmt(v) -> str:
    """Format metric value, handling None."""
    if v is None:
        return "null"
    return f"{v:.4f}"


if __name__ == "__main__":
    main()
