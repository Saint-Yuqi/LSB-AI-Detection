#!/usr/bin/env python3
"""
Evaluate a SAM3 checkpoint against one of the three benchmarks:

    fbox_gold_satellites   satellites only, ROI-restricted
    firebox_dr1_streams    streams only, full-frame (SBlim31.5)
    gt_canonical           streams + satellites, full-frame

All inference, matching, post and overlay run on a fixed 1024x1024 grid.

Usage:
    conda run -n sam3 python scripts/eval/evaluate_checkpoint.py \
        --config configs/eval_checkpoint.yaml \
        --mode fbox_gold_satellites
    conda run -n sam3 python scripts/eval/evaluate_checkpoint.py \
        --config configs/eval_checkpoint.yaml \
        --mode gt_canonical --max-samples 1

Output layout (under ``output.root``):
    {mode}/{condition}[/{noise_profile}]/{variant}/{base_key}/
        predictions_raw.json
        predictions_post_pred_only.json
        post_pred_only_satellite_stage_trace.json   (fbox_gold_satellites only)
        predictions_post_gt_aware.json   (gt_canonical only)
        overlays/gt_contour.png
    {mode}/{condition}[/{noise_profile}]/{variant}/
        report.json                      (per-sample reports + summary)

    To regenerate all overlay PNGs from existing predictions JSON (no SAM3, no
    report write):

        conda run -n sam3 python scripts/eval/evaluate_checkpoint.py \\
            --config configs/eval_checkpoint.yaml --mode fbox_gold_satellites \\
            --overlays-only

    To recompute report.json, diagnostics.json, and diagnostics_summary from
    on-disk ``predictions_*.json`` only (no SAM3 inference, no overlay write):

        conda run -n sam3 python scripts/eval/evaluate_checkpoint.py \\
            --config configs/eval_checkpoint.yaml --mode fbox_gold_satellites \\
            --from-disk

    To rebuild post-processing from existing ``predictions_raw.json`` only
    (no SAM3 inference), reading raw predictions from one run and writing a
    fresh self-contained eval tree somewhere else:

        conda run -n sam3 python scripts/eval/evaluate_checkpoint.py \\
            --config configs/eval_checkpoint.yaml --mode fbox_gold_satellites \\
            --post-from-raw \\
            --source-output-root outputs/eval_checkpoint \\
            --output-root outputs/eval_checkpoint_no_core
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.evaluation.checkpoint_eval import (  # noqa: E402
    SATELLITE_POST_STAGES,
    TARGET_H,
    TARGET_W,
    aggregate,
    apply_post_gt_aware,
    apply_post_pred_only,
    apply_satellite_post_with_trace,
    compute_sample_report,
    json_default,
    load_benchmark,
    run_sam3_on_sample,
)
from src.evaluation.satellite_diagnostics import aggregate_diagnostics  # noqa: E402
from src.analysis.mask_metrics import append_metrics_to_masks  # noqa: E402
from src.inference.sam3_prompt_runner import SAM3PromptRunner  # noqa: E402
from src.pipelines.unified_dataset.artifacts import (  # noqa: E402
    assign_stable_ids,
    load_predictions_json,
    save_predictions_json,
)
from src.visualization.overlay import (  # noqa: E402
    save_eval_prediction_overlay,
    save_gt_contour_only_overlay,
)

logger = logging.getLogger(__name__)


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _filter_gt_type_map(gt_map: np.ndarray, gt_type_of_id: dict[int, str], want: str) -> np.ndarray:
    keep = [i for i, t in gt_type_of_id.items() if t == want]
    if not keep:
        return np.zeros_like(gt_map, dtype=np.int32)
    return np.where(np.isin(gt_map, keep), gt_map, 0).astype(np.int32)


def _output_dir(root: Path, mode: str, render_cfg: dict, base_key: str | None = None) -> Path:
    """Build the output subtree path mirroring render/condition/variant."""
    condition = render_cfg.get("condition", "current")
    variant = render_cfg.get("variant", "linear_magnitude")
    parts: list[str] = [mode, condition]
    if condition == "noisy":
        noise_profile = render_cfg.get("noise_profile")
        if not noise_profile:
            raise ValueError("render.noise_profile must be set when condition == 'noisy'")
        parts.append(noise_profile)
    parts.append(variant)
    if base_key:
        parts.append(base_key)
    return root.joinpath(*parts)


def _config_for_report(cfg: dict, mode: str) -> dict:
    """Return only the active benchmark config needed to interpret a report."""
    report_cfg = deepcopy(cfg)
    prompts = report_cfg.get("prompts")
    if isinstance(prompts, dict) and mode in prompts:
        report_cfg["prompts"] = {mode: prompts[mode]}

    if mode == "fbox_gold_satellites":
        pred_only = (
            report_cfg.get("post", {})
            .get("pred_only", {})
        )
        for key in (
            "enable_streams_sanity",
            "streams_sanity",
            "enable_cross_type_conflict",
            "cross_type_conflict",
        ):
            pred_only.pop(key, None)
    return report_cfg


def _label_for_overlay(sample, num_gt_sat: int, num_gt_str: int) -> str:
    # ASCII-only: OpenCV's Hershey font renders non-ASCII (e.g. U+00B7) as '?'.
    return (
        f"{sample.base_key} | GT={num_gt_sat + num_gt_str} "
        f"| SAT={num_gt_sat} | STR={num_gt_str}"
    )


def _write_sample_overlays(
    sample_dir: Path,
    sample,
    render_rgb: np.ndarray,
    raw_masks: list[dict],
    po_pair: tuple[list[dict], list[dict]],
    ga_pair: tuple[list[dict], list[dict]] | None,
    sample_label: str,
) -> None:
    """Write the 3-or-4 evaluation overlays for one sample.

    Always writes:
        overlays/gt_contour.png
        overlays/raw_overlay.png
        overlays/post_pred_only_overlay.png
    Additionally writes, iff ``ga_pair is not None``:
        overlays/post_gt_aware_overlay.png

    ``post_gt_aware`` is sourced strictly from ``ga_pair`` so that this
    function is decoupled from the ``save_post_predictions`` JSON branch.
    """
    overlays_dir = sample_dir / "overlays"
    overlays_dir.mkdir(exist_ok=True)

    roi_bbox = sample.roi_bbox_1024  # (y0, x0, y1, x1) | None

    save_gt_contour_only_overlay(
        overlays_dir / "gt_contour.png",
        render_rgb,
        sample.gt_instance_map_1024,
        sample_label,
        sample.gt_type_of_id,
        roi_bbox=roi_bbox,
    )

    save_eval_prediction_overlay(
        overlays_dir / "raw_overlay.png",
        render_rgb,
        sample.gt_instance_map_1024,
        sample.gt_type_of_id,
        raw_masks,
        roi_bbox,
        layer_label="raw",
        sample_label=sample_label,
    )

    po_streams, po_sats = po_pair
    save_eval_prediction_overlay(
        overlays_dir / "post_pred_only_overlay.png",
        render_rgb,
        sample.gt_instance_map_1024,
        sample.gt_type_of_id,
        po_streams + po_sats,
        roi_bbox,
        layer_label="post_pred_only",
        sample_label=sample_label,
    )

    if ga_pair is not None:
        ga_streams, ga_sats = ga_pair
        save_eval_prediction_overlay(
            overlays_dir / "post_gt_aware_overlay.png",
            render_rgb,
            sample.gt_instance_map_1024,
            sample.gt_type_of_id,
            ga_streams + ga_sats,
            roi_bbox,
            layer_label="post_gt_aware",
            sample_label=sample_label,
        )


def _write_satellite_stage_trace_sidecar(
    sample_dir: Path,
    mode: str,
    sats_raw: list[dict],
    po_sats: list[dict],
    trace_records: list[dict] | None,
) -> None:
    """Write the pure-satellite post trace sidecar for fbox samples."""
    if trace_records is None:
        return

    n_kept_trace = sum(1 for record in trace_records if record["final_status"] == "kept")
    trace_doc = {
        "schema_version": "1.0.0",
        "benchmark_mode": mode,
        "layer": "post_pred_only",
        "stage_order": list(SATELLITE_POST_STAGES),
        "n_raw_satellites": len(sats_raw),
        "n_kept_post": n_kept_trace,
        "n_removed_post": len(sats_raw) - n_kept_trace,
        "records": trace_records,
    }
    assert len(trace_records) == len(sats_raw)
    assert n_kept_trace == len(po_sats)
    (sample_dir / "post_pred_only_satellite_stage_trace.json").write_text(
        json.dumps(trace_doc, indent=2, default=json_default)
    )


def _split_stream_sat_masks(
    masks: list[dict],
) -> tuple[list[dict], list[dict]]:
    streams = [m for m in masks if m.get("type_label") == "streams"]
    sats = [m for m in masks if m.get("type_label") == "satellites"]
    return streams, sats


def regenerate_overlays_from_disk(
    cfg: dict,
    samples: list,
    mode: str,
    render_cfg: dict,
) -> None:
    """Re-write overlay PNGs from existing ``predictions_*.json`` on disk.

    Does not run SAM3, does not write ``report.json``. Requires
    ``predictions_raw.json`` and ``predictions_post_pred_only.json`` per
    sample. If ``predictions_post_gt_aware.json`` exists, a fourth overlay
    is written (typical for ``gt_canonical`` runs).
    """
    out_root = _resolve(cfg["output"]["root"])
    layer_out_dir = _output_dir(out_root, mode, render_cfg)
    if not layer_out_dir.is_dir():
        raise FileNotFoundError(
            f"eval output directory does not exist: {layer_out_dir}. "
            "Run a full eval first or check output.root in the config."
        )

    n_ok = 0
    n_skip = 0
    for idx, sample in enumerate(samples):
        sample_dir = layer_out_dir / sample.base_key
        raw_p = sample_dir / "predictions_raw.json"
        post_p = sample_dir / "predictions_post_pred_only.json"
        if not raw_p.is_file() or not post_p.is_file():
            logger.warning(
                "[%d/%d] skip %s — missing %s or %s",
                idx + 1, len(samples), sample.base_key, raw_p.name, post_p.name,
            )
            n_skip += 1
            continue

        _, raw_masks = load_predictions_json(raw_p)
        _, post_masks = load_predictions_json(post_p)
        po_streams, po_sats = _split_stream_sat_masks(post_masks)

        ga_path = sample_dir / "predictions_post_gt_aware.json"
        if ga_path.is_file():
            _, ga_masks = load_predictions_json(ga_path)
            ga_pair = _split_stream_sat_masks(ga_masks)
        else:
            ga_pair = None

        render_bgr = cv2.imread(str(sample.render_1024_path))
        if render_bgr is None:
            logger.warning(
                "[%d/%d] skip %s — cannot read render %s",
                idx + 1, len(samples), sample.base_key, sample.render_1024_path,
            )
            n_skip += 1
            continue
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)

        num_gt_sat = sum(1 for t in sample.gt_type_of_id.values() if t == "satellites")
        num_gt_str = sum(1 for t in sample.gt_type_of_id.values() if t == "streams")
        _write_sample_overlays(
            sample_dir,
            sample,
            render_rgb,
            raw_masks,
            (po_streams, po_sats),
            ga_pair,
            _label_for_overlay(sample, num_gt_sat, num_gt_str),
        )
        n_ok += 1
        logger.info("[%d/%d] overlays OK %s", idx + 1, len(samples), sample.base_key)

    logger.info("overlays-only: %d written, %d skipped", n_ok, n_skip)


def regenerate_report_from_disk(
    cfg: dict,
    samples: list,
    mode: str,
    render_cfg: dict,
) -> None:
    """Recompute ``report.json`` + per-sample ``diagnostics.json`` from on-disk
    predictions (no SAM3, no ``predictions_*.json`` rewrite, no overlays).

    Loads ``predictions_raw.json`` and ``predictions_post_pred_only.json`` per
    sample; loads ``predictions_post_gt_aware.json`` when ``gt_canonical`` and
    the file exists. Masks are metrics-enriched the same way as a fresh
    inference run, then :func:`compute_sample_report` is called with
    ``render_signal`` from the render PNG.
    """
    out_root = _resolve(cfg["output"]["root"])
    layer_out_dir = _output_dir(out_root, mode, render_cfg)
    if not layer_out_dir.is_dir():
        raise FileNotFoundError(
            f"eval output directory does not exist: {layer_out_dir}. "
            "Point output.root at your existing run or run a full eval first."
        )

    reports: list[dict] = []
    per_sample_diag_rows: list[list[dict]] = []
    per_sample_diag_post_counts: list[dict | None] = []
    per_sample_diag_post_counts_roi: list[dict | None] = []
    t0 = time.time()
    n_ok = 0
    n_skip = 0

    for idx, sample in enumerate(samples):
        sample_dir = layer_out_dir / sample.base_key
        raw_p = sample_dir / "predictions_raw.json"
        post_p = sample_dir / "predictions_post_pred_only.json"
        if not raw_p.is_file() or not post_p.is_file():
            logger.warning(
                "[%d/%d] skip %s — missing %s or %s",
                idx + 1, len(samples), sample.base_key, raw_p.name, post_p.name,
            )
            n_skip += 1
            continue

        _, raw_masks = load_predictions_json(raw_p)
        if raw_masks:
            append_metrics_to_masks(raw_masks, TARGET_H, TARGET_W, compute_hull=True)
        assign_stable_ids(raw_masks)

        _, post_masks = load_predictions_json(post_p)
        if post_masks:
            append_metrics_to_masks(post_masks, TARGET_H, TARGET_W, compute_hull=True)
        po_streams, po_sats = _split_stream_sat_masks(post_masks)

        ga_path = sample_dir / "predictions_post_gt_aware.json"
        if mode == "gt_canonical" and ga_path.is_file():
            _, ga_masks = load_predictions_json(ga_path)
            if ga_masks:
                append_metrics_to_masks(ga_masks, TARGET_H, TARGET_W, compute_hull=True)
            ga_streams, ga_sats = _split_stream_sat_masks(ga_masks)
            ga_pair = (ga_streams, ga_sats)
        else:
            ga_pair = None

        render_bgr = cv2.imread(str(sample.render_1024_path))
        if render_bgr is None:
            logger.warning(
                "[%d/%d] skip %s — cannot read render %s",
                idx + 1, len(samples), sample.base_key, sample.render_1024_path,
            )
            n_skip += 1
            continue
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        render_signal = render_rgb.mean(axis=2).astype(np.float32)

        report, diag_report = compute_sample_report(
            sample, raw_masks, (po_streams, po_sats), ga_pair, cfg,
            render_signal=render_signal,
        )
        report["sam3_inference_ms"] = 0.0
        reports.append(report)

        if diag_report is not None:
            per_sample_diag_rows.append(diag_report["per_candidate"])
            per_sample_diag_post_counts.append(diag_report["counts_post_by_label"])
            per_sample_diag_post_counts_roi.append(diag_report["counts_post_by_label_roi"])
            (sample_dir / "diagnostics.json").write_text(
                json.dumps(diag_report, indent=2, default=json_default)
            )

        trace_records = None
        if mode == "fbox_gold_satellites" and raw_masks:
            _streams_raw, sats_raw = _split_stream_sat_masks(raw_masks)
            traced_sats, trace_records = apply_satellite_post_with_trace(
                sats_raw, TARGET_H, TARGET_W, cfg["post"]["pred_only"]
            )
            assert len(traced_sats) == len(po_sats), (
                f"trace/post mismatch: traced_kept={len(traced_sats)} "
                f"vs po_sats={len(po_sats)} for {sample.base_key}"
            )
            _write_satellite_stage_trace_sidecar(
                sample_dir=sample_dir,
                mode=mode,
                sats_raw=sats_raw,
                po_sats=po_sats,
                trace_records=trace_records,
            )
        n_ok += 1
        logger.info(
            "[%d/%d] from-disk OK %s (diag=%s)",
            idx + 1, len(samples), sample.base_key, diag_report is not None,
        )

    if not reports:
        logger.error("from-disk: no sample produced a report; aborting")
        return

    summary = aggregate(reports)
    diagnostics_summary = (
        aggregate_diagnostics(
            per_sample_diag_rows,
            per_sample_diag_post_counts,
            per_sample_diag_post_counts_roi,
        )
        if per_sample_diag_rows
        else None
    )
    report_doc = {
        "config": _config_for_report(cfg, mode),
        "benchmark_mode": mode,
        "n_samples": len(reports),
        "elapsed_seconds": time.time() - t0,
        "summary": summary,
        "diagnostics_summary": diagnostics_summary,
        "per_sample": reports,
    }
    report_path = layer_out_dir / "report.json"
    report_path.write_text(json.dumps(report_doc, indent=2, default=json_default))
    logger.info(
        "wrote %s (from-disk: %d ok, %d skipped)", report_path, n_ok, n_skip
    )


def rebuild_post_from_raw_predictions(
    cfg: dict,
    samples: list,
    mode: str,
    render_cfg: dict,
    *,
    source_output_root: Path | None = None,
    allow_inplace: bool = False,
) -> None:
    """Rebuild post outputs from on-disk ``predictions_raw.json`` only.

    This path reuses stored raw SAM3 predictions, reapplies the current
    ``post`` config, and writes a fresh self-contained eval tree:

        predictions_raw.json
        predictions_post_pred_only.json
        predictions_post_gt_aware.json   (gt_canonical only)
        diagnostics.json
        post_pred_only_satellite_stage_trace.json   (fbox only)
        overlays/*                       (when enabled)
        report.json

    No SAM3 inference is run. By default, in-place overwrite is refused so
    callers can safely preserve an existing run for side-by-side comparison.
    """
    dest_out_root = _resolve(cfg["output"]["root"])
    source_out_root = source_output_root or dest_out_root

    source_layer_out_dir = _output_dir(source_out_root, mode, render_cfg)
    dest_layer_out_dir = _output_dir(dest_out_root, mode, render_cfg)

    if not source_layer_out_dir.is_dir():
        raise FileNotFoundError(
            f"source eval output directory does not exist: {source_layer_out_dir}. "
            "Point --source-output-root at an existing run with predictions_raw.json."
        )

    if (
        source_layer_out_dir.resolve() == dest_layer_out_dir.resolve()
        and not allow_inplace
    ):
        raise RuntimeError(
            "refusing in-place post-from-raw overwrite. "
            "Set --output-root to a new location to preserve the old run, "
            "or pass --allow-inplace to overwrite post artifacts in place."
        )

    dest_layer_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("source dir: %s", source_layer_out_dir)
    logger.info("dest dir:   %s", dest_layer_out_dir)

    reports: list[dict] = []
    per_sample_diag_rows: list[list[dict]] = []
    per_sample_diag_post_counts: list[dict | None] = []
    per_sample_diag_post_counts_roi: list[dict | None] = []
    t0 = time.time()
    n_ok = 0
    n_skip = 0

    for idx, sample in enumerate(samples):
        source_sample_dir = source_layer_out_dir / sample.base_key
        raw_p = source_sample_dir / "predictions_raw.json"
        if not raw_p.is_file():
            logger.warning(
                "[%d/%d] skip %s — missing %s in source run",
                idx + 1, len(samples), sample.base_key, raw_p,
            )
            n_skip += 1
            continue

        _, raw_masks = load_predictions_json(raw_p)
        if raw_masks:
            append_metrics_to_masks(raw_masks, TARGET_H, TARGET_W, compute_hull=True)
        assign_stable_ids(raw_masks)

        streams_raw, sats_raw = _split_stream_sat_masks(raw_masks)
        po_streams, po_sats = apply_post_pred_only(
            streams_raw, sats_raw, TARGET_H, TARGET_W, cfg["post"]["pred_only"]
        )

        trace_records = None
        if mode == "fbox_gold_satellites" and sats_raw:
            traced_sats, trace_records = apply_satellite_post_with_trace(
                sats_raw, TARGET_H, TARGET_W, cfg["post"]["pred_only"]
            )
            assert len(traced_sats) == len(po_sats), (
                f"trace/post mismatch: traced_kept={len(traced_sats)} "
                f"vs po_sats={len(po_sats)} for {sample.base_key}"
            )

        ga_pair = None
        if mode == "gt_canonical":
            streams_gt_map = _filter_gt_type_map(
                sample.gt_instance_map_1024, sample.gt_type_of_id, "streams"
            )
            ga_pair = apply_post_gt_aware(
                po_streams, po_sats, streams_gt_map, TARGET_H, TARGET_W,
                cfg["post"]["gt_aware"],
            )

        render_bgr = cv2.imread(str(sample.render_1024_path))
        if render_bgr is None:
            logger.warning(
                "[%d/%d] skip %s — cannot read render %s",
                idx + 1, len(samples), sample.base_key, sample.render_1024_path,
            )
            n_skip += 1
            continue
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        render_signal = render_rgb.mean(axis=2).astype(np.float32)

        report, diag_report = compute_sample_report(
            sample, raw_masks, (po_streams, po_sats), ga_pair, cfg,
            render_signal=render_signal,
        )
        report["sam3_inference_ms"] = 0.0
        reports.append(report)

        sample_dir = dest_layer_out_dir / sample.base_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        if cfg["output"].get("save_raw_predictions", True):
            save_predictions_json(
                sample_dir / "predictions_raw.json",
                raw_masks, TARGET_H, TARGET_W, engine="sam3", layer="raw",
            )
        if cfg["output"].get("save_post_predictions", True):
            save_predictions_json(
                sample_dir / "predictions_post_pred_only.json",
                po_streams + po_sats, TARGET_H, TARGET_W,
                engine="sam3", layer="post_pred_only",
            )
            if ga_pair is not None:
                ga_streams, ga_sats = ga_pair
                save_predictions_json(
                    sample_dir / "predictions_post_gt_aware.json",
                    ga_streams + ga_sats, TARGET_H, TARGET_W,
                    engine="sam3", layer="post_gt_aware",
                )

        if diag_report is not None:
            per_sample_diag_rows.append(diag_report["per_candidate"])
            per_sample_diag_post_counts.append(diag_report["counts_post_by_label"])
            per_sample_diag_post_counts_roi.append(diag_report["counts_post_by_label_roi"])
            (sample_dir / "diagnostics.json").write_text(
                json.dumps(diag_report, indent=2, default=json_default)
            )

        _write_satellite_stage_trace_sidecar(
            sample_dir=sample_dir,
            mode=mode,
            sats_raw=sats_raw,
            po_sats=po_sats,
            trace_records=trace_records,
        )

        if cfg["output"].get("save_overlays", True) and cfg.get("overlay", {}).get("enabled", True):
            num_gt_sat = sum(1 for t in sample.gt_type_of_id.values() if t == "satellites")
            num_gt_str = sum(1 for t in sample.gt_type_of_id.values() if t == "streams")
            _write_sample_overlays(
                sample_dir,
                sample,
                render_rgb,
                raw_masks,
                (po_streams, po_sats),
                ga_pair,
                _label_for_overlay(sample, num_gt_sat, num_gt_str),
            )

        n_ok += 1
        logger.info(
            "[%d/%d] post-from-raw OK %s (diag=%s)",
            idx + 1, len(samples), sample.base_key, diag_report is not None,
        )

    if not reports:
        logger.error("post-from-raw: no sample produced a report; aborting")
        return

    summary = aggregate(reports)
    diagnostics_summary = (
        aggregate_diagnostics(
            per_sample_diag_rows,
            per_sample_diag_post_counts,
            per_sample_diag_post_counts_roi,
        )
        if per_sample_diag_rows
        else None
    )
    report_doc = {
        "config": _config_for_report(cfg, mode),
        "benchmark_mode": mode,
        "n_samples": len(reports),
        "elapsed_seconds": time.time() - t0,
        "summary": summary,
        "diagnostics_summary": diagnostics_summary,
        "per_sample": reports,
    }
    report_path = dest_layer_out_dir / "report.json"
    report_path.write_text(json.dumps(report_doc, indent=2, default=json_default))
    logger.info(
        "wrote %s (post-from-raw: %d ok, %d skipped)", report_path, n_ok, n_skip
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True, help="Path to configs/eval_checkpoint.yaml")
    ap.add_argument("--mode", default=None,
                    choices=["fbox_gold_satellites", "firebox_dr1_streams", "gt_canonical"],
                    help="Override benchmark.mode")
    ap.add_argument("--condition", default=None, choices=["current", "noisy"],
                    help="Override render.condition")
    ap.add_argument("--variant", default=None, help="Override render.variant")
    ap.add_argument("--noise-profile", default=None, help="Override render.noise_profile")
    ap.add_argument(
        "--output-root", default=None,
        help="Override output.root for the destination run tree.",
    )
    ap.add_argument(
        "--source-output-root", default=None,
        help="When using --post-from-raw, read predictions_raw.json from this "
        "existing output.root instead of the destination output.root.",
    )
    ap.add_argument("--max-samples", type=int, default=None,
                    help="Process only the first N samples (after loader)")
    ap.add_argument(
        "--overlays-only",
        action="store_true",
        help="Regenerate overlay PNGs from existing predictions_*.json only "
        "(no SAM3 inference, no report.json).",
    )
    ap.add_argument(
        "--from-disk",
        action="store_true",
        help="Recompute report.json, per-sample diagnostics.json, and "
        "diagnostics_summary from existing predictions_*.json and renders "
        "(no SAM3 inference, does not rewrite predictions or overlays).",
    )
    ap.add_argument(
        "--post-from-raw",
        action="store_true",
        help="Reapply the current post config to existing predictions_raw.json "
        "and write a fresh eval tree (no SAM3 inference).",
    )
    ap.add_argument(
        "--allow-inplace",
        action="store_true",
        help="Allow --post-from-raw to overwrite post artifacts in the same "
        "run tree. By default this is refused to preserve the old run.",
    )
    ap.add_argument(
        "--disable-core-policy",
        action="store_true",
        help="Override post.pred_only.enable_core_policy=false for this run.",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.mode:
        cfg["benchmark"]["mode"] = args.mode
    if args.condition:
        cfg["render"]["condition"] = args.condition
    if args.variant:
        cfg["render"]["variant"] = args.variant
    if args.noise_profile:
        cfg["render"]["noise_profile"] = args.noise_profile
    if args.output_root:
        cfg["output"]["root"] = args.output_root
    if args.disable_core_policy:
        cfg.setdefault("post", {}).setdefault("pred_only", {})["enable_core_policy"] = False

    mode = cfg["benchmark"]["mode"]
    render_cfg = cfg["render"]

    # --- load samples ---
    samples = load_benchmark(cfg)
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info("loaded %d samples for benchmark=%s", len(samples), mode)

    active_no_infer_modes = sum(
        bool(flag) for flag in (args.overlays_only, args.from_disk, args.post_from_raw)
    )
    if active_no_infer_modes > 1:
        raise SystemExit(
            "choose only one of --overlays-only, --from-disk, or --post-from-raw"
        )
    if args.overlays_only:
        if not (cfg["output"].get("save_overlays", True) and cfg.get("overlay", {}).get("enabled", True)):
            logger.warning("overlay is disabled in config; no-op")
            return
        t0 = time.time()
        regenerate_overlays_from_disk(cfg, samples, mode, render_cfg)
        logger.info("overlays-only finished in %.1fs", time.time() - t0)
        return
    if args.from_disk:
        t0 = time.time()
        regenerate_report_from_disk(cfg, samples, mode, render_cfg)
        logger.info("from-disk finished in %.1fs", time.time() - t0)
        return
    if args.post_from_raw:
        t0 = time.time()
        rebuild_post_from_raw_predictions(
            cfg,
            samples,
            mode,
            render_cfg,
            source_output_root=(
                _resolve(args.source_output_root) if args.source_output_root else None
            ),
            allow_inplace=args.allow_inplace,
        )
        logger.info("post-from-raw finished in %.1fs", time.time() - t0)
        return

    # --- build runner once ---
    sam3_cfg = cfg["sam3"]
    ckpt_path = _resolve(cfg["checkpoint"])
    runner = SAM3PromptRunner(
        checkpoint=ckpt_path,
        bpe_path=sam3_cfg["bpe_path"],
        confidence_threshold=float(sam3_cfg.get("confidence_threshold", 0.18)),
        resolution=int(sam3_cfg.get("resolution", 1008)),
        device=sam3_cfg.get("device", "cuda"),
        target_size=(TARGET_H, TARGET_W),
    )

    prompts = cfg["prompts"][mode]
    out_root = _resolve(cfg["output"]["root"])
    layer_out_dir = _output_dir(out_root, mode, render_cfg)
    layer_out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("output dir: %s", layer_out_dir)

    reports: list[dict] = []
    per_sample_diag_rows: list[list[dict]] = []
    per_sample_diag_post_counts: list[dict | None] = []
    per_sample_diag_post_counts_roi: list[dict | None] = []
    t0 = time.time()

    for idx, sample in enumerate(samples):
        logger.info("[%d/%d] %s", idx + 1, len(samples), sample.base_key)
        raw_masks, time_ms = run_sam3_on_sample(runner, sample, prompts)

        # Stamp raw_index + candidate_id ONCE on the full list (streams +
        # satellites in original order) BEFORE splitting or diagnostics.
        # These IDs survive into predictions_raw.json and diagnostics.json.
        assign_stable_ids(raw_masks)

        streams_raw = [m for m in raw_masks if m.get("type_label") == "streams"]
        sats_raw = [m for m in raw_masks if m.get("type_label") == "satellites"]

        po_streams, po_sats = apply_post_pred_only(
            streams_raw, sats_raw, TARGET_H, TARGET_W, cfg["post"]["pred_only"]
        )

        # --- satellite stage trace sidecar (fbox_gold_satellites only) ---
        trace_records = None
        if mode == "fbox_gold_satellites" and sats_raw:
            traced_sats, trace_records = apply_satellite_post_with_trace(
                sats_raw, TARGET_H, TARGET_W, cfg["post"]["pred_only"]
            )
            # Reconciliation: trace helper must agree with apply_post_pred_only.
            assert len(traced_sats) == len(po_sats), (
                f"trace/post mismatch: traced_kept={len(traced_sats)} "
                f"vs po_sats={len(po_sats)} for {sample.base_key}"
            )

        ga_pair = None
        if mode == "gt_canonical":
            streams_gt_map = _filter_gt_type_map(
                sample.gt_instance_map_1024, sample.gt_type_of_id, "streams"
            )
            ga_pair = apply_post_gt_aware(
                po_streams, po_sats, streams_gt_map, TARGET_H, TARGET_W,
                cfg["post"]["gt_aware"],
            )

        # Read the render once per sample; reused by diagnostics AND overlays.
        render_bgr = cv2.imread(str(sample.render_1024_path))
        if render_bgr is None:
            raise FileNotFoundError(f"cv2 failed to read render: {sample.render_1024_path}")
        render_rgb = cv2.cvtColor(render_bgr, cv2.COLOR_BGR2RGB)
        # The linear_magnitude render writes identical R=G=B channels, so a
        # plain channel mean recovers the single-channel intensity SAM3 saw.
        render_signal = render_rgb.mean(axis=2).astype(np.float32)

        report, diag_report = compute_sample_report(
            sample, raw_masks, (po_streams, po_sats), ga_pair, cfg,
            render_signal=render_signal,
        )
        report["sam3_inference_ms"] = time_ms
        reports.append(report)

        sample_dir = layer_out_dir / sample.base_key
        sample_dir.mkdir(parents=True, exist_ok=True)

        if cfg["output"].get("save_raw_predictions", True):
            save_predictions_json(
                sample_dir / "predictions_raw.json",
                raw_masks, TARGET_H, TARGET_W, engine="sam3", layer="raw",
            )
        if cfg["output"].get("save_post_predictions", True):
            save_predictions_json(
                sample_dir / "predictions_post_pred_only.json",
                po_streams + po_sats, TARGET_H, TARGET_W,
                engine="sam3", layer="post_pred_only",
            )
            if ga_pair is not None:
                ga_streams, ga_sats = ga_pair
                save_predictions_json(
                    sample_dir / "predictions_post_gt_aware.json",
                    ga_streams + ga_sats, TARGET_H, TARGET_W,
                    engine="sam3", layer="post_gt_aware",
                )

        # Full diagnostics go to a sidecar (keeps report.json small).
        if diag_report is not None:
            per_sample_diag_rows.append(diag_report["per_candidate"])
            per_sample_diag_post_counts.append(diag_report["counts_post_by_label"])
            per_sample_diag_post_counts_roi.append(diag_report["counts_post_by_label_roi"])
            (sample_dir / "diagnostics.json").write_text(
                json.dumps(diag_report, indent=2, default=json_default)
            )

        # Stage trace sidecar.
        _write_satellite_stage_trace_sidecar(
            sample_dir=sample_dir,
            mode=mode,
            sats_raw=sats_raw,
            po_sats=po_sats,
            trace_records=trace_records,
        )

        if cfg["output"].get("save_overlays", True) and cfg.get("overlay", {}).get("enabled", True):
            num_gt_sat = sum(1 for t in sample.gt_type_of_id.values() if t == "satellites")
            num_gt_str = sum(1 for t in sample.gt_type_of_id.values() if t == "streams")
            _write_sample_overlays(
                sample_dir,
                sample,
                render_rgb,
                raw_masks,
                (po_streams, po_sats),
                ga_pair,
                _label_for_overlay(sample, num_gt_sat, num_gt_str),
            )

    summary = aggregate(reports)
    diagnostics_summary = (
        aggregate_diagnostics(
            per_sample_diag_rows,
            per_sample_diag_post_counts,
            per_sample_diag_post_counts_roi,
        )
        if per_sample_diag_rows
        else None
    )
    report_doc = {
        "config": _config_for_report(cfg, mode),
        "benchmark_mode": mode,
        "n_samples": len(reports),
        "elapsed_seconds": time.time() - t0,
        "summary": summary,
        "diagnostics_summary": diagnostics_summary,
        "per_sample": reports,
    }
    report_path = layer_out_dir / "report.json"
    report_path.write_text(json.dumps(report_doc, indent=2, default=json_default))
    logger.info("wrote %s", report_path)
    logger.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
