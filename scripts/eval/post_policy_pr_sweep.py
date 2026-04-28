#!/usr/bin/env python3
"""Sweep score_gate threshold and produce a post-policy PR curve.

For each threshold tau in the sweep:
    1. Load each sample's predictions_raw.json from an existing eval output.
    2. Override score_gate.{small,medium,large}_min_score = tau.
    3. Run the full post pipeline (apply_post_pred_only) — score_gate +
       prior_filter (+ core_policy + cross_type_conflict if enabled in cfg).
    4. Classify resulting sats via the satellite diagnostic taxonomy and
       count matched / unmatched / unique-GT-covered, pooled across samples.

Output: post_policy_pr_curve.{json,csv,png} alongside the input report.json.

Usage:
    python scripts/eval/post_policy_pr_sweep.py \\
        --config configs/eval_checkpoint.yaml \\
        --mode fbox_gold_satellites \\
        --eval-dir outputs/eval_checkpoint/fbox_gold_satellites/current/linear_magnitude
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.analysis.mask_metrics import append_metrics_to_masks  # noqa: E402
from src.evaluation.checkpoint_eval import (  # noqa: E402
    TARGET_H,
    TARGET_W,
    apply_post_pred_only,
    load_benchmark,
)
from src.evaluation.satellite_diagnostics import (  # noqa: E402
    MATCHED_LABELS,
    DiagnosticCfg,
    classify_candidates,
)
from src.pipelines.unified_dataset.artifacts import (  # noqa: E402
    assign_stable_ids,
    load_predictions_json,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _resolve(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else (_PROJECT_ROOT / p).resolve()


def _filter_gt_by_type(
    gt_instance_map: np.ndarray,
    gt_type_of_id: dict[int, str],
    target_type: str,
) -> np.ndarray:
    """Return a masked instance map keeping only instances of ``target_type``.

    Mirrors src.evaluation.checkpoint_eval._filter_gt_by_type but is
    re-implemented here to avoid importing a private helper.
    """
    out = np.zeros_like(gt_instance_map)
    for inst_id, type_label in gt_type_of_id.items():
        if type_label != target_type:
            continue
        out = np.where(gt_instance_map == inst_id, inst_id, out)
    return out


def _diag_cfg_from_dict(cfg: dict) -> DiagnosticCfg:
    sat_cfg = (cfg.get("diagnostics") or {}).get("satellites") or {}
    return DiagnosticCfg(
        min_purity_for_match=float(sat_cfg.get("min_purity_for_match", 0.50)),
        completeness_complete=float(sat_cfg.get("completeness_complete", 0.50)),
        complete_one_to_one_min_completeness=float(
            sat_cfg.get("complete_one_to_one_min_completeness", 0.95)
        ),
        complete_one_to_one_max_seed_ratio=float(
            sat_cfg.get("complete_one_to_one_max_seed_ratio", 3.0)
        ),
        annulus_r_in_frac=float(sat_cfg.get("annulus_r_in_frac", 1.2)),
        annulus_r_out_frac=float(sat_cfg.get("annulus_r_out_frac", 2.0)),
        radial_n_rings=int(sat_cfg.get("radial_n_rings", 6)),
    )


def _f1(p: float | None, r: float | None) -> float | None:
    if p is None or r is None:
        return None
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def _evaluate_one_threshold(
    samples_data: list[dict],
    base_post_cfg: dict,
    tau: float,
    diag_cfg: DiagnosticCfg,
) -> dict:
    """Run post-pipeline at score_gate threshold tau, return aggregate metrics."""
    cfg_pred_only = deepcopy(base_post_cfg)
    cfg_pred_only.setdefault("score_gate", {})
    cfg_pred_only["score_gate"]["small_min_score"] = tau
    cfg_pred_only["score_gate"]["medium_min_score"] = tau
    cfg_pred_only["score_gate"]["large_min_score"] = tau
    cfg_pred_only["enable_score_gate"] = True

    # Pooled (micro) full_frame and ROI accounting.
    ff = {"matched": 0, "unmatched": 0, "covered_gt": 0, "num_gt": 0, "num_pred": 0}
    roi = {"matched": 0, "unmatched": 0, "covered_gt": 0, "num_gt": 0, "num_pred": 0}

    for s in samples_data:
        sat_post_masks = apply_post_pred_only(
            s["raw_streams"], s["raw_sats"], TARGET_H, TARGET_W, cfg_pred_only
        )[1]
        if not sat_post_masks:
            ff["num_gt"] += s["num_gt_full"]
            roi["num_gt"] += s["num_gt_roi"]
            continue

        entries = classify_candidates(
            sat_post_masks, s["gt_sat_map"], TARGET_H, TARGET_W, diag_cfg,
            roi_bbox=s["roi_bbox"],
        )

        # Full-frame slice
        ff_covered: set[int] = set()
        ff["num_pred"] += len(entries)
        ff["num_gt"] += s["num_gt_full"]
        for e in entries:
            if e.taxonomy_label in MATCHED_LABELS:
                ff["matched"] += 1
                if e.matched_gt_id is not None:
                    ff_covered.add(int(e.matched_gt_id))
            else:
                ff["unmatched"] += 1
        ff["covered_gt"] += len(ff_covered)

        # ROI slice
        roi_entries = [e for e in entries if e.intersects_roi]
        roi_covered: set[int] = set()
        roi["num_pred"] += len(roi_entries)
        roi["num_gt"] += s["num_gt_roi"]
        for e in roi_entries:
            if e.taxonomy_label in MATCHED_LABELS:
                roi["matched"] += 1
                if e.matched_gt_id is not None:
                    roi_covered.add(int(e.matched_gt_id))
            else:
                roi["unmatched"] += 1
        roi["covered_gt"] += len(roi_covered)

    def _block(b: dict) -> dict:
        precision = (b["matched"] / b["num_pred"]) if b["num_pred"] > 0 else None
        recall = (b["covered_gt"] / b["num_gt"]) if b["num_gt"] > 0 else None
        return {
            "precision": precision,
            "recall": recall,
            "f1": _f1(precision, recall),
            "matched": b["matched"],
            "unmatched": b["unmatched"],
            "unique_gt_covered": b["covered_gt"],
            "num_gt": b["num_gt"],
            "num_pred": b["num_pred"],
        }

    return {
        "threshold": tau,
        "full_frame": _block(ff),
        "roi": _block(roi),
    }


def _auc_pr(rows: list[dict], slice_key: str) -> float | None:
    """AUC under PR using step-sum (sklearn convention).

    Sort rows by recall ascending; sum (R_i - R_{i-1}) * P_i. Skip rows with
    None precision/recall.
    """
    pts = [
        (r[slice_key]["recall"], r[slice_key]["precision"])
        for r in rows
        if r[slice_key]["recall"] is not None and r[slice_key]["precision"] is not None
    ]
    if not pts:
        return None
    pts.sort(key=lambda x: x[0])
    auc = 0.0
    prev_r = 0.0
    for r, p in pts:
        auc += (r - prev_r) * p
        prev_r = r
    return float(auc)


def _render_plot(rows: list[dict], out_path: Path) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    for slice_key, color in (("full_frame", "#1f77b4"), ("roi", "#d62728")):
        pts = [
            (r[slice_key]["recall"], r[slice_key]["precision"], r["threshold"])
            for r in rows
            if r[slice_key]["recall"] is not None and r[slice_key]["precision"] is not None
        ]
        if not pts:
            continue
        pts.sort(key=lambda x: x[0])
        recalls = [p[0] for p in pts]
        precisions = [p[1] for p in pts]
        auc = _auc_pr(rows, slice_key)
        auc_s = f"{auc:.4f}" if auc is not None else "n/a"
        ax.plot(recalls, precisions, color=color, linewidth=1.6, marker="o",
                markersize=3, label=f"{slice_key}  AUC-PR={auc_s}")

    ax.set_xlabel("Recall (post)")
    ax.set_ylabel("Precision (post)")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)
    ax.set_title(
        "Satellite post-policy PR curve\n"
        "(sweep score_gate {small,medium,large}_min_score = tau)"
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Existing eval output dir (where predictions_raw.json live)")
    ap.add_argument("--thresholds", type=str,
                    default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,"
                            "0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
                    help="Comma-separated tau values to sweep")
    ap.add_argument("--max-samples", type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    cfg["benchmark"]["mode"] = args.mode

    base_post_cfg = cfg["post"]["pred_only"]
    diag_cfg = _diag_cfg_from_dict(cfg)

    samples = load_benchmark(cfg)
    if args.max_samples is not None:
        samples = samples[: args.max_samples]
    logger.info("loaded %d samples for mode=%s", len(samples), args.mode)

    eval_dir = args.eval_dir.resolve()

    # Pre-load raw masks + GT once per sample.
    samples_data: list[dict] = []
    for sample in samples:
        raw_p = eval_dir / sample.base_key / "predictions_raw.json"
        if not raw_p.is_file():
            logger.warning("skip %s — missing %s", sample.base_key, raw_p)
            continue
        _, raw_masks = load_predictions_json(raw_p)
        if raw_masks:
            append_metrics_to_masks(raw_masks, TARGET_H, TARGET_W, compute_hull=True)
        assign_stable_ids(raw_masks)

        raw_streams = [m for m in raw_masks if m.get("type_label") == "streams"]
        raw_sats = [m for m in raw_masks if m.get("type_label") == "satellites"]

        gt_full = sample.gt_instance_map_1024
        gt_sat_full = _filter_gt_by_type(gt_full, sample.gt_type_of_id, "satellites")
        num_gt_full = int(sum(1 for g in np.unique(gt_sat_full) if g != 0))

        # ROI-restricted GT (used only to count ROI N_GT; matching itself uses
        # the full sat map and ROI membership flag from classify_candidates)
        if sample.roi_bbox_1024 is not None:
            y0, x0, y1, x1 = sample.roi_bbox_1024
            gt_sat_roi = np.zeros_like(gt_sat_full)
            gt_sat_roi[y0:y1, x0:x1] = gt_sat_full[y0:y1, x0:x1]
            num_gt_roi = int(sum(1 for g in np.unique(gt_sat_roi) if g != 0))
        else:
            num_gt_roi = num_gt_full

        samples_data.append({
            "base_key": sample.base_key,
            "raw_streams": raw_streams,
            "raw_sats": raw_sats,
            "gt_sat_map": gt_sat_full,
            "roi_bbox": sample.roi_bbox_1024,
            "num_gt_full": num_gt_full,
            "num_gt_roi": num_gt_roi,
        })

    logger.info("pre-loaded %d samples with raw masks", len(samples_data))

    taus = sorted({float(x) for x in args.thresholds.split(",")})
    rows: list[dict] = []
    for tau in taus:
        row = _evaluate_one_threshold(samples_data, base_post_cfg, tau, diag_cfg)
        rows.append(row)
        ff = row["full_frame"]
        roi_b = row["roi"]
        logger.info(
            "tau=%.3f  ff:P=%s R=%s F1=%s   roi:P=%s R=%s F1=%s",
            tau,
            f"{ff['precision']:.4f}" if ff["precision"] is not None else "n/a",
            f"{ff['recall']:.4f}" if ff["recall"] is not None else "n/a",
            f"{ff['f1']:.4f}" if ff["f1"] is not None else "n/a",
            f"{roi_b['precision']:.4f}" if roi_b["precision"] is not None else "n/a",
            f"{roi_b['recall']:.4f}" if roi_b["recall"] is not None else "n/a",
            f"{roi_b['f1']:.4f}" if roi_b["f1"] is not None else "n/a",
        )

    auc_full = _auc_pr(rows, "full_frame")
    auc_roi = _auc_pr(rows, "roi")

    out_dir = eval_dir / "satellite_tradeoff"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_out = out_dir / "post_policy_pr_curve.json"
    json_out.write_text(json.dumps({
        "eval_dir": str(eval_dir),
        "mode": args.mode,
        "n_samples": len(samples_data),
        "thresholds": taus,
        "auc_pr_full_frame": auc_full,
        "auc_pr_roi": auc_roi,
        "rows": rows,
        "post_cfg_base": base_post_cfg,
    }, indent=2))
    logger.info("wrote %s", json_out)

    csv_out = out_dir / "post_policy_pr_curve.csv"
    with csv_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "threshold",
            "ff_precision", "ff_recall", "ff_f1", "ff_num_pred", "ff_matched", "ff_unmatched",
            "roi_precision", "roi_recall", "roi_f1", "roi_num_pred", "roi_matched", "roi_unmatched",
        ])
        for r in rows:
            ff = r["full_frame"]
            ro = r["roi"]
            w.writerow([
                r["threshold"],
                ff["precision"], ff["recall"], ff["f1"], ff["num_pred"], ff["matched"], ff["unmatched"],
                ro["precision"], ro["recall"], ro["f1"], ro["num_pred"], ro["matched"], ro["unmatched"],
            ])
    logger.info("wrote %s", csv_out)

    png_out = out_dir / "post_policy_pr_curve.png"
    _render_plot(rows, png_out)
    logger.info("wrote %s", png_out)

    print()
    auc_full_s = f"{auc_full:.4f}" if auc_full is not None else "n/a"
    auc_roi_s = f"{auc_roi:.4f}" if auc_roi is not None else "n/a"
    print(f"AUC-PR (post pipeline)  full_frame = {auc_full_s}   roi = {auc_roi_s}")
    print(f"(raw, for reference)    full_frame = 0.5562          roi = 0.8903")
    print()
    # Find best F1 + first row meeting both >= 0.9
    for slice_key in ("full_frame", "roi"):
        feasible = [
            r for r in rows
            if r[slice_key]["precision"] is not None and r[slice_key]["recall"] is not None
            and r[slice_key]["precision"] >= 0.90 and r[slice_key]["recall"] >= 0.90
        ]
        best = max(
            (r for r in rows if r[slice_key]["f1"] is not None),
            key=lambda r: r[slice_key]["f1"],
            default=None,
        )
        print(f"--- {slice_key} ---")
        if best is not None:
            b = best[slice_key]
            print(f"  best F1 at tau={best['threshold']:.3f}  "
                  f"P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}")
        if feasible:
            for r in feasible:
                b = r[slice_key]
                print(f"  P>=0.9 & R>=0.9 at tau={r['threshold']:.3f}  "
                      f"P={b['precision']:.4f} R={b['recall']:.4f} F1={b['f1']:.4f}")
        else:
            print(f"  no tau achieves P>=0.9 AND R>=0.9 on {slice_key}")


if __name__ == "__main__":
    main()
