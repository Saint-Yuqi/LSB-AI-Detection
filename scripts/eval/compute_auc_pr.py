#!/usr/bin/env python3
"""Compute AUC-PR (Average Precision) for raw satellite predictions.

Reads existing eval output:
    <eval_dir>/<base_key>/diagnostics.json   (per_candidate rows)
    <eval_dir>/report.json                   (per_sample.num_gt_satellites)

Each prediction is treated as TP iff its taxonomy_label is matched
(``compact_complete`` or ``diffuse_core``); the diagnostic taxonomy already
enforces one-to-one GT assignment, so each TP corresponds to a unique GT.
Missed GTs (FN) are appended as phantom (y_true=1, y_score=-inf) entries so
the recall denominator equals total N_GT (not N_TP).

Two slices: full_frame and roi (filter by ``intersects_roi``).
Two aggregations:
  - micro: pool all predictions across samples, single PR sweep.
  - macro: per-sample AP, then mean.

Also writes the (precision, recall) curve points so downstream plotting can
re-render without recomputing.

Usage:
    python scripts/eval/compute_auc_pr.py \\
        --eval-dir outputs/eval_checkpoint_no_core_policy/fbox_gold_satellites/current/linear_magnitude
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve

MATCHED_LABELS = ("compact_complete", "diffuse_core")


def _collect_rows(diag_path: Path, *, roi: bool) -> tuple[list[float], list[int]]:
    """Return (scores, is_tp) for one sample, optionally restricted to ROI."""
    doc = json.loads(diag_path.read_text())
    rows = doc.get("per_candidate", [])
    scores: list[float] = []
    is_tp: list[int] = []
    for r in rows:
        if roi and not r.get("intersects_roi", False):
            continue
        scores.append(float(r["confidence_score"]))
        is_tp.append(1 if r["taxonomy_label"] in MATCHED_LABELS else 0)
    return scores, is_tp


_PHANTOM_SCORE = -1.0  # sentinel below any real confidence in [0, 1]


def _pad_with_phantom_fn(
    scores: list[float], is_tp: list[int], n_gt: int
) -> tuple[np.ndarray, np.ndarray]:
    """Append (y_true=1, y_score=-1) rows so sum(y_true) == n_gt.

    Confidence scores are in [0, 1]; -1 is below all of them so phantom rows
    sort to the bottom of the PR sweep and only contribute to the recall
    denominator (representing missed GTs as un-recoverable FN).
    """
    n_tp = sum(is_tp)
    n_fn = max(n_gt - n_tp, 0)
    if n_fn > 0:
        scores = scores + [_PHANTOM_SCORE] * n_fn
        is_tp = is_tp + [1] * n_fn
    return np.asarray(is_tp, dtype=np.int8), np.asarray(scores, dtype=np.float64)


def _ap_safe(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    if y_true.size == 0 or y_true.sum() == 0:
        return None
    return float(average_precision_score(y_true, y_score))


def _curve_safe(
    y_true: np.ndarray, y_score: np.ndarray
) -> dict[str, list] | None:
    if y_true.size == 0 or y_true.sum() == 0:
        return None
    p, r, t = precision_recall_curve(y_true, y_score)
    return {
        "precision": [float(x) for x in p],
        "recall": [float(x) for x in r],
        # thresholds has len(p)-1; last slot is the +inf PR sweep pad (JSON null).
        "thresholds": [float(x) for x in t] + [None],
    }


def compute_slice(
    eval_dir: Path,
    samples: list[dict],
    *,
    roi: bool,
) -> dict:
    """Compute AUC-PR for one slice (full_frame or roi)."""
    pooled_scores: list[float] = []
    pooled_tp: list[int] = []
    pooled_n_gt = 0
    per_sample_aps: list[float] = []
    n_samples_used = 0
    n_samples_skipped = 0

    for s in samples:
        bk = s["base_key"]
        n_gt = int(s.get("num_gt_satellites", 0))
        if roi and n_gt == 0:
            # Sample with no GT in ROI contributes only FPs to micro; skip macro.
            pass
        diag_path = eval_dir / bk / "diagnostics.json"
        if not diag_path.is_file():
            n_samples_skipped += 1
            continue
        scores, is_tp = _collect_rows(diag_path, roi=roi)

        # Per-sample AP (macro)
        if n_gt > 0:
            yt_s, ys_s = _pad_with_phantom_fn(scores, is_tp, n_gt)
            ap_s = _ap_safe(yt_s, ys_s)
            if ap_s is not None:
                per_sample_aps.append(ap_s)

        # Pool for micro
        pooled_scores.extend(scores)
        pooled_tp.extend(is_tp)
        pooled_n_gt += n_gt
        n_samples_used += 1

    yt, ys = _pad_with_phantom_fn(pooled_scores, pooled_tp, pooled_n_gt)
    micro_ap = _ap_safe(yt, ys)
    micro_curve = _curve_safe(yt, ys)
    macro_ap = mean(per_sample_aps) if per_sample_aps else None

    return {
        "micro_auc_pr": micro_ap,
        "macro_auc_pr": macro_ap,
        "n_samples_used": n_samples_used,
        "n_samples_skipped": n_samples_skipped,
        "n_samples_with_gt": len(per_sample_aps),
        "n_predictions_pooled": len(pooled_scores),
        "n_tp_pooled": int(sum(pooled_tp)),
        "n_fp_pooled": int(len(pooled_tp) - sum(pooled_tp)),
        "n_gt_pooled": int(pooled_n_gt),
        "n_fn_pooled": int(pooled_n_gt - sum(pooled_tp)),
        "pr_curve": micro_curve,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval-dir", type=Path, required=True,
                    help="Eval output dir containing report.json and per-sample subdirs")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output JSON path (default: <eval-dir>/auc_pr.json)")
    args = ap.parse_args()

    eval_dir = args.eval_dir.resolve()
    report_path = eval_dir / "report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"report.json not found in {eval_dir}")

    report = json.loads(report_path.read_text())
    samples = report.get("per_sample", [])
    if not samples:
        raise ValueError(f"no per_sample entries in {report_path}")

    result = {
        "eval_dir": str(eval_dir),
        "benchmark_mode": report.get("benchmark_mode"),
        "n_samples": len(samples),
        "matched_labels": list(MATCHED_LABELS),
        "full_frame": compute_slice(eval_dir, samples, roi=False),
        "roi": compute_slice(eval_dir, samples, roi=True),
    }

    out_path = args.out or (eval_dir / "auc_pr.json")
    out_path.write_text(json.dumps(result, indent=2))

    print(f"wrote {out_path}")
    for slc in ("full_frame", "roi"):
        s = result[slc]
        micro_s = f"{s['micro_auc_pr']:.4f}" if s["micro_auc_pr"] is not None else "n/a"
        macro_s = f"{s['macro_auc_pr']:.4f}" if s["macro_auc_pr"] is not None else "n/a"
        print(
            f"  {slc:10s}  micro AUC-PR = {micro_s}   macro AUC-PR = {macro_s}   "
            f"(N_pred={s['n_predictions_pooled']}, N_TP={s['n_tp_pooled']}, "
            f"N_FP={s['n_fp_pooled']}, N_GT={s['n_gt_pooled']})"
        )


if __name__ == "__main__":
    main()
