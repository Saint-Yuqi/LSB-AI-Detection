#!/usr/bin/env python3
"""Build a (confidence_threshold, precision, recall, F1) table from auc_pr.json.

Two outputs:
  - CSV with every distinct threshold seen in the PR curve (full table).
  - Console summary: best F1, best threshold meeting both
    precision >= --min-precision and recall >= --min-recall, plus a
    coarse-grained threshold grid for quick reading.

Usage:
    python scripts/eval/auc_pr_threshold_table.py \\
        --auc-pr-json outputs/.../satellite_tradeoff/auc_pr.json \\
        --slice roi --min-precision 0.9 --min-recall 0.9
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _f1(p: float, r: float) -> float:
    if p + r == 0.0:
        return 0.0
    return 2.0 * p * r / (p + r)


def _build_rows(curve: dict) -> list[dict]:
    """Each curve point i (precision[i], recall[i]) corresponds to the threshold
    'keep predictions with score >= thresholds[i]'. The last padded value
    (null in JSON, +inf in sklearn) is the no-predictions corner where precision is undefined
    by sklearn convention; skip it.
    """
    p = curve["precision"]
    r = curve["recall"]
    t = curve["thresholds"]
    rows: list[dict] = []
    for i in range(len(p)):
        thr = t[i]
        if thr is None or thr == float("inf"):
            continue
        rows.append({
            "confidence_threshold": float(thr),
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": _f1(float(p[i]), float(r[i])),
        })
    rows.sort(key=lambda x: x["confidence_threshold"])
    return rows


def _print_grid(rows: list[dict], grid: list[float]) -> None:
    """Print precision/recall/F1 at the smallest threshold >= each grid point."""
    print(f"  {'threshold':>10}  {'precision':>10}  {'recall':>10}  {'F1':>10}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}")
    for g in grid:
        # Find first row with threshold >= g
        match = next((r for r in rows if r["confidence_threshold"] >= g), None)
        if match is None:
            print(f"  {g:>10.3f}  {'n/a':>10}  {'n/a':>10}  {'n/a':>10}")
            continue
        print(
            f"  {match['confidence_threshold']:>10.4f}  "
            f"{match['precision']:>10.4f}  "
            f"{match['recall']:>10.4f}  "
            f"{match['f1']:>10.4f}"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auc-pr-json", type=Path, required=True)
    ap.add_argument("--slice", choices=("full_frame", "roi"), default="roi",
                    help="Which slice to analyse (default: roi)")
    ap.add_argument("--out-csv", type=Path, default=None,
                    help="Output CSV (default: <dir>/threshold_table_<slice>.csv)")
    ap.add_argument("--min-precision", type=float, default=0.90)
    ap.add_argument("--min-recall", type=float, default=0.90)
    args = ap.parse_args()

    data = json.loads(args.auc_pr_json.read_text())
    slice_data = data[args.slice]
    curve = slice_data.get("pr_curve")
    if curve is None:
        raise ValueError(f"slice {args.slice} has no pr_curve")

    rows = _build_rows(curve)

    out_csv = args.out_csv or (
        args.auc_pr_json.parent / f"threshold_table_{args.slice}.csv"
    )
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["confidence_threshold", "precision", "recall", "f1"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"slice = {args.slice}")
    print(f"micro AUC-PR = {slice_data['micro_auc_pr']:.4f}")
    print(
        f"N_pred={slice_data['n_predictions_pooled']}, "
        f"N_TP={slice_data['n_tp_pooled']}, "
        f"N_FP={slice_data['n_fp_pooled']}, "
        f"N_GT={slice_data['n_gt_pooled']}"
    )
    print(f"wrote full table -> {out_csv}  ({len(rows)} rows)")
    print()

    print("Coarse grid (smallest threshold >= grid value):")
    _print_grid(rows, [0.18, 0.30, 0.40, 0.50, 0.55, 0.60,
                       0.65, 0.70, 0.75, 0.80, 0.85, 0.90])
    print()

    # Best F1 overall
    best_f1 = max(rows, key=lambda r: r["f1"])
    print(
        f"Best F1: threshold={best_f1['confidence_threshold']:.4f}  "
        f"P={best_f1['precision']:.4f}  R={best_f1['recall']:.4f}  "
        f"F1={best_f1['f1']:.4f}"
    )

    # Lowest threshold that meets both gates (gives highest recall under both gates)
    feasible = [
        r for r in rows
        if r["precision"] >= args.min_precision and r["recall"] >= args.min_recall
    ]
    if not feasible:
        print(
            f"\nNo threshold satisfies precision >= {args.min_precision:.2f} "
            f"AND recall >= {args.min_recall:.2f} simultaneously on slice "
            f"'{args.slice}'."
        )
        # Closest by F1 above one or the other
        max_p = max(rows, key=lambda r: (r["precision"], r["recall"]))
        max_r = max(rows, key=lambda r: (r["recall"], r["precision"]))
        print(
            f"  best precision = {max_p['precision']:.4f} "
            f"(R={max_p['recall']:.4f}, thr={max_p['confidence_threshold']:.4f})"
        )
        print(
            f"  best recall    = {max_r['recall']:.4f} "
            f"(P={max_r['precision']:.4f}, thr={max_r['confidence_threshold']:.4f})"
        )
        return

    # Lowest threshold maximises recall while still satisfying both gates;
    # highest threshold maximises precision.
    feasible_sorted = sorted(feasible, key=lambda r: r["confidence_threshold"])
    lo, hi = feasible_sorted[0], feasible_sorted[-1]
    best_in_window = max(feasible, key=lambda r: r["f1"])

    print(
        f"\nThresholds satisfying P >= {args.min_precision:.2f} AND "
        f"R >= {args.min_recall:.2f}:  {len(feasible)} rows"
    )
    print(
        f"  lowest threshold:  {lo['confidence_threshold']:.4f}  "
        f"P={lo['precision']:.4f}  R={lo['recall']:.4f}  F1={lo['f1']:.4f}"
    )
    print(
        f"  highest threshold: {hi['confidence_threshold']:.4f}  "
        f"P={hi['precision']:.4f}  R={hi['recall']:.4f}  F1={hi['f1']:.4f}"
    )
    print(
        f"  best F1 in window: {best_in_window['confidence_threshold']:.4f}  "
        f"P={best_in_window['precision']:.4f}  R={best_in_window['recall']:.4f}  "
        f"F1={best_in_window['f1']:.4f}"
    )


if __name__ == "__main__":
    main()
