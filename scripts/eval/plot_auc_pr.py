#!/usr/bin/env python3
"""Plot the precision-recall curve from auc_pr.json.

Renders one figure with two curves (full_frame and roi), each annotated with
its micro/macro AUC-PR value. Saves as PNG next to auc_pr.json.

Usage:
    python scripts/eval/plot_auc_pr.py \\
        --auc-pr-json outputs/.../satellite_tradeoff/auc_pr.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def _plot_slice(ax, slice_data: dict, label: str, color: str) -> None:
    curve = slice_data.get("pr_curve")
    if curve is None:
        return
    p = curve["precision"]
    r = curve["recall"]
    micro = slice_data["micro_auc_pr"]
    macro = slice_data["macro_auc_pr"]
    micro_s = f"{micro:.4f}" if micro is not None else "n/a"
    macro_s = f"{macro:.4f}" if macro is not None else "n/a"
    legend = (
        f"{label}  micro AP={micro_s}  macro AP={macro_s}  "
        f"(N_TP={slice_data['n_tp_pooled']}, "
        f"N_FP={slice_data['n_fp_pooled']}, "
        f"N_GT={slice_data['n_gt_pooled']})"
    )
    ax.plot(r, p, color=color, linewidth=1.6, label=legend)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--auc-pr-json", type=Path, required=True,
                    help="Path to auc_pr.json produced by compute_auc_pr.py")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output PNG (default: <dir>/auc_pr_curve.png)")
    ap.add_argument("--title", type=str, default=None,
                    help="Figure title (default: derived from eval_dir)")
    args = ap.parse_args()

    data = json.loads(args.auc_pr_json.read_text())

    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    _plot_slice(ax, data["full_frame"], "full_frame", "#1f77b4")
    _plot_slice(ax, data["roi"], "roi", "#d62728")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0.0, 1.02)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.85)

    title = args.title or (
        f"Satellite raw-prediction PR curve\n"
        f"{data.get('benchmark_mode', '')} (n_samples={data.get('n_samples', 0)})"
    )
    ax.set_title(title, fontsize=11)

    out = args.out or (args.auc_pr_json.parent / "auc_pr_curve.png")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
