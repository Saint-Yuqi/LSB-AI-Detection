"""
Visualize SAM3 evaluation metrics across SNR tiers — type-aware.

Reads eval_results_*.json (latest by timestamp) from each results directory
and produces comparison charts for streams, satellites, and combined metrics.

Usage:
    python scripts/visualize_eval_metrics.py \
        --results-dirs outputs/eval_sam3 outputs/eval_sam3_snr05 \
                       outputs/eval_sam3_snr10 outputs/eval_sam3_snr20 outputs/eval_sam3_snr50 \
        --labels clean snr05 snr10 snr20 snr50 \
        --output-dir outputs/eval_sam3_comparison

Output:
    {output_dir}/metrics_bar_{type}_{layer}.png  — [0,1] metrics grouped bars
    {output_dir}/hd95_bar_{type}_{layer}.png     — HD95 separate scale
    {output_dir}/degradation_curves.png          — Per-type metric vs SNR
    {output_dir}/summary.csv                     — Tabular summary
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Metrics with [0,1] scale
_NORM_METRICS = [
    ("macro_mean_dice", "Dice"),
    ("macro_mean_precision", "Precision"),
    ("macro_mean_recall_pixel", "Recall"),
    ("macro_mean_matched_iou", "Matched IoU"),
    ("macro_mean_instance_recall", "Instance Recall"),
]

# SNR value for ordering in line plots (clean → ∞)
_SNR_MAP = {"clean": float("inf"), "snr05": 5, "snr10": 10, "snr20": 20, "snr50": 50}

_TYPE_KEYS = ("streams", "satellites", "combined")


def _load_latest_results(results_dir: Path) -> dict:
    """Load the most recent eval_results_*.json from a directory."""
    pattern = str(results_dir / "eval_results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No eval_results_*.json in {results_dir}")
    with open(files[-1]) as f:
        return json.load(f)


def _extract_all_metrics(doc: dict) -> dict[str, dict[str, dict]]:
    """
    Extract metrics from both new and legacy JSON schemas.

    Returns: {type_key: {layer: {metric_key: value}}}
    """
    overall = doc["summary"]["overall"]
    if "streams" in overall:
        # New schema: overall.{streams,satellites,combined}.{raw,post}
        return {
            tk: {
                ly: overall.get(tk, {}).get(ly, {})
                for ly in ("raw", "post")
            }
            for tk in _TYPE_KEYS
        }
    else:
        # Legacy schema: overall.{raw,post} → streams only
        # combined left EMPTY — legacy was streams-only
        return {
            "streams": {ly: overall.get(ly, {}) for ly in ("raw", "post")},
            "satellites": {ly: {} for ly in ("raw", "post")},
            "combined": {ly: {} for ly in ("raw", "post")},
        }


def _get_metric(layer_data: dict, key: str) -> float:
    """Get metric value, return np.nan for missing (not 0.0)."""
    v = layer_data.get(key)
    if v is None:
        return np.nan
    return float(v)


def plot_grouped_bars(
    labels: list[str],
    all_layer_data: list[dict],
    output_path: Path,
    type_key: str,
    layer: str,
) -> None:
    """Grouped bar chart for [0,1]-scale metrics."""
    metric_keys = _NORM_METRICS
    n_groups = len(labels)
    n_bars = len(metric_keys)
    x = np.arange(n_groups)
    width = 0.8 / n_bars

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 1.5), 6))

    for i, (key, display) in enumerate(metric_keys):
        values = [_get_metric(m, key) for m in all_layer_data]
        bars = ax.bar(x + i * width, values, width, label=display, zorder=3)
        for bar, v in zip(bars, values):
            if not np.isnan(v) and v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("SNR Tier", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title(f"{type_key.upper()} — {layer.upper()} layer", fontsize=14, fontweight="bold")
    ax.set_xticks(x + width * (n_bars - 1) / 2)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_hd95_bar(
    labels: list[str],
    all_layer_data: list[dict],
    output_path: Path,
    type_key: str,
    layer: str,
) -> None:
    """Separate bar chart for HD95 (different scale)."""
    key = "macro_mean_capped_hausdorff95"
    values = [_get_metric(m, key) for m in all_layer_data]

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 1.2), 5))
    bars = ax.bar(labels, values, color="#e74c3c", alpha=0.85, zorder=3)
    for bar, v in zip(bars, values):
        if not np.isnan(v) and v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("SNR Tier", fontsize=12)
    ax.set_ylabel("Hausdorff95 (pixels)", fontsize=12)
    ax.set_title(f"HD95 — {type_key.upper()} / {layer.upper()}", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3, zorder=0)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_degradation_curves(
    labels: list[str],
    all_metrics: dict[str, list[dict]],
    output_path: Path,
) -> None:
    """
    Line plot: metrics vs SNR for each type (streams/satellites/combined),
    raw layer only.
    """
    # Sort by SNR value (ascending)
    snr_values = [_SNR_MAP.get(l, 0) for l in labels]
    order = np.argsort(snr_values)
    sorted_labels = [labels[i] for i in order]

    active_types = [tk for tk in _TYPE_KEYS if any(
        all_metrics[tk][i] for i in range(len(labels))
    )]

    n_types = len(active_types)
    if n_types == 0:
        return

    fig, axes = plt.subplots(1, n_types, figsize=(7 * n_types, 6), sharey=True,
                             squeeze=False)

    for ax_idx, type_key in enumerate(active_types):
        ax = axes[0][ax_idx]
        sorted_data = [all_metrics[type_key][i] for i in order]
        x_pos = np.arange(len(sorted_labels))

        for key, display in _NORM_METRICS:
            values = [_get_metric(m, key) for m in sorted_data]
            ax.plot(x_pos, values, "o-", label=display, linewidth=2, markersize=6)

        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_labels, fontsize=10)
        ax.set_xlabel("SNR Tier (ascending noise →)", fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"{type_key.upper()} (raw)", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8, loc="lower left")
        ax.grid(alpha=0.3)

    fig.suptitle("Metric Degradation vs Noise Level", fontsize=15, fontweight="bold", y=1.02)
    fig.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def save_summary_csv(
    labels: list[str],
    all_metrics: dict[str, dict[str, list[dict]]],
    output_path: Path,
) -> None:
    """Save tabular summary of all metrics."""
    all_keys = [k for k, _ in _NORM_METRICS] + ["macro_mean_capped_hausdorff95"]
    all_display = [d for _, d in _NORM_METRICS] + ["HD95"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["tier", "type", "layer"] + all_display
        writer.writerow(header)
        for label_idx, label in enumerate(labels):
            for tk in _TYPE_KEYS:
                for ly in ("raw", "post"):
                    m = all_metrics[tk].get(ly, [{}] * len(labels))[label_idx]
                    row = [label, tk, ly]
                    for k in all_keys:
                        v = _get_metric(m, k)
                        row.append(f"{v:.4f}" if not np.isnan(v) else "")
                    writer.writerow(row)
    print(f"  Saved: {output_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dirs", nargs="+", required=True,
                    help="Paths to eval output directories (one per tier)")
    ap.add_argument("--labels", nargs="+", required=True,
                    help="Labels for each tier (e.g. clean snr05 snr10 snr20 snr50)")
    ap.add_argument("--output-dir", default="outputs/eval_sam3_comparison",
                    help="Output directory for charts")
    args = ap.parse_args()

    if len(args.results_dirs) != len(args.labels):
        print("ERROR: --results-dirs and --labels must have same length")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load all results, organized by type → layer → [per-tier data]
    # Structure: {type_key: {layer: [layer_data_per_tier]}}
    all_metrics: dict[str, dict[str, list[dict]]] = {
        tk: {ly: [] for ly in ("raw", "post")}
        for tk in _TYPE_KEYS
    }

    for rdir, label in zip(args.results_dirs, args.labels):
        rdir = Path(rdir)
        try:
            doc = _load_latest_results(rdir)
        except FileNotFoundError as e:
            print(f"WARNING: {e} — skipping '{label}'")
            for tk in _TYPE_KEYS:
                for ly in ("raw", "post"):
                    all_metrics[tk][ly].append({})
            continue

        extracted = _extract_all_metrics(doc)
        for tk in _TYPE_KEYS:
            for ly in ("raw", "post"):
                all_metrics[tk][ly].append(extracted.get(tk, {}).get(ly, {}))
        print(f"  Loaded: {label} from {rdir}")

    # --- Generate charts ---
    # Per-type bar charts (raw layer only to reduce chart count)
    for tk in _TYPE_KEYS:
        raw_data = all_metrics[tk]["raw"]
        # Skip if all tiers have empty data for this type
        if all(not d for d in raw_data):
            print(f"  SKIP {tk}: no data across any tier")
            continue
        plot_grouped_bars(args.labels, raw_data,
                          out_dir / f"metrics_bar_{tk}_raw.png", tk, "raw")
        plot_hd95_bar(args.labels, raw_data,
                      out_dir / f"hd95_bar_{tk}_raw.png", tk, "raw")

    # Degradation curves (all types, raw layer)
    raw_per_type = {tk: all_metrics[tk]["raw"] for tk in _TYPE_KEYS}
    plot_degradation_curves(args.labels, raw_per_type,
                            out_dir / "degradation_curves.png")

    # Summary CSV
    save_summary_csv(args.labels, all_metrics, out_dir / "summary.csv")

    print(f"\nAll charts saved to {out_dir}/")


if __name__ == "__main__":
    main()
