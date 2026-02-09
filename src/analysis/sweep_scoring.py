"""
Sweep scoring – aggregate per-image CSVs → summary.json + config ranking.

Usage:
    from src.analysis.sweep_scoring import aggregate_and_rank
    ranking = aggregate_and_rank(output_root)   # writes ranking.json

Metrics in summary.json per config:
    mean_time_ms, time_p50, time_p95
    mean_N_raw, mean_N_keep_prior, mean_N_keep_final
    mean_core_rate
    CV_N_keep_final (std / mean)
    mean_stability_score_of_kept
    score (composite)
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _load_csv(csv_path: Path) -> list[dict[str, str]]:
    """Minimal CSV reader (no pandas dependency)."""
    with open(csv_path) as f:
        lines = f.read().strip().split("\n")
    if len(lines) < 2:
        return []
    header = lines[0].split(",")
    rows = []
    for line in lines[1:]:
        vals = line.split(",")
        rows.append({k: v for k, v in zip(header, vals)})
    return rows


def summarise_config(config_dir: Path) -> dict[str, Any]:
    """Read per_image_metrics.csv and compute aggregated summary."""
    csv_path = config_dir / "per_image_metrics.csv"
    if not csv_path.exists():
        return {}

    rows = _load_csv(csv_path)
    if not rows:
        return {}

    # Required columns
    times = np.array([float(r.get("time_ms", 0)) for r in rows])
    n_raw = np.array([int(r.get("N_raw", 0)) for r in rows])
    n_keep_prior = np.array([int(r.get("N_keep_prior", 0)) for r in rows])
    n_keep_final = np.array([int(r.get("N_keep_final", 0)) for r in rows])
    n_core_rej = np.array([int(r.get("N_core_rejected", 0)) for r in rows])

    # Optional stability score (mean of kept masks)
    stab_scores = []
    for r in rows:
        v = r.get("mean_stability_score_kept", "")
        if v:
            stab_scores.append(float(v))
    mean_stab = float(np.mean(stab_scores)) if stab_scores else None

    # Core rate = core_rejected / N_raw
    core_rates = np.where(n_raw > 0, n_core_rej / n_raw, 0.0)

    # CV = std / mean (coefficient of variation)
    mean_keep = float(np.mean(n_keep_final))
    std_keep = float(np.std(n_keep_final))
    cv_keep = std_keep / mean_keep if mean_keep > 0 else 0.0

    summary: dict[str, Any] = {
        "n_images": len(rows),
        "mean_time_ms": round(float(np.mean(times)), 2),
        "time_p50": round(float(np.percentile(times, 50)), 2),
        "time_p95": round(float(np.percentile(times, 95)), 2),
        "mean_N_raw": round(float(np.mean(n_raw)), 2),
        "mean_N_keep_prior": round(float(np.mean(n_keep_prior)), 2),
        "mean_N_keep_final": round(mean_keep, 2),
        "std_N_keep_final": round(std_keep, 2),
        "CV_N_keep_final": round(cv_keep, 4),
        "mean_core_rate": round(float(np.mean(core_rates)), 4),
        "mean_stability_score_of_kept": round(mean_stab, 4) if mean_stab else None,
    }

    # Composite score (lower is better) --------------------------------------
    # Penalise high core rate and high CV. Reward moderate N_keep.
    # score = a*time + b*core_rate + c*CV - d*N_keep
    a_time, b_core, c_cv, d_keep = 0.01, 10.0, 5.0, 0.05
    score = (
        a_time * summary["mean_time_ms"]
        + b_core * summary["mean_core_rate"]
        + c_cv * summary["CV_N_keep_final"]
        - d_keep * summary["mean_N_keep_final"]
    )
    summary["score"] = round(score, 4)

    return summary


def aggregate_and_rank(output_root: Path) -> list[dict[str, Any]]:
    """
    Walk config subdirs under output_root, compute summary.json for each,
    then rank by score (ascending) and write ranking.json.

    Returns sorted list.
    """
    output_root = Path(output_root)
    summaries: list[dict[str, Any]] = []

    for config_dir in sorted(output_root.iterdir()):
        if not config_dir.is_dir():
            continue
        summary = summarise_config(config_dir)
        if not summary:
            continue
        summary["config_name"] = config_dir.name
        # Write per-config summary.json
        (config_dir / "summary.json").write_text(json.dumps(summary, indent=2))
        summaries.append(summary)

    # Sort by score ascending (lower = better)
    summaries.sort(key=lambda x: x.get("score", 1e9))

    # Write ranking.json
    (output_root / "ranking.json").write_text(json.dumps(summaries, indent=2))
    return summaries
