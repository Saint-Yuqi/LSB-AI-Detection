#!/usr/bin/env python3
"""
Sweep SAM2 AutoMask configurations and rank by proxy metrics.

Usage:
    python scripts/sweep_automask_configs.py \
        --image_root data/02_processed/sam2_prepared/img_folder \
        --configs_yaml configs/automask_sweep.yaml \
        --output_dir outputs/automask_sweep \
        [--subset_yaml configs/sweep_subset.yaml] \
        [--overlay_n 5] \
        [--stats_json outputs/mask_stats/mask_stats_summary.json]

Pipeline:
    AutoMask → metrics(cheap) → grouping → selection → metrics(hull) → prior → core → overlay

Env:
    CUDA, PyTorch with bf16 support.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.sam2_automask_runner import AutoMaskRunner
from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
from src.postprocess.core_exclusion_filter import CoreExclusionFilter
from src.postprocess.candidate_grouping import group_by_centroid
from src.postprocess.representative_selection import select_representatives, load_area_target
from src.analysis.mask_metrics import append_metrics_to_masks
from src.analysis.sweep_scoring import aggregate_and_rank
from src.visualization.overlay import save_overlay


def _load_yaml(path: Path) -> dict | list:
    """Minimal YAML loader (requires pyyaml)."""
    import yaml
    with open(path) as f:
        return yaml.safe_load(f)


def _collect_images(image_root: Path, subset_yaml: Path | None) -> list[Path]:
    """Collect image folders matching *_SB32_streams or from subset yaml."""
    if subset_yaml and subset_yaml.exists():
        data = _load_yaml(subset_yaml)
        folders = data.get("folders", [])
        return [image_root / f for f in folders if (image_root / f).exists()]

    # Default: *_SB32_streams
    return sorted(image_root.glob("*_SB32_streams"))


def _parse_folder_name(name: str) -> tuple[str, str] | None:
    """Parse galaxy_id and orientation from folder name."""
    import re
    m = re.match(r"^(\d+)_(eo|fo)_SB[\d.]+_\w+$", name)
    if m:
        return m.group(1), m.group(2)
    return None


def _write_config_snapshot(
    config_dir: Path,
    gen_config: dict,
    filter_cfg: dict,
    core_cfg: dict,
    grouping_cfg: dict,
    selection_cfg: dict,
    checkpoint: str,
):
    """Save reproducibility snapshot."""
    snapshot = {
        "generator": gen_config,
        "prior_filter": filter_cfg,
        "core_exclusion": core_cfg,
        "grouping": grouping_cfg,
        "selection": selection_cfg,
        "checkpoint": checkpoint,
        "checkpoint_md5": _md5_first_1mb(checkpoint),
    }
    (config_dir / "config_used.json").write_text(json.dumps(snapshot, indent=2))


def _md5_first_1mb(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return hashlib.md5(f.read(1024 * 1024)).hexdigest()
    except Exception:
        return ""


def run_sweep(
    image_root: Path,
    configs: list[dict[str, Any]],
    output_dir: Path,
    subset_yaml: Path | None,
    stats_json: Path,
    overlay_n: int = 5,
):
    """Main sweep loop with grouping/selection pipeline."""
    image_folders = _collect_images(image_root, subset_yaml)
    if not image_folders:
        print("No images found!")
        return

    print(f"Found {len(image_folders)} image folders.")

    # Determine overlay samples (same across all configs)
    overlay_indices = set(range(0, len(image_folders), max(1, len(image_folders) // overlay_n)))

    # Runner (loads model once)
    runner = AutoMaskRunner()

    # Warm-up on first image
    first_img_path = image_folders[0] / "0000.png"
    first_img = np.array(Image.open(first_img_path).convert("RGB"))
    runner.warmup(first_img, n=2)

    # Load area_target from stats_json
    area_target = load_area_target(stats_json)
    print(f"Loaded area_target={area_target:.1f} from {stats_json}")

    for cfg_entry in configs:
        cfg_name = cfg_entry.get("name", "default")
        gen_config: dict = cfg_entry.get("generator", {})
        radius_frac = cfg_entry.get("core_radius_frac", 0.08)

        # Grouping config
        grouping_cfg = cfg_entry.get("grouping", {})
        centroid_dist_px = grouping_cfg.get("centroid_dist_px", 15.0)

        # Selection config (inject area_target)
        selection_cfg = cfg_entry.get("selection", {})
        if "area_target" not in selection_cfg:
            selection_cfg["area_target"] = area_target

        # Prior config
        prior_cfg_entry = cfg_entry.get("prior", {})
        ambiguous_factor = prior_cfg_entry.get("ambiguous_factor", 0.25)

        print(f"\n=== Config: {cfg_name} ===")

        config_dir = output_dir / cfg_name
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "overlays").mkdir(exist_ok=True)

        # Load prior filter config from stats
        prior_cfg = load_filter_cfg(stats_json)
        prior_cfg["ambiguous_factor"] = ambiguous_factor
        prior_cfg["core_radius_frac"] = radius_frac

        prior_flt = SatellitePriorFilter(prior_cfg)
        core_flt = CoreExclusionFilter(radius_frac=radius_frac)

        # Snapshot
        _write_config_snapshot(
            config_dir, gen_config, prior_cfg,
            {"radius_frac": radius_frac},
            grouping_cfg, selection_cfg,
            runner.checkpoint.as_posix()
        )

        per_image_rows: list[dict[str, Any]] = []

        for idx, folder in enumerate(image_folders):
            img_path = folder / "0000.png"
            if not img_path.exists():
                continue
            image = np.array(Image.open(img_path).convert("RGB"))
            H, W = image.shape[:2]

            parsed = _parse_folder_name(folder.name)
            gal_id, ori = parsed if parsed else (folder.name, "unk")

            # === PIPELINE ===

            # 1. Run AutoMask
            masks, time_ms = runner.run(image, gen_config)
            n_raw = len(masks)

            if n_raw == 0:
                per_image_rows.append({
                    "galaxy_id": gal_id,
                    "orientation": ori,
                    "N_raw": 0,
                    "N_groups": 0,
                    "N_dup_rejected": 0,
                    "dup_reject_rate": 0.0,
                    "N_reps": 0,
                    "N_keep_prior": 0,
                    "N_ambiguous": 0,
                    "ambiguous_rate": 0.0,
                    "N_core_rejected": 0,
                    "N_keep_final": 0,
                    "time_ms": round(time_ms, 2),
                    "mean_stability_score_kept": 0.0,
                    # core_diag keys (must match non-empty rows)
                    "R_exclude": None,
                    "dist_p05": None,
                    "dist_p50": None,
                    "dist_p95": None,
                    "core_area_min": None,
                    "core_area_max": None,
                    "core_area_mean": None,
                    "core_solidity_mean": None,
                })
                continue

            # 2. Cheap metrics (no hull)
            append_metrics_to_masks(masks, H, W, compute_hull=False)

            # 3. Grouping
            group_by_centroid(masks, dist_px=centroid_dist_px)
            n_groups = len(set(m.get("group_id", 0) for m in masks))

            # 4. Representative selection (no solidity in score)
            reps, dups = select_representatives(masks, selection_cfg)
            n_dup_rejected = len(dups)
            dup_reject_rate = n_dup_rejected / n_raw if n_raw > 0 else 0.0

            # 5. Hull metrics for reps only
            append_metrics_to_masks(reps, H, W, compute_hull=True)

            # 6. Prior filter (uses solidity now)
            kept_prior, rej_prior, ambig = prior_flt.filter(reps)
            n_keep_prior = len(kept_prior)
            n_ambiguous = len(ambig)
            n_reps = len(reps)
            ambiguous_rate = n_ambiguous / n_reps if n_reps > 0 else 0.0

            # 7. Core exclusion
            kept_final, core_hits, core_diag = core_flt.filter(kept_prior, H, W)
            n_keep_final = len(kept_final)
            n_core_rej = len(core_hits)

            # Mean stability score of kept masks
            stab_scores = [m.get("stability_score", 0) for m in kept_final]
            mean_stab = float(np.mean(stab_scores)) if stab_scores else 0.0

            per_image_rows.append({
                "galaxy_id": gal_id,
                "orientation": ori,
                "N_raw": n_raw,
                "N_groups": n_groups,
                "N_dup_rejected": n_dup_rejected,
                "dup_reject_rate": round(dup_reject_rate, 4),
                "N_reps": n_reps,
                "N_keep_prior": n_keep_prior,
                "N_ambiguous": n_ambiguous,
                "ambiguous_rate": round(ambiguous_rate, 4),
                "N_core_rejected": n_core_rej,
                "N_keep_final": n_keep_final,
                "time_ms": round(time_ms, 2),
                "mean_stability_score_kept": round(mean_stab, 4),
                **{k: round(v, 4) if isinstance(v, float) else v for k, v in core_diag.items()},
            })

            # Overlay (subset)
            if idx in overlay_indices:
                ov_path = config_dir / "overlays" / f"{gal_id}_{ori}_overlay.png"
                save_overlay(
                    image, kept_final,
                    core_rejected=core_hits,
                    prior_rejected=rej_prior,
                    duplicate_rejected=dups,
                    ambiguous=ambig,
                    out_path=ov_path,
                    draw_prior=True,
                    draw_duplicate=True,
                    draw_ambiguous=True,
                )

            if (idx + 1) % 20 == 0:
                print(f"  Processed {idx + 1}/{len(image_folders)}")

        # Write per_image_metrics.csv
        csv_path = config_dir / "per_image_metrics.csv"
        if per_image_rows:
            fieldnames = list(per_image_rows[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(per_image_rows)

    # Aggregate & rank
    ranking = aggregate_and_rank(output_dir)
    print("\n=== Ranking (by score ascending) ===")
    for i, r in enumerate(ranking[:5]):
        print(f"  {i+1}. {r['config_name']} | score={r['score']} | N_keep={r['mean_N_keep_final']} | core_rate={r['mean_core_rate']}")


def main():
    parser = argparse.ArgumentParser(description="Sweep AutoMask configs")
    parser.add_argument("--image_root", type=Path, default=PROJECT_ROOT / "data/02_processed/sam2_prepared/img_folder")
    parser.add_argument("--configs_yaml", type=Path, default=PROJECT_ROOT / "configs/automask_sweep.yaml")
    parser.add_argument("--output_dir", type=Path, default=PROJECT_ROOT / "outputs/automask_sweep")
    parser.add_argument("--subset_yaml", type=Path, default=None)
    parser.add_argument("--stats_json", type=Path, default=PROJECT_ROOT / "outputs/mask_stats/mask_stats_summary.json")
    parser.add_argument("--overlay_n", type=int, default=5, help="Number of overlay samples per config")
    args = parser.parse_args()

    configs = _load_yaml(args.configs_yaml)
    if isinstance(configs, dict):
        configs = configs.get("configs", [])

    run_sweep(args.image_root, configs, args.output_dir, args.subset_yaml, args.stats_json, args.overlay_n)


if __name__ == "__main__":
    main()
