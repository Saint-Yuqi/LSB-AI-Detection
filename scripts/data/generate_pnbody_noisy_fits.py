#!/usr/bin/env python3
"""
Batch-inject pNbody physics noise into clean SB magnitude FITS.

Wraps the ``mockimgs_sb_add_noise`` CLI from the pNbody package, which uses
``--SB_limit`` to control noise strength via a physically correct Gaussian
noise model derived from a surface-brightness detection threshold.

Usage:
    conda run -n sam3 python scripts/data/generate_pnbody_noisy_fits.py \
        --config configs/noise_profiles_pnbody_physics.yaml
    conda run -n sam3 python scripts/data/generate_pnbody_noisy_fits.py \
        --config configs/noise_profiles_fbox_gold.yaml \
        --galaxies 11 --profiles sb31.5
    conda run -n sam3 python scripts/data/generate_pnbody_noisy_fits.py \
        --config ... --dry-run

I/O notes:
    - Input .fits.gz is read natively by the CLI (no decompression needed).
    - ``-o`` must receive a .fits path; the CLI auto-gzips to {path}.gz and
      deletes the intermediate .fits.
    - Output: {output_root}/{profile}/{gid:05d}/magnitudes-Fbox-{gid}-{view}-{SUFFIX}.fits.gz

Config schema (backward compatible with noise_profiles_pnbody_physics.yaml):
    paths.firebox_root, paths.output_root
    source.filename_suffix:  "VIS2" (default) | "VIS"
    source.sb_maps_layout:   "nested" (default; {gid:05d}/) | "flat"
    source.enumeration.mode: "grid" (default) | "manifest" | "mask_glob"
        grid:       data_selection.galaxy_ids + data_selection.views
        manifest:   source.enumeration.manifest -> dataset_manifest.json
        mask_glob:  source.enumeration.mask_eo_glob + mask_fo_glob (+ mask_name_regex)
    profiles: [{name, sb_limit}, ...]
    reproducibility.random_seed
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

_PROJECT_ROOT = Path(__file__).resolve().parents[2]

logger = logging.getLogger(__name__)

_DEFAULT_MASK_REGEX = r"ark_features-(\d+)-(eo|fo)-SBlim[\d.]+\.fits\.gz"


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve project-relative config paths against the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return _PROJECT_ROOT / path


def _resolve_noise_cli() -> list[str]:
    """Locate the ``mockimgs_sb_add_noise`` CLI."""
    installed = shutil.which("mockimgs_sb_add_noise")
    if installed:
        return [installed]

    bin_dir = Path(sys.executable).parent
    candidate = bin_dir / "mockimgs_sb_add_noise"
    if candidate.exists():
        return [str(candidate)]

    raise FileNotFoundError(
        "mockimgs_sb_add_noise not found on PATH or next to the Python interpreter"
    )


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _compute_seed(base_seed: int, gid: int, view: str, profile_name: str) -> int:
    """Deterministic per-file seed: base + sha256(gid, view, profile) mod 2^31."""
    key_str = f"{gid}_{view}_{profile_name}"
    return base_seed + int(
        hashlib.sha256(key_str.encode()).hexdigest(), 16
    ) % (2**31)


def _enumerate_pairs(cfg: dict, firebox_root: Path) -> list[tuple[int, str]]:
    """Produce (gid, view) pairs per the configured enumeration mode.

    Fails fast on unknown modes or missing mode-specific keys.
    """
    source = cfg.get("source", {}) or {}
    enumeration = source.get("enumeration", {}) or {}
    mode = enumeration.get("mode", "grid")

    if mode == "grid":
        galaxy_ids = cfg["data_selection"]["galaxy_ids"]
        views = (
            cfg["data_selection"].get("views")
            or cfg["data_selection"]["orientations"]
        )
        return [(int(g), str(v)) for g in galaxy_ids for v in views]

    if mode == "manifest":
        manifest_path = enumeration.get("manifest")
        if not manifest_path:
            raise KeyError("source.enumeration.manifest is required for mode=manifest")
        with open(_resolve_project_path(manifest_path)) as f:
            manifest = json.load(f)
        pairs: list[tuple[int, str]] = []
        for entry in manifest["samples"]:
            pairs.append((int(entry["galaxy_id"]), str(entry["view"])))
        return pairs

    if mode == "mask_glob":
        mask_regex = re.compile(enumeration.get("mask_name_regex", _DEFAULT_MASK_REGEX))
        pairs_set: set[tuple[int, str]] = set()
        for glob_key in ("mask_eo_glob", "mask_fo_glob"):
            rel_glob = enumeration.get(glob_key)
            if not rel_glob:
                raise KeyError(f"source.enumeration.{glob_key} is required for mode=mask_glob")
            for path in sorted(firebox_root.glob(rel_glob)):
                m = mask_regex.match(path.name)
                if not m:
                    continue
                pairs_set.add((int(m.group(1)), m.group(2)))
        return sorted(pairs_set)

    raise ValueError(
        f"Unknown enumeration mode: {mode!r}. Expected: grid | manifest | mask_glob"
    )


def _resolve_input_path(
    sb_dir: Path,
    gid: int,
    view: str,
    suffix: str,
    sb_maps_layout: str,
) -> Path:
    fname = f"magnitudes-Fbox-{gid}-{view}-{suffix}.fits.gz"
    if sb_maps_layout == "nested":
        return sb_dir / f"{gid:05d}" / fname
    if sb_maps_layout == "flat":
        return sb_dir / fname
    raise ValueError(f"Unknown sb_maps_layout: {sb_maps_layout!r}. Expected: nested | flat")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--config", required=True,
                    help="Path to a noise profile YAML")
    ap.add_argument("--galaxies", default=None,
                    help="Comma-separated galaxy IDs (subset)")
    ap.add_argument("--profiles", default=None,
                    help="Comma-separated profile names (subset)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print commands without executing")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = _load_config(args.config)

    firebox_root = _resolve_project_path(cfg["paths"]["firebox_root"])
    output_root = _resolve_project_path(cfg["paths"]["output_root"])
    sb_dir = firebox_root / "SB_maps"

    source = cfg.get("source", {}) or {}
    filename_suffix = source.get("filename_suffix", "VIS2")
    sb_maps_layout = source.get("sb_maps_layout", "nested")

    profiles = cfg["profiles"]
    base_seed = cfg["reproducibility"]["random_seed"]

    pairs = _enumerate_pairs(cfg, firebox_root)

    if args.galaxies:
        subset = {int(g.strip()) for g in args.galaxies.split(",")}
        pairs = [p for p in pairs if p[0] in subset]
    if args.profiles:
        subset = set(args.profiles.split(","))
        profiles = [p for p in profiles if p["name"] in subset]

    noise_cli = _resolve_noise_cli()
    logger.info("Using CLI: %s", " ".join(noise_cli))
    logger.info(
        "source: suffix=%s layout=%s enumeration=%s -> %d (gid, view) pairs",
        filename_suffix, sb_maps_layout,
        (source.get("enumeration", {}) or {}).get("mode", "grid"),
        len(pairs),
    )

    total = len(pairs) * len(profiles)
    logger.info(
        "Generating %d noisy FITS (%d samples x %d profiles)",
        total, len(pairs), len(profiles),
    )

    done = 0
    skipped = 0
    t0 = time.time()

    for prof in profiles:
        prof_name = prof["name"]
        sb_limit = float(prof["sb_limit"])

        prof_dir = output_root / prof_name
        prof_dir.mkdir(parents=True, exist_ok=True)

        for gid, view in pairs:
            src_path = _resolve_input_path(sb_dir, gid, view, filename_suffix, sb_maps_layout)

            if not src_path.exists():
                logger.warning("SKIP (missing): %s", src_path)
                continue

            # -o receives .fits; CLI auto-produces .fits.gz
            prof_galaxy_dir = prof_dir / f"{gid:05d}"
            prof_galaxy_dir.mkdir(parents=True, exist_ok=True)
            out_fname = f"magnitudes-Fbox-{gid}-{view}-{filename_suffix}.fits"
            out_stem = prof_galaxy_dir / out_fname
            out_final = Path(str(out_stem) + ".gz")

            if out_final.exists():
                skipped += 1
                done += 1
                continue

            file_seed = _compute_seed(base_seed, gid, view, prof_name)

            cmd = noise_cli + [
                str(src_path),
                "--SB_limit", str(sb_limit),
                "--random-seed", str(file_seed),
                "-o", str(out_stem),
            ]

            logger.info("CMD: %s", " ".join(cmd))
            if not args.dry_run:
                env = os.environ.copy()
                subprocess.run(cmd, check=True, env=env)

            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            logger.info(
                "[%d/%d] %s/%s  SB_limit=%.1f  seed=%d  (%.1f files/s)",
                done, total, prof_name, src_path.name, sb_limit, file_seed, rate,
            )

    elapsed = time.time() - t0
    logger.info(
        "Done. %d generated, %d skipped (exist), %d total in %.1fs",
        done - skipped, skipped, done, elapsed,
    )


if __name__ == "__main__":
    main()
