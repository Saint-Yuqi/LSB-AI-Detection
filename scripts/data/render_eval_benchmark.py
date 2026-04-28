#!/usr/bin/env python3
"""
Render evaluation-time renders for the Fbox Gold and DR1 benchmarks.

Writes 1024x1024 PNGs under data/02_processed/renders_eval/{benchmark}/...
from the benchmark's own clean SB_maps and the pre-generated noisy FITS
(produced by scripts/data/generate_pnbody_noisy_fits.py).

Output layout:
    data/02_processed/renders_eval/{benchmark}/
        current/{variant}/{base_key}/0000.png
        noisy/{variant}/{profile}/{base_key}/0000.png

Variants + params are read from configs/unified_data_prep.yaml so renders
are byte-identical to training preprocessing.

Usage:
    conda run -n sam3 python scripts/data/render_eval_benchmark.py \
        --benchmark fbox_gold_satellites
    conda run -n sam3 python scripts/data/render_eval_benchmark.py \
        --benchmark firebox_dr1_streams --conditions noisy --variants linear_magnitude
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from astropy.io import fits

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT))

from src.pipelines.unified_dataset.preprocessor_factory import (  # noqa: E402
    create_preprocessor as _create_preprocessor,
)

logger = logging.getLogger(__name__)

_DEFAULT_MASK_REGEX = r"ark_features-(\d+)-(eo|fo)-SBlim[\d.]+\.fits\.gz"

# Catalog of supported benchmarks: maps --benchmark to its noise profile config.
# Both fields in the noise config (source.*, paths.*, profiles) are reused here
# to avoid duplicating sample-enumeration logic.
_BENCHMARK_CATALOG: dict[str, str] = {
    "fbox_gold_satellites": "configs/noise_profiles_fbox_gold.yaml",
    "firebox_dr1_streams": "configs/noise_profiles_dr1_streams.yaml",
}


def _resolve_project_path(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else _PROJECT_ROOT / path


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_fits_data(filepath: Path) -> np.ndarray:
    """Load FITS data from gzipped file."""
    with gzip.open(filepath, "rb") as f:
        with fits.open(f) as hdul:
            return hdul[0].data.copy()


def _enumerate_pairs(noise_cfg: dict, firebox_root: Path) -> list[tuple[int, str]]:
    """(gid, view) pairs per the noise config's enumeration mode.

    Mirrors the enumeration logic of generate_pnbody_noisy_fits.py so that
    the two scripts walk the same sample universe.
    """
    source = noise_cfg.get("source", {}) or {}
    enumeration = source.get("enumeration", {}) or {}
    mode = enumeration.get("mode", "grid")

    if mode == "grid":
        galaxy_ids = noise_cfg["data_selection"]["galaxy_ids"]
        views = (
            noise_cfg["data_selection"].get("views")
            or noise_cfg["data_selection"]["orientations"]
        )
        return [(int(g), str(v)) for g in galaxy_ids for v in views]

    if mode == "manifest":
        manifest_path = enumeration.get("manifest")
        if not manifest_path:
            raise KeyError("source.enumeration.manifest is required for mode=manifest")
        with open(_resolve_project_path(manifest_path)) as f:
            manifest = json.load(f)
        return [(int(e["galaxy_id"]), str(e["view"])) for e in manifest["samples"]]

    if mode == "mask_glob":
        mask_regex = re.compile(enumeration.get("mask_name_regex", _DEFAULT_MASK_REGEX))
        pairs_set: set[tuple[int, str]] = set()
        for glob_key in ("mask_eo_glob", "mask_fo_glob"):
            rel_glob = enumeration.get(glob_key)
            if not rel_glob:
                raise KeyError(
                    f"source.enumeration.{glob_key} is required for mode=mask_glob"
                )
            for path in sorted(firebox_root.glob(rel_glob)):
                m = mask_regex.match(path.name)
                if not m:
                    continue
                pairs_set.add((int(m.group(1)), m.group(2)))
        return sorted(pairs_set)

    raise ValueError(
        f"Unknown enumeration mode: {mode!r}. Expected: grid | manifest | mask_glob"
    )


def _clean_fits_path(
    sb_dir: Path, gid: int, view: str, suffix: str, sb_maps_layout: str
) -> Path:
    fname = f"magnitudes-Fbox-{gid}-{view}-{suffix}.fits.gz"
    if sb_maps_layout == "nested":
        return sb_dir / f"{gid:05d}" / fname
    if sb_maps_layout == "flat":
        return sb_dir / fname
    raise ValueError(f"Unknown sb_maps_layout: {sb_maps_layout!r}")


def _noisy_fits_path(
    noise_output_root: Path,
    profile: str,
    gid: int,
    view: str,
    suffix: str,
) -> Path:
    return (
        noise_output_root
        / profile
        / f"{gid:05d}"
        / f"magnitudes-Fbox-{gid}-{view}-{suffix}.fits.gz"
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--benchmark", required=True, choices=sorted(_BENCHMARK_CATALOG.keys())
    )
    ap.add_argument(
        "--data-config", default="configs/unified_data_prep.yaml",
        help="Preprocessor variant params + target_size",
    )
    ap.add_argument(
        "--conditions", nargs="+", default=["current", "noisy"],
        choices=["current", "noisy"],
    )
    ap.add_argument(
        "--variants", nargs="+", default=["asinh_stretch", "linear_magnitude"],
    )
    ap.add_argument(
        "--profiles", nargs="+", default=None,
        help="Noise profile subset (default: all in noise config). Ignored for current.",
    )
    ap.add_argument(
        "--output-root", default="data/02_processed/renders_eval",
        help="Root for renders_eval output tree",
    )
    ap.add_argument("--galaxies", default=None, help="Comma-separated gid subset")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    noise_cfg_path = _resolve_project_path(_BENCHMARK_CATALOG[args.benchmark])
    noise_cfg = _load_yaml(noise_cfg_path)
    data_cfg = _load_yaml(_resolve_project_path(args.data_config))

    firebox_root = _resolve_project_path(noise_cfg["paths"]["firebox_root"])
    noise_output_root = _resolve_project_path(noise_cfg["paths"]["output_root"])
    sb_dir = firebox_root / "SB_maps"

    source = noise_cfg.get("source", {}) or {}
    filename_suffix = source.get("filename_suffix", "VIS2")
    sb_maps_layout = source.get("sb_maps_layout", "nested")

    target_size = tuple(data_cfg["processing"]["target_size"])
    if target_size != (1024, 1024):
        raise ValueError(
            f"render_eval_benchmark.py is 1024-only; got target_size={target_size} "
            f"from {args.data_config}"
        )

    variant_params: dict[str, dict] = {
        v["name"]: v.get("params", {}) for v in data_cfg["preprocessing_variants"]
    }
    for vname in args.variants:
        if vname not in variant_params:
            raise KeyError(
                f"variant {vname!r} not in {args.data_config}; "
                f"available: {list(variant_params)}"
            )

    preprocessors = {
        name: _create_preprocessor(name, variant_params[name], target_size)
        for name in args.variants
    }

    pairs = _enumerate_pairs(noise_cfg, firebox_root)
    if args.galaxies:
        subset = {int(g.strip()) for g in args.galaxies.split(",")}
        pairs = [p for p in pairs if p[0] in subset]

    profile_names = [p["name"] for p in noise_cfg["profiles"]]
    if args.profiles:
        profile_names = [p for p in profile_names if p in set(args.profiles)]

    output_root = _resolve_project_path(args.output_root) / args.benchmark

    logger.info(
        "benchmark=%s suffix=%s layout=%s samples=%d variants=%s conditions=%s profiles=%s",
        args.benchmark, filename_suffix, sb_maps_layout, len(pairs),
        args.variants, args.conditions, profile_names,
    )

    done = 0
    skipped = 0
    missing = 0
    t0 = time.time()

    for gid, view in pairs:
        base_key = f"{gid:05d}_{view}"
        clean_path = _clean_fits_path(sb_dir, gid, view, filename_suffix, sb_maps_layout)

        # current
        if "current" in args.conditions:
            sb_map = None
            for vname, proc in preprocessors.items():
                out_dir = output_root / "current" / vname / base_key
                out_path = out_dir / "0000.png"
                if out_path.exists():
                    skipped += 1
                    done += 1
                    continue
                if not clean_path.exists():
                    logger.warning("SKIP (missing clean): %s", clean_path)
                    missing += 1
                    continue
                if sb_map is None:
                    sb_map = _load_fits_data(clean_path)
                out_dir.mkdir(parents=True, exist_ok=True)
                rgb = proc.process(sb_map)
                cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                done += 1
                rate = done / max(time.time() - t0, 1e-9)
                logger.info(
                    "[%d] current/%s/%s  (%.1f img/s)", done, vname, base_key, rate
                )

        # noisy
        if "noisy" in args.conditions:
            for profile in profile_names:
                noisy_path = _noisy_fits_path(
                    noise_output_root, profile, gid, view, filename_suffix
                )
                sb_map = None
                for vname, proc in preprocessors.items():
                    out_dir = output_root / "noisy" / vname / profile / base_key
                    out_path = out_dir / "0000.png"
                    if out_path.exists():
                        skipped += 1
                        done += 1
                        continue
                    if not noisy_path.exists():
                        logger.warning("SKIP (missing noisy): %s", noisy_path)
                        missing += 1
                        continue
                    if sb_map is None:
                        sb_map = _load_fits_data(noisy_path)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    rgb = proc.process(sb_map)
                    cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
                    done += 1
                    rate = done / max(time.time() - t0, 1e-9)
                    logger.info(
                        "[%d] noisy/%s/%s/%s  (%.1f img/s)",
                        done, vname, profile, base_key, rate,
                    )

    elapsed = time.time() - t0
    logger.info(
        "Done. rendered=%d skipped_existing=%d missing_inputs=%d elapsed=%.1fs",
        done - skipped, skipped, missing, elapsed,
    )


if __name__ == "__main__":
    main()
