"""
Render noise-injected FITS to PNG using preprocessing variants from unified config.

Usage:
    python scripts/render_noisy_fits.py \
        --noise-config configs/noise_profiles.yaml \
        --data-config configs/unified_data_prep.yaml
    python scripts/render_noisy_fits.py ... --variants asinh_stretch linear_magnitude
    python scripts/render_noisy_fits.py ... --profiles snr10 snr20

Args:
    --noise-config : Path to noise_profiles.yaml (input FITS paths)
    --data-config  : Path to unified_data_prep.yaml (preprocessor params)
    --variants     : Preprocessing variants to render (default: asinh_stretch)
    --profiles     : SNR profile subset (default: all)

Output:
    data/02_processed/renders/noisy/{variant}/{profile}/{base_key}/0000.png
    Directory naming uses zero-padded galaxy IDs: {gid:05d}_{orient}
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import yaml
from astropy.io import fits

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.data.preprocessing import (
    LSBPreprocessor,
    LinearMagnitudePreprocessor,
    MultiExposurePreprocessor,
)

# ---------------------------------------------------------------------------
# Preprocessor factory — mirrors prepare_unified_dataset.py exactly
# ---------------------------------------------------------------------------

_FNAME_RE = re.compile(r"magnitudes-Fbox-(\d+)-(eo|fo)-VIS2\.fits\.gz")


def _create_preprocessor(name: str, params: dict, target_size: tuple[int, int]):
    """Factory matching prepare_unified_dataset.py::create_preprocessor."""
    if name == "asinh_stretch":
        return LSBPreprocessor(
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 50.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            target_size=target_size,
        )
    elif name == "linear_magnitude":
        return LinearMagnitudePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            target_size=target_size,
        )
    elif name == "multi_exposure":
        return MultiExposurePreprocessor(
            global_mag_min=params.get("global_mag_min", 20.0),
            global_mag_max=params.get("global_mag_max", 35.0),
            zeropoint=params.get("zeropoint", 22.5),
            nonlinearity=params.get("nonlinearity", 300.0),
            clip_percentile=params.get("clip_percentile", 99.5),
            gamma=params.get("gamma", 0.5),
            b_mode=params.get("b_mode", "gamma"),
            target_size=target_size,
        )
    else:
        raise ValueError(f"Unknown preprocessor: {name}")


def _load_fits_data(filepath: Path) -> np.ndarray:
    """Load FITS data from gzipped file."""
    with gzip.open(filepath, "rb") as f:
        with fits.open(f) as hdul:
            return hdul[0].data.copy()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--noise-config", required=True,
                    help="Path to noise_profiles.yaml")
    ap.add_argument("--data-config", default="configs/unified_data_prep.yaml",
                    help="Path to unified_data_prep.yaml (preprocessor params)")
    ap.add_argument("--variants", nargs="+", default=["asinh_stretch"],
                    help="Preprocessing variants to render (default: asinh_stretch)")
    ap.add_argument("--profiles", nargs="+", default=None,
                    help="SNR profile subset (default: all)")
    ap.add_argument("--output-root", default=None,
                    help="Override output root (default: from data-config)")
    args = ap.parse_args()

    # --- Load configs ---
    with open(args.noise_config) as f:
        noise_cfg = yaml.safe_load(f)
    with open(args.data_config) as f:
        data_cfg = yaml.safe_load(f)

    noise_root = Path(noise_cfg["paths"]["output_root"])
    output_root = Path(args.output_root or data_cfg["paths"]["output_root"])
    target_size = tuple(data_cfg["processing"]["target_size"])

    # Build variant → params lookup from unified config
    variant_params: dict[str, dict] = {}
    for v in data_cfg["preprocessing_variants"]:
        variant_params[v["name"]] = v.get("params", {})

    # Validate requested variants exist in config
    for vname in args.variants:
        if vname not in variant_params:
            print(f"ERROR: variant '{vname}' not found in {args.data_config} "
                  f"preprocessing_variants. Available: {list(variant_params.keys())}")
            sys.exit(1)

    # Build preprocessors — params read from unified config, not hardcoded
    preprocessors = {
        name: _create_preprocessor(name, variant_params[name], target_size)
        for name in args.variants
    }

    # Collect noise profiles
    profiles = [p["name"] for p in noise_cfg["profiles"]]
    if args.profiles:
        profiles = [p for p in profiles if p in set(args.profiles)]

    # --- Enumerate input FITS ---
    total = 0
    work_items: list[tuple[str, Path]] = []  # (profile, fits_path)
    for prof in profiles:
        prof_dir = noise_root / prof
        if not prof_dir.exists():
            print(f"  SKIP (missing dir): {prof_dir}")
            continue
        fits_files = sorted(prof_dir.glob("magnitudes-Fbox-*-VIS2.fits.gz"))
        work_items.extend((prof, f) for f in fits_files)
    total = len(work_items) * len(preprocessors)

    print(f"Rendering {total} images "
          f"({len(work_items)} FITS × {len(preprocessors)} variants)")

    done = 0
    skipped = 0
    t0 = time.time()

    for prof, fits_path in work_items:
        # Parse galaxy_id and orientation from filename
        m = _FNAME_RE.match(fits_path.name)
        if not m:
            print(f"  SKIP (bad filename): {fits_path.name}")
            continue
        gid = int(m.group(1))
        orient = m.group(2)
        # Zero-padded base_key matching BaseKey.__str__
        base_key = f"{gid:05d}_{orient}"

        # Lazy-load FITS only if at least one variant needs rendering
        sb_map = None

        for vname, proc in preprocessors.items():
            out_dir = output_root / "renders" / "noisy" / vname / prof / base_key
            out_path = out_dir / "0000.png"

            if out_path.exists():
                skipped += 1
                done += 1
                continue

            # Load on first use
            if sb_map is None:
                sb_map = _load_fits_data(fits_path)

            out_dir.mkdir(parents=True, exist_ok=True)
            rgb = proc.process(sb_map)
            cv2.imwrite(str(out_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"  [{done}/{total}] {vname}/{prof}/{base_key}  ({rate:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"\nDone. {done - skipped} rendered, {skipped} skipped (exist) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
