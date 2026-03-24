"""
Batch-generate noise-injected FITS from clean SB magnitude maps.

Usage:
    python scripts/generate_noisy_fits.py --config configs/noise_profiles.yaml
    python scripts/generate_noisy_fits.py --config ... --galaxies 11,13 --profiles snr10

Env:
    Requires: numpy, astropy, pyyaml

Output:
    data/04_noise/{profile_name}/magnitudes-Fbox-{gid}-{orient}-VIS2.fits.gz
    Original FITS headers preserved + NOISSNR / NOISSCL / NOISSKY / NOISRDN cards.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

# Resolve project root so `src` is importable
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

from src.noise.forward_observation import ForwardObservationModel


def _load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _load_fits_with_header(filepath: Path):
    """Return (data: ndarray, header: fits.Header)."""
    with gzip.open(filepath, "rb") as f:
        with fits.open(f) as hdul:
            return hdul[0].data.copy(), hdul[0].header.copy()


def _save_fits_gz(data: np.ndarray, header: fits.Header, outpath: Path) -> None:
    outpath.parent.mkdir(parents=True, exist_ok=True)
    hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
    hdul = fits.HDUList([hdu])
    with gzip.open(outpath, "wb") as f:
        hdul.writeto(f, overwrite=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True, help="Path to noise_profiles.yaml")
    ap.add_argument("--galaxies", default=None, help="Comma-separated galaxy IDs (subset)")
    ap.add_argument("--profiles", default=None, help="Comma-separated profile names (subset)")
    args = ap.parse_args()

    cfg = _load_config(args.config)

    firebox_root = Path(cfg["paths"]["firebox_root"])
    output_root = Path(cfg["paths"]["output_root"])
    sb_dir = firebox_root / "SB_maps"

    galaxy_ids = cfg["data_selection"]["galaxy_ids"]
    orientations = cfg["data_selection"].get("views") or cfg["data_selection"]["orientations"]
    profiles = cfg["profiles"]
    noise_cfg = cfg["noise_model"]
    base_seed = cfg["reproducibility"]["random_seed"]

    # CLI subset filters
    if args.galaxies:
        subset = set(int(g) for g in args.galaxies.split(","))
        galaxy_ids = [g for g in galaxy_ids if g in subset]
    if args.profiles:
        subset = set(args.profiles.split(","))
        profiles = [p for p in profiles if p["name"] in subset]

    total = len(galaxy_ids) * len(orientations) * len(profiles)
    print(f"Generating {total} noisy FITS "
          f"({len(galaxy_ids)} galaxies × {len(orientations)} orient × {len(profiles)} profiles)")

    done = 0
    t0 = time.time()

    for prof in profiles:
        prof_name = prof["name"]
        target_snr = float(prof["target_snr"])

        for gid in galaxy_ids:
            for orient in orientations:
                fname = f"magnitudes-Fbox-{gid}-{orient}-VIS2.fits.gz"
                src_path = sb_dir / fname

                if not src_path.exists():
                    print(f"  SKIP (missing): {src_path}")
                    continue

                sb_map, header = _load_fits_with_header(src_path)

                # Deterministic per-file seed: base + sha256(gid, orient, profile)
                key_str = f"{gid}_{orient}_{prof_name}"
                file_seed = base_seed + int(
                    hashlib.sha256(key_str.encode()).hexdigest(), 16
                ) % (2**31)

                model = ForwardObservationModel.from_target_snr(
                    target_snr=target_snr,
                    sb_map=sb_map,
                    zeropoint=noise_cfg["zeropoint"],
                    sky_level=noise_cfg["sky_level"],
                    read_noise=noise_cfg["read_noise"],
                    signal_quantile=noise_cfg["signal_quantile"],
                    background_quantile=noise_cfg["background_quantile"],
                    seed=file_seed,
                )

                noisy_mag = model.inject(sb_map)

                # Annotate FITS header with noise params
                header["NOISSNR"] = (target_snr, "Target SNR for noise injection")
                header["NOISSCL"] = (model.signal_scale, "Effective signal_scale (gain)")
                header["NOISSKY"] = (model.sky_level, "Sky background (counts/pixel)")
                header["NOISRDN"] = (model.read_noise, "CCD read noise sigma (e-/pixel)")
                header["NOISSQN"] = (model.signal_quantile, "Signal quantile for SNR")
                header["NOISBQN"] = (model.background_quantile, "Bkg quantile for SNR")
                header["NOISSED"] = (file_seed, "RNG seed for this file")

                # Measured analytic SNR for QA
                measured_snr = model.expected_snr(sb_map)
                header["NOISMSN"] = (round(measured_snr, 2), "Measured analytic SNR")

                out_path = output_root / prof_name / fname
                _save_fits_gz(noisy_mag, header, out_path)

                done += 1
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                nan_frac = float(np.isnan(noisy_mag).sum()) / noisy_mag.size
                print(f"  [{done}/{total}] {prof_name}/{fname}  "
                      f"scale={model.signal_scale:.1f}  "
                      f"SNR_analytic={measured_snr:.1f}  "
                      f"NaN_frac={nan_frac:.3f}  "
                      f"({rate:.1f} files/s)")

    print(f"\nDone. {done} files written to {output_root}/ in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
