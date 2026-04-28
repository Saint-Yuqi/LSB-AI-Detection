#!/usr/bin/env python3
"""
Generate physics SB-noisy FITS without depending on pNbody.

Replicates the noise model used by ``mockimgs_sb_add_noise`` from
``pNbody.Mockimgs.noise``:

    sigma = pixfov_arcsec * (sqrt(area) / SN) * 10**(-(SB_limit - zp) / 2.5)

Then converts mag/arcsec^2 -> flux per pixel, adds Gaussian(0, sigma)
noise, and converts back to mag/arcsec^2. The output FITS carries the
same header bookkeeping keys (NOISESTD, SB_LIMIT, SB_AREA, SB_SN) as
the pNbody binary, so downstream consumers (renderer, header probes)
behave identically.

Why a separate script: the installed pNbody package is incompatible
with NumPy 2.x in the sam3 conda env (``ImportError: numpy.core.multiarray``),
so the binary cannot be invoked. The math is small and self-contained,
so we inline it instead of fighting the conda env.

Usage:
    conda run -n sam3 python scripts/data/generate_sb_noisy_fits_inline.py \
        --config configs/noise_profiles_dr1_physics.yaml --profiles sb32

Output: ``{paths.output_root}/{profile}/magnitudes-Fbox-{gid}-{view}-VIS2.fits.gz``
(flat layout, matching the existing dr1_physics tree).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import logging
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def _resolve(p: str) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _compute_seed(base_seed: int, gid: int, view: str, profile_name: str) -> int:
    """Same seeding rule as scripts/data/generate_pnbody_noisy_fits.py."""
    key = f"{gid}_{view}_{profile_name}"
    return base_seed + int(hashlib.sha256(key.encode()).hexdigest(), 16) % (2**31)


def _gaussian_sigma(sb_limit: float, pixfov: float, area: float, sn: float, zp: float) -> float:
    """``pNbody.Mockimgs.noise.get_std_for_SB_limit`` (one-line port)."""
    return pixfov * (np.sqrt(area) / sn) * 10 ** (-(sb_limit - zp) / 2.5)


def inject_sb_noise(
    in_fits_gz: Path,
    out_fits_gz: Path,
    sb_limit: float,
    seed: int,
    sb_area: float = 100.0,
    sb_sn: float = 3.0,
    zp: float = 0.0,
) -> dict:
    """Inject Gaussian noise sized by the SB-limit detection model.

    Mirrors mockimgs_sb_add_noise:
      mag/arcsec^2 -> flux per pixel -> + N(0, sigma) -> back to mag/arcsec^2.

    Negative flux values are floored at 1e-40 before the magnitude conversion
    (same epsilon as the pNbody binary) so log10 stays finite.
    """
    with gzip.open(in_fits_gz, "rb") as f:
        hdul = fits.open(f)
        data = hdul[0].data.copy()
        header = hdul[0].header.copy()
        hdul.close()

    pixfov = float(header["PIXFOVX"])
    pixarea = float(header["PIXAREA"])
    naxis1 = int(header["NAXIS1"])
    naxis2 = int(header["NAXIS2"])
    units = str(header["UNITS"]).strip()
    if units != "mag/arcsec^2":
        raise ValueError(
            f"{in_fits_gz}: expected UNITS='mag/arcsec^2', got {units!r}"
        )

    # mag -> flux per pixel
    flux = pixarea * 10 ** (-(data - zp) / 2.5)

    sigma = _gaussian_sigma(sb_limit, pixfov, sb_area, sb_sn, zp)
    rng = np.random.default_rng(seed)
    noise = rng.normal(loc=0.0, scale=sigma, size=(naxis1, naxis2))
    flux_noisy = flux + noise

    # back to mag/arcsec^2 with the pNbody binary's negative-floor convention
    flux_clipped = np.where(flux_noisy < 0, 1e-40, flux_noisy)
    data_noisy = -2.5 * np.log10(flux_clipped / pixarea) + zp

    # Stamp provenance keys (parity with mockimgs_sb_add_noise)
    header["NOISESTD"] = (sigma, "inline noise injector: noise std")
    header["SB_LIMIT"] = (sb_limit, "inline noise injector: target SB limit")
    header["SB_AREA"] = (sb_area, "inline noise injector: SB area arcsec^2")
    header["SB_SN"] = (sb_sn, "inline noise injector: SN target")
    header["NSEED"] = (seed, "inline noise injector: rng seed")

    hdu = fits.PrimaryHDU(data_noisy.astype(np.float32))
    hdu.header = header

    # Write directly to .fits then gzip-compress (same dance as pNbody binary
    # so the on-disk filename ends in .fits.gz).
    out_fits_gz.parent.mkdir(parents=True, exist_ok=True)
    tmp_fits = out_fits_gz.with_suffix("")  # strip ".gz"
    if tmp_fits.exists():
        tmp_fits.unlink()
    hdu.writeto(tmp_fits)
    with open(tmp_fits, "rb") as f_in:
        with gzip.open(out_fits_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    tmp_fits.unlink()

    return {"sigma": sigma, "seed": seed, "sb_limit": sb_limit, "out": str(out_fits_gz)}


def _resolve_input(sb_dir: Path, gid: int, view: str, suffix: str, layout: str) -> Path:
    fname = f"magnitudes-Fbox-{gid}-{view}-{suffix}.fits.gz"
    if layout == "nested":
        return sb_dir / f"{gid:05d}" / fname
    if layout == "flat":
        return sb_dir / fname
    raise ValueError(f"Unknown sb_maps_layout {layout!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True,
                    help="noise_profiles_*.yaml")
    ap.add_argument("--profiles", nargs="+", default=None,
                    help="Subset of profile names to render (default: all)")
    ap.add_argument("--galaxies", type=str, default=None,
                    help="Comma-separated galaxy id subset")
    ap.add_argument("--force", action="store_true",
                    help="Re-generate even if output already exists")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    firebox_root = _resolve(cfg["paths"]["firebox_root"])
    output_root = _resolve(cfg["paths"]["output_root"])
    sb_dir = firebox_root / "SB_maps"

    source = cfg.get("source", {}) or {}
    suffix = source.get("filename_suffix", "VIS2")
    layout = source.get("sb_maps_layout", "flat")  # DR1 SB_maps is flat

    galaxy_ids = cfg["data_selection"]["galaxy_ids"]
    if args.galaxies:
        keep = {int(g.strip()) for g in args.galaxies.split(",")}
        galaxy_ids = [g for g in galaxy_ids if g in keep]

    views = cfg["data_selection"].get("views") or cfg["data_selection"].get("orientations")
    profiles = cfg["profiles"]
    if args.profiles:
        keep = set(args.profiles)
        profiles = [p for p in profiles if p["name"] in keep]

    base_seed = int(cfg["reproducibility"]["random_seed"])

    pairs = [(int(g), str(v)) for g in galaxy_ids for v in views]
    total = len(pairs) * len(profiles)
    logger.info(
        "Generating %d noisy FITS via inline injector (%d pairs x %d profiles)",
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
            src = _resolve_input(sb_dir, gid, view, suffix, layout)
            if not src.exists():
                logger.warning("MISSING: %s", src)
                continue
            out = prof_dir / f"magnitudes-Fbox-{gid}-{view}-{suffix}.fits.gz"
            if out.exists() and not args.force:
                skipped += 1
                done += 1
                continue
            seed = _compute_seed(base_seed, gid, view, prof_name)
            result = inject_sb_noise(src, out, sb_limit, seed)
            done += 1
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[%d/%d] %s/%s sigma=%.3e seed=%d (%.1f files/s)",
                done, total, prof_name, src.name, result["sigma"], seed, rate,
            )

    logger.info(
        "Done. %d generated, %d skipped (exist), %d total in %.1fs",
        done - skipped, skipped, total, time.time() - t0,
    )


if __name__ == "__main__":
    main()
