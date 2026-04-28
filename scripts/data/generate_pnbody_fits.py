#!/usr/bin/env python3
"""
Batch-generate PNbody FITS for 24 LOS per halo.

For each galaxy_id × LOS direction, calls ``mockimgs_sb_compute_images`` with
``--los x y z`` and ``-o`` to produce the canonical
``magnitudes-Fbox-{gid}-los{nn}-VIS2.fits.gz`` file directly.

Usage:
    python scripts/data/generate_pnbody_fits.py --config configs/pnbody/firebox_pnbody_24los.yaml
    python scripts/data/generate_pnbody_fits.py --config ... --galaxies 11,13
    python scripts/data/generate_pnbody_fits.py --config ... --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import gzip
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
N_LOS = 24

logger = logging.getLogger(__name__)


def _resolve_project_path(path_value: str | Path) -> Path:
    """Resolve project-relative config paths against the repository root."""
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _resolve_existing_path(path_value: str | Path, *, label: str) -> Path:
    """Resolve an input path and fall back to a close filename match when possible."""
    requested = _resolve_project_path(path_value)
    if requested.exists():
        return requested

    parent = requested.parent
    if parent.exists():
        pattern = f"{requested.stem}*{requested.suffix}"
        matches = sorted(parent.glob(pattern))
        if len(matches) == 1:
            logger.warning("%s not found at %s; using %s", label, requested, matches[0])
            return matches[0]

    raise FileNotFoundError(f"{label} not found: {requested}")


def _resolve_pnbody_cli() -> list[str]:
    """Prefer installed CLI, otherwise fall back to the repo-local script."""
    installed = shutil.which("mockimgs_sb_compute_images")
    if installed:
        return [installed]

    local_cli = PROJECT_ROOT / "scripts" / "mockimgs_sb_compute_images"
    if local_cli.exists():
        logger.warning(
            "mockimgs_sb_compute_images not found on PATH; using local script via %s",
            sys.executable,
        )
        return [sys.executable, str(local_cli)]

    raise FileNotFoundError(
        "mockimgs_sb_compute_images not found on PATH and local fallback script is missing"
    )


def _load_los_vectors(los_file: Path) -> list[tuple[float, float, float]]:
    """Read LOS (x, y, z) vectors from the HEALPix nside=2 centres file."""
    vectors = []
    with open(los_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            vectors.append((float(parts[0]), float(parts[1]), float(parts[2])))
    if len(vectors) != N_LOS:
        raise ValueError(f"Expected {N_LOS} LOS vectors in {los_file}, got {len(vectors)}")
    return vectors


def _gzip_file(src: Path, dst: Path) -> None:
    """Compress *src* to *dst* (.gz) and remove the original."""
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    src.unlink()


def _run_pnbody(
    halo_path: Path,
    instrument_file: Path,
    los_vector: tuple[float, float, float],
    distance: float,
    rsp_opts: dict,
    output_path: Path,
    dry_run: bool = False,
) -> None:
    """Call mockimgs_sb_compute_images for one halo + one LOS direction.

    The upstream CLI triggers its FITS-save branch only when ``-o`` ends with
    ``.fits`` (``os.path.splitext`` check).  We therefore pass a ``.fits``
    name and gzip the result afterwards if *output_path* ends with ``.gz``.
    """
    needs_gzip = output_path.suffix == ".gz"
    fits_path = output_path.with_suffix("") if needs_gzip else output_path

    lx, ly, lz = los_vector
    cmd = _resolve_pnbody_cli() + [
        str(halo_path),
        "--instrument", str(instrument_file),
        "--los", str(lx), str(ly), str(lz),
        "--distance", str(distance),
        "--rsp_mode", str(rsp_opts.get("rsp_mode", "None")),
        "--rsp_fac", str(rsp_opts.get("rsp_fac", 0.6)),
        "-o", str(fits_path),
    ]
    logger.info("CMD: %s", " ".join(cmd))
    if dry_run:
        return
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    subprocess.run(cmd, check=True, env=env)
    if needs_gzip and fits_path.exists():
        _gzip_file(fits_path, output_path)
        logger.info("Compressed %s -> %s", fits_path.name, output_path.name)



def main():
    parser = argparse.ArgumentParser(description="Generate PNbody 24-view FITS")
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--galaxies", type=str, default=None,
                        help="Comma-separated galaxy ID subset for smoke tests")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    config_path = args.config.resolve()

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    halo_root = _resolve_project_path(cfg["halo_root"])
    halo_pattern = cfg["halo_pattern"]
    galaxy_ids = cfg["galaxy_ids"]
    los_file = _resolve_existing_path(cfg["los_file"], label="LOS file")
    distance = cfg["distance"]
    rsp_opts = cfg.get("rsp_opts", {})
    instrument_file = _resolve_existing_path(cfg["instrument_file"], label="Instrument file")
    output_root = _resolve_project_path(cfg["output_root"])
    metadata_root = _resolve_project_path(cfg["metadata_root"])

    if args.galaxies:
        subset = {int(g.strip()) for g in args.galaxies.split(",")}
        galaxy_ids = [g for g in galaxy_ids if g in subset]

    los_vectors = _load_los_vectors(los_file)

    output_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    views_csv_path = metadata_root / "views.csv"
    csv_rows: list[dict] = []
    manifest_entries: list[dict] = []

    for gid in galaxy_ids:
        halo_name = halo_pattern.format(galaxy_id=gid)
        halo_path = halo_root / halo_name
        if not halo_path.exists():
            logger.warning("Halo not found, skipping: %s", halo_path)
            continue

        logger.info("Processing galaxy %d: %s", gid, halo_path.name)

        galaxy_dir = output_root / f"{gid:05d}"
        galaxy_dir.mkdir(parents=True, exist_ok=True)

        for los_idx, los_vec in enumerate(los_vectors):
            canonical = galaxy_dir / f"magnitudes-Fbox-{gid}-los{los_idx:02d}-VIS2.fits.gz"
            _run_pnbody(
                halo_path=halo_path,
                instrument_file=instrument_file,
                los_vector=los_vec,
                distance=distance,
                rsp_opts=rsp_opts,
                output_path=canonical,
                dry_run=args.dry_run,
            )
            lx, ly, lz = los_vec
            csv_rows.append({
                "galaxy_id": gid,
                "view_id": f"los{los_idx:02d}",
                "los_x": lx, "los_y": ly, "los_z": lz,
                "source_hdf5": str(halo_path),
                "output_fits": str(canonical.relative_to(output_root)),
            })

        manifest_entries.append({
            "galaxy_id": gid,
            "halo_file": str(halo_path),
            "n_views": N_LOS,
            "views": [f"los{i:02d}" for i in range(N_LOS)],
        })

    if csv_rows:
        with open(views_csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
        logger.info("Wrote %s (%d rows)", views_csv_path, len(csv_rows))

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "config": str(args.config),
        "distance_mpc": distance,
        "rsp_opts": rsp_opts,
        "instrument_file": str(instrument_file),
        "n_galaxies": len(manifest_entries),
        "galaxies": manifest_entries,
    }
    manifest_path = metadata_root / "generation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("Wrote %s", manifest_path)


if __name__ == "__main__":
    main()
