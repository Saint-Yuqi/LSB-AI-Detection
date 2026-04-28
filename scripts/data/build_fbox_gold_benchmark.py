#!/usr/bin/env python3
"""
Build Fbox Gold Satellites Benchmark V1.

Usage:
    python scripts/data/build_fbox_gold_benchmark.py                    # full build
    python scripts/data/build_fbox_gold_benchmark.py --verify-only      # check only
    python scripts/data/build_fbox_gold_benchmark.py --copy             # copy FITS
    python scripts/data/build_fbox_gold_benchmark.py --write-checksums  # embed SHA256

Args:
    --verify-only     Check existing benchmark integrity without writing.
    --copy            Copy FITS files instead of symlinking.
    --allow-overlap   Disable seg_ids overlap fail-fast (last-writer-wins).
    --write-checksums Write SHA256 checksums into dataset_manifest.json.

Env:
    Requires NFS mount at /shares/feldmann.ics.mnf.uzh/ for default mode.
    Use --copy to build offline-portable benchmark.

Note:
    Bypasses SatelliteInstance.from_dict — full-property extraction from raw
    pickle dict. This script is self-contained and does not modify src/.
"""

import argparse
import hashlib
import json
import pickle
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# ─── Constants ───────────────────────────────────────────────────────────────

BENCHMARK_NAME = "Fbox_Gold_Satellites"
BENCHMARK_VERSION = "1.0"
SOURCE_PICKLE_PATH = Path(
    "/shares/feldmann.ics.mnf.uzh/Lucas/pNbody/satellites/fbox/"
    "props_gals_Fbox_new.pkl"
)
SOURCE_SBLIM = "SBlim31.5"
SOURCE_FILTER = "VIS"
IMAGE_SHAPE = (2051, 2051)

# Verified baseline statistics
EXPECTED = dict(galaxies=66, samples=132, instances=426, empty_samples=9)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_ROOT = (
    PROJECT_ROOT / "data" / "01_raw" / "LSB_and_Satellites" / "Fbox_Gold_Satellites"
)
# Resolve through existing fbox symlink to get real NFS path
FBOX_SB_MAPS = (
    PROJECT_ROOT / "data" / "01_raw" / "LSB_and_Satellites" / "fbox" / "sb_maps"
).resolve()

ROI_DEFINITION = {
    "physical_analysis_region_kpc": 300,
    "source_image_size": list(IMAGE_SHAPE),
    "pixel_scale_kpc_per_pixel": 0.64,
    "roi_half_width_source_pixels": 468.75,
    "roi_full_width_source_pixels": 937.5,
    "note": (
        "Benchmark stores full-frame truth. "
        "ROI-restricted evaluation is implemented downstream."
    ),
}

COORDINATE_CONVENTION = {
    "frame": "full-resolution source-frame",
    "indexing": "zero-based",
    "origin": "top-left",
    "mask_index_order": "[y, x]",
    "centroid_fields": (
        "x/y and geo_x/geo_y are float pixel coordinates "
        "in the same 2051x2051 full frame"
    ),
}

# Pickle fields to extract (excluding seg_map, seg_ids)
INSTANCE_FIELDS = [
    "x", "y", "geo-x", "geo-y",
    "area", "dmin", "dmax", "dmed",
    "gini", "axis_ratio", "orientation_angle",
    "mag_fltr", "sb_fltr", "mag_g", "mag_r", "mag_g_nodust", "mag_r_nodust",
]
FIELD_RENAME = {"geo-x": "geo_x", "geo-y": "geo_y"}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _die(msg: str):
    print(f"FATAL: {msg}", file=sys.stderr)
    sys.exit(1)


def _to_native(val: Any) -> Any:
    """numpy scalar -> Python native with controlled float precision."""
    if isinstance(val, np.integer):
        return int(val)
    if isinstance(val, np.floating):
        return round(float(val), 8)
    if isinstance(val, np.ndarray) and val.ndim == 0:
        return _to_native(val.item())
    return val


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj: Any):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _parse_key(key: str) -> Tuple[int, str]:
    """'11, eo' -> (11, 'eo')"""
    parts = key.replace(" ", "").split(",")
    return int(parts[0]), parts[1]


# ─── Phase 1: Load & Validate ───────────────────────────────────────────────

def load_and_validate(
    pkl_path: Path,
) -> Tuple[dict, List[dict]]:
    """Load pickle, enumerate samples, validate VIS images, assert stats."""
    if not pkl_path.exists():
        _die(f"Pickle not found: {pkl_path}")
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    print(f"  Loaded pickle: {len(data)} keys")

    # Enumerate samples
    samples: List[dict] = []
    for key in data:
        gid, view = _parse_key(key)
        sb_data = data[key].get(SOURCE_SBLIM, {})
        sat_ids = sorted(
            [k for k in sb_data if k.startswith("id") and isinstance(sb_data[k], dict)],
            key=lambda x: int(x[2:]),
        )
        fname = f"magnitudes-Fbox-{gid}-{view}-{SOURCE_FILTER}.fits.gz"
        src_path = FBOX_SB_MAPS / fname
        if not src_path.exists():
            _die(f"VIS image missing for {gid},{view}: {src_path}")

        samples.append({
            "galaxy_id": gid,
            "view": view,
            "source_key": key,
            "sample_id": f"{gid:05d}_{view}",
            "sat_ids": sat_ids,
            "instance_count": len(sat_ids),
            "is_empty_sample": len(sat_ids) == 0,
            "source_image_filename": fname,
            "source_image_realpath": str(src_path.resolve()),
        })

    samples.sort(key=lambda s: (s["galaxy_id"], s["view"]))

    # Assert stats
    gids = {s["galaxy_id"] for s in samples}
    total_inst = sum(s["instance_count"] for s in samples)
    n_empty = sum(1 for s in samples if s["is_empty_sample"])
    for label, actual, exp in [
        ("galaxies", len(gids), EXPECTED["galaxies"]),
        ("samples", len(samples), EXPECTED["samples"]),
        ("instances", total_inst, EXPECTED["instances"]),
        ("empty_samples", n_empty, EXPECTED["empty_samples"]),
    ]:
        if actual != exp:
            _die(f"{label}: expected {exp}, got {actual}")

    print(
        f"  ✓ {len(gids)} galaxies, {len(samples)} samples, "
        f"{total_inst} instances, {n_empty} empty"
    )
    return data, samples


# ─── Phase 2–3: Directories & Image Entries ─────────────────────────────────

def create_structure(samples: List[dict], copy: bool):
    """Create directory tree + symlinks/copies for VIS FITS."""
    BENCHMARK_ROOT.mkdir(parents=True, exist_ok=True)
    (BENCHMARK_ROOT / "SB_maps").mkdir(exist_ok=True)
    (BENCHMARK_ROOT / "MASKS").mkdir(exist_ok=True)

    for s in samples:
        (BENCHMARK_ROOT / "SB_maps" / f"{s['galaxy_id']:05d}").mkdir(exist_ok=True)
        (BENCHMARK_ROOT / "MASKS" / s["sample_id"]).mkdir(exist_ok=True)

        link = BENCHMARK_ROOT / "SB_maps" / f"{s['galaxy_id']:05d}" / s["source_image_filename"]
        target = Path(s["source_image_realpath"])
        if link.exists() or link.is_symlink():
            link.unlink()
        if copy:
            shutil.copy2(target, link)
        else:
            link.symlink_to(target)

    mode = "copied" if copy else "symlinked"
    print(f"  ✓ Structure created, {len(samples)} VIS images {mode}")


# ─── Phase 4: Per-Sample Artifacts ──────────────────────────────────────────

def build_sample_artifacts(
    data: dict, samples: List[dict], allow_overlap: bool,
):
    """Build instance_map.npy, instances.json, sample_manifest.json."""
    for s in samples:
        mask_dir = BENCHMARK_ROOT / "MASKS" / s["sample_id"]
        sb_data = data[s["source_key"]][SOURCE_SBLIM]

        instance_map = np.zeros(IMAGE_SHAPE, dtype=np.uint16)
        instances: List[dict] = []

        # Occupancy for overlap detection (only needed if >1 satellite)
        need_overlap_check = (not allow_overlap) and s["instance_count"] > 1
        if need_overlap_check:
            occupancy = np.zeros(IMAGE_SHAPE, dtype=np.uint16)

        for bench_id, sat_key in enumerate(s["sat_ids"], start=1):
            sat = sb_data[sat_key]
            seg_ids = sat.get("seg_ids")

            if seg_ids is not None and len(seg_ids) > 0:
                yc = seg_ids[:, 0].astype(np.intp)
                xc = seg_ids[:, 1].astype(np.intp)

                if need_overlap_check:
                    occupied = occupancy[yc, xc]
                    overlap = occupied > 0
                    if overlap.any():
                        conflicts = np.unique(occupied[overlap]).tolist()
                        _die(
                            f"Pixel overlap in {s['sample_id']}: "
                            f"{sat_key} (id={bench_id}) overlaps bench_id(s) "
                            f"{conflicts}, {int(overlap.sum())} pixels"
                        )
                    occupancy[yc, xc] = bench_id

                instance_map[yc, xc] = bench_id

            rec = {"id": bench_id, "type": "satellites", "source_object_id": sat_key}
            for field in INSTANCE_FIELDS:
                bm_key = FIELD_RENAME.get(field, field)
                val = sat.get(field)
                rec[bm_key] = _to_native(val) if val is not None else None
            instances.append(rec)

        np.save(mask_dir / "instance_map.npy", instance_map)
        _write_json(mask_dir / "instances.json", instances)

        img_rel = f"SB_maps/{s['galaxy_id']:05d}/{s['source_image_filename']}"
        s["image_relpath"] = img_rel
        s["mask_relpath"] = f"MASKS/{s['sample_id']}/instance_map.npy"
        s["instances_relpath"] = f"MASKS/{s['sample_id']}/instances.json"
        s["sample_manifest_relpath"] = f"MASKS/{s['sample_id']}/sample_manifest.json"

        _write_json(mask_dir / "sample_manifest.json", {
            "sample_id": s["sample_id"],
            "galaxy_id": s["galaxy_id"],
            "view": s["view"],
            "source_key": s["source_key"],
            "source_pickle": str(SOURCE_PICKLE_PATH),
            "source_sblim": SOURCE_SBLIM,
            "source_image_filter": SOURCE_FILTER,
            "image_relpath": img_rel,
            "source_image_realpath": s["source_image_realpath"],
            "image_shape": list(IMAGE_SHAPE),
            "coordinate_convention": COORDINATE_CONVENTION,
            "instance_count": s["instance_count"],
            "is_empty_sample": s["is_empty_sample"],
        })

    print(f"  ✓ {len(samples)} sample artifacts built")


# ─── Phase 5: Global Metadata ───────────────────────────────────────────────

def write_global_metadata(samples: List[dict], write_checksums: bool):
    """Write dataset_manifest.json, roi_definition.json, README.md."""
    manifest_samples = [
        {
            "sample_id": s["sample_id"],
            "galaxy_id": s["galaxy_id"],
            "view": s["view"],
            "source_key": s["source_key"],
            "image_relpath": s["image_relpath"],
            "mask_relpath": s["mask_relpath"],
            "instances_relpath": s["instances_relpath"],
            "sample_manifest_relpath": s["sample_manifest_relpath"],
            "instance_count": s["instance_count"],
            "is_empty_sample": s["is_empty_sample"],
        }
        for s in samples
    ]

    dataset_manifest: Dict[str, Any] = {
        "benchmark_name": BENCHMARK_NAME,
        "benchmark_version": BENCHMARK_VERSION,
        "source_pickle_path": str(SOURCE_PICKLE_PATH),
        "source_image_filter": SOURCE_FILTER,
        "source_sblim": SOURCE_SBLIM,
        "image_shape": list(IMAGE_SHAPE),
        "total_galaxies": EXPECTED["galaxies"],
        "total_samples": EXPECTED["samples"],
        "total_satellite_instances": EXPECTED["instances"],
        "empty_sample_count": EXPECTED["empty_samples"],
        "views": ["eo", "fo"],
        "naming_convention": {
            "sample_id_format": "{galaxy_id:05d}_{view}",
            "sample_id_example": "00011_eo",
            "sb_maps_directory": "SB_maps/{galaxy_id:05d}/ (zero-padded)",
            "image_filename": (
                "magnitudes-Fbox-{gid}-{view}-VIS.fits.gz "
                "(original, non-padded galaxy ID)"
            ),
            "note": (
                "Directory names are zero-padded to 5 digits. "
                "Image filenames preserve the original non-padded galaxy ID."
            ),
        },
        "samples": manifest_samples,
    }

    if write_checksums:
        checksums = {}
        for s in samples:
            mask_dir = BENCHMARK_ROOT / "MASKS" / s["sample_id"]
            for fname in ["instance_map.npy", "instances.json", "sample_manifest.json"]:
                fpath = mask_dir / fname
                if fpath.exists():
                    checksums[f"MASKS/{s['sample_id']}/{fname}"] = _sha256(fpath)
        dataset_manifest["generated_artifact_checksums"] = checksums

    _write_json(BENCHMARK_ROOT / "dataset_manifest.json", dataset_manifest)
    _write_json(BENCHMARK_ROOT / "roi_definition.json", ROI_DEFINITION)
    _write_readme()
    print("  ✓ Global metadata written")


def _write_readme():
    readme = """\
# Fbox Gold Satellites Benchmark V1

## What is this benchmark?

A **satellites-only** benchmark derived from the FIREbox cosmological simulation.
It provides **full-resolution gold annotations** for 132 samples (66 galaxies × 2
viewing angles) at the `SBlim31.5` surface-brightness detection threshold, totaling
426 satellite instances. 9 samples contain zero detected satellites and are explicitly
marked as empty.

This benchmark defines the **ground-truth data assets and protocol** only.
It does **not** include any evaluation implementation (metrics, ROI cropping, matching).
Evaluation code consumes this benchmark downstream.

## Where does it come from?

- **Source pickle**: `/shares/feldmann.ics.mnf.uzh/Lucas/pNbody/satellites/fbox/props_gals_Fbox_new.pkl`
- **Surface-brightness level**: `SBlim31.5`
- **Image filter**: `VIS` (Euclid VIS-band equivalent)
- **Image source**: `magnitudes-Fbox-{gid}-{view}-VIS.fits.gz` from the flat `sb_maps/` directory
- **Simulation**: FIREbox (Hopkins et al.), post-processed with pNbody/SKIRT radiative transfer

## What is the ground-truth coordinate system?

All coordinates are in the **full-resolution source frame**:

| Property | Value |
|---|---|
| Frame | 2051 × 2051 pixel, full FITS frame |
| Origin | Top-left corner (row 0, col 0) |
| Indexing | Zero-based |
| Mask index order | `[y, x]` (row, column) |
| Centroid fields | `x`, `y`, `geo_x`, `geo_y` — all float pixel coords in the same frame |

## Why is ±300 kpc important?

The physical analysis region of interest is a ±300 kpc box centered on each host galaxy.
At the simulation pixel scale of **0.64 kpc/pixel**, this corresponds to:

- Half-width: 468.75 source pixels
- Full width: 937.5 source pixels

The benchmark stores **full-frame** (2051 × 2051) truth annotations.
ROI restriction to the ±300 kpc region is applied **at evaluation time**, not here.
See `roi_definition.json` for the exact protocol parameters.

## What does this benchmark provide vs. not provide?

### Benchmark definition (this repository)

- `dataset_manifest.json` — global metadata and sample index
- `roi_definition.json` — ROI protocol parameters
- `SB_maps/{gid:05d}/` — canonical image entry points (symlinks to source FITS)
- `MASKS/{sample_id}/` — per-sample truth:
  - `instance_map.npy` — (2051, 2051) uint16 instance segmentation mask
  - `instances.json` — full physical properties for each satellite instance
  - `sample_manifest.json` — per-sample provenance and metadata

### Evaluation implementation (NOT included)

- Metric computation (IoU, detection F1, etc.)
- ROI cropping and boundary handling
- Prediction ↔ ground-truth matching
- Score thresholding and post-processing

## How to verify benchmark integrity

```bash
python scripts/data/build_fbox_gold_benchmark.py --verify-only
```

This checks: file existence, symlink validity, manifest consistency,
instance_map shape/dtype/ID continuity, and global statistic assertions.

## NFS dependency

By default, image entries under `SB_maps/` are **absolute-path symlinks** pointing to
files on the NFS mount at `/shares/feldmann.ics.mnf.uzh/`. If you need an offline-portable
copy, rebuild with:

```bash
python scripts/data/build_fbox_gold_benchmark.py --copy
```

## Instance map construction

- Instance IDs are assigned `1..N` in **numeric order** of the original pickle keys
  (`id1`, `id2`, ...), ensuring deterministic, reproducible builds.
- Pixel values are written from `seg_ids` coordinates; `seg_map` is not copied.
- By default, the build script **fails fast** if any two satellites within the same
  sample share overlapping pixels. Use `--allow-overlap` to permit last-writer-wins
  (higher numeric ID overwrites lower).
- `dtype=uint16`; V1 instance counts are well below the 65535 upper bound.
"""
    with open(BENCHMARK_ROOT / "README.md", "w") as f:
        f.write(readme)


# ─── Phase 6: Verification ──────────────────────────────────────────────────

def verify_benchmark(root: Path) -> bool:
    """Verify benchmark integrity. Returns True if all checks pass."""
    errors: List[str] = []

    # Root assets
    for name in ["README.md", "dataset_manifest.json", "roi_definition.json"]:
        if not (root / name).exists():
            errors.append(f"Missing root file: {name}")
    for dname in ["SB_maps", "MASKS"]:
        if not (root / dname).is_dir():
            errors.append(f"Missing root directory: {dname}")
    if errors:
        for e in errors:
            print(f"  ✗ {e}", file=sys.stderr)
        return False

    with open(root / "dataset_manifest.json") as f:
        manifest = json.load(f)

    # Global stats
    for field, exp in [
        ("total_galaxies", EXPECTED["galaxies"]),
        ("total_samples", EXPECTED["samples"]),
        ("total_satellite_instances", EXPECTED["instances"]),
        ("empty_sample_count", EXPECTED["empty_samples"]),
    ]:
        if manifest.get(field) != exp:
            errors.append(f"Manifest {field}: expected {exp}, got {manifest.get(field)}")

    samples = manifest.get("samples", [])
    if len(samples) != EXPECTED["samples"]:
        errors.append(f"Sample list length: expected {EXPECTED['samples']}, got {len(samples)}")

    total_inst = 0
    n_empty = 0
    galaxy_ids = set()

    for s in samples:
        sid = s["sample_id"]
        galaxy_ids.add(s["galaxy_id"])

        # Image entry
        img_path = root / s["image_relpath"]
        if not img_path.exists():
            tag = "broken symlink" if img_path.is_symlink() else "missing"
            errors.append(f"{sid}: image {tag}: {s['image_relpath']}")

        # Mask directory files
        mask_dir = root / "MASKS" / sid
        for fname in ["instance_map.npy", "instances.json", "sample_manifest.json"]:
            if not (mask_dir / fname).exists():
                errors.append(f"{sid}: missing {fname}")

        # instance_map checks
        imap_path = mask_dir / "instance_map.npy"
        if imap_path.exists():
            imap = np.load(imap_path)
            if imap.shape != IMAGE_SHAPE:
                errors.append(f"{sid}: instance_map shape {imap.shape} != {IMAGE_SHAPE}")
            if imap.dtype != np.uint16:
                errors.append(f"{sid}: instance_map dtype {imap.dtype} != uint16")

            # instances.json consistency
            inst_path = mask_dir / "instances.json"
            if inst_path.exists():
                with open(inst_path) as f:
                    instances = json.load(f)
                n_inst = len(instances)
                total_inst += n_inst

                if s["is_empty_sample"]:
                    n_empty += 1
                    if imap.max() != 0:
                        errors.append(f"{sid}: empty sample but instance_map max={imap.max()}")
                    if n_inst != 0:
                        errors.append(f"{sid}: empty sample but {n_inst} instances")
                else:
                    if int(imap.max()) != n_inst:
                        errors.append(
                            f"{sid}: max(instance_map)={imap.max()} != "
                            f"len(instances)={n_inst}"
                        )
                    # ID continuity
                    unique_ids = set(np.unique(imap)) - {0}
                    expected_ids = set(range(1, n_inst + 1))
                    if unique_ids != expected_ids:
                        errors.append(f"{sid}: non-contiguous IDs: {unique_ids} vs {expected_ids}")

        # sample_manifest cross-check
        sm_path = mask_dir / "sample_manifest.json"
        if sm_path.exists():
            with open(sm_path) as f:
                sm = json.load(f)
            if sm["galaxy_id"] != s["galaxy_id"]:
                errors.append(f"{sid}: galaxy_id mismatch manifest vs sample_manifest")
            if sm["view"] != s["view"]:
                errors.append(f"{sid}: view mismatch")
            if sm["instance_count"] != s["instance_count"]:
                errors.append(f"{sid}: instance_count mismatch")

    # Aggregate checks
    if len(galaxy_ids) != EXPECTED["galaxies"]:
        errors.append(f"Galaxy count: {len(galaxy_ids)} != {EXPECTED['galaxies']}")
    if total_inst != EXPECTED["instances"]:
        errors.append(f"Total instances: {total_inst} != {EXPECTED['instances']}")
    if n_empty != EXPECTED["empty_samples"]:
        errors.append(f"Empty samples: {n_empty} != {EXPECTED['empty_samples']}")

    if errors:
        print(f"\n  ✗ {len(errors)} verification errors:")
        for e in errors:
            print(f"    • {e}", file=sys.stderr)
        return False

    print("  ✓ All verification checks passed")
    return True


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build Fbox Gold Satellites Benchmark V1"
    )
    parser.add_argument("--verify-only", action="store_true",
                        help="Check existing benchmark integrity only")
    parser.add_argument("--copy", action="store_true",
                        help="Copy FITS files instead of symlinking")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="Allow seg_ids pixel overlap (last-writer-wins)")
    parser.add_argument("--write-checksums", action="store_true",
                        help="Embed SHA256 checksums in dataset_manifest.json")
    args = parser.parse_args()

    if args.verify_only:
        print("═══ Fbox Gold Benchmark — Verify Only ═══")
        if not BENCHMARK_ROOT.is_dir():
            _die(f"Benchmark root not found: {BENCHMARK_ROOT}")
        ok = verify_benchmark(BENCHMARK_ROOT)
        sys.exit(0 if ok else 1)

    print("═══ Fbox Gold Benchmark — Build ═══")
    print("\n── Phase 1: Load & Validate ──")
    data, samples = load_and_validate(SOURCE_PICKLE_PATH)

    print("\n── Phase 2–3: Structure & Image Entries ──")
    create_structure(samples, copy=args.copy)

    print("\n── Phase 4: Sample Artifacts ──")
    build_sample_artifacts(data, samples, allow_overlap=args.allow_overlap)

    print("\n── Phase 5: Global Metadata ──")
    write_global_metadata(samples, write_checksums=args.write_checksums)

    print("\n── Phase 6: Verification ──")
    ok = verify_benchmark(BENCHMARK_ROOT)
    if not ok:
        _die("Post-build verification failed")

    print("\n═══ Build Complete ═══")
    print(f"  Benchmark root: {BENCHMARK_ROOT}")


if __name__ == "__main__":
    main()
