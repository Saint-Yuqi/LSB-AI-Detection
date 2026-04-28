"""
Prepare a SAM3 evaluation dataset from the publicly available MATLAS HiPS
images plus the Sola+ 2025 polygon annotations (Table B1 / C2 / regions_all_features.reg).

Input
-----
- electronic_tables/Table_B1_galaxy_sample.txt    (475 galaxies, survey labels)
- electronic_tables/regions_all_features.reg      (DS9 polygons in ICRS, all 5 feature types)
- CDS HiPS service (hips2fits)                    (MATLAS r-band FITS + color JPG)

Output (under <out_root>/)
-------------------------
- fits/{galaxy}.fits.gz               mag/arcsec^2 FITS with WCS, matching the
                                      user's pipeline format
- png_linear_magnitude/{galaxy}.png   1024x1024 RGB PNG via LinearMagnitudePreprocessor
                                      (matches eval_checkpoint.yaml render config)
- png_asinh_stretch/{galaxy}.png      1024x1024 RGB PNG via LSBPreprocessor (asinh)
- png_color_preview/{galaxy}.jpg      MATLAS native 3-band RGB cutout (sanity-check)
- annotations/per_image/{galaxy}.json polygons in (RA,Dec) AND pixel coords + class
- annotations/instances_coco.json     COCO instance segmentation for the whole set
- annotations/taxonomy.json           class name -> id mapping
- debug_overlays/{galaxy}.png         polygons drawn on the linear-mag PNG
- manifest.csv                        per-galaxy bookkeeping

Phases
------
Phases run in order; each is idempotent (skipped if outputs exist) and
selectable via --phase.
    parse      : parse regions + Table_B1, write galaxy_index.json
    download   : pull FITS + color JPG from hips2fits per galaxy
    magnitude  : convert raw HiPS FITS -> mag/arcsec^2 FITS
    render     : run user's LinearMagnitude + Asinh preprocessors -> PNG
    annotate   : compute polygon pixel coords + per-image JSON + COCO + overlays
    manifest   : write manifest.csv
    all        : run everything (default)
"""
from __future__ import annotations

import argparse
import csv
import gzip
import io
import json
import logging
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import requests
from astropy.io import fits
from astropy.wcs import WCS

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.preprocessing import LinearMagnitudePreprocessor, LSBPreprocessor  # noqa: E402

LOG = logging.getLogger("matlas_sola")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# CDS hips2fits service (returns FITS or JPEG re-projected to a TAN WCS we specify).
HIPS2FITS_URL = "https://alasky.cds.unistra.fr/hips-image-services/hips2fits"
HIPS_R = "CDS/P/MATLAS/r"
HIPS_G = "CDS/P/MATLAS/g"
HIPS_I = "CDS/P/MATLAS/i"
HIPS_COLOR = "CDS/P/MATLAS/color"

# Effective AB zeropoint for the MATLAS HiPS counts.
#
# IMPORTANT: the MATLAS HiPS does NOT advertise a photometric zeropoint in
# its metadata, so the value below is *empirically calibrated* against the
# 136 Stream/Tail features in Sola+ 2025 Table_C2 (each row publishes the
# author's median SB in mag/arcsec^2 inside the polygon). With ZP = 23.45,
# our SB values match Table_C2 with bias ~ 0.0 mag and 1-sigma scatter 0.4 mag.
# Naive textbook "MegaPipe AB ZP = 30" gave a systematic +6.6 mag offset
# (renders too dim, sky values pushed into the [20, 35] linear_magnitude
# saturation floor). Override at the CLI with --zeropoint if you re-derive it.
#
# Conversion:
#   mag/arcsec^2 = -2.5*log10(counts) + ZP + 2.5*log10(1 / pix_area_arcsec^2)
# with native MegaCam pixel area (0.187"/pix)^2 = 0.0349 arcsec^2.
DEFAULT_ZP = 23.45
NATIVE_PIX_ARCSEC = 0.187  # MATLAS / MegaCam

# Cutout geometry strategy ----------------------------------------------------
# Per Elisabeth Sola (private comm., 2026-04-28): annotators always worked on
# a fixed 31' × 31' cutout per host galaxy in Jafar. We replicate exactly that
# field so polygons-outside-image and image-outside-inspected-region cases
# both vanish — i.e. polygon-free pixels inside the 31' frame are confirmed
# feature-free, not "untouched".
TARGET_PIXELS = 1024              # final image side (matches SAM3 target_size)
FIXED_FOV_ARCMIN = 31.0           # Sola+ Jafar standard cutout
COLOR_PREVIEW_PX = 1024

# Five-class taxonomy. IDs 0-4. SAM3 fine-tune used "stellar streams" only;
# the user wants raw labels to enable diagnostic re-binding later.
TAXONOMY: dict[str, int] = {
    "Inner": 0,
    "Halo":  1,
    "Shells": 2,
    "Stream": 3,
    "Tail":  4,
}
TAXONOMY_INV = {v: k for k, v in TAXONOMY.items()}

# Surveys we can pull from MATLAS HiPS. CFIS,MATLAS galaxies are in the MATLAS
# footprint per Sola+2025 (5 of them) and we pull them too. CFIS-only / VESTIGE /
# NGVS are proprietary -> skipped.
ELIGIBLE_SURVEYS = {"MATLAS", "CFIS,MATLAS"}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Polygon:
    feature_id: int
    feature_type: str            # one of TAXONOMY keys
    galaxy: str
    radec: list[tuple[float, float]]   # ICRS (deg, deg)
    shape: str = "polygon"       # "polygon" (closed area) or "line" (open polyline; Shells)


@dataclass
class GalaxyInfo:
    name: str
    ra_deg: float
    dec_deg: float
    survey: str
    bands: str
    polygons: list[Polygon] = field(default_factory=list)

    # Filled by phase=parse / select_cutout
    fov_arcmin: float = 0.0
    pix_arcsec: float = 0.0
    bbox_radec: tuple[float, float, float, float] | None = None  # (ra_min, ra_max, dec_min, dec_max)


# ---------------------------------------------------------------------------
# Phase: parse regions + galaxy table
# ---------------------------------------------------------------------------

# Header line: "# Feature<id> <galaxy> <type...>"  (type may be multi-word, e.g. "Inner Galaxy")
REGION_LINE_RE = re.compile(
    r"^#\s*Feature(?P<fid>\d+)\s+(?P<gal>\S+)\s+(?P<type>.+?)\s*$"
)
SHAPE_LINE_RE = re.compile(r"^(?P<shape>polygon|line)\(([^)]+)\)")

# Sola+ uses "Inner Galaxy" in the .reg; map to single-word taxonomy key "Inner".
TYPE_NORMALIZE = {
    "Inner Galaxy": "Inner",
    "Inner":  "Inner",
    "Halo":   "Halo",
    "Shells": "Shells",
    "Stream": "Stream",
    "Tail":   "Tail",
}


def parse_regions(reg_path: Path) -> list[Polygon]:
    polys: list[Polygon] = []
    pending: tuple[int, str, str] | None = None
    skipped_unknown_type = 0
    with reg_path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            m = REGION_LINE_RE.match(line)
            if m:
                ftype_raw = m.group("type").strip()
                ftype = TYPE_NORMALIZE.get(ftype_raw)
                if ftype is None:
                    LOG.warning("unknown feature type %r — skipping Feature%s",
                                ftype_raw, m.group("fid"))
                    skipped_unknown_type += 1
                    pending = None
                    continue
                pending = (int(m.group("fid")), m.group("gal"), ftype)
                continue
            m = SHAPE_LINE_RE.match(line)
            if m and pending is not None:
                fid, gal, ftype = pending
                raw_vals = m.group(2).split(",")
                try:
                    vals = [float(x) for x in raw_vals]
                except ValueError:
                    LOG.warning("non-numeric coords for Feature%d %s — skipping", fid, gal)
                    pending = None
                    continue
                if len(vals) < 4 or len(vals) % 2:
                    LOG.warning("bad coord count for Feature%d %s (n=%d) — skipping",
                                fid, gal, len(vals))
                    pending = None
                    continue
                pts = list(zip(vals[0::2], vals[1::2]))
                polys.append(Polygon(fid, ftype, gal, pts, shape=m.group("shape")))
                pending = None
    LOG.info("parsed %d shapes (%d polygons, %d lines), skipped %d unknown",
             len(polys),
             sum(1 for p in polys if p.shape == "polygon"),
             sum(1 for p in polys if p.shape == "line"),
             skipped_unknown_type)
    return polys


def parse_table_b1(path: Path) -> dict[str, dict[str, Any]]:
    """Parse fixed-width Table_B1 per its ReadMe byte-by-byte schema."""
    out: dict[str, dict[str, Any]] = {}
    with path.open() as fh:
        for raw in fh:
            if len(raw) < 78:
                continue
            galaxy = raw[0:8].strip()
            try:
                ra = float(raw[9:16].strip())
                dec = float(raw[17:24].strip())
            except ValueError:
                continue
            survey = raw[63:75].strip()
            bands = raw[76:78].strip()
            out[galaxy] = {
                "name": galaxy,
                "ra_deg": ra,
                "dec_deg": dec,
                "survey": survey,
                "bands": bands,
            }
    LOG.info("parsed %d galaxies from %s", len(out), path)
    return out


def assemble_galaxies(table_b1: dict[str, dict[str, Any]],
                      polygons: list[Polygon]) -> dict[str, GalaxyInfo]:
    """Filter to MATLAS-eligible galaxies and attach polygons."""
    galaxies: dict[str, GalaxyInfo] = {}
    for name, row in table_b1.items():
        if row["survey"] not in ELIGIBLE_SURVEYS:
            continue
        galaxies[name] = GalaxyInfo(
            name=name, ra_deg=row["ra_deg"], dec_deg=row["dec_deg"],
            survey=row["survey"], bands=row["bands"],
        )
    for p in polygons:
        if p.galaxy in galaxies:
            galaxies[p.galaxy].polygons.append(p)
    LOG.info("eligible galaxies: %d (with polygons: %d)",
             len(galaxies), sum(1 for g in galaxies.values() if g.polygons))
    return galaxies


def compute_cutout_geometry(g: GalaxyInfo) -> None:
    """Assign the fixed 31' × 31' Sola+ Jafar cutout to every galaxy."""
    g.fov_arcmin = FIXED_FOV_ARCMIN
    g.pix_arcsec = (g.fov_arcmin * 60.0) / TARGET_PIXELS  # ~1.816"/pix
    if g.polygons:
        ras = np.array([pt[0] for p in g.polygons for pt in p.radec])
        decs = np.array([pt[1] for p in g.polygons for pt in p.radec])
        g.bbox_radec = (float(ras.min()), float(ras.max()),
                        float(decs.min()), float(decs.max()))
    else:
        g.bbox_radec = None


def write_galaxy_index(galaxies: dict[str, GalaxyInfo], path: Path) -> None:
    payload = {
        name: {
            "name": g.name,
            "ra_deg": g.ra_deg,
            "dec_deg": g.dec_deg,
            "survey": g.survey,
            "bands": g.bands,
            "fov_arcmin": g.fov_arcmin,
            "pix_arcsec": g.pix_arcsec,
            "bbox_radec": g.bbox_radec,
            "polygons": [
                {"feature_id": p.feature_id, "type": p.feature_type,
                 "shape": p.shape, "radec": p.radec}
                for p in g.polygons
            ],
        }
        for name, g in galaxies.items()
    }
    path.write_text(json.dumps(payload, indent=2))
    LOG.info("wrote galaxy index: %s", path)


def load_galaxy_index(path: Path) -> dict[str, GalaxyInfo]:
    raw = json.loads(path.read_text())
    galaxies: dict[str, GalaxyInfo] = {}
    for name, d in raw.items():
        polys = [
            Polygon(int(p["feature_id"]), p["type"], name,
                    [tuple(pt) for pt in p["radec"]],
                    shape=p.get("shape", "polygon"))
            for p in d.get("polygons", [])
        ]
        galaxies[name] = GalaxyInfo(
            name=d["name"], ra_deg=d["ra_deg"], dec_deg=d["dec_deg"],
            survey=d["survey"], bands=d["bands"],
            fov_arcmin=d["fov_arcmin"], pix_arcsec=d["pix_arcsec"],
            bbox_radec=tuple(d["bbox_radec"]) if d.get("bbox_radec") else None,
            polygons=polys,
        )
    return galaxies


# ---------------------------------------------------------------------------
# Phase: download
# ---------------------------------------------------------------------------


def hips2fits_request(hips: str, ra: float, dec: float, fov_deg: float,
                      width: int, height: int, fmt: str = "fits",
                      stretch: str | None = None,
                      cmap: str | None = None,
                      session: requests.Session | None = None,
                      timeout: float = 180.0,
                      retries: int = 3) -> bytes:
    params = {
        "hips": hips,
        "ra": f"{ra:.6f}",
        "dec": f"{dec:.6f}",
        "fov": f"{fov_deg:.6f}",
        "width": str(width),
        "height": str(height),
        "projection": "TAN",
        "coordsys": "icrs",
        "format": fmt,
    }
    if stretch:
        params["stretch"] = stretch
    if cmap:
        params["cmap"] = cmap
    sess = session or requests.Session()
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            r = sess.get(HIPS2FITS_URL, params=params, timeout=timeout)
            r.raise_for_status()
            return r.content
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = 5 * (attempt + 1)
            LOG.warning("hips2fits attempt %d failed (%s) — retrying in %ds", attempt + 1, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"hips2fits failed after {retries} retries: {last_exc}")


def download_one(g: GalaxyInfo, out_dir: Path, raw_subdir: str = "raw_fits",
                 force: bool = False) -> dict[str, Any]:
    """Pull r-band FITS + color JPG. Returns small status dict."""
    raw_dir = out_dir / raw_subdir
    raw_dir.mkdir(parents=True, exist_ok=True)
    color_dir = out_dir / "png_color_preview"
    color_dir.mkdir(parents=True, exist_ok=True)

    fits_path = raw_dir / f"{g.name}_r_raw.fits"
    color_path = color_dir / f"{g.name}.jpg"

    fov_deg = g.fov_arcmin / 60.0
    status = {"galaxy": g.name, "fits_ok": fits_path.exists(),
              "color_ok": color_path.exists()}

    sess = requests.Session()

    if force or not fits_path.exists():
        LOG.info("[download] %s r-band FITS  (%.1f arcmin, %.2f\"/px)",
                 g.name, g.fov_arcmin, g.pix_arcsec)
        body = hips2fits_request(HIPS_R, g.ra_deg, g.dec_deg, fov_deg,
                                 TARGET_PIXELS, TARGET_PIXELS,
                                 fmt="fits", session=sess)
        fits_path.write_bytes(body)
        status["fits_ok"] = True

    if force or not color_path.exists():
        LOG.info("[download] %s color JPG", g.name)
        body = hips2fits_request(HIPS_COLOR, g.ra_deg, g.dec_deg, fov_deg,
                                 COLOR_PREVIEW_PX, COLOR_PREVIEW_PX,
                                 fmt="jpg", session=sess)
        color_path.write_bytes(body)
        status["color_ok"] = True

    return status


def run_download_phase(galaxies: dict[str, GalaxyInfo], out_dir: Path,
                        max_workers: int = 4, force: bool = False) -> None:
    items = list(galaxies.values())
    LOG.info("downloading %d galaxies (workers=%d)", len(items), max_workers)
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(download_one, g, out_dir, force=force): g.name for g in items}
        for fut in as_completed(futs):
            name = futs[fut]
            try:
                fut.result()
            except Exception as exc:  # noqa: BLE001
                LOG.error("download FAILED for %s: %s", name, exc)
                failures.append(name)
    if failures:
        LOG.warning("download failures: %s", failures)
        (out_dir / "download_failures.json").write_text(json.dumps(failures, indent=2))


# ---------------------------------------------------------------------------
# Phase: magnitude conversion
# ---------------------------------------------------------------------------


def counts_to_sb(counts: np.ndarray, zeropoint: float = DEFAULT_ZP,
                 pix_arcsec: float = NATIVE_PIX_ARCSEC) -> np.ndarray:
    """
    Convert HiPS r-band counts to surface brightness mag/arcsec^2.

    HiPS preserves per-native-pixel flux values regardless of the resampled
    output grid, so the per-arcsec^2 conversion uses MATLAS' native pixel area.

    Pixels with non-positive counts are mapped to 35.0 (~ at/below the SB
    detection limit) so the user's preprocessor renders them as black.
    """
    pix_area = pix_arcsec ** 2
    out = np.full_like(counts, 35.0, dtype=np.float32)
    pos = counts > 0
    out[pos] = (-2.5 * np.log10(counts[pos]) + zeropoint
                - 2.5 * np.log10(pix_area)).astype(np.float32)
    out = np.where(np.isfinite(out), out, 35.0).astype(np.float32)
    return out


def write_sb_fits_gz(sb_map: np.ndarray, header: fits.Header, path: Path) -> None:
    """Write float32 mag/arcsec^2 FITS, gzipped, preserving WCS."""
    hdu = fits.PrimaryHDU(data=sb_map.astype(np.float32), header=header)
    hdu.header["BUNIT"] = ("mag/arcsec^2", "surface brightness")
    hdu.header["COMMENT"] = "MATLAS HiPS r-band counts converted to mag/arcsec^2"
    hdu.header["COMMENT"] = f"ZP={DEFAULT_ZP} (AB), native pix={NATIVE_PIX_ARCSEC}\""
    buf = io.BytesIO()
    hdu.writeto(buf, overwrite=True)
    buf.seek(0)
    with gzip.open(path, "wb") as gz:
        gz.write(buf.read())


def run_magnitude_phase(galaxies: dict[str, GalaxyInfo], out_dir: Path,
                        zeropoint: float = DEFAULT_ZP, force: bool = False) -> None:
    raw_dir = out_dir / "raw_fits"
    sb_dir = out_dir / "fits"
    sb_dir.mkdir(parents=True, exist_ok=True)
    for g in galaxies.values():
        raw_path = raw_dir / f"{g.name}_r_raw.fits"
        sb_path = sb_dir / f"{g.name}.fits.gz"
        if not raw_path.exists():
            LOG.warning("[magnitude] missing raw FITS for %s — skipping", g.name)
            continue
        if sb_path.exists() and not force:
            continue
        with fits.open(raw_path) as hdul:
            data = hdul[0].data
            header = hdul[0].header.copy()
        sb = counts_to_sb(np.asarray(data, dtype=np.float64), zeropoint=zeropoint,
                          pix_arcsec=NATIVE_PIX_ARCSEC)
        write_sb_fits_gz(sb, header, sb_path)
        LOG.info("[magnitude] %s -> %s (median SB=%.2f)",
                 g.name, sb_path.name, float(np.nanmedian(sb)))


# ---------------------------------------------------------------------------
# Phase: render PNGs through user's preprocessors
# ---------------------------------------------------------------------------


def run_render_phase(galaxies: dict[str, GalaxyInfo], out_dir: Path,
                     force: bool = False) -> None:
    import cv2
    sb_dir = out_dir / "fits"
    lin_dir = out_dir / "png_linear_magnitude"
    asinh_dir = out_dir / "png_asinh_stretch"
    lin_dir.mkdir(parents=True, exist_ok=True)
    asinh_dir.mkdir(parents=True, exist_ok=True)

    lin_proc = LinearMagnitudePreprocessor(global_mag_min=20.0, global_mag_max=35.0,
                                           target_size=(TARGET_PIXELS, TARGET_PIXELS))
    asinh_proc = LSBPreprocessor(zeropoint=22.5, nonlinearity=200.0,
                                 clip_percentile=99.5,
                                 target_size=(TARGET_PIXELS, TARGET_PIXELS))

    for g in galaxies.values():
        sb_path = sb_dir / f"{g.name}.fits.gz"
        if not sb_path.exists():
            continue
        with gzip.open(sb_path, "rb") as gz, fits.open(gz) as hdul:
            sb = hdul[0].data.astype(np.float64)

        lin_path = lin_dir / f"{g.name}.png"
        if force or not lin_path.exists():
            rgb = lin_proc.process(sb)
            cv2.imwrite(str(lin_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

        asinh_path = asinh_dir / f"{g.name}.png"
        if force or not asinh_path.exists():
            rgb = asinh_proc.process(sb)
            cv2.imwrite(str(asinh_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    LOG.info("[render] done")


# ---------------------------------------------------------------------------
# Phase: annotate (polygon RA/Dec -> pixel, COCO + per-image JSON + overlays)
# ---------------------------------------------------------------------------


def polygons_to_pixel(g: GalaxyInfo, wcs: WCS) -> list[dict[str, Any]]:
    """Convert a galaxy's polygons (RA/Dec) to pixel coords using WCS."""
    out = []
    for p in g.polygons:
        ras = np.array([pt[0] for pt in p.radec])
        decs = np.array([pt[1] for pt in p.radec])
        x, y = wcs.all_world2pix(ras, decs, 0)
        out.append({
            "feature_id": p.feature_id,
            "type": p.feature_type,
            "type_id": TAXONOMY[p.feature_type],
            "shape": p.shape,
            "radec": p.radec,
            "polygon_xy": list(zip([float(v) for v in x], [float(v) for v in y])),
        })
    return out


# Per Elisabeth Sola (private comm.): Shells were drawn as single-pixel-wide
# polylines in the native MATLAS frame (0.187"/px), i.e. the polyline IS the
# shell — not a region. At our 31'/1024 = 1.816"/px output scale, native
# 1 px ≈ 0.10 output px, so a thickness=1 raster is the strictest faithful
# reading. For IoU-based evaluation against blob-shaped predictions, we
# expose --shell-eval-thickness-px (default 3) which dilates the line in
# the COCO mask so it can be matched. The polyline geometry itself is
# preserved verbatim in annotations/per_image/*.json.
SHELL_NATIVE_THICKNESS_PX = 1
DEFAULT_SHELL_EVAL_THICKNESS_PX = 3


def rasterize_shape(poly_xy: list[tuple[float, float]],
                    shape_kind: str,
                    image_shape: tuple[int, int],
                    shell_thickness_px: int = DEFAULT_SHELL_EVAL_THICKNESS_PX) -> np.ndarray:
    """
    Rasterize either a closed polygon (filled) or a Sola+ Shell polyline
    (open polyline with author-confirmed single-pixel-wide-by-design width,
    optionally dilated for IoU evaluation).
    """
    import cv2
    mask = np.zeros(image_shape, dtype=np.uint8)
    pts = np.array(poly_xy, dtype=np.int32).reshape(-1, 1, 2)
    if shape_kind == "polygon":
        cv2.fillPoly(mask, [pts], color=1)
    else:  # Sola+ shell: native 1-px polyline, dilated to thickness for IoU
        cv2.polylines(mask, [pts], isClosed=False, color=1,
                      thickness=max(1, shell_thickness_px))
    return mask


def encode_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a binary mask as uncompressed COCO RLE (column-major)."""
    pixels = mask.flatten(order="F")
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": runs.tolist()}


def draw_overlay(galaxy: GalaxyInfo, base_png: Path, polygons_xy: list[dict[str, Any]],
                 out_path: Path) -> None:
    """
    Draw polygon outlines on the given base PNG (asinh_stretch by default —
    higher dynamic range, stream/shell structure visible). Per-polygon
    feature IDs are listed once in a top-left legend instead of being
    written next to each shape, to keep the overlay readable.
    """
    import cv2
    img = cv2.imread(str(base_png))
    if img is None:
        LOG.warning("overlay skipped: missing PNG %s", base_png)
        return
    # OpenCV is BGR; pick distinct colors per class.
    color_map = {
        "Inner":  (0, 255, 255),    # cyan
        "Halo":   (0, 200, 255),    # orange
        "Shells": (0, 255, 255),    # yellow
        "Stream": (0, 255, 0),      # green
        "Tail":   (255, 0, 255),    # magenta
    }
    # 1. Draw shapes only (no per-shape labels)
    for p in polygons_xy:
        pts = np.array(p["polygon_xy"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(img, [pts],
                      isClosed=(p.get("shape", "polygon") == "polygon"),
                      color=color_map.get(p["type"], (255, 255, 255)),
                      thickness=2)

    # 2. Top-left legend: galaxy name + per-class feature-ID list
    h, w = img.shape[:2]
    pad = 8
    line_h = 18
    legend_lines: list[tuple[str, tuple[int, int, int]]] = [
        (f"{galaxy.name}  ({len(polygons_xy)} features, "
         f"FoV={galaxy.fov_arcmin:.1f}'  px={galaxy.pix_arcsec:.2f}\")",
         (255, 255, 255)),
    ]
    by_class: dict[str, list[int]] = {}
    for p in polygons_xy:
        by_class.setdefault(p["type"], []).append(p["feature_id"])
    for cls in ("Inner", "Halo", "Shells", "Stream", "Tail"):
        if cls not in by_class:
            continue
        ids = sorted(by_class[cls])
        ids_str = ",".join(f"#{i}" for i in ids)
        legend_lines.append((f"{cls}: {ids_str}", color_map[cls]))

    # background bar so text stays readable on bright PNGs
    bar_h = pad * 2 + line_h * len(legend_lines)
    bar_w = min(w - 2 * pad, 720)
    overlay = img.copy()
    cv2.rectangle(overlay, (pad, pad), (pad + bar_w, pad + bar_h),
                  (0, 0, 0), thickness=-1)
    img = cv2.addWeighted(overlay, 0.55, img, 0.45, 0)

    y = pad + line_h
    for text, color in legend_lines:
        cv2.putText(img, text, (pad + 6, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    color, 1, cv2.LINE_AA)
        y += line_h

    cv2.imwrite(str(out_path), img)


def run_annotate_phase(galaxies: dict[str, GalaxyInfo], out_dir: Path,
                       force: bool = False,
                       overlay_variant: str = "asinh_stretch",
                       shell_thickness_px: int = DEFAULT_SHELL_EVAL_THICKNESS_PX) -> None:
    sb_dir = out_dir / "fits"
    overlay_base_dir = out_dir / f"png_{overlay_variant}"
    per_img_dir = out_dir / "annotations" / "per_image"
    overlay_dir = out_dir / "debug_overlays"
    per_img_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    coco = {
        "info": {
            "description": "MATLAS Sola+2025 tidal-feature evaluation set",
            "source": "Sola+2025 MNRAS Table B1/C2 + regions_all_features.reg, "
                       "MATLAS HiPS via CDS hips2fits",
        },
        "categories": [{"id": v, "name": k, "supercategory": "lsb_feature"}
                       for k, v in TAXONOMY.items()],
        "images": [],
        "annotations": [],
    }
    img_id = 0
    ann_id = 0

    for g in sorted(galaxies.values(), key=lambda x: x.name):
        sb_path = sb_dir / f"{g.name}.fits.gz"
        if not sb_path.exists():
            continue
        with gzip.open(sb_path, "rb") as gz, fits.open(gz) as hdul:
            header = hdul[0].header
        wcs = WCS(header)

        polys_xy = polygons_to_pixel(g, wcs)

        per_img_payload = {
            "galaxy": g.name,
            "ra_deg": g.ra_deg,
            "dec_deg": g.dec_deg,
            "fov_arcmin": g.fov_arcmin,
            "pix_arcsec": g.pix_arcsec,
            "image_size": [TARGET_PIXELS, TARGET_PIXELS],
            "taxonomy": TAXONOMY,
            "polygons": polys_xy,
        }
        per_img_path = per_img_dir / f"{g.name}.json"
        if force or not per_img_path.exists():
            per_img_path.write_text(json.dumps(per_img_payload, indent=2))

        overlay_path = overlay_dir / f"{g.name}.png"
        if force or not overlay_path.exists():
            draw_overlay(g, overlay_base_dir / f"{g.name}.png", polys_xy, overlay_path)

        # COCO image record
        img_id += 1
        coco["images"].append({
            "id": img_id,
            "file_name": f"png_linear_magnitude/{g.name}.png",
            "width": TARGET_PIXELS,
            "height": TARGET_PIXELS,
            "galaxy": g.name,
            "ra_deg": g.ra_deg,
            "dec_deg": g.dec_deg,
            "fov_arcmin": g.fov_arcmin,
            "pix_arcsec": g.pix_arcsec,
        })
        for p in polys_xy:
            mask = rasterize_shape(p["polygon_xy"], p["shape"],
                                   (TARGET_PIXELS, TARGET_PIXELS),
                                   shell_thickness_px=shell_thickness_px)
            if mask.sum() == 0:
                continue
            ys, xs = np.where(mask)
            x0, x1 = int(xs.min()), int(xs.max())
            y0, y1 = int(ys.min()), int(ys.max())
            ann_id += 1
            poly_flat = [c for pt in p["polygon_xy"] for c in pt]
            coco["annotations"].append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": p["type_id"],
                "feature_id": p["feature_id"],
                "type": p["type"],
                "shape": p["shape"],
                "iscrowd": 0,
                "bbox": [x0, y0, x1 - x0 + 1, y1 - y0 + 1],
                "area": int(mask.sum()),
                "segmentation": [poly_flat],
                "rle": encode_rle(mask),
            })

    coco_path = out_dir / "annotations" / "instances_coco.json"
    coco_path.write_text(json.dumps(coco))
    LOG.info("[annotate] wrote COCO with %d images / %d annotations",
             len(coco["images"]), len(coco["annotations"]))

    tax_path = out_dir / "annotations" / "taxonomy.json"
    tax_path.write_text(json.dumps({"name_to_id": TAXONOMY,
                                    "id_to_name": TAXONOMY_INV},
                                   indent=2))


# ---------------------------------------------------------------------------
# Phase: manifest
# ---------------------------------------------------------------------------


def run_manifest_phase(galaxies: dict[str, GalaxyInfo], out_dir: Path) -> None:
    rows = []
    for g in sorted(galaxies.values(), key=lambda x: x.name):
        counts = {k: 0 for k in TAXONOMY}
        for p in g.polygons:
            counts[p.feature_type] = counts.get(p.feature_type, 0) + 1
        rows.append({
            "galaxy": g.name, "ra_deg": g.ra_deg, "dec_deg": g.dec_deg,
            "survey": g.survey, "bands": g.bands,
            "fov_arcmin": round(g.fov_arcmin, 3),
            "pix_arcsec": round(g.pix_arcsec, 4),
            "n_polygons": len(g.polygons),
            **{f"n_{k.lower()}": counts[k] for k in TAXONOMY},
        })
    path = out_dir / "manifest.csv"
    if not rows:
        LOG.warning("manifest empty (no galaxies)")
        return
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOG.info("[manifest] wrote %s (%d rows)", path, len(rows))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--electronic-tables",
                    default="/shares/feldmann.ics.mnf.uzh/data/LSB_and_Satellites/electronic_tables",
                    type=Path,
                    help="Directory containing Table_B1 and regions_all_features.reg")
    ap.add_argument("--out",
                    default=str(PROJECT_ROOT / "data/01_raw/MATLAS_Sola2025_eval"),
                    type=Path)
    ap.add_argument("--phase", choices=["parse", "download", "magnitude",
                                         "render", "annotate", "manifest", "all"],
                    default="all")
    ap.add_argument("--galaxies", default=None,
                    help="Comma-separated subset of galaxy names (debug)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--zeropoint", type=float, default=DEFAULT_ZP)
    ap.add_argument("--shell-eval-thickness-px", type=int,
                    default=DEFAULT_SHELL_EVAL_THICKNESS_PX,
                    help="Dilation thickness when rasterizing Sola+ shell polylines "
                         "for IoU evaluation. Native authoring width is 1 px in the "
                         "MATLAS frame; default 3 px makes the GT mask matchable "
                         "against blob-shaped predictions.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download and re-render even if outputs exist")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(message)s",
    )

    args.out.mkdir(parents=True, exist_ok=True)
    index_path = args.out / "galaxy_index.json"

    # ----- parse phase -----
    if args.phase in ("parse", "all"):
        polys = parse_regions(args.electronic_tables / "regions_all_features.reg")
        b1 = parse_table_b1(args.electronic_tables / "Table_B1_galaxy_sample.txt")
        galaxies = assemble_galaxies(b1, polys)
        for g in galaxies.values():
            compute_cutout_geometry(g)
        write_galaxy_index(galaxies, index_path)
    else:
        if not index_path.exists():
            LOG.error("galaxy_index.json missing — run --phase parse first")
            return 2
        galaxies = load_galaxy_index(index_path)

    if args.galaxies:
        keep = set(args.galaxies.split(","))
        galaxies = {k: v for k, v in galaxies.items() if k in keep}
        LOG.info("subset: %d galaxies", len(galaxies))

    # ----- subsequent phases -----
    if args.phase in ("download", "all"):
        run_download_phase(galaxies, args.out, max_workers=args.workers,
                           force=args.force)
    if args.phase in ("magnitude", "all"):
        run_magnitude_phase(galaxies, args.out, zeropoint=args.zeropoint,
                            force=args.force)
    if args.phase in ("render", "all"):
        run_render_phase(galaxies, args.out, force=args.force)
    if args.phase in ("annotate", "all"):
        run_annotate_phase(galaxies, args.out, force=args.force,
                           shell_thickness_px=args.shell_eval_thickness_px)
    if args.phase in ("manifest", "all"):
        run_manifest_phase(galaxies, args.out)

    LOG.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
