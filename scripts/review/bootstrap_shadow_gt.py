#!/usr/bin/env python3
"""
Bootstrap a Shadow GT root for the satellite-override migration flow.

The Shadow GT is a physically separate ``gt_canonical/current`` tree used to
host the post-review authoritative labels. It is built by:

    1. File-level symlinking the render inputs needed by the SAM3 inference
       pipeline (``{preprocessing}/{base_key}/0000.png``) from the authoritative
       render root.
    2. Copying ``streams_instance_map.npy`` and ``manifest.json`` from the
       authoritative GT so streams stay aligned.
    3. NEVER copying authoritative satellite artifacts
       (``instance_map_uint8.png``, ``instances.json``, ``id_map.json``,
       ``sam3_predictions_*.json``, overlays, diagnostics), so the subsequent
       pure SAM3 evaluate run writes clean satellite artifacts from native
       checkpoint output only.

After this script, point a DR1 SAM3 ``inference`` run at
``--output-root {shadow_root}``; afterwards apply
``scripts/review/migrate_satellite_overrides.py`` to import reviewed
exceptions as explicit human-adopted instances.

Usage:
    conda run --no-capture-output -n sam3 python scripts/review/bootstrap_shadow_gt.py \\
        --shadow-root scratch/gt_shadow \\
        --preprocessing linear_magnitude
    conda run --no-capture-output -n sam3 python scripts/review/bootstrap_shadow_gt.py --galaxies 11,13
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import setup_logger
from src.utils.runtime_env import assert_expected_conda_env

DEFAULT_SOURCE_GT = PROJECT_ROOT / "data" / "02_processed" / "gt_canonical" / "current"
DEFAULT_SOURCE_RENDERS = PROJECT_ROOT / "data" / "02_processed" / "renders" / "current"
DEFAULT_SHADOW_ROOT = PROJECT_ROOT / "scratch" / "gt_shadow"

_SATELLITE_ARTIFACTS_THAT_MUST_NOT_BE_COPIED = (
    "instance_map_uint8.png",
    "instances.json",
    "id_map.json",
    "sam3_predictions_raw.json",
    "sam3_predictions_post.json",
    "sam3_raw_overlay.png",
    "sam3_eval_overlay.png",
    "sam3_satellite_diagnostics.json",
    "overlay.png",
)


def _file_symlink(src: Path, dst: Path) -> None:
    """Create or refresh a file-level symlink; refuses to overwrite real files."""
    if not src.exists():
        raise FileNotFoundError(f"symlink source missing: {src}")
    if not src.is_file():
        raise ValueError(f"refusing to create directory symlink for {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    elif dst.exists():
        raise FileExistsError(
            f"refusing to overwrite non-symlink file at {dst}; "
            "delete manually if you really intend to rebuild the shadow"
        )
    dst.symlink_to(src.resolve())


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)


def _assert_no_satellite_leakage(shadow_gt_dir: Path) -> None:
    for name in _SATELLITE_ARTIFACTS_THAT_MUST_NOT_BE_COPIED:
        path = shadow_gt_dir / name
        if path.exists():
            raise RuntimeError(
                f"Shadow GT leak: {path} exists. The shadow must be rebuilt "
                f"from scratch without authoritative satellite artifacts."
            )


def _list_base_keys(source_gt: Path, galaxy_filter: set[int] | None) -> list[str]:
    keys = sorted(p.name for p in source_gt.iterdir() if p.is_dir())
    if galaxy_filter:
        keys = [k for k in keys if int(k.split("_")[0]) in galaxy_filter]
    return keys


def bootstrap(
    *,
    source_gt: Path,
    source_renders: Path,
    shadow_root: Path,
    preprocessing: str,
    galaxy_filter: set[int] | None,
    logger,
) -> dict[str, int]:
    shadow_gt_root = shadow_root / "gt_canonical" / "current"
    shadow_render_root = shadow_root / "renders" / "current" / preprocessing
    source_render_variant = source_renders / preprocessing

    base_keys = _list_base_keys(source_gt, galaxy_filter)
    if not base_keys:
        raise RuntimeError(f"No base keys discovered under {source_gt}")

    n_render_links = 0
    n_streams_copies = 0
    n_manifest_copies = 0

    for base_key in base_keys:
        render_src = source_render_variant / base_key / "0000.png"
        render_dst = shadow_render_root / base_key / "0000.png"
        _file_symlink(render_src, render_dst)
        n_render_links += 1

        streams_src = source_gt / base_key / "streams_instance_map.npy"
        streams_dst = shadow_gt_root / base_key / "streams_instance_map.npy"
        if not streams_src.exists():
            raise FileNotFoundError(
                f"authoritative streams_instance_map missing for {base_key}: {streams_src}"
            )
        _copy_file(streams_src, streams_dst)
        n_streams_copies += 1

        manifest_src = source_gt / base_key / "manifest.json"
        manifest_dst = shadow_gt_root / base_key / "manifest.json"
        if manifest_src.exists():
            _copy_file(manifest_src, manifest_dst)
        else:
            manifest_dst.parent.mkdir(parents=True, exist_ok=True)
            manifest_dst.write_text("{}\n")
        n_manifest_copies += 1

        _assert_no_satellite_leakage(shadow_gt_root / base_key)

    logger.info(
        "Shadow GT bootstrap at %s: %d render symlinks, %d streams copies, "
        "%d manifest copies across %d base keys",
        shadow_root, n_render_links, n_streams_copies, n_manifest_copies, len(base_keys),
    )
    return {
        "n_base_keys": len(base_keys),
        "n_render_links": n_render_links,
        "n_streams_copies": n_streams_copies,
        "n_manifest_copies": n_manifest_copies,
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--source-gt", type=Path, default=DEFAULT_SOURCE_GT,
                    help="Authoritative gt_canonical/current root.")
    ap.add_argument("--source-renders", type=Path, default=DEFAULT_SOURCE_RENDERS,
                    help="Authoritative renders/current root (contains {preprocessing}/).")
    ap.add_argument("--shadow-root", type=Path, default=DEFAULT_SHADOW_ROOT,
                    help="Destination shadow root; creates gt_canonical/current under it.")
    ap.add_argument("--preprocessing", type=str, default="linear_magnitude",
                    help="Render preprocessing variant to scaffold (must match inference config).")
    ap.add_argument("--galaxies", type=str, default=None,
                    help="Comma-separated galaxy IDs to restrict bootstrap to (smoke runs).")
    return ap.parse_args()


def main() -> int:
    assert_expected_conda_env(context="scripts/review/bootstrap_shadow_gt.py")
    args = parse_args()
    args.shadow_root.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("bootstrap_shadow_gt", args.shadow_root / "_bootstrap_logs")

    galaxy_filter: set[int] | None = None
    if args.galaxies:
        galaxy_filter = {int(g.strip()) for g in args.galaxies.split(",") if g.strip()}

    bootstrap(
        source_gt=args.source_gt,
        source_renders=args.source_renders,
        shadow_root=args.shadow_root,
        preprocessing=args.preprocessing,
        galaxy_filter=galaxy_filter,
        logger=logger,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
