#!/usr/bin/env python3
"""
Group existing PNbody clean/noisy FITS by galaxy and leave reverse compat symlinks.

Canonical targets:
    clean  -> data/01_raw/LSB_and_Satellites/FIREbox_PNbody/SB_maps/{gid:05d}/...
    noisy  -> data/04_noise/pnbody_physics/{profile}/{gid:05d}/...

The old flat file paths are replaced by symlinks pointing back to the grouped files.
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

_FNAME_RE = re.compile(r"magnitudes-Fbox-(\d+)-([^_]+)-VIS2\.fits\.gz$")


def _scan_flat_files(root: Path) -> list[Path]:
    return sorted(
        path
        for path in root.glob("magnitudes-Fbox-*-VIS2.fits.gz")
        if path.is_file() or path.is_symlink()
    )


def _group_target(root: Path, file_path: Path) -> tuple[int, str, Path]:
    match = _FNAME_RE.match(file_path.name)
    if not match:
        raise ValueError(f"Unexpected FITS name: {file_path.name}")
    galaxy_id = int(match.group(1))
    view_id = match.group(2)
    return galaxy_id, view_id, root / f"{galaxy_id:05d}" / file_path.name


def _migrate_root(
    *,
    dataset: str,
    condition: str,
    root: Path,
    dry_run: bool,
) -> tuple[list[dict], dict[int, set[str]]]:
    entries: list[dict] = []
    galaxy_views: dict[int, set[str]] = defaultdict(set)

    for src_path in _scan_flat_files(root):
        galaxy_id, view_id, dst_path = _group_target(root, src_path)
        galaxy_views[galaxy_id].add(view_id)

        symlink_path = src_path
        entry = {
            "dataset": dataset,
            "condition": condition,
            "galaxy_id": galaxy_id,
            "view_id": view_id,
            "src_path": str(src_path),
            "dst_path": str(dst_path),
            "symlink_path": str(symlink_path),
        }

        if not dry_run:
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if not dst_path.exists():
                src_path.replace(dst_path)
            if symlink_path.exists() and not symlink_path.is_symlink():
                symlink_path.unlink()
            if not symlink_path.exists():
                symlink_path.symlink_to(dst_path.relative_to(symlink_path.parent))

        entries.append(entry)

    return entries, galaxy_views


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--clean-root",
        type=Path,
        default=Path("data/01_raw/LSB_and_Satellites/FIREbox_PNbody/SB_maps"),
    )
    parser.add_argument(
        "--noisy-root",
        type=Path,
        default=Path("data/04_noise/pnbody_physics"),
    )
    parser.add_argument(
        "--profiles",
        nargs="+",
        default=["sb30", "sb31.5"],
        help="Noise profile subdirectories to migrate",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("outputs/pnbody_grouped_fits_migration_manifest.json"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    all_entries: list[dict] = []
    clean_entries, clean_views = _migrate_root(
        dataset="pnbody",
        condition="clean",
        root=args.clean_root,
        dry_run=args.dry_run,
    )
    all_entries.extend(clean_entries)

    complete_summary = {
        "clean": {
            gid: {"n_views_found": len(views), "is_complete": len(views) == 24}
            for gid, views in clean_views.items()
        }
    }

    for profile in args.profiles:
        profile_root = args.noisy_root / profile
        profile_entries, profile_views = _migrate_root(
            dataset="pnbody",
            condition=profile,
            root=profile_root,
            dry_run=args.dry_run,
        )
        all_entries.extend(profile_entries)
        complete_summary[profile] = {
            gid: {"n_views_found": len(views), "is_complete": len(views) == 24}
            for gid, views in profile_views.items()
        }

    for entry in all_entries:
        summary = complete_summary[entry["condition"]].get(entry["galaxy_id"], {})
        entry["n_views_found"] = summary.get("n_views_found", 0)
        entry["is_complete"] = summary.get("is_complete", False)

    payload = {
        "dataset": "pnbody",
        "created_at": datetime.now().isoformat(),
        "dry_run": args.dry_run,
        "entries": all_entries,
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(payload, indent=2))
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
