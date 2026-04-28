"""
Asset manager: shared bare-context dedup, crop caching, and manifests.

Dedup invariant: 100 candidates from the same ``sample_id`` share
exactly 1 bare context file and at most 100 crop files (less if
spatial crops overlap).
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.review.render_spec import RenderSpec
from src.review.review_render import (
    save_bare_context,
    save_crop,
    save_ev_full_image,
)
from src.review.schemas import CropSpec


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class AssetManager:
    """Manage review-render assets with dedup and caching."""

    def __init__(self, asset_root: Path) -> None:
        self._root = Path(asset_root)
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    # ------------------------------------------------------------------ bare context
    def ensure_bare_context(
        self,
        sample_id: str,
        variant: str,
        full_image: np.ndarray,
    ) -> tuple[str, str]:
        """Return ``(path_str, sha1)`` for the shared bare context.

        Creates the file only on the first call for this *sample_id*.
        """
        rel = Path("review_context") / f"{sample_id}_{variant}.png"
        abspath = self._root / rel
        if not abspath.exists():
            save_bare_context(full_image, abspath)
        return str(rel), _sha1(abspath)

    # ------------------------------------------------------------------ crop
    def ensure_crop(
        self,
        sample_id: str,
        crop_spec: CropSpec,
        spec: RenderSpec,
        full_image: np.ndarray,
        candidate_rle: dict,
    ) -> str:
        """Return the relative path to the cached crop PNG.

        Cache key: ``(sample_id, crop_spec.content_key, spec.spec_id)``.
        """
        fname = f"{crop_spec.content_key}_{spec.spec_id}.png"
        rel = Path("review_crops") / sample_id / fname
        abspath = self._root / rel
        if not abspath.exists():
            save_crop(full_image, candidate_rle, crop_spec, spec, abspath)
        return str(rel)

    # ------------------------------------------------------------------ EV
    def ensure_ev_image(
        self,
        sample_id: str,
        variant: str,
        spec: RenderSpec,
        full_image: np.ndarray,
        fragment_hints: list[dict] | None = None,
        state_key: str | None = None,
    ) -> tuple[str, str]:
        """Return ``(path_str, sha1)`` for an EV review image.

        *state_key* disambiguates multi-variant renders of the same image
        (e.g. ``synthetic_variant_id``).  When set, the key is appended to
        the filename so that each variant gets its own cached file.
        """
        suffix = f"_{state_key}" if state_key else ""
        fname = f"{sample_id}_{variant}_{spec.spec_id}{suffix}.png"
        rel = Path("review_ev") / fname
        abspath = self._root / rel
        if not abspath.exists():
            save_ev_full_image(full_image, fragment_hints, spec, abspath)
        return str(rel), _sha1(abspath)

    # ------------------------------------------------------------------ manifest
    def build_asset_manifest(self, round_dir: Path | None = None) -> dict:
        """Scan all assets under *root* and return a per-file sha1 manifest."""
        scan_root = round_dir or self._root
        manifest: dict[str, str] = {}
        for p in sorted(scan_root.rglob("*.png")):
            manifest[str(p.relative_to(scan_root))] = _sha1(p)
        return {
            "scanned_at": datetime.now(timezone.utc).isoformat(),
            "files": manifest,
        }

    def write_asset_manifest(self, round_dir: Path) -> Path:
        """Write ``asset_manifest.json`` to *round_dir*."""
        manifest = self.build_asset_manifest(round_dir)
        out = round_dir / "asset_manifest.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(manifest, f, indent=2)
        return out
