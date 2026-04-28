"""
Round artifact assembly and manifest generation.

Content/provenance separation:
  - Content artifacts: ``verifier_examples_*.jsonl``, ``verifier_chat_*.jsonl``,
    ``correction_link_map.json`` — deterministic, hashable.
  - Provenance artifacts: ``round_manifest.json``, ``asset_manifest.json``
    — timestamped metadata.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.review.schemas import TaskFamily


# ---------------------------------------------------------------------------
#  Training stages
# ---------------------------------------------------------------------------

STAGES = ("A", "B", "C")
STAGE_DESCRIPTIONS = {
    "A": "Silver-only bootstrap SFT",
    "B": "80% gold + 20% high-precision silver replay",
    "C": "Incremental gold append per round",
}


# ---------------------------------------------------------------------------
#  Round assembly
# ---------------------------------------------------------------------------

class RoundManager:
    """Assemble and manage round artifacts."""

    def __init__(self, rounds_root: Path) -> None:
        self._root = rounds_root
        self._root.mkdir(parents=True, exist_ok=True)

    def round_dir(self, round_id: str) -> Path:
        return self._root / round_id

    def assemble_round(
        self,
        round_id: str,
        examples_files: dict[str, Path],
        chat_files: dict[str, Path],
        correction_link_map_path: Path | None = None,
        asset_manifest_path: Path | None = None,
        config_snapshot: dict[str, Any] | None = None,
        stage: str = "A",
    ) -> dict[str, Any]:
        """Copy content and provenance artifacts into a round directory.

        Parameters
        ----------
        examples_files
            ``{family: path}`` mapping for business JSONL files.
        chat_files
            ``{family: path}`` mapping for chat JSONL files.
        correction_link_map_path
            Path to existing ``correction_link_map.json``, if any.
        asset_manifest_path
            Path to existing ``asset_manifest.json``, if any.
        config_snapshot
            Config dict to embed in round manifest.
        stage
            Training stage tag (``"A"``, ``"B"``, ``"C"``).

        Returns
        -------
        Round manifest dict.
        """
        rd = self.round_dir(round_id)
        rd.mkdir(parents=True, exist_ok=True)

        content_hashes: dict[str, str] = {}

        for family, src in examples_files.items():
            dst = rd / f"verifier_examples_{family}.jsonl"
            shutil.copy2(src, dst)
            content_hashes[dst.name] = self._sha256(dst)

        for family, src in chat_files.items():
            dst = rd / f"verifier_chat_{family}.jsonl"
            shutil.copy2(src, dst)
            content_hashes[dst.name] = self._sha256(dst)

        if correction_link_map_path and correction_link_map_path.exists():
            dst = rd / "correction_link_map.json"
            shutil.copy2(correction_link_map_path, dst)
            content_hashes[dst.name] = self._sha256(dst)

        if asset_manifest_path and asset_manifest_path.exists():
            shutil.copy2(asset_manifest_path, rd / "asset_manifest.json")

        manifest = {
            "round_id": round_id,
            "stage": stage,
            "stage_description": STAGE_DESCRIPTIONS.get(stage, ""),
            "created_at": datetime.now(timezone.utc).isoformat(),
            "families": list(examples_files.keys()),
            "content_hashes": content_hashes,
            "config_snapshot": config_snapshot or {},
        }

        with open(rd / "round_manifest.json", "w") as fh:
            json.dump(manifest, fh, indent=2)

        return manifest

    @staticmethod
    def _sha256(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 16), b""):
                h.update(chunk)
        return h.hexdigest()

    def list_rounds(self) -> list[str]:
        """List available round IDs."""
        return sorted(
            d.name for d in self._root.iterdir()
            if d.is_dir() and (d / "round_manifest.json").exists()
        )

    def load_manifest(self, round_id: str) -> dict[str, Any]:
        """Load a round's manifest."""
        with open(self.round_dir(round_id) / "round_manifest.json") as fh:
            return json.load(fh)
