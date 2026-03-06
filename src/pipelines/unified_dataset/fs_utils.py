"""
Shared filesystem helpers for the unified dataset pipeline.
"""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA1 of file (first chunk_size bytes for large files)."""
    h = hashlib.sha1()
    try:
        with open(path, "rb") as f:
            h.update(f.read(chunk_size))
    except Exception:
        return ""
    return h.hexdigest()
