"""
ETL: business JSONL → two-turn chat JSONL.

Deterministic transform with no ``system`` role.  The stamped context
(per-candidate coloured bbox + candidate_id) is rendered at ETL time from
the shared bare context, resolving the dedup contradiction.

Content artifacts are byte-deterministic.  Provenance metadata
(``etl_manifest.json``) carries timestamps.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.review.prompt_registry import PromptRegistry
from src.review.render_spec import RenderSpec, RenderSpecRegistry
from src.review.review_render import save_stamped_context
from src.review.schemas import CropSpec, TaskFamily


# ---------------------------------------------------------------------------
#  Internal helpers
# ---------------------------------------------------------------------------

def _ensure_stamped_context(
    bare_context_path: Path,
    candidate_bbox_xywh: tuple[int, int, int, int],
    candidate_id: str,
    spec: RenderSpec,
    output_root: Path,
    sample_id: str,
) -> str:
    """Render and save stamped context; return relative output path."""
    rel = (
        Path("review_stamped")
        / sample_id
        / f"{candidate_id}_{spec.spec_id}.png"
    )
    abspath = output_root / rel
    if not abspath.exists():
        bare = np.array(Image.open(bare_context_path).convert("RGB"))
        save_stamped_context(bare, candidate_bbox_xywh, candidate_id, spec, abspath)
    return str(rel)


# ---------------------------------------------------------------------------
#  Single-record transform
# ---------------------------------------------------------------------------

def transform_example(
    record: dict[str, Any],
    prompt_registry: PromptRegistry,
    spec_registry: RenderSpecRegistry,
    asset_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Convert one business JSONL record to a chat JSONL record."""
    family = TaskFamily(record["task_family"])
    spec = spec_registry.get(record["render_spec_id"])
    prompt_text = prompt_registry.render(record["fixed_prompt_id"])
    arefs = record["asset_refs"]

    content: list[dict[str, Any]] = [{"type": "text", "text": prompt_text}]

    if family == TaskFamily.SATELLITE_MV:
        crop_spec = CropSpec(**arefs["crop_spec"])
        crop_fname = f"{crop_spec.content_key}_{spec.spec_id}.png"
        crop_path = str(
            Path("review_crops") / record["sample_id"] / crop_fname
        )
        content.append({"type": "image", "path": crop_path})

        bare_abs = asset_root / arefs["bare_context_path"]
        bbox = tuple(arefs["candidate_bbox_xywh"])
        stamped_path = _ensure_stamped_context(
            bare_abs, bbox, arefs["candidate_id"],
            spec, output_root, record["sample_id"],
        )
        content.append({"type": "image", "path": stamped_path})
    else:
        content.append({"type": "image", "path": arefs["review_image_path"]})

    chat: dict[str, Any] = {
        "messages": [
            {"role": "user", "content": content},
            {"role": "assistant", "content": record["decision_label"]},
        ],
        "metadata": {
            "example_id": record["example_id"],
            "fixed_prompt_id": record["fixed_prompt_id"],
            "revision_hash": record["revision_hash"],
        },
    }
    return chat


# ---------------------------------------------------------------------------
#  Batch transform
# ---------------------------------------------------------------------------

def run_etl(
    examples_path: Path,
    prompt_registry: PromptRegistry,
    spec_registry: RenderSpecRegistry,
    asset_root: Path,
    output_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Transform all records and write chat JSONL + manifest.

    Returns the ETL manifest dict (provenance metadata).
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    with open(examples_path) as fh:
        for line in fh:
            records.append(json.loads(line))

    content_hash = hashlib.sha256()

    with open(output_path, "w") as out:
        for rec in records:
            chat = transform_example(
                rec, prompt_registry, spec_registry, asset_root, output_root,
            )
            line = json.dumps(chat, sort_keys=True) + "\n"
            content_hash.update(line.encode())
            out.write(line)

    input_hash = hashlib.sha256(
        Path(examples_path).read_bytes()
    ).hexdigest()

    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "input_file": str(examples_path),
        "input_sha256": input_hash,
        "output_file": str(output_path),
        "output_sha256": content_hash.hexdigest(),
        "num_records": len(records),
    }

    manifest_path = output_path.parent / "etl_manifest.json"
    with open(manifest_path, "w") as fh:
        json.dump(manifest, fh, indent=2)

    return manifest
