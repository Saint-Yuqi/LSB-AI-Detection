#!/usr/bin/env python3
"""Assemble round artifacts and generate manifest.

Thin CLI wrapper around ``src.review.round_manager``.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.review.round_manager import RoundManager


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--examples", type=Path, nargs="+", required=True,
                        help="verifier_examples_*.jsonl files")
    parser.add_argument("--chats", type=Path, nargs="+", required=True,
                        help="verifier_chat_*.jsonl files")
    parser.add_argument("--link-map", type=Path, default=None)
    parser.add_argument("--asset-manifest", type=Path, default=None)
    parser.add_argument("--stage", default="A", choices=["A", "B", "C"])
    parser.add_argument("--config-snapshot", type=Path, default=None,
                        help="JSON file with config to embed")
    args = parser.parse_args(argv)

    def _family_from_path(p: Path) -> str:
        stem = p.stem
        for prefix in ("verifier_examples_", "verifier_chat_"):
            if stem.startswith(prefix):
                return stem[len(prefix):]
        return stem

    examples_map = {_family_from_path(p): p for p in args.examples}
    chat_map = {_family_from_path(p): p for p in args.chats}

    config_snap = None
    if args.config_snapshot and args.config_snapshot.exists():
        with open(args.config_snapshot) as f:
            config_snap = json.load(f)

    mgr = RoundManager(args.output_dir)
    manifest = mgr.assemble_round(
        round_id=args.round_id,
        examples_files=examples_map,
        chat_files=chat_map,
        correction_link_map_path=args.link_map,
        asset_manifest_path=args.asset_manifest,
        config_snapshot=config_snap,
        stage=args.stage,
    )

    print(f"Round {args.round_id} assembled: {manifest['created_at']}")
    print(f"  Families: {manifest['families']}")
    print(f"  Stage: {manifest['stage']} ({manifest['stage_description']})")


if __name__ == "__main__":
    main()
