#!/usr/bin/env python3
"""Manual authoritative GT editor for SAM3 satellite instances."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.config import load_config
from src.review.authoritative_gt import (
    adopt_raw_candidate,
    delete_authoritative_instance,
    parse_base_key,
)


def _common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="Dataset config YAML")
    parser.add_argument("--dataset", type=str, default=None, help="Override dataset_name")
    parser.add_argument("--condition", type=str, default=None, help="Authoritative condition")
    parser.add_argument("--base-key", type=str, required=True, help="e.g. 00066_fo or 00011_los00")
    parser.add_argument("--note", type=str, default=None, help="Optional operator note")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    adopt = sub.add_parser("adopt-raw", help="Adopt one raw satellite candidate into authoritative GT")
    _common_args(adopt)
    adopt.add_argument("--source-json", type=Path, default=None, help="Optional external sam3_predictions_raw.json")
    select_group = adopt.add_mutually_exclusive_group(required=True)
    select_group.add_argument("--candidate-id", type=str, default=None)
    select_group.add_argument("--raw-index", type=int, default=None)
    select_group.add_argument(
        "--candidate-rle-sha1",
        type=str,
        default=None,
        help="Select candidate by canonical RLE SHA1 (most stable across re-inference).",
    )
    adopt.add_argument("--min-area-px", type=int, default=None)

    delete = sub.add_parser("delete-instance", help="Delete one surviving authoritative satellite instance")
    _common_args(delete)
    delete_group = delete.add_mutually_exclusive_group(required=True)
    delete_group.add_argument("--instance-id", type=int, default=None)
    delete_group.add_argument("--source-candidate-id", type=str, default=None)

    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.dataset:
        config["dataset_name"] = args.dataset

    key = parse_base_key(args.base_key)

    if args.command == "adopt-raw":
        result = adopt_raw_candidate(
            config,
            key=key,
            dataset=args.dataset,
            condition=args.condition,
            source_json=args.source_json,
            candidate_id=args.candidate_id,
            raw_index=args.raw_index,
            candidate_rle_sha1=args.candidate_rle_sha1,
            min_area_px=args.min_area_px,
            manual_note=args.note,
        )
        print(
            f"Adopted {result['base_key']} -> new instance_id={result['assigned_instance_id']} "
            f"from {result['source_prediction_path']}"
        )
        return

    result = delete_authoritative_instance(
        config,
        key=key,
        dataset=args.dataset,
        condition=args.condition,
        instance_id=args.instance_id,
        source_candidate_id=args.source_candidate_id,
        manual_note=args.note,
    )
    print(f"Deleted {result['base_key']} -> instance_id={result['deleted_instance_id']}")


if __name__ == "__main__":
    main()
