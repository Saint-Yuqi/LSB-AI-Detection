#!/usr/bin/env python3
"""Correction import with revision_hash validation.

Reads a corrections JSONL file where each line contains the original
example record and the corrected data payload.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from src.review.correction import CorrectionLinkMap, import_correction


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", type=Path, required=True,
                        help="JSONL file with correction records")
    parser.add_argument("--gt-dir", type=Path, required=True)
    parser.add_argument("--link-map", type=Path, required=True,
                        help="correction_link_map.json (created if missing)")
    args = parser.parse_args(argv)

    link_map = CorrectionLinkMap.load(args.link_map)
    re_export: list[str] = []

    with open(args.corrections) as fh:
        for lineno, line in enumerate(fh, 1):
            entry = json.loads(line)
            record = entry["original_record"]

            corrected: dict = {}
            if "corrected_mask" in entry:
                corrected["corrected_mask"] = np.array(
                    entry["corrected_mask"], dtype=np.uint8,
                )
            if "corrected_map" in entry:
                corrected["corrected_map"] = np.array(
                    entry["corrected_map"], dtype=np.uint8,
                )
                corrected["corrected_instances"] = entry["corrected_instances"]
            if "corrected_stream_masks" in entry:
                corrected["corrected_stream_masks"] = {
                    int(k): np.array(v, dtype=np.uint8)
                    for k, v in entry["corrected_stream_masks"].items()
                }

            try:
                result = import_correction(record, corrected, args.gt_dir, link_map)
                re_export.extend(result["re_export_keys"])
                print(f"  L{lineno}: OK -> {result['new_task_revision_id']}")
            except ValueError as e:
                print(f"  L{lineno}: REJECTED: {e}", file=sys.stderr)

    link_map.save(args.link_map)

    if re_export:
        print(f"\nRe-export required for: {sorted(set(re_export))}")
    print("Done.")


if __name__ == "__main__":
    main()
