#!/usr/bin/env python3
"""Business JSONL → chat JSONL deterministic transform.

Thin CLI wrapper around ``src.review.etl.run_etl``.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from src.review.etl import run_etl
from src.review.prompt_registry import PromptRegistry
from src.review.render_spec import RenderSpecRegistry


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True,
                        help="Task family (used for output naming)")
    parser.add_argument("--examples", type=Path, required=True,
                        help="verifier_examples_{family}.jsonl")
    parser.add_argument("--render-spec-config", type=Path, required=True)
    parser.add_argument("--prompt-config", type=Path, required=True)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Root dir for stamped context renders")
    parser.add_argument("--output", type=Path, required=True,
                        help="verifier_chat_{family}.jsonl")
    args = parser.parse_args(argv)

    prompt_reg = PromptRegistry.from_yaml(args.prompt_config)
    spec_reg = RenderSpecRegistry.from_yaml(args.render_spec_config)

    manifest = run_etl(
        args.examples, prompt_reg, spec_reg,
        args.asset_root, args.output_root, args.output,
    )
    print(f"ETL complete: {manifest['num_records']} records -> {args.output}")


if __name__ == "__main__":
    main()
