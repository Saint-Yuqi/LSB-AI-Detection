"""
Tests for holdout splitting: group-wise by halo_id, no correlated view
leakage, no candidate-level randomization.

Usage:
    pytest tests/test_holdout.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.holdout import holdout_split_examples


def _make_examples(
    n_galaxies: int = 10,
    views_per_galaxy: int = 2,
    candidates_per_view: int = 3,
) -> list[dict]:
    examples = []
    for gid in range(1, n_galaxies + 1):
        for v in range(views_per_galaxy):
            for c in range(candidates_per_view):
                examples.append({
                    "example_id": f"ex_{gid}_{v}_{c}",
                    "halo_id": gid,
                    "view_id": f"view_{v}",
                    "sample_id": f"{gid:05d}_view_{v}",
                })
    return examples


class TestHoldoutSplit:
    def test_no_leakage(self):
        examples = _make_examples(20)
        train, holdout = holdout_split_examples(examples, holdout_ratio=0.2)

        train_halos = {ex["halo_id"] for ex in train}
        holdout_halos = {ex["halo_id"] for ex in holdout}
        assert train_halos.isdisjoint(holdout_halos)

    def test_all_views_same_side(self):
        examples = _make_examples(10, views_per_galaxy=4)
        train, holdout = holdout_split_examples(examples, holdout_ratio=0.3)

        for split_name, split_data in [("train", train), ("holdout", holdout)]:
            halo_views: dict[int, set] = {}
            for ex in split_data:
                halo_views.setdefault(ex["halo_id"], set()).add(ex["view_id"])
            for halo, views in halo_views.items():
                all_examples_for_halo = [
                    e for e in examples if e["halo_id"] == halo
                ]
                all_views = {e["view_id"] for e in all_examples_for_halo}
                assert views == all_views, (
                    f"Halo {halo} leaked across splits in {split_name}"
                )

    def test_all_candidates_same_side(self):
        examples = _make_examples(10, candidates_per_view=5)
        train, holdout = holdout_split_examples(examples)

        for split in [train, holdout]:
            by_sample: dict[str, list] = {}
            for ex in split:
                by_sample.setdefault(ex["sample_id"], []).append(ex)
            for sample_id, exs in by_sample.items():
                all_for_sample = [
                    e for e in examples if e["sample_id"] == sample_id
                ]
                assert len(exs) == len(all_for_sample)

    def test_deterministic(self):
        examples = _make_examples(50)
        t1, h1 = holdout_split_examples(examples, holdout_ratio=0.2, seed=42)
        t2, h2 = holdout_split_examples(examples, holdout_ratio=0.2, seed=42)

        assert [e["example_id"] for e in t1] == [e["example_id"] for e in t2]
        assert [e["example_id"] for e in h1] == [e["example_id"] for e in h2]

    def test_different_seeds(self):
        examples = _make_examples(50)
        _, h1 = holdout_split_examples(examples, holdout_ratio=0.3, seed=42)
        _, h2 = holdout_split_examples(examples, holdout_ratio=0.3, seed=99)

        halos_1 = {e["halo_id"] for e in h1}
        halos_2 = {e["halo_id"] for e in h2}
        assert halos_1 != halos_2

    def test_preserves_total_count(self):
        examples = _make_examples(20)
        train, holdout = holdout_split_examples(examples)
        assert len(train) + len(holdout) == len(examples)

    def test_empty_input(self):
        train, holdout = holdout_split_examples([])
        assert train == []
        assert holdout == []

    def test_halo_id_equals_galaxy_id(self):
        """V1 invariant: halo_id = galaxy_id for both DR1 and PNbody."""
        examples = _make_examples(10)
        train, holdout = holdout_split_examples(examples)
        for ex in train + holdout:
            assert ex["halo_id"] == int(ex["sample_id"].split("_")[0])
