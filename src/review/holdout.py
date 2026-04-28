"""
Group-wise holdout splitting via key_adapter.

Groups by ``halo_id`` (= ``galaxy_id`` in V1).  All views/candidates
of the same ``halo_id`` go to the same partition (train or holdout).
Uses the same deterministic hash-assignment pattern as
``src.pipelines.unified_dataset.split._hash_assign``.
"""
from __future__ import annotations

import hashlib
from typing import Any, Callable


def _hash_assign(seed: int, group_key: int, train_ratio: float) -> str:
    h = hashlib.sha256(f"{seed}:{group_key}".encode()).hexdigest()
    frac = int(h[:8], 16) / 0xFFFFFFFF
    return "train" if frac < train_ratio else "holdout"


def holdout_split_examples(
    examples: list[dict[str, Any]],
    holdout_ratio: float = 0.2,
    seed: int = 42,
    key_fn: Callable[[dict[str, Any]], int] = lambda ex: ex["halo_id"],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Group-wise holdout split.

    Parameters
    ----------
    examples
        List of ``VerifierExample`` records (as dicts).
    holdout_ratio
        Fraction held out for evaluation.
    seed
        Deterministic hash seed.
    key_fn
        Extracts the grouping key (default: ``halo_id``).

    Returns
    -------
    (train_examples, holdout_examples)
    """
    train_ratio = 1.0 - holdout_ratio

    groups: dict[int, list[dict[str, Any]]] = {}
    for ex in examples:
        gk = key_fn(ex)
        groups.setdefault(gk, []).append(ex)

    assignment: dict[int, str] = {}
    for gk in groups:
        assignment[gk] = _hash_assign(seed, gk, train_ratio)

    train: list[dict[str, Any]] = []
    holdout: list[dict[str, Any]] = []
    for gk, exs in groups.items():
        if assignment[gk] == "train":
            train.extend(exs)
        else:
            holdout.extend(exs)

    return train, holdout
