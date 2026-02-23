"""
Candidate grouping – cluster same-target masks by centroid proximity.

Usage:
    from src.postprocess.candidate_grouping import group_by_centroid
    group_by_centroid(masks, dist_px=15.0)
    # Each mask now has 'group_id' (int, 0..K-1)

Algorithm:
    1. Extract centroid_xy from each mask (must be computed beforehand)
    2. Build cKDTree for O(n log n) neighbor search
    3. query_pairs(r=dist_px) to find all close pairs
    4. Union-Find to merge transitive groups
    5. Assign group_id to each mask

Edge cases:
    - Empty masks list: no-op
    - Single mask: group_id = 0
    - Missing centroid_xy: raises KeyError (caller must ensure metrics computed)
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy.spatial import cKDTree


class UnionFind:
    """Simple Union-Find with path compression."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            rx, ry = ry, rx
        self.parent[ry] = rx
        if self.rank[rx] == self.rank[ry]:
            self.rank[rx] += 1


def group_by_centroid(
    masks: list[dict[str, Any]],
    dist_px: float = 15.0,
) -> None:
    """
    In-place add 'group_id' to each mask based on centroid clustering.

    Args:
        masks: list of mask dicts with 'centroid_xy' already computed.
        dist_px: maximum distance (px) for two centroids to be in same group.

    Complexity: O(n log n) via cKDTree.

    Side effects:
        - Each mask gets 'group_id': int (0..K-1 for K groups).
    """
    n = len(masks)

    # Edge cases
    if n == 0:
        return
    if n == 1:
        masks[0]["group_id"] = 0
        return

    # Extract centroids
    centroids = np.array([m["centroid_xy"] for m in masks], dtype=np.float64)

    # Build spatial index
    tree = cKDTree(centroids)

    # Find all pairs within dist_px
    pairs = tree.query_pairs(r=dist_px)

    # Union-Find
    uf = UnionFind(n)
    for i, j in pairs:
        uf.union(i, j)

    # Map roots to compact group IDs
    root_to_gid: dict[int, int] = {}
    for i in range(n):
        root = uf.find(i)
        if root not in root_to_gid:
            root_to_gid[root] = len(root_to_gid)
        masks[i]["group_id"] = root_to_gid[root]
