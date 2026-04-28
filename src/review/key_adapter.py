"""
Mapping layer between pipeline-canonical ``BaseKey(galaxy_id, view_id)``
and the verifier protocol's ``sample_id`` / ``halo_id`` identifiers.

V1 scope
--------
Both DR1 and PNbody use ``halo_id = galaxy_id`` (1:1 identity).
A future ``KeyAdapterConfig`` with a ``halo_mapping`` override can be
added without breaking existing callers.
"""
from __future__ import annotations

from src.pipelines.unified_dataset.keys import BaseKey


def base_key_to_sample_id(key: BaseKey) -> str:
    """``sample_id`` = ``str(BaseKey)`` = ``'{galaxy_id:05d}_{view_id}'``."""
    return str(key)


def base_key_to_halo_id(key: BaseKey) -> int:
    """V1: ``halo_id = galaxy_id`` for both DR1 and PNbody."""
    return key.galaxy_id


def halo_id_to_galaxy_ids(halo_id: int) -> list[int]:
    """V1: trivial 1:1 identity."""
    return [halo_id]


def sample_id_to_base_key(sample_id: str) -> BaseKey:
    """Inverse of :func:`base_key_to_sample_id`."""
    parts = sample_id.split("_", 1)
    if len(parts) != 2:
        raise ValueError(f"Cannot parse sample_id {sample_id!r}")
    return BaseKey(galaxy_id=int(parts[0]), view_id=parts[1])
