"""F4: legacy configs (default gt_subdir) keep working unchanged.

The new path requires ``inference_phase.sam3.prior_filter`` to be set.
Legacy configs that omit it MUST still load by sourcing thresholds from
``satellites.prior.stats_json`` exactly as before.
"""
from __future__ import annotations

from src.pipelines.unified_dataset.paths import PathResolver


def _legacy_config(tmp_path) -> dict:
    return {
        "paths": {
            "firebox_root": str(tmp_path),
            "output_root": str(tmp_path / "out"),
            # No gt_subdir / pseudo_gt_subdir — defaults must keep legacy behavior.
        },
        "processing": {"target_size": [1024, 1024]},
        "data_sources": {"streams": {"image_pattern": "{galaxy_id}-{view_id}.fits"}},
    }


def _new_config(tmp_path) -> dict:
    cfg = _legacy_config(tmp_path)
    cfg["paths"]["gt_subdir"] = "gt_canonical_tidal_v1"
    cfg["paths"]["pseudo_gt_subdir"] = "pseudo_gt_canonical_tidal_v1"
    return cfg


def test_path_resolver_legacy_defaults(tmp_path):
    cfg = _legacy_config(tmp_path)
    resolver = PathResolver(cfg)
    assert resolver.gt_subdir == "gt_canonical"
    assert resolver.pseudo_gt_subdir is None
    assert resolver.is_new_path() is False


def test_path_resolver_detects_new_path_via_gt_subdir(tmp_path):
    cfg = _new_config(tmp_path)
    cfg["paths"].pop("pseudo_gt_subdir")  # only gt_subdir non-default
    resolver = PathResolver(cfg)
    assert resolver.is_new_path() is True


def test_path_resolver_detects_new_path_via_pseudo_gt_subdir(tmp_path):
    """F18: pseudo-only configs (PNbody) must also activate the new path."""
    cfg = _legacy_config(tmp_path)
    cfg["paths"]["pseudo_gt_subdir"] = "pseudo_gt_canonical_tidal_v1"
    resolver = PathResolver(cfg)
    assert resolver.is_new_path() is True


def test_path_resolver_getter_does_not_create_dirs(tmp_path):
    """F15: getters return paths only; writers create directories."""
    cfg = _new_config(tmp_path)
    resolver = PathResolver(cfg)

    from src.pipelines.unified_dataset.keys import BaseKey
    key = BaseKey(galaxy_id=11, view_id="eo")
    expected = resolver.output_root / "gt_canonical_tidal_v1" / "current" / str(key)
    assert not expected.exists()

    # Calling the getter does not create the directory.
    p = resolver.get_gt_dir(key)
    assert p == expected
    assert not p.exists()
