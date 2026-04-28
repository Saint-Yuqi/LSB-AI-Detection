"""Tests for the one-shot legacy override migration into the Shadow GT.

The migration script (``scripts/review/migrate_satellite_overrides.py``) is the
only bridge from archived review YAML to explicit, auditable GT edits. These
tests cover the three behaviors called out in the migration plan:

  * force_keep from native shadow ``sam3_predictions_raw.json``
  * force_keep from an external ``inject_from_json`` probe JSON
  * force_drop resolving to a surviving shadow instance by sort policy
"""
from __future__ import annotations

import importlib.util
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipelines.unified_dataset.artifacts import save_predictions_json
from src.pipelines.unified_dataset.keys import BaseKey
from src.utils.coco_utils import mask_to_rle


def _load_migration_module():
    path = PROJECT_ROOT / "scripts" / "review" / "migrate_satellite_overrides.py"
    spec = importlib.util.spec_from_file_location("migrate_satellite_overrides", path)
    module = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


MIG = _load_migration_module()


def _sat_mask(seg: np.ndarray, score: float = 0.9) -> dict:
    ys, xs = np.where(seg)
    return {
        "type_label": "satellites",
        "segmentation": seg.astype(bool),
        "score": score,
        "area": int(seg.sum()),
        "bbox": [
            int(xs.min()),
            int(ys.min()),
            int(xs.max() - xs.min() + 1),
            int(ys.max() - ys.min() + 1),
        ],
        "centroid_x": float(xs.mean()),
        "centroid_y": float(ys.mean()),
    }


def _make_shadow_gt(shadow_root: Path, key: BaseKey, shape: tuple[int, int] = (64, 64)) -> Path:
    """Scaffold a minimal shadow GT dir with the streams scaffolding only."""
    gt_dir = shadow_root / "gt_canonical" / "current" / str(key)
    gt_dir.mkdir(parents=True, exist_ok=True)

    streams_map = np.zeros(shape, dtype=np.uint8)
    streams_map[40:45, 40:45] = 2
    np.save(gt_dir / "streams_instance_map.npy", streams_map)

    imap = np.zeros(shape, dtype=np.uint8)
    imap[40:45, 40:45] = 2
    Image.fromarray(imap).save(gt_dir / "instance_map_uint8.png")
    (gt_dir / "instances.json").write_text(
        json.dumps([{"id": 2, "type": "streams"}], indent=2)
    )

    render_dir = shadow_root / "renders" / "current" / "linear_magnitude" / str(key)
    render_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((shape[0], shape[1], 3), dtype=np.uint8)).save(
        render_dir / "0000.png"
    )
    return gt_dir


def _write_stats_json(path: Path, *, min_area: int = 5) -> None:
    path.write_text(
        json.dumps(
            {
                "filter_recommendations": {
                    "satellites": {
                        "min_area": min_area,
                        "min_solidity": 0.83,
                        "aspect_sym_moment_max": 1.75,
                    }
                }
            }
        )
    )


@pytest.fixture
def shadow_dr1_config(tmp_path: Path) -> tuple[dict, Path]:
    shadow_root = tmp_path / "shadow_root"
    shadow_root.mkdir()
    stats_path = tmp_path / "mask_stats_summary.json"
    _write_stats_json(stats_path)
    cfg = {
        "paths": {"firebox_root": str(tmp_path / "raw"), "output_root": str(shadow_root)},
        "data_selection": {"galaxy_ids": [11], "views": ["eo"]},
        "processing": {"target_size": [64, 64]},
        "preprocessing_variants": [{"name": "linear_magnitude"}],
        "satellites": {
            "prior": {"stats_json": str(stats_path)},
            "satellite_sort_policy": ["area_desc", "centroid_x_asc", "centroid_y_asc"],
        },
        "data_sources": {"streams": {"image_pattern": "unused-{orientation}.fits.gz"}},
    }
    return cfg, shadow_root


class TestForceKeep:
    def test_force_keep_from_native_raw(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key = BaseKey(11, "eo")
        gt_dir = _make_shadow_gt(shadow_root, key)

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        raw_path = gt_dir / "sam3_predictions_raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        overrides = {
            str(key): {
                "force_keep": [
                    {"candidate_id": "sat_0000", "note": "migrated-from-native"},
                ],
            }
        }
        logger = logging.getLogger("test-migrate-native")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter=None,
            dry_run=False,
            logger=logger,
        )

        assert summary["force_keep"]["ok"] == 1
        assert summary["force_keep"]["failed"] == 0

        instances = json.loads((gt_dir / "instances.json").read_text())
        adopted = [i for i in instances if i["type"] == "satellites"]
        assert len(adopted) == 1
        assert adopted[0]["provenance"]["manual_note"].startswith("migrated-from-native")

    def test_force_keep_from_external_inject_json(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key = BaseKey(11, "eo")
        gt_dir = _make_shadow_gt(shadow_root, key)

        native_seg = np.zeros((64, 64), dtype=bool)
        native_seg[5:10, 5:10] = True
        save_predictions_json(
            gt_dir / "sam3_predictions_raw.json",
            [_sat_mask(native_seg)],
            64,
            64,
            layer="raw",
        )

        ext_seg = np.zeros((64, 64), dtype=bool)
        ext_seg[20:28, 20:28] = True
        ext_path = shadow_root / "external_probe.json"
        save_predictions_json(ext_path, [_sat_mask(ext_seg)], 64, 64, layer="raw")

        from src.review.authoritative_gt import rle_sha1

        ext_sha = rle_sha1(mask_to_rle(ext_seg.astype(np.uint8)))
        overrides = {
            str(key): {
                "inject_from_json": [
                    {"candidate_rle_sha1": ext_sha, "path": str(ext_path)},
                ],
                "force_keep": [
                    {"candidate_rle_sha1": ext_sha, "note": "migrated-from-external"},
                ],
            }
        }
        logger = logging.getLogger("test-migrate-external")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter=None,
            dry_run=False,
            logger=logger,
        )

        assert summary["force_keep"]["ok"] == 1
        instances = json.loads((gt_dir / "instances.json").read_text())
        adopted = [i for i in instances if i["type"] == "satellites"]
        assert len(adopted) == 1
        assert adopted[0]["provenance"]["source_candidate_rle_sha1"] == ext_sha

    def test_dry_run_does_not_mutate_shadow_gt(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key = BaseKey(11, "eo")
        gt_dir = _make_shadow_gt(shadow_root, key)

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        save_predictions_json(
            gt_dir / "sam3_predictions_raw.json",
            [_sat_mask(seg)],
            64,
            64,
            layer="raw",
        )
        original_instances = (gt_dir / "instances.json").read_text()

        overrides = {str(key): {"force_keep": [{"candidate_id": "sat_0000"}]}}
        logger = logging.getLogger("test-migrate-dry")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter=None,
            dry_run=True,
            logger=logger,
        )

        assert summary["force_keep"]["skipped_dry_run"] == 1
        assert summary["force_keep"]["ok"] == 0
        assert (gt_dir / "instances.json").read_text() == original_instances


class TestForceDrop:
    def test_force_drop_resolves_by_sort_policy_and_deletes(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key = BaseKey(11, "eo")
        gt_dir = _make_shadow_gt(shadow_root, key)

        # Two surviving shadow satellites. area_desc then centroid_x_asc: the
        # larger one (sat_A) should get the first instance_id after streams.
        seg_a = np.zeros((64, 64), dtype=bool)
        seg_a[10:18, 10:18] = True  # area 64
        seg_b = np.zeros((64, 64), dtype=bool)
        seg_b[30:35, 30:35] = True  # area 25

        # Stamp them into the shadow image-map as though merge_instances had run.
        imap = np.zeros((64, 64), dtype=np.uint8)
        streams_map = np.load(gt_dir / "streams_instance_map.npy")
        imap[streams_map > 0] = streams_map[streams_map > 0]
        imap[seg_a] = 3  # max_stream_id(=2) + 1
        imap[seg_b] = 4
        Image.fromarray(imap).save(gt_dir / "instance_map_uint8.png")
        (gt_dir / "instances.json").write_text(
            json.dumps(
                [
                    {"id": 2, "type": "streams"},
                    {"id": 3, "type": "satellites", "provenance": {"source_candidate_id": "sat_A"}},
                    {"id": 4, "type": "satellites", "provenance": {"source_candidate_id": "sat_B"}},
                ],
                indent=2,
            )
        )

        from src.review.authoritative_gt import rle_sha1

        sha_b = rle_sha1(mask_to_rle(seg_b.astype(np.uint8)))
        save_predictions_json(
            gt_dir / "sam3_predictions_post.json",
            [_sat_mask(seg_a), _sat_mask(seg_b)],
            64,
            64,
            layer="post",
        )

        overrides = {
            str(key): {
                "force_drop": [
                    {"candidate_rle_sha1": sha_b, "note": "migrated-drop"},
                ],
            }
        }
        logger = logging.getLogger("test-migrate-drop")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter=None,
            dry_run=False,
            logger=logger,
        )

        assert summary["force_drop"]["ok"] == 1
        assert summary["force_drop"]["failed"] == 0

        instances = json.loads((gt_dir / "instances.json").read_text())
        remaining_ids = {i["id"] for i in instances}
        # instance_id 4 corresponds to the smaller seg_b in area_desc order.
        assert 4 not in remaining_ids
        assert 3 in remaining_ids
        assert 2 in remaining_ids

    def test_force_drop_unresolvable_is_summarized_failure(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key = BaseKey(11, "eo")
        gt_dir = _make_shadow_gt(shadow_root, key)

        save_predictions_json(
            gt_dir / "sam3_predictions_post.json", [], 64, 64, layer="post"
        )

        overrides = {
            str(key): {
                "force_drop": [{"candidate_rle_sha1": "deadbeefdeadbeef", "note": "missing"}],
            }
        }
        logger = logging.getLogger("test-migrate-drop-fail")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter=None,
            dry_run=False,
            logger=logger,
        )
        assert summary["force_drop"]["ok"] == 0
        assert summary["force_drop"]["failed"] == 1
        assert summary["failures"][0]["rule"] == "force_drop"


class TestBaseKeyFilter:
    def test_base_key_filter_restricts_migration(self, shadow_dr1_config) -> None:
        cfg, shadow_root = shadow_dr1_config
        key_a = BaseKey(11, "eo")
        key_b = BaseKey(13, "eo")
        cfg["data_selection"]["galaxy_ids"] = [11, 13]
        gt_a = _make_shadow_gt(shadow_root, key_a)
        _make_shadow_gt(shadow_root, key_b)

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        save_predictions_json(
            gt_a / "sam3_predictions_raw.json", [_sat_mask(seg)], 64, 64, layer="raw"
        )

        overrides = {
            str(key_a): {"force_keep": [{"candidate_id": "sat_0000"}]},
            str(key_b): {"force_keep": [{"candidate_id": "sat_0000"}]},
        }
        logger = logging.getLogger("test-migrate-filter")
        summary = MIG.migrate(
            config=cfg,
            shadow_root=shadow_root,
            overrides=overrides,
            base_key_filter={str(key_a)},
            dry_run=False,
            logger=logger,
        )
        assert summary["n_base_keys"] == 1
        assert summary["force_keep"]["ok"] == 1
