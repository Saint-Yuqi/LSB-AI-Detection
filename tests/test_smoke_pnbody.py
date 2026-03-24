#!/usr/bin/env python3
"""
Smoke tests for the PNbody 24-view integration and pseudo-label pipeline.

Covers: config loading, PathResolver dual format-key, instrument factory,
generate_pnbody_fits.py dry-run, split_annotations --output-prefix,
pseudo-label rasterize + overlay + completeness gate, generate_noisy_fits
views compat.

Usage:
    pytest tests/test_smoke_pnbody.py -v
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ── 1. Config loading ────────────────────────────────────────────────────

class TestConfigLoading:
    def test_load_pnbody_generation_config(self):
        """firebox_pnbody_24los.yaml loads and has 184 galaxy_ids."""
        from src.pipelines.unified_dataset.config import load_config
        cfg = load_config(PROJECT_ROOT / "configs" / "pnbody" / "firebox_pnbody_24los.yaml")
        assert len(cfg["galaxy_ids"]) == 184
        assert cfg["distance"] == 35
        assert "los_file" in cfg
        assert "instrument_file" in cfg

    def test_load_pnbody_unified_config(self):
        """unified_data_prep_pnbody.yaml loads, has gt_phase disabled, pseudo_label mode."""
        from src.pipelines.unified_dataset.config import load_config, generate_base_keys
        cfg = load_config(PROJECT_ROOT / "configs" / "unified_data_prep_pnbody.yaml")

        assert cfg["gt_phase"]["enabled"] is False
        assert cfg["inference_phase"]["run_mode"] == "pseudo_label"
        assert cfg["export_phase"]["annotations_filename"] == "annotations_pnbody_pseudo.json"

        keys = generate_base_keys(cfg, galaxy_filter=[11])
        assert len(keys) == 24
        assert str(keys[0]) == "00011_los00"
        assert str(keys[23]) == "00011_los23"

    def test_load_legacy_dr1_config(self):
        """Original unified_data_prep.yaml still loads correctly with views key."""
        from src.pipelines.unified_dataset.config import load_config, generate_base_keys
        cfg = load_config(PROJECT_ROOT / "configs" / "unified_data_prep.yaml")

        keys = generate_base_keys(cfg, galaxy_filter=[11])
        assert len(keys) == 2
        assert str(keys[0]) == "00011_eo"
        assert str(keys[1]) == "00011_fo"

    def test_legacy_orientations_compat(self):
        """Config with data_selection.orientations still works, emits deprecation warning."""
        from src.pipelines.unified_dataset.config import generate_base_keys
        cfg = {
            "data_selection": {
                "galaxy_ids": [11],
                "orientations": ["eo", "fo"],
            }
        }
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            keys = generate_base_keys(cfg)
            assert len(keys) == 2
            assert any("deprecated" in str(warning.message).lower() for warning in w)


# ── 2. PathResolver dual format-key ──────────────────────────────────────

class TestPathResolverDualFormatKey:
    def test_legacy_orientation_pattern(self):
        """PathResolver works with {orientation} pattern (old DR1 config)."""
        from src.pipelines.unified_dataset.paths import PathResolver
        from src.pipelines.unified_dataset.keys import BaseKey

        config = {
            "paths": {"firebox_root": "/data/FIREbox-DR1", "output_root": "/out"},
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "magnitudes-Fbox-{galaxy_id}-{orientation}-VIS2.fits.gz",
                    "mask_subdir_eo": "MASKS_EO",
                    "mask_subdir_fo": "MASKS_FO",
                    "mask_pattern": "ark_features-{galaxy_id}-{orientation}-SBlim{threshold}.fits.gz",
                }
            },
            "processing": {"target_size": [1024, 1024]},
        }
        resolver = PathResolver(config)
        key_eo = BaseKey(11, "eo")
        key_fo = BaseKey(11, "fo")

        fits_eo = resolver.get_fits_path(key_eo)
        assert "magnitudes-Fbox-11-eo-VIS2.fits.gz" in str(fits_eo)

        fits_fo = resolver.get_fits_path(key_fo)
        assert "magnitudes-Fbox-11-fo-VIS2.fits.gz" in str(fits_fo)

        mask_eo = resolver.get_mask_path(key_eo, 32.0)
        assert "MASKS_EO" in str(mask_eo)
        assert "ark_features-11-eo-SBlim32.fits.gz" in str(mask_eo)

        mask_fo = resolver.get_mask_path(key_fo, 32.0)
        assert "MASKS_FO" in str(mask_fo)

    def test_new_view_id_pattern(self):
        """PathResolver works with {view_id} pattern (new PNbody config)."""
        from src.pipelines.unified_dataset.paths import PathResolver
        from src.pipelines.unified_dataset.keys import BaseKey

        config = {
            "paths": {"firebox_root": "/data/FIREbox_PNbody", "output_root": "/out"},
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "magnitudes-Fbox-{galaxy_id}-{view_id}-VIS2.fits.gz",
                }
            },
            "processing": {"target_size": [1024, 1024]},
        }
        resolver = PathResolver(config)
        key = BaseKey(11, "los00")

        fits_path = resolver.get_fits_path(key)
        assert "magnitudes-Fbox-11-los00-VIS2.fits.gz" in str(fits_path)

    def test_mask_returns_none_when_absent(self):
        """PathResolver.get_mask_path returns None when mask config is absent."""
        from src.pipelines.unified_dataset.paths import PathResolver
        from src.pipelines.unified_dataset.keys import BaseKey

        config = {
            "paths": {"firebox_root": "/data/PNbody", "output_root": "/out"},
            "data_sources": {"streams": {"image_subdir": "SB_maps", "image_pattern": "x.fits.gz"}},
            "processing": {"target_size": [1024, 1024]},
        }
        resolver = PathResolver(config)
        key = BaseKey(11, "los00")
        assert resolver.get_mask_path(key, 32.0) is None

    def test_mask_subdir_map(self):
        """PathResolver uses mask_subdir_map for flexible subdir lookup."""
        from src.pipelines.unified_dataset.paths import PathResolver
        from src.pipelines.unified_dataset.keys import BaseKey

        config = {
            "paths": {"firebox_root": "/data", "output_root": "/out"},
            "data_sources": {
                "streams": {
                    "image_subdir": "SB_maps",
                    "image_pattern": "x.fits.gz",
                    "mask_subdir_map": {"eo": "MASKS_EO", "fo": "MASKS_FO"},
                    "mask_pattern": "mask-{galaxy_id}-{view_id}.fits.gz",
                }
            },
            "processing": {"target_size": [1024, 1024]},
        }
        resolver = PathResolver(config)

        mask_eo = resolver.get_mask_path(BaseKey(11, "eo"), 32.0)
        assert mask_eo is not None
        assert "MASKS_EO" in str(mask_eo)

        mask_los = resolver.get_mask_path(BaseKey(11, "los00"), 32.0)
        assert mask_los is None  # los00 not in map


# ── 3. Instrument factory ────────────────────────────────────────────────

class TestInstruments:
    def test_custom_instruments_importable(self):
        """instruments/custom_instruments.py is importable."""
        sys.path.insert(0, str(PROJECT_ROOT))
        # Can't actually build the instrument without pNbody installed,
        # but we can verify the module loads
        import importlib
        spec = importlib.util.spec_from_file_location(
            "custom_instruments",
            PROJECT_ROOT / "instruments" / "custom_instruments.py",
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "build_arrakihs_vis2_legacy")


# ── 4. generate_pnbody_fits.py --dry-run ─────────────────────────────────

class TestGeneratePnbodyFits:
    def test_dry_run_single_galaxy(self):
        """generate_pnbody_fits.py --dry-run runs without error for 1 galaxy."""
        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "data" / "generate_pnbody_fits.py"),
                "--config", str(PROJECT_ROOT / "configs" / "pnbody" / "firebox_pnbody_24los.yaml"),
                "--galaxies", "11",
                "--dry-run",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"
        assert "Processing galaxy 11" in result.stderr or "Processing galaxy 11" in result.stdout \
            or "CMD:" in result.stderr  # dry-run prints CMD but doesn't execute


# ── 5. split_annotations.py --output-prefix ──────────────────────────────

class TestSplitAnnotationsPrefix:
    def test_output_prefix(self, tmp_path):
        """--output-prefix produces {prefix}_train.json / {prefix}_val.json."""
        from src.pipelines.unified_dataset.split import galaxy_split_coco

        coco = {
            "info": {"description": "test"},
            "images": [],
            "annotations": [],
            "categories": [{"id": 1, "name": "stellar stream", "supercategory": "lsb"}],
        }
        img_id = 0
        ann_id = 0
        for gid in [11, 13, 19, 22, 24]:
            for view in ["los00", "los01"]:
                img_id += 1
                coco["images"].append({
                    "id": img_id,
                    "file_name": f"images/{gid:05d}_{view}_linear_magnitude.png",
                    "width": 64, "height": 64,
                    "galaxy_id": gid, "view_id": view, "orientation": view,
                    "variant": "linear_magnitude",
                    "base_key": f"{gid:05d}_{view}",
                })
                ann_id += 1
                coco["annotations"].append({
                    "id": ann_id, "image_id": img_id,
                    "category_id": 1, "bbox": [0, 0, 10, 10],
                    "area": 100, "iscrowd": 0,
                })

        ann_path = tmp_path / "annotations_pnbody_pseudo.json"
        ann_path.write_text(json.dumps(coco))

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "data" / "split_annotations.py"),
                "--annotations", str(ann_path),
                "--output-dir", str(tmp_path),
                "--output-prefix", "annotations_pnbody_pseudo",
                "--seed", "42",
            ],
            capture_output=True, text=True, cwd=str(PROJECT_ROOT),
        )
        assert result.returncode == 0, f"STDERR: {result.stderr}"

        assert (tmp_path / "annotations_pnbody_pseudo_train.json").exists()
        assert (tmp_path / "annotations_pnbody_pseudo_val.json").exists()
        assert (tmp_path / "split_manifest.json").exists()

        train = json.loads((tmp_path / "annotations_pnbody_pseudo_train.json").read_text())
        val = json.loads((tmp_path / "annotations_pnbody_pseudo_val.json").read_text())
        assert len(train["images"]) + len(val["images"]) == 10

        manifest = json.loads((tmp_path / "split_manifest.json").read_text())
        train_gids = set(manifest["train_galaxy_ids"])
        val_gids = set(manifest["val_galaxy_ids"])
        assert train_gids.isdisjoint(val_gids)


# ── 6. Pseudo-label flow: rasterize + overlay + completeness gate ────────

class TestPseudoLabelFlow:
    def test_rasterize_and_save(self, tmp_path):
        """Full pseudo-label artifact pipeline: rasterize -> save -> completeness."""
        from src.pipelines.unified_dataset.artifacts import save_pseudo_gt
        from src.pipelines.unified_dataset.inference_sam3 import _pseudo_label_complete, _PSEUDO_LABEL_ARTIFACTS
        from src.visualization.overlay import save_pseudo_label_overlay

        H, W = 64, 64
        masks = [
            {"segmentation": np.zeros((H, W), dtype=bool), "type_label": "streams",
             "predicted_iou": 0.92, "bbox": [5, 5, 10, 10]},
            {"segmentation": np.zeros((H, W), dtype=bool), "type_label": "satellites",
             "predicted_iou": 0.88, "bbox": [30, 30, 10, 10]},
        ]
        masks[0]["segmentation"][5:15, 5:15] = True
        masks[1]["segmentation"][30:40, 30:40] = True

        gt_dir = tmp_path / "gt_dir"
        gt_dir.mkdir()

        # 1) rasterize + save
        instance_map, instances = save_pseudo_gt(gt_dir, masks, H, W)
        assert (gt_dir / "instance_map_uint8.png").exists()
        assert (gt_dir / "instances.json").exists()
        assert instance_map.dtype == np.uint8
        assert len(instances) == 2

        # 2) pred-only QA overlay
        image = np.zeros((H, W, 3), dtype=np.uint8)
        overlay_path = gt_dir / "sam3_pseudo_label_overlay.png"
        save_pseudo_label_overlay(overlay_path, image, masks)
        assert overlay_path.exists()

        # 3) completeness gate — still incomplete (missing manifest, raw, post JSONs)
        assert not _pseudo_label_complete(gt_dir)

        # 4) create remaining artifacts
        (gt_dir / "manifest.json").write_text("{}")
        (gt_dir / "sam3_predictions_raw.json").write_text("{}")
        (gt_dir / "sam3_predictions_post.json").write_text("{}")
        assert _pseudo_label_complete(gt_dir)

    def test_overlay_runs_without_gt(self, tmp_path):
        """save_pseudo_label_overlay works with empty predictions."""
        from src.visualization.overlay import save_pseudo_label_overlay

        image = np.zeros((64, 64, 3), dtype=np.uint8)
        out = tmp_path / "overlay.png"
        save_pseudo_label_overlay(out, image, [])
        assert out.exists()


# ── 7. GT phase skip ─────────────────────────────────────────────────────

class TestGtPhaseSkip:
    def test_gt_run_returns_early_when_disabled(self):
        """run_gt_phase returns immediately when gt_phase.enabled=false."""
        import logging
        from src.pipelines.unified_dataset.gt import run_gt_phase
        from src.pipelines.unified_dataset.keys import BaseKey

        config = {
            "gt_phase": {"enabled": False},
            "paths": {"firebox_root": "/dummy", "output_root": "/dummy"},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [64, 64]},
            "data_selection": {"canonical_sb_threshold": 32},
        }
        logger = logging.getLogger("test_gt_skip")
        # Should return without error even though config has no mask paths
        run_gt_phase(config, [BaseKey(11, "los00")], logger)


# ── 8. generate_noisy_fits config compat ─────────────────────────────────

class TestGenerateNoisyFitsCompat:
    def test_views_key_read(self):
        """generate_noisy_fits.py reads data_selection.views (not only orientations)."""
        from src.pipelines.unified_dataset.config import load_config
        cfg = load_config(PROJECT_ROOT / "configs" / "noise_profiles.yaml")
        views = cfg["data_selection"].get("views") or cfg["data_selection"].get("orientations")
        assert views is not None
        assert "eo" in views


# ── 9. Export with configurable annotations_filename ─────────────────────

class TestExportAnnotationsFilename:
    def test_custom_annotations_filename(self, tmp_path):
        """run_export_phase writes to configured annotations_filename."""
        import logging
        from src.pipelines.unified_dataset.export import run_export_phase
        from src.pipelines.unified_dataset.keys import BaseKey

        output_root = tmp_path / "output"
        key = BaseKey(11, "los00")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        gt_dir.mkdir(parents=True)

        H, W = 64, 64
        instance_map = np.zeros((H, W), dtype=np.uint8)
        instance_map[5:15, 5:15] = 1
        Image.fromarray(instance_map).save(gt_dir / "instance_map_uint8.png")
        (gt_dir / "instances.json").write_text(json.dumps([{"id": 1, "type": "streams"}]))

        render_dir = output_root / "renders" / "current" / "linear_magnitude" / str(key)
        render_dir.mkdir(parents=True)
        Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(render_dir / "0000.png")

        config = {
            "paths": {"firebox_root": "/dummy", "output_root": str(output_root)},
            "data_sources": {"streams": {}},
            "processing": {"target_size": [H, W]},
            "preprocessing_variants": [{"name": "linear_magnitude"}],
            "data_selection": {"galaxy_ids": [11], "views": ["los00"]},
            "export_phase": {"annotations_filename": "annotations_pnbody_pseudo.json"},
        }
        logger = logging.getLogger("test_export")
        run_export_phase(config, [key], logger)

        ann_path = output_root / "sam3_prepared" / "annotations_pnbody_pseudo.json"
        assert ann_path.exists()
        coco = json.loads(ann_path.read_text())
        assert len(coco["images"]) == 1
        assert coco["images"][0]["view_id"] == "los00"
        assert coco["images"][0]["orientation"] == "los00"  # mirror field
