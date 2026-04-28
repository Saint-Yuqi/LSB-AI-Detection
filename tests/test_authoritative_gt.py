from __future__ import annotations

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
from src.pipelines.unified_dataset.export import run_export_phase
from src.pipelines.unified_dataset.keys import BaseKey
from src.review.authoritative_gt import (
    AlreadyDeletedError,
    DuplicateAdoptionError,
    LockAcquisitionError,
    adopt_raw_candidate,
    delete_authoritative_instance,
    directory_update_lock,
    parse_base_key,
)
from src.utils.coco_utils import mask_to_rle


def _write_stats_json(path: Path, *, min_area: int = 30) -> None:
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


def _write_render(output_root: Path, key: BaseKey, variant: str = "linear_magnitude") -> None:
    render_dir = output_root / "renders" / "current" / variant / str(key)
    render_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros((64, 64, 3), dtype=np.uint8)).save(render_dir / "0000.png")


def _write_gt_case(
    gt_dir: Path,
    *,
    satellites: list[tuple[int, tuple[slice, slice], dict | None]] | None = None,
    streams: list[tuple[int, tuple[slice, slice]]] | None = None,
    shape: tuple[int, int] = (64, 64),
) -> tuple[np.ndarray, list[dict]]:
    satellites = satellites or []
    streams = streams or []

    gt_dir.mkdir(parents=True, exist_ok=True)
    imap = np.zeros(shape, dtype=np.uint8)
    instances: list[dict] = []

    for inst_id, (rows, cols) in streams:
        imap[rows, cols] = inst_id
        instances.append({"id": inst_id, "type": "streams"})

    for inst_id, (rows, cols), provenance in satellites:
        imap[rows, cols] = inst_id
        inst = {"id": inst_id, "type": "satellites"}
        if provenance is not None:
            inst["provenance"] = provenance
        instances.append(inst)

    Image.fromarray(imap).save(gt_dir / "instance_map_uint8.png")
    (gt_dir / "instances.json").write_text(json.dumps(instances, indent=2))
    return imap, instances


def _sat_mask(seg: np.ndarray, score: float = 0.9) -> dict:
    ys, xs = np.where(seg)
    return {
        "type_label": "satellites",
        "segmentation": seg.astype(bool),
        "score": score,
        "area": int(seg.sum()),
        "bbox": [int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)],
    }


@pytest.fixture
def dr1_config(tmp_path: Path) -> dict:
    output_root = tmp_path / "output"
    stats_path = tmp_path / "mask_stats_summary.json"
    _write_stats_json(stats_path, min_area=5)
    return {
        "paths": {"firebox_root": str(tmp_path / "raw"), "output_root": str(output_root)},
        "data_selection": {"galaxy_ids": [11], "views": ["eo"]},
        "processing": {"target_size": [64, 64]},
        "preprocessing_variants": [{"name": "linear_magnitude"}],
        "satellites": {"prior": {"stats_json": str(stats_path)}},
        "data_sources": {"streams": {"image_pattern": "unused-{orientation}.fits.gz"}},
    }


@pytest.fixture
def pnbody_config(tmp_path: Path) -> dict:
    output_root = tmp_path / "output"
    stats_path = tmp_path / "mask_stats_summary.json"
    _write_stats_json(stats_path, min_area=5)
    return {
        "dataset_name": "pnbody",
        "paths": {"firebox_root": str(tmp_path / "raw"), "output_root": str(output_root)},
        "data_selection": {"galaxy_ids": [11], "views": ["los00"]},
        "processing": {"target_size": [64, 64]},
        "preprocessing_variants": [{"name": "linear_magnitude"}],
        "satellites": {"prior": {"stats_json": str(stats_path)}},
        "data_sources": {"streams": {"image_pattern": "unused-{view_id}.fits.gz"}},
        "data_conditions": {"clean": {"label_mode": "authoritative"}},
    }


class TestDirectoryUpdateLock:
    def test_lock_timeout_exposes_owner(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "00011_eo"
        gt_dir.mkdir()
        lock_dir = gt_dir / ".update.lock"
        lock_dir.mkdir()
        (lock_dir / "owner.json").write_text(json.dumps({"hostname": "node-a", "pid": 7}))

        with pytest.raises(LockAcquisitionError, match="node-a"):
            with directory_update_lock(gt_dir, operation="test", retries=1, initial_delay_s=0.0, max_delay_s=0.0):
                pass

    def test_lock_is_released_on_exception(self, tmp_path: Path) -> None:
        gt_dir = tmp_path / "00011_eo"
        gt_dir.mkdir()

        with pytest.raises(RuntimeError, match="boom"):
            with directory_update_lock(gt_dir, operation="test", retries=0):
                raise RuntimeError("boom")

        assert not (gt_dir / ".update.lock").exists()


class TestManualAuthoritativeGt:
    def test_adopt_old_schema_does_not_mutate_source(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[(1, (slice(5, 10), slice(5, 10)), None)],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )

        seg = np.zeros((64, 64), dtype=np.uint8)
        seg[20:28, 20:28] = 1
        legacy_raw = {
            "schema_version": 1,
            "predictions": [
                {
                    "type_label": "satellites",
                    "score": 0.91,
                    "area": int(seg.sum()),
                    "bbox_xywh": [20, 20, 8, 8],
                    "rle": mask_to_rle(seg),
                }
            ],
        }
        source_path = output_root / "legacy_raw.json"
        original_text = json.dumps(legacy_raw, indent=2)
        source_path.write_text(original_text)

        result = adopt_raw_candidate(
            dr1_config,
            key=key,
            source_json=source_path,
            candidate_id="sat_0000",
            min_area_px=5,
            manual_note="manual add",
        )

        assert result["assigned_instance_id"] == 3
        assert source_path.read_text() == original_text

        instances = json.loads((gt_dir / "instances.json").read_text())
        adopted = next(inst for inst in instances if inst["id"] == 3)
        assert adopted["provenance"]["source_candidate_id"] == "sat_0000"
        assert adopted["provenance"]["manual_note"] == "manual add"

    def test_adopt_by_candidate_rle_sha1_selector(self, dr1_config: dict) -> None:
        """--candidate-rle-sha1 is the preferred stable selector across reruns."""
        from src.review.authoritative_gt import rle_sha1

        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(gt_dir, satellites=[(1, (slice(5, 10), slice(5, 10)), None)])

        seg_a = np.zeros((64, 64), dtype=bool)
        seg_a[20:28, 20:28] = True
        seg_b = np.zeros((64, 64), dtype=bool)
        seg_b[40:45, 40:45] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(
            raw_path, [_sat_mask(seg_a), _sat_mask(seg_b)], 64, 64, layer="raw"
        )

        target_sha = rle_sha1(mask_to_rle(seg_b.astype(np.uint8)))
        result = adopt_raw_candidate(
            dr1_config,
            key=key,
            source_json=raw_path,
            candidate_rle_sha1=target_sha,
            min_area_px=1,
            manual_note="selected-by-sha",
        )
        instances = json.loads((gt_dir / "instances.json").read_text())
        adopted = next(i for i in instances if i["id"] == result["assigned_instance_id"])
        assert adopted["provenance"]["source_candidate_rle_sha1"] == target_sha

    def test_adopt_by_candidate_rle_sha1_conflict_with_id_raises(
        self, dr1_config: dict
    ) -> None:
        """If sha1 and id disagree, select_prediction_candidate rejects the selection."""
        from src.review.authoritative_gt import rle_sha1

        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(gt_dir, satellites=[(1, (slice(5, 10), slice(5, 10)), None)])

        seg_a = np.zeros((64, 64), dtype=bool)
        seg_a[20:28, 20:28] = True
        seg_b = np.zeros((64, 64), dtype=bool)
        seg_b[40:45, 40:45] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(
            raw_path, [_sat_mask(seg_a), _sat_mask(seg_b)], 64, 64, layer="raw"
        )

        sha_a = rle_sha1(mask_to_rle(seg_a.astype(np.uint8)))
        with pytest.raises(ValueError, match="No prediction matches"):
            adopt_raw_candidate(
                dr1_config,
                key=key,
                source_json=raw_path,
                candidate_id="sat_0001",
                candidate_rle_sha1=sha_a,
                min_area_px=1,
            )

    def test_duplicate_adoption_is_rejected(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(gt_dir, satellites=[(1, (slice(5, 10), slice(5, 10)), None)])

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        adopt_raw_candidate(dr1_config, key=key, source_json=raw_path, candidate_id="sat_0000", min_area_px=5)
        with pytest.raises(DuplicateAdoptionError):
            adopt_raw_candidate(dr1_config, key=key, source_json=raw_path, candidate_id="sat_0000", min_area_px=5)

    def test_delete_tail_then_adopt_does_not_reuse_id(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[
                (1, (slice(5, 10), slice(5, 10)), {"source_candidate_id": "sat_0000"}),
                (3, (slice(20, 25), slice(20, 25)), {"source_candidate_id": "sat_0001"}),
            ],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )

        delete_result = delete_authoritative_instance(dr1_config, key=key, instance_id=3, manual_note="fp")
        assert delete_result["deleted_instance_id"] == 3

        seg = np.zeros((64, 64), dtype=bool)
        seg[30:36, 30:36] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        adopt_result = adopt_raw_candidate(
            dr1_config,
            key=key,
            source_json=raw_path,
            candidate_id="sat_0000",
            min_area_px=5,
        )
        assert adopt_result["assigned_instance_id"] == 4

    def test_prepared_log_reserves_future_id(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[(1, (slice(5, 10), slice(5, 10)), None)],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )
        (gt_dir / "manual_corrections.jsonl").write_text(
            json.dumps(
                {
                    "operation_id": "abc",
                    "status": "prepared",
                    "operation": "adopt_raw_satellite",
                    "assigned_instance_id": 3,
                }
            )
            + "\n"
        )

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        result = adopt_raw_candidate(
            dr1_config,
            key=key,
            source_json=raw_path,
            candidate_id="sat_0000",
            min_area_px=5,
        )
        assert result["assigned_instance_id"] == 4

    def test_overflow_guard_uses_history(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(gt_dir)
        (gt_dir / "manual_corrections.jsonl").write_text(
            json.dumps(
                {
                    "operation_id": "abc",
                    "status": "prepared",
                    "operation": "adopt_raw_satellite",
                    "assigned_instance_id": 255,
                }
            )
            + "\n"
        )

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        with pytest.raises(OverflowError):
            adopt_raw_candidate(
                dr1_config,
                key=key,
                source_json=raw_path,
                candidate_id="sat_0000",
                min_area_px=5,
            )

    def test_bbox_fill_warning_is_written_into_note(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(gt_dir)

        seg = np.zeros((64, 64), dtype=bool)
        diag = [(10, 10), (11, 12), (12, 14), (13, 16), (15, 19)]
        for y, x in diag:
            seg[y, x] = True
        raw_path = output_root / "raw.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        adopt_raw_candidate(
            dr1_config,
            key=key,
            source_json=raw_path,
            candidate_id="sat_0000",
            min_area_px=1,
            manual_note="reviewed",
        )

        instances = json.loads((gt_dir / "instances.json").read_text())
        note = next(inst["provenance"]["manual_note"] for inst in instances if inst["type"] == "satellites")
        assert "reviewed" in note
        assert "warning:bbox_fill_lt_0.10" in note

    def test_delete_logs_snapshot_and_is_idempotent(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[(1, (slice(5, 10), slice(5, 10)), {"source_candidate_id": "sat_0000"})],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )

        delete_authoritative_instance(dr1_config, key=key, instance_id=1, manual_note="fp")
        log_entries = [
            json.loads(line)
            for line in (gt_dir / "manual_corrections.jsonl").read_text().splitlines()
        ]
        committed = [entry for entry in log_entries if entry["status"] == "committed"]
        assert committed[0]["deleted_instance_id"] == 1
        assert committed[0]["deleted_area_px"] > 0
        assert committed[0]["deleted_instance_provenance"]["source_candidate_id"] == "sat_0000"

        with pytest.raises(AlreadyDeletedError):
            delete_authoritative_instance(dr1_config, key=key, instance_id=1)

    def test_delete_removes_annotation_from_export(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[(1, (slice(5, 10), slice(5, 10)), {"source_candidate_id": "sat_0000"})],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )
        _write_render(output_root, key)

        delete_authoritative_instance(dr1_config, key=key, instance_id=1)
        run_export_phase(dr1_config, [key], logger=logging.getLogger("test"))

        coco = json.loads((output_root / "sam3_prepared" / "annotations.json").read_text())
        cat_ids = [ann["category_id"] for ann in coco["annotations"]]
        assert cat_ids == [1]

    def test_export_provenance_fields_are_sparse(self, dr1_config: dict) -> None:
        output_root = Path(dr1_config["paths"]["output_root"])
        key = BaseKey(11, "eo")
        gt_dir = output_root / "gt_canonical" / "current" / str(key)
        _write_gt_case(
            gt_dir,
            satellites=[
                (
                    1,
                    (slice(5, 10), slice(5, 10)),
                    {
                        "label_source": "manual_adopted",
                        "human_fix": True,
                        "human_fix_kind": "adopt_from_raw",
                        "source_candidate_id": "sat_0007",
                        "source_raw_index": 7,
                        "source_candidate_rle_sha1": "deadbeefdeadbeef",
                        "source_prediction_path": "/tmp/source.json",
                        "final_mask_rle_sha1": "feedfacefeedface",
                        "manual_note": "reviewed",
                    },
                )
            ],
            streams=[(2, (slice(40, 45), slice(40, 45)))],
        )
        _write_render(output_root, key)
        run_export_phase(dr1_config, [key], logger=logging.getLogger("test"))

        coco = json.loads((output_root / "sam3_prepared" / "annotations.json").read_text())
        satellite_ann = next(ann for ann in coco["annotations"] if ann["category_id"] == 2)
        stream_ann = next(ann for ann in coco["annotations"] if ann["category_id"] == 1)
        assert satellite_ann["source_candidate_id"] == "sat_0007"
        assert satellite_ann["human_fix"] is True
        assert "source_candidate_id" not in stream_ann

    def test_adopt_raw_pnbody_clean_authoritative(self, pnbody_config: dict) -> None:
        output_root = Path(pnbody_config["paths"]["output_root"])
        key = parse_base_key("00011_los00")
        gt_dir = output_root / "pseudo_gt_canonical" / "pnbody" / "clean" / "current" / str(key)
        _write_gt_case(gt_dir, satellites=[(1, (slice(5, 10), slice(5, 10)), None)])

        seg = np.zeros((64, 64), dtype=bool)
        seg[20:28, 20:28] = True
        raw_path = output_root / "raw_pnbody.json"
        save_predictions_json(raw_path, [_sat_mask(seg)], 64, 64, layer="raw")

        result = adopt_raw_candidate(
            pnbody_config,
            key=key,
            condition="clean",
            source_json=raw_path,
            candidate_id="sat_0000",
            min_area_px=5,
        )
        assert result["assigned_instance_id"] == 2
