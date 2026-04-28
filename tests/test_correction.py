"""
Tests for correction module: stale revision hash rejection, superseded status,
correction eligibility rules, and per-family write-back.

Usage:
    pytest tests/test_correction.py -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.correction import (
    CorrectionLinkMap,
    import_correction,
    validate_revision_hash,
    _write_back_satellite_mv,
    _write_back_stream_ev,
)
from src.review.schemas import (
    CorrectionLink,
    CropSpec,
    TaskFamily,
    compute_revision_hash_sat_mv,
)
from src.utils.coco_utils import mask_to_rle


@pytest.fixture
def gt_dir(tmp_path):
    bk_dir = tmp_path / "00011_eo"
    bk_dir.mkdir()

    imap = np.zeros((100, 100), dtype=np.uint8)
    imap[20:40, 20:40] = 1  # satellite instance 1
    imap[60:80, 60:80] = 2  # stream instance 2
    Image.fromarray(imap).save(bk_dir / "instance_map_uint8.png")

    instances = [
        {"id": 1, "type": "satellites"},
        {"id": 2, "type": "streams"},
    ]
    (bk_dir / "instances.json").write_text(json.dumps(instances))

    return tmp_path


def _make_sat_mv_record(gt_dir: Path) -> dict:
    imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
    mask = (imap == 1).astype(np.uint8)
    rle = mask_to_rle(mask)
    from src.utils.coco_utils import get_bbox_from_mask
    bbox = get_bbox_from_mask(mask)
    bbox_xywh = (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))

    crop_spec = CropSpec(center_x=30, center_y=30, size=384)

    rev_hash = compute_revision_hash_sat_mv(
        task_family=TaskFamily.SATELLITE_MV,
        sample_id="00011_eo",
        task_revision_id="rev_00",
        render_spec_id="sat_mv_v1",
        fixed_prompt_id="sat_mv_v1",
        candidate_rle=rle,
        candidate_bbox_xywh=bbox_xywh,
        crop_spec=crop_spec,
        bare_context_sha1="fake_sha1",
        candidate_source="authoritative",
        authoritative_instance_id=1,
    )

    return {
        "example_id": "satell_000001",
        "task_family": "satellite_mv",
        "sample_id": "00011_eo",
        "halo_id": 11,
        "view_id": "eo",
        "task_revision_id": "rev_00",
        "render_spec_id": "sat_mv_v1",
        "fixed_prompt_id": "sat_mv_v1",
        "decision_label": "accept",
        "revision_hash": rev_hash,
        "asset_refs": {
            "bare_context_path": "review_context/00011_eo_asinh.png",
            "bare_context_sha1": "fake_sha1",
            "crop_spec": {"center_x": 30, "center_y": 30, "size": 384, "pad_mode": "zero"},
            "candidate_bbox_xywh": list(bbox_xywh),
            "candidate_rle": rle,
            "candidate_id": "inst_001",
            "authoritative_instance_id": 1,
            "candidate_source": "authoritative",
        },
    }


class TestCorrectionLinkMap:
    def test_mark_superseded(self):
        clm = CorrectionLinkMap()
        cl = clm.mark_superseded("rev_00", "h1", "rev_01", "h2")
        assert cl.status == "superseded"
        assert clm.is_superseded("rev_00")

    def test_save_and_load(self, tmp_path):
        clm = CorrectionLinkMap()
        clm.mark_superseded("rev_00", "h1", "rev_01", "h2")
        path = tmp_path / "clm.json"
        clm.save(path)

        loaded = CorrectionLinkMap.load(path)
        assert loaded.is_superseded("rev_00")
        assert not loaded.is_superseded("rev_99")

    def test_load_missing_file(self, tmp_path):
        clm = CorrectionLinkMap.load(tmp_path / "nonexistent.json")
        assert not clm.is_superseded("anything")


class TestWriteBackSatMv:
    def test_replaces_mask(self, gt_dir):
        record = _make_sat_mv_record(gt_dir)
        new_mask = np.zeros((100, 100), dtype=np.uint8)
        new_mask[25:45, 25:45] = 1

        _write_back_satellite_mv(record, new_mask, gt_dir)

        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        assert imap[22, 22] == 0  # old position cleared
        assert imap[30, 30] == 1  # new position written

    def test_rejects_pre_merge_rejected(self, gt_dir):
        record = _make_sat_mv_record(gt_dir)
        record["asset_refs"]["candidate_source"] = "pre_merge_rejected"
        record["asset_refs"]["authoritative_instance_id"] = None

        with pytest.raises(ValueError, match="pre_merge_rejected"):
            _write_back_satellite_mv(record, np.zeros((100, 100)), gt_dir)


class TestWriteBackStreamEv:
    def test_preserves_satellites(self, gt_dir):
        record = {
            "sample_id": "00011_eo",
            "task_family": "stream_ev",
        }
        new_stream = np.zeros((100, 100), dtype=np.uint8)
        new_stream[65:85, 65:85] = 1

        _write_back_stream_ev(record, {2: new_stream}, gt_dir)

        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        assert imap[30, 30] == 1  # satellite preserved

    def test_rejects_satellite_modification(self, gt_dir):
        record = {"sample_id": "00011_eo", "task_family": "stream_ev"}
        sat_mask = np.zeros((100, 100), dtype=np.uint8)
        sat_mask[25:45, 25:45] = 1

        with pytest.raises(ValueError, match="not a stream"):
            _write_back_stream_ev(record, {1: sat_mask}, gt_dir)


class TestImportCorrection:
    def test_stale_hash_rejected(self, gt_dir):
        record = _make_sat_mv_record(gt_dir)
        record["revision_hash"] = "wrong_hash"
        link_map = CorrectionLinkMap()

        with pytest.raises(ValueError, match="[Ss]tale"):
            import_correction(
                record,
                {"corrected_mask": np.ones((100, 100), dtype=np.uint8)},
                gt_dir,
                link_map,
            )

    def test_successful_correction(self, gt_dir):
        record = _make_sat_mv_record(gt_dir)
        link_map = CorrectionLinkMap()

        new_mask = np.zeros((100, 100), dtype=np.uint8)
        new_mask[25:45, 25:45] = 1

        result = import_correction(
            record, {"corrected_mask": new_mask}, gt_dir, link_map,
        )

        assert "new_task_revision_id" in result
        assert link_map.is_superseded("rev_00")
        assert "00011_eo" in result["re_export_keys"]


class TestSyntheticEvWriteBackGuard:
    """Drop_* variants rejected; gt_complete/gt_empty/None allowed."""

    def _make_ev_record(self, gt_dir, variant_id):
        """Create a satellite_ev record with a given synthetic_variant_id."""
        from src.review.schemas import compute_revision_hash_ev
        import hashlib

        imap = np.array(Image.open(gt_dir / "00011_eo" / "instance_map_uint8.png"))
        with open(gt_dir / "00011_eo" / "instances.json") as f:
            instances = json.load(f)
        ann_hash = hashlib.sha256(
            imap.tobytes() + json.dumps(instances, sort_keys=True).encode()
        ).hexdigest()

        rev_hash = compute_revision_hash_ev(
            task_family=TaskFamily.SATELLITE_EV,
            sample_id="00011_eo",
            task_revision_id="rev_00",
            render_spec_id="sat_ev_v1",
            fixed_prompt_id="sat_ev_v1",
            review_image_sha1="fake",
            annotation_state_hash=ann_hash,
            synthetic_variant_id=variant_id,
        )
        return {
            "task_family": "satellite_ev",
            "sample_id": "00011_eo",
            "task_revision_id": "rev_00",
            "render_spec_id": "sat_ev_v1",
            "fixed_prompt_id": "sat_ev_v1",
            "revision_hash": rev_hash,
            "asset_refs": {
                "review_image_path": "review_ev/test.png",
                "review_image_sha1": "fake",
                "annotation_state_hash": ann_hash,
                "synthetic_variant_id": variant_id,
            },
        }

    def test_drop_variant_rejected(self, gt_dir):
        record = self._make_ev_record(gt_dir, "drop_top3")
        link_map = CorrectionLinkMap()

        with pytest.raises(ValueError, match="not write-back safe"):
            import_correction(
                record,
                {"corrected_map": np.zeros((100, 100), dtype=np.uint8),
                 "corrected_instances": []},
                gt_dir, link_map,
            )

    def test_gt_complete_allowed(self, gt_dir):
        record = self._make_ev_record(gt_dir, "gt_complete")
        link_map = CorrectionLinkMap()

        result = import_correction(
            record,
            {"corrected_map": np.zeros((100, 100), dtype=np.uint8),
             "corrected_instances": []},
            gt_dir, link_map,
        )
        assert "new_task_revision_id" in result

    def test_none_variant_allowed(self, gt_dir):
        record = self._make_ev_record(gt_dir, None)
        link_map = CorrectionLinkMap()

        result = import_correction(
            record,
            {"corrected_map": np.zeros((100, 100), dtype=np.uint8),
             "corrected_instances": []},
            gt_dir, link_map,
        )
        assert "new_task_revision_id" in result
