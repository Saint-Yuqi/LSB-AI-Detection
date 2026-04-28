"""
Tests for review schemas, label-space enforcement, revision-hash determinism,
and key-adapter mapping.

Usage:
    pytest tests/test_review_schemas.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.review.schemas import (
    LABEL_SPACES,
    CorrectionLink,
    CropSpec,
    EvAssetRefs,
    LabelSource,
    SatMvAssetRefs,
    TaskFamily,
    VerifierExample,
    compute_revision_hash_ev,
    compute_revision_hash_sat_mv,
    validate_label,
)
from src.review.key_adapter import (
    base_key_to_halo_id,
    base_key_to_sample_id,
    halo_id_to_galaxy_ids,
    sample_id_to_base_key,
)
from src.pipelines.unified_dataset.keys import BaseKey


# ---------------------------------------------------------------------------
#  Frozen dataclass invariants
# ---------------------------------------------------------------------------

class TestCropSpec:
    def test_frozen(self):
        cs = CropSpec(center_x=100, center_y=200, size=384)
        with pytest.raises(AttributeError):
            cs.center_x = 50

    def test_content_key(self):
        cs = CropSpec(center_x=128, center_y=256, size=384, pad_mode="zero")
        assert cs.content_key == "128_256_384_zero"

    def test_content_key_deterministic(self):
        a = CropSpec(center_x=10, center_y=20)
        b = CropSpec(center_x=10, center_y=20)
        assert a.content_key == b.content_key


class TestSatMvAssetRefs:
    def test_frozen(self):
        ar = SatMvAssetRefs(
            bare_context_path="ctx.png", bare_context_sha1="abc",
            crop_spec=CropSpec(0, 0), candidate_bbox_xywh=(10, 20, 30, 40),
            candidate_rle={"size": [100, 100], "counts": "abc"},
            candidate_id="inst_001",
            authoritative_instance_id=1, candidate_source="authoritative",
        )
        with pytest.raises(AttributeError):
            ar.candidate_id = "inst_002"

    def test_bbox_and_crop_are_distinct(self):
        ar = SatMvAssetRefs(
            bare_context_path="ctx.png", bare_context_sha1="abc",
            crop_spec=CropSpec(50, 60, 384),
            candidate_bbox_xywh=(10, 20, 30, 40),
            candidate_rle={}, candidate_id="inst_001",
            authoritative_instance_id=1, candidate_source="authoritative",
        )
        assert ar.candidate_bbox_xywh != (
            ar.crop_spec.center_x, ar.crop_spec.center_y,
            ar.crop_spec.size, ar.crop_spec.size,
        )

    def test_reject_has_none_instance_id(self):
        ar = SatMvAssetRefs(
            bare_context_path="ctx.png", bare_context_sha1="abc",
            crop_spec=CropSpec(0, 0), candidate_bbox_xywh=(0, 0, 10, 10),
            candidate_rle={}, candidate_id="rej_prior_000",
            authoritative_instance_id=None,
            candidate_source="pre_merge_rejected",
        )
        assert ar.authoritative_instance_id is None
        assert ar.candidate_source == "pre_merge_rejected"


# ---------------------------------------------------------------------------
#  Label space enforcement
# ---------------------------------------------------------------------------

class TestLabelSpaces:
    def test_satellite_mv_valid(self):
        for label in ("accept", "minor_fix", "reject", "route_to_ev"):
            validate_label(TaskFamily.SATELLITE_MV, label)

    def test_satellite_mv_invalid(self):
        with pytest.raises(ValueError, match="invalid"):
            validate_label(TaskFamily.SATELLITE_MV, "confirm_complete")

    def test_stream_ev_valid(self):
        for label in LABEL_SPACES[TaskFamily.STREAM_EV]:
            validate_label(TaskFamily.STREAM_EV, label)

    def test_verifier_example_rejects_bad_label(self):
        with pytest.raises(ValueError):
            VerifierExample(
                example_id="x", task_family=TaskFamily.SATELLITE_MV,
                sample_id="00011_eo", halo_id=11, view_id="eo",
                task_revision_id="rev_00",
                asset_refs=SatMvAssetRefs(
                    bare_context_path="p", bare_context_sha1="h",
                    crop_spec=CropSpec(0, 0),
                    candidate_bbox_xywh=(0, 0, 1, 1),
                    candidate_rle={}, candidate_id="c",
                    authoritative_instance_id=1,
                    candidate_source="authoritative",
                ),
                render_spec_id="sat_mv_v1", fixed_prompt_id="sat_mv_v1",
                decision_label="INVALID_LABEL",
                label_source=LabelSource.SILVER,
                source_round="r0", source_checkpoint="cp0",
                revision_hash="abc",
            )


class TestCorrectionLink:
    def test_valid_statuses(self):
        cl = CorrectionLink("orig", "new", "h1", "h2", "superseded")
        assert cl.status == "superseded"
        cl2 = CorrectionLink("orig", "new", "h1", "h2", "active")
        assert cl2.status == "active"

    def test_invalid_status(self):
        with pytest.raises(ValueError):
            CorrectionLink("orig", "new", "h1", "h2", "invalid")


# ---------------------------------------------------------------------------
#  Revision hash determinism
# ---------------------------------------------------------------------------

class TestRevisionHash:
    def test_sat_mv_deterministic(self):
        kwargs = dict(
            task_family=TaskFamily.SATELLITE_MV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_mv_v1", fixed_prompt_id="sat_mv_v1",
            candidate_rle={"size": [100, 100], "counts": "abc"},
            candidate_bbox_xywh=(10, 20, 30, 40),
            crop_spec=CropSpec(25, 40, 384),
            bare_context_sha1="deadbeef",
            candidate_source="authoritative",
            authoritative_instance_id=5,
        )
        h1 = compute_revision_hash_sat_mv(**kwargs)
        h2 = compute_revision_hash_sat_mv(**kwargs)
        assert h1 == h2
        assert len(h1) == 64  # sha256 hex

    def test_sat_mv_changes_with_bbox(self):
        base = dict(
            task_family=TaskFamily.SATELLITE_MV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_mv_v1", fixed_prompt_id="sat_mv_v1",
            candidate_rle={}, candidate_bbox_xywh=(10, 20, 30, 40),
            crop_spec=CropSpec(25, 40, 384), bare_context_sha1="abc",
            candidate_source="authoritative", authoritative_instance_id=1,
        )
        h1 = compute_revision_hash_sat_mv(**base)
        base["candidate_bbox_xywh"] = (11, 20, 30, 40)
        h2 = compute_revision_hash_sat_mv(**base)
        assert h1 != h2

    def test_sat_mv_reject_vs_accept(self):
        base = dict(
            task_family=TaskFamily.SATELLITE_MV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_mv_v1", fixed_prompt_id="sat_mv_v1",
            candidate_rle={}, candidate_bbox_xywh=(10, 20, 30, 40),
            crop_spec=CropSpec(25, 40, 384), bare_context_sha1="abc",
        )
        h_auth = compute_revision_hash_sat_mv(
            **base, candidate_source="authoritative",
            authoritative_instance_id=5,
        )
        h_rej = compute_revision_hash_sat_mv(
            **base, candidate_source="pre_merge_rejected",
            authoritative_instance_id=None,
        )
        assert h_auth != h_rej

    def test_ev_deterministic(self):
        kwargs = dict(
            task_family=TaskFamily.SATELLITE_EV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_ev_v1", fixed_prompt_id="sat_ev_v1",
            review_image_sha1="imgsha1",
            annotation_state_hash="annhash",
        )
        h1 = compute_revision_hash_ev(**kwargs)
        h2 = compute_revision_hash_ev(**kwargs)
        assert h1 == h2

    def test_stream_ev_with_fragment_hash(self):
        base = dict(
            task_family=TaskFamily.STREAM_EV,
            sample_id="s", task_revision_id="r",
            render_spec_id="stream_ev_v1", fixed_prompt_id="stream_ev_v1",
            review_image_sha1="img", annotation_state_hash="ann",
        )
        h_no_frag = compute_revision_hash_ev(**base)
        h_with_frag = compute_revision_hash_ev(**base, fragment_hints_hash="frag")
        assert h_no_frag != h_with_frag

    def test_ev_omit_variant_equals_explicit_none(self):
        """Backward compat: omitting synthetic_variant_id == passing None."""
        base = dict(
            task_family=TaskFamily.SATELLITE_EV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_ev_v1", fixed_prompt_id="sat_ev_v1",
            review_image_sha1="imgsha1",
            annotation_state_hash="annhash",
        )
        h_omit = compute_revision_hash_ev(**base)
        h_none = compute_revision_hash_ev(**base, synthetic_variant_id=None)
        assert h_omit == h_none

    def test_ev_different_variant_id_different_hash(self):
        base = dict(
            task_family=TaskFamily.SATELLITE_EV,
            sample_id="00011_eo", task_revision_id="rev_00",
            render_spec_id="sat_ev_v1", fixed_prompt_id="sat_ev_v1",
            review_image_sha1="img", annotation_state_hash="ann",
        )
        h_none = compute_revision_hash_ev(**base)
        h_complete = compute_revision_hash_ev(**base, synthetic_variant_id="gt_complete")
        h_drop = compute_revision_hash_ev(**base, synthetic_variant_id="drop_0")
        assert h_none != h_complete
        assert h_complete != h_drop
        assert h_none != h_drop


class TestSatelliteEvLabelSpace:
    def test_tightened_label_space(self):
        allowed = LABEL_SPACES[TaskFamily.SATELLITE_EV]
        assert allowed == frozenset({"confirm_complete", "add_missing", "confirm_empty"})
        assert "remove_fp" not in allowed
        assert "redraw" not in allowed


# ---------------------------------------------------------------------------
#  Key adapter
# ---------------------------------------------------------------------------

class TestKeyAdapter:
    def test_base_key_to_sample_id(self):
        bk = BaseKey(galaxy_id=11, view_id="eo")
        assert base_key_to_sample_id(bk) == "00011_eo"

    def test_base_key_to_halo_id(self):
        bk = BaseKey(galaxy_id=11, view_id="eo")
        assert base_key_to_halo_id(bk) == 11

    def test_halo_id_to_galaxy_ids_identity(self):
        assert halo_id_to_galaxy_ids(11) == [11]

    def test_sample_id_round_trip(self):
        bk = BaseKey(galaxy_id=42, view_id="los05")
        sid = base_key_to_sample_id(bk)
        bk2 = sample_id_to_base_key(sid)
        assert bk2.galaxy_id == bk.galaxy_id
        assert bk2.view_id == bk.view_id

    def test_sample_id_parse_invalid(self):
        with pytest.raises(ValueError):
            sample_id_to_base_key("no_underscore")

    def test_halo_id_same_for_dr1_and_pnbody(self):
        dr1 = BaseKey(galaxy_id=100, view_id="eo")
        pnb = BaseKey(galaxy_id=100, view_id="los00")
        assert base_key_to_halo_id(dr1) == base_key_to_halo_id(pnb)
