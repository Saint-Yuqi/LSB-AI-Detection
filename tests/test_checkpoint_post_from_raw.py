from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.eval.evaluate_checkpoint import _output_dir, rebuild_post_from_raw_predictions
from src.evaluation.checkpoint_eval import Sample
from src.pipelines.unified_dataset.artifacts import load_predictions_json, save_predictions_json


MODE = "fbox_gold_satellites"
RENDER_CFG = {"condition": "current", "variant": "linear_magnitude"}


def _make_center_satellite(
    shape: tuple[int, int] = (1024, 1024),
    box: tuple[int, int, int, int] = (497, 497, 527, 527),
) -> dict:
    seg = np.zeros(shape, dtype=bool)
    y0, x0, y1, x1 = box
    seg[y0:y1, x0:x1] = True
    return {
        "type_label": "satellites",
        "segmentation": seg,
        "score": 0.95,
    }


def _make_sample(tmp_path: Path) -> Sample:
    render_path = tmp_path / "render.png"
    Image.fromarray(np.zeros((1024, 1024, 3), dtype=np.uint8)).save(render_path)

    gt = np.zeros((1024, 1024), dtype=np.int32)
    gt[497:527, 497:527] = 1
    return Sample(
        base_key="00011_eo",
        galaxy_id=11,
        view="eo",
        benchmark_mode=MODE,
        render_1024_path=render_path,
        gt_instance_map_1024=gt,
        gt_type_of_id={1: "satellites"},
        roi_bbox_1024=(0, 0, 1024, 1024),
    )


def _base_cfg(output_root: Path, *, enable_core_policy: bool) -> dict:
    return {
        "matching": {"iou_threshold": 0.5},
        "diagnostics": {"enabled": False},
        "post": {
            "pred_only": {
                "enable_streams_sanity": True,
                "enable_score_gate": True,
                "score_gate": {},
                "enable_prior_filter": True,
                "prior_filter": {
                    "area_min": 30,
                    "solidity_min": 0.83,
                    "aspect_sym_max": 1.75,
                },
                "enable_core_policy": enable_core_policy,
                "core_policy": {},
                "enable_cross_type_conflict": False,
            },
            "gt_aware": {"enable_gt_stream_conflict": True},
        },
        "overlay": {"enabled": False},
        "output": {
            "root": str(output_root),
            "save_raw_predictions": True,
            "save_post_predictions": True,
            "save_overlays": False,
        },
    }


def _write_source_raw_run(source_root: Path, sample: Sample) -> Path:
    sample_dir = _output_dir(source_root, MODE, RENDER_CFG, sample.base_key)
    sample_dir.mkdir(parents=True, exist_ok=True)
    save_predictions_json(
        sample_dir / "predictions_raw.json",
        [_make_center_satellite()],
        1024,
        1024,
        engine="sam3",
        layer="raw",
    )
    return sample_dir


def test_post_from_raw_keeps_hard_center_filtered_when_core_policy_is_disabled(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    source_root = tmp_path / "source_eval"
    _write_source_raw_run(source_root, sample)

    strict_root = tmp_path / "strict_eval"
    strict_cfg = _base_cfg(strict_root, enable_core_policy=True)
    rebuild_post_from_raw_predictions(
        strict_cfg,
        [sample],
        MODE,
        RENDER_CFG,
        source_output_root=source_root,
    )

    nocore_root = tmp_path / "nocore_eval"
    nocore_cfg = _base_cfg(nocore_root, enable_core_policy=False)
    rebuild_post_from_raw_predictions(
        nocore_cfg,
        [sample],
        MODE,
        RENDER_CFG,
        source_output_root=source_root,
    )

    strict_post = _output_dir(strict_root, MODE, RENDER_CFG, sample.base_key) / "predictions_post_pred_only.json"
    nocore_post = _output_dir(nocore_root, MODE, RENDER_CFG, sample.base_key) / "predictions_post_pred_only.json"
    strict_trace = _output_dir(strict_root, MODE, RENDER_CFG, sample.base_key) / "post_pred_only_satellite_stage_trace.json"
    nocore_trace = _output_dir(nocore_root, MODE, RENDER_CFG, sample.base_key) / "post_pred_only_satellite_stage_trace.json"

    assert strict_post.is_file()
    assert nocore_post.is_file()
    assert strict_trace.is_file()
    assert nocore_trace.is_file()

    _strict_doc, strict_masks = load_predictions_json(strict_post)
    _nocore_doc, nocore_masks = load_predictions_json(nocore_post)
    assert len(strict_masks) == 0
    assert len(nocore_masks) == 0

    strict_trace_doc = json.loads(strict_trace.read_text())
    nocore_trace_doc = json.loads(nocore_trace.read_text())
    strict_prior = strict_trace_doc["records"][0]["stage_results"][1]
    strict_core = strict_trace_doc["records"][0]["stage_results"][2]
    nocore_prior = nocore_trace_doc["records"][0]["stage_results"][1]
    nocore_core = nocore_trace_doc["records"][0]["stage_results"][2]
    assert strict_prior["stage"] == "prior_filter"
    assert strict_prior["outcome"] == "drop"
    assert strict_prior["reason"] == "prior_hard_center"
    assert strict_core["stage"] == "core_policy"
    assert strict_core["outcome"] == "not_reached"
    assert strict_core["reason"] is None
    assert nocore_prior["stage"] == "prior_filter"
    assert nocore_prior["outcome"] == "drop"
    assert nocore_prior["reason"] == "prior_hard_center"
    assert nocore_core["stage"] == "core_policy"
    assert nocore_core["outcome"] == "not_reached"
    assert nocore_core["reason"] is None


def test_post_from_raw_refuses_inplace_overwrite_by_default(tmp_path: Path) -> None:
    sample = _make_sample(tmp_path)
    source_root = tmp_path / "source_eval"
    _write_source_raw_run(source_root, sample)

    cfg = _base_cfg(source_root, enable_core_policy=False)
    with pytest.raises(RuntimeError, match="refusing in-place post-from-raw overwrite"):
        rebuild_post_from_raw_predictions(
            cfg,
            [sample],
            MODE,
            RENDER_CFG,
            source_output_root=source_root,
        )
