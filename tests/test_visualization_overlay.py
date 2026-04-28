from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.visualization.overlay import (
    save_eval_prediction_overlay,
    save_evaluation_overlay,
    save_gt_contour_only_overlay,
)


def _prediction(seg: np.ndarray, *, type_label: str, score: float) -> dict:
    rows, cols = np.where(seg)
    return {
        "segmentation": seg,
        "type_label": type_label,
        "score": score,
        "bbox": [
            int(cols.min()),
            int(rows.min()),
            int(cols.max() - cols.min() + 1),
            int(rows.max() - rows.min() + 1),
        ],
    }


def test_save_evaluation_overlay_accepts_instance_map(tmp_path) -> None:
    image = np.full((32, 32, 3), 255, dtype=np.uint8)
    gt_streams = np.zeros((32, 32), dtype=np.int32)
    gt_streams[8:18, 6:20] = 1
    pred_seg = np.zeros((32, 32), dtype=bool)
    pred_seg[9:17, 8:19] = True

    out_path = tmp_path / "eval_overlay_map.png"
    save_evaluation_overlay(
        out_path,
        image,
        gt_streams=gt_streams,
        predictions=[_prediction(pred_seg, type_label="streams", score=0.81)],
    )

    assert out_path.exists()
    rendered = np.array(Image.open(out_path).convert("RGB"))
    assert rendered.shape == image.shape


def test_save_evaluation_overlay_accepts_gt_mask_lists(tmp_path) -> None:
    image = np.full((32, 32, 3), 180, dtype=np.uint8)
    gt_stream_seg = np.zeros((32, 32), dtype=bool)
    gt_stream_seg[5:14, 5:16] = True
    gt_sat_seg = np.zeros((32, 32), dtype=bool)
    gt_sat_seg[20:26, 20:27] = True

    pred_stream = np.zeros((32, 32), dtype=bool)
    pred_stream[6:13, 7:15] = True
    pred_sat = np.zeros((32, 32), dtype=bool)
    pred_sat[20:25, 21:26] = True

    out_path = tmp_path / "eval_overlay_mask_list.png"
    save_evaluation_overlay(
        out_path,
        image,
        gt_streams=[{"segmentation": gt_stream_seg}],
        predictions=[
            _prediction(pred_stream, type_label="streams", score=0.77),
            _prediction(pred_sat, type_label="satellites", score=0.63),
        ],
        gt_satellites=[{"segmentation": gt_sat_seg}],
    )

    assert out_path.exists()


def test_save_evaluation_overlay_runs_without_gt_satellites(tmp_path) -> None:
    image = np.full((24, 24, 3), 200, dtype=np.uint8)
    gt_streams = np.zeros((24, 24), dtype=np.int32)
    gt_streams[4:12, 4:12] = 1

    out_path = tmp_path / "eval_overlay_no_sat_gt.png"
    save_evaluation_overlay(
        out_path,
        image,
        gt_streams=gt_streams,
        predictions=[],
        gt_satellites=None,
    )

    assert out_path.exists()


def test_save_evaluation_overlay_draws_legend_backdrop(tmp_path) -> None:
    image = np.full((48, 48, 3), 255, dtype=np.uint8)
    gt_streams = np.zeros((48, 48), dtype=np.int32)
    gt_streams[20:30, 18:31] = 1

    pred_seg = np.zeros((48, 48), dtype=bool)
    pred_seg[21:29, 20:29] = True

    out_path = tmp_path / "eval_overlay_legend.png"
    save_evaluation_overlay(
        out_path,
        image,
        gt_streams=gt_streams,
        predictions=[_prediction(pred_seg, type_label="streams", score=0.92)],
    )

    rendered = np.array(Image.open(out_path).convert("RGB"))
    assert np.any(rendered[10:40, 10:40] != image[10:40, 10:40]), (
        "legend backdrop should visibly alter the top-left region"
    )


# --------------------------------------------------------------------------- #
#  New overlay helpers: type-aware GT, ROI box, per-type prediction palette
# --------------------------------------------------------------------------- #


def _solid_grey_canvas(h: int, w: int, value: int = 120) -> np.ndarray:
    return np.full((h, w, 3), value, dtype=np.uint8)


def _has_near_colour(
    img: np.ndarray,
    target: tuple[int, int, int],
    tol: int = 15,
) -> bool:
    """Any pixel within ``tol`` of ``target`` in every channel."""
    r, g, b = target
    return bool(
        np.any(
            (np.abs(img[..., 0].astype(int) - r) <= tol)
            & (np.abs(img[..., 1].astype(int) - g) <= tol)
            & (np.abs(img[..., 2].astype(int) - b) <= tol)
        )
    )


def test_save_gt_contour_only_overlay_type_aware_and_roi(tmp_path) -> None:
    """Streams drawn white, satellites + ROI drawn yellow, coord order preserved.

    Uses a 512x512 canvas so the legend backdrop (semi-transparent dark fill
    at the top-left) does not overlap with the contour-pixel probe regions
    in the lower-right and the ROI bottom edge.
    """
    h = w = 512
    gt_map = np.zeros((h, w), dtype=np.int32)
    # id=1 streams upper-right (outside legend zone which starts at x=10,y=18).
    gt_map[40:80, 400:460] = 1
    # id=2 satellites bottom-right.
    gt_map[380:430, 380:470] = 2
    gt_type_of_id = {1: "streams", 2: "satellites"}

    # Deliberately non-square ROI with distinct y0, x0, y1, x1 values to
    # catch any (y, x) transposition bug.
    roi_bbox = (200, 100, 470, 500)  # (y0, x0, y1, x1)

    image = _solid_grey_canvas(h, w, value=80)
    out_path = tmp_path / "gt_contour.png"
    save_gt_contour_only_overlay(
        out_path,
        image,
        gt_map,
        sample_label="test",
        gt_type_of_id=gt_type_of_id,
        roi_bbox=roi_bbox,
    )

    rendered = np.array(Image.open(out_path).convert("RGB"))
    assert rendered.shape == image.shape

    # Streams contour in the upper-right -> white.
    stream_strip = rendered[40:80, 400:460]
    assert _has_near_colour(stream_strip, (255, 255, 255), tol=5), (
        "streams contour should be white in its bounding-box region"
    )

    # Satellites contour in the bottom-right -> yellow.
    sat_strip = rendered[380:430, 380:470]
    assert _has_near_colour(sat_strip, (255, 255, 0), tol=5), (
        "satellites contour should be yellow in its bounding-box region"
    )

    # ROI bottom edge at y=470. Probe a strip well away from the satellite
    # contour (cols < 380 or cols > 470) so a positive match must come from
    # the ROI rectangle itself, not the sat contour.
    bottom_edge_left = rendered[469:472, 100:370]
    assert _has_near_colour(bottom_edge_left, (255, 255, 0), tol=5), (
        "ROI bottom edge (y=470) should be yellow for x in [100, 370]"
    )

    # Coordinate-order sanity: ROI left edge is at x=100, not x=0.
    # Probe x in [0, 95] (well clear of the thickness=2 contour at x=100)
    # and y in [220, 450] (below the legend zone, above the bottom edge).
    # If someone swapped y/x order, the rectangle would be drawn as
    # cv2.rectangle((200, 100), (470, 500)) vs the correct
    # (100, 200)-(500, 470), leaking colour into x<100.
    left_of_roi = rendered[220:451, 0:96]
    original_left = image[220:451, 0:96]
    assert np.array_equal(left_of_roi, original_left), (
        "ROI must not leak to x<96 (coord order: (y0, x0, y1, x1))"
    )

    # Swap-safety: ROI top edge sits at y=200. A y0<->x0 swap would put
    # the top edge at y=100 and colour rendered[96:99, 220:450].
    above_roi_strip = rendered[96:99, 220:451]
    original_above = image[96:99, 220:451]
    assert np.array_equal(above_roi_strip, original_above), (
        "ROI top edge must be at y=200, not y=100 (swap-safety check)"
    )


def test_save_eval_prediction_overlay_labels_and_palette_deterministic(
    tmp_path,
) -> None:
    """Pred overlay draws GT + preds with deterministic palette + legend.

    Uses a 512x512 canvas so contour probe regions sit outside the
    top-left legend backdrop.
    """
    h = w = 512
    gt_map = np.zeros((h, w), dtype=np.int32)
    gt_map[40:80, 400:460] = 1
    gt_map[380:430, 380:470] = 2
    gt_type_of_id = {1: "streams", 2: "satellites"}

    pred_stream_seg = np.zeros((h, w), dtype=bool)
    pred_stream_seg[50:70, 410:450] = True
    pred_sat_seg = np.zeros((h, w), dtype=bool)
    pred_sat_seg[390:420, 390:460] = True

    predictions = [
        {
            "segmentation": pred_stream_seg,
            "type_label": "streams",
            "score": 0.81,
        },
        {
            "segmentation": pred_sat_seg,
            "type_label": "satellites",
            "score": 0.42,
        },
    ]

    image = _solid_grey_canvas(h, w, value=80)
    out_a = tmp_path / "pred_a.png"
    out_b = tmp_path / "pred_b.png"

    save_eval_prediction_overlay(
        out_a,
        image,
        gt_map,
        gt_type_of_id,
        predictions,
        roi_bbox=None,
        layer_label="post_gt_aware",
        sample_label="test",
    )
    save_eval_prediction_overlay(
        out_b,
        image,
        gt_map,
        gt_type_of_id,
        predictions,
        roi_bbox=None,
        layer_label="post_gt_aware",
        sample_label="test",
    )

    assert out_a.exists()
    assert out_b.exists()

    # Palette determinism: identical inputs -> identical PNG bytes.
    assert out_a.read_bytes() == out_b.read_bytes(), (
        "save_eval_prediction_overlay must be deterministic under fixed seed"
    )

    rendered = np.array(Image.open(out_a).convert("RGB"))
    assert rendered.shape == image.shape

    # GT streams still white in the upper-right contour region.
    assert _has_near_colour(rendered[40:80, 400:460], (255, 255, 255), tol=5), (
        "GT streams contour should be white"
    )

    # Legend was drawn (top-left region changed from flat grey).
    assert np.any(rendered[10:80, 10:200] != image[10:80, 10:200]), (
        "legend backdrop should visibly alter the top-left region"
    )

    # Satellites GT bounding box must contain yellow (GT contour) and the
    # warm-palette prediction contour somewhere in the region.
    sat_region = rendered[380:430, 380:470]
    assert _has_near_colour(sat_region, (255, 255, 0), tol=5), (
        "GT satellite contour should be yellow"
    )
    warm = (
        (sat_region[..., 0] >= 200)
        & (sat_region[..., 1] >= 80)
        & (sat_region[..., 1] <= 200)
        & (sat_region[..., 2] <= 120)
    )
    assert bool(warm.any()), "warm-palette satellite prediction contour missing"


def test_write_sample_overlays_gates_post_gt_aware(tmp_path) -> None:
    """`post_gt_aware_overlay.png` written iff ga_pair is not None."""
    # Local import so the test does not force a module-level dependency on
    # the eval script (which pulls in SAM3 runner etc).
    from scripts.eval.evaluate_checkpoint import _write_sample_overlays

    @dataclass
    class _FakeSample:
        base_key: str = "00000_eo"
        gt_instance_map_1024: np.ndarray = field(
            default_factory=lambda: np.zeros((32, 32), dtype=np.int32)
        )
        gt_type_of_id: dict = field(default_factory=dict)
        roi_bbox_1024: Optional[tuple] = None

    sample = _FakeSample()
    # A single streams GT instance so the GT contour path exercises something.
    sample.gt_instance_map_1024[6:14, 6:20] = 1
    sample.gt_type_of_id = {1: "streams"}

    render_rgb = _solid_grey_canvas(32, 32, value=90)
    raw_masks: list[dict] = []
    po_pair: tuple[list[dict], list[dict]] = ([], [])

    # Case 1: ga_pair is None -> no post_gt_aware overlay.
    sample_dir_a = tmp_path / "no_gt_aware"
    sample_dir_a.mkdir()
    _write_sample_overlays(
        sample_dir_a,
        sample,
        render_rgb,
        raw_masks,
        po_pair,
        ga_pair=None,
        sample_label="test",
    )
    overlays_a = sample_dir_a / "overlays"
    assert (overlays_a / "gt_contour.png").exists()
    assert (overlays_a / "raw_overlay.png").exists()
    assert (overlays_a / "post_pred_only_overlay.png").exists()
    assert not (overlays_a / "post_gt_aware_overlay.png").exists(), (
        "post_gt_aware_overlay.png must NOT be written when ga_pair is None"
    )

    # Case 2: ga_pair is a tuple -> file is written.
    sample_dir_b = tmp_path / "with_gt_aware"
    sample_dir_b.mkdir()
    _write_sample_overlays(
        sample_dir_b,
        sample,
        render_rgb,
        raw_masks,
        po_pair,
        ga_pair=([], []),
        sample_label="test",
    )
    overlays_b = sample_dir_b / "overlays"
    assert (overlays_b / "post_gt_aware_overlay.png").exists(), (
        "post_gt_aware_overlay.png must be written when ga_pair is a tuple"
    )
