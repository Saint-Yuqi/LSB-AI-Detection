"""
Checkpoint evaluation core — single SAM3-only, 1024-grid module for
benchmarking a checkpoint against three benchmarks:

    fbox_gold_satellites   satellites only, ROI-restricted (470x470 at 1024)
    firebox_dr1_streams    streams only, full-frame, SBlim31.5
    gt_canonical           streams + satellites, full-frame, post-retrain

Working grid is fixed at 1024. Native 2051x2051 benchmarks (fbox, DR1)
are downsampled once at load time and guarded by a positive-ID-set
preservation check; gt_canonical is already 1024 and passes through.

Three report layers per sample:
    raw              untouched SAM3 predictions
    post_pred_only   pure prediction post (streams_sanity, score_gate,
                     prior_filter, core_policy, cross_type_conflict);
                     applied to ALL benchmarks
    post_gt_aware    pred_only + GT-stream conflict resolver;
                     populated ONLY for gt_canonical

All filter stages are gated by enable_* flags in the config.

Public API:
    load_benchmark(cfg) -> list[Sample]
    run_sam3_on_sample(runner, sample, prompts) -> list[mask_dict]
    apply_post_pred_only(stream_masks, sat_masks, H, W, cfg_pred_only)
    apply_satellite_post_with_trace(sat_masks, H, W, cfg_pred_only)
    apply_post_gt_aware(stream_masks, sat_masks, streams_gt_map, cfg_gt_aware)
    compute_sample_report(sample, raw_masks, post_pred_only_masks,
                          post_gt_aware_masks_or_none, cfg) -> dict
    aggregate(reports) -> dict
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from astropy.io import fits
from PIL import Image

from src.analysis.mask_metrics import append_metrics_to_masks
from src.evaluation.metrics import (
    calculate_optimal_instance_metrics,
    calculate_optimal_instance_metrics_rle,
    calculate_pixel_metrics,
    rasterize_per_class_rles,
)
from src.pipelines.unified_dataset.taxonomy import (
    INNER_GALAXY,
    SATELLITES,
    TIDAL_FEATURES,
    normalize_type_label,
)
from src.utils.coco_utils import mask_to_rle
from src.evaluation.satellite_diagnostics import (
    MATCHED_LABELS,
    TAXONOMY_LABELS,
    DiagnosticCfg,
    SatelliteDiagnosticReport,
    TaxonomyEntry,
    build_candidate_table,
    classify_candidates,
    matched_unmatched_counts,
)
from src.postprocess.satellite_conflict_resolver import SatelliteConflictResolver
from src.postprocess.satellite_core_policy import SatelliteCorePolicy
from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
from src.postprocess.satellite_score_gate import SatelliteScoreGate
from src.postprocess.stream_satellite_conflict_filter import StreamSatelliteConflictFilter
from src.postprocess.streams_sanity_filter import StreamsSanityFilter, load_streams_cfg

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# =========================================================================== #
#  Frozen constants (§1, §3b of the eval plan)
# =========================================================================== #

#: Working grid resolution for ALL evaluation steps. 2051 is rejected.
TARGET_H: int = 1024
TARGET_W: int = 1024

#: Fbox_Gold_Satellites ROI on the 1024 grid (half-open [y0:y1, x0:x1]).
#: Derived from roi_definition.json (468.75 px half-width at 2051) scaled by
#: 1024/2051 -> center (511.75, 511.75), half 234.32 -> [277:747, 277:747].
FBOX_ROI_1024: tuple[int, int, int, int] = (277, 277, 747, 747)

#: ROI instance-membership rule.
ROI_RULE: str = "mask_centroid_in_roi"

#: Mask filename regex for FIREbox-DR1 SBlim31.5 streams.
_DR1_MASK_REGEX = re.compile(
    r"ark_features-(\d+)-(eo|fo)-SBlim31\.5\.fits\.gz"
)


# =========================================================================== #
#  Sample dataclass
# =========================================================================== #


@dataclass
class Sample:
    """One evaluation sample on the 1024 working grid.

    Two metric paths share this dataclass:

    - ``gt_path_version="legacy"`` (default): ``gt_instance_map_1024`` is the
      packed int32 GT loaded from ``instance_map_uint8.png``. Existing
      taxonomy + pixel + Hungarian raster paths consume it.

    - ``gt_path_version="tidal_v1"``: ``gt_rles_by_type`` carries per-class
      lists of COCO RLE dicts (preserves within-class overlap).
      ``gt_instance_map_1024`` is a zeros placeholder so legacy fields
      that always assumed an int32 array do not raise on attribute access.
    """

    base_key: str
    galaxy_id: int
    view: str
    benchmark_mode: str  # fbox_gold_satellites | firebox_dr1_streams | gt_canonical
    render_1024_path: Path
    gt_instance_map_1024: np.ndarray  # (H, W) int32, 0=bg (zeros on tidal_v1)
    gt_type_of_id: dict[int, str]      # {instance_id: type_label}
    roi_bbox_1024: Optional[tuple[int, int, int, int]]  # (y0, x0, y1, x1) or None
    dropped_instance_ids: list[int] = field(default_factory=list)
    # F11: per-class RLE GT for the new (tidal_v1) path. None when legacy.
    gt_rles_by_type: Optional[dict[str, list[dict[str, Any]]]] = None
    gt_path_version: str = "legacy"


# =========================================================================== #
#  Benchmark loaders
# =========================================================================== #


def _resolve(p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _downsample_instance_map(
    gt_native: np.ndarray,
    base_key: str,
    allow_instance_drop: bool,
) -> tuple[np.ndarray, list[int]]:
    """Resize an integer instance map from native (2051) to 1024x1024 using
    nearest neighbour, then compare positive ID sets and either fail fast or
    record dropped IDs.
    """
    ids_native = set(int(x) for x in np.unique(gt_native) if x != 0)
    gt_1024 = cv2.resize(
        gt_native.astype(np.int32),
        (TARGET_W, TARGET_H),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.int32)
    ids_1024 = set(int(x) for x in np.unique(gt_1024) if x != 0)
    dropped = sorted(ids_native - ids_1024)
    if dropped and not allow_instance_drop:
        raise RuntimeError(
            f"[{base_key}] positive-ID-set preservation guard tripped: "
            f"{len(dropped)} instance(s) lost during 2051->1024 downsample "
            f"(ids={dropped}). Set benchmark.allow_instance_drop=true to proceed."
        )
    if dropped:
        logger.warning(
            "[%s] %d instance(s) lost in 2051->1024 downsample (ids=%s); "
            "allow_instance_drop=true — continuing.",
            base_key, len(dropped), dropped,
        )
    return gt_1024, dropped


def _load_fits_int_map(path: Path) -> np.ndarray:
    """Load a gzipped FITS as an int32 instance map (preserve IDs)."""
    with fits.open(str(path)) as hdul:
        data = hdul[0].data
    if data is None:
        raise ValueError(f"Empty FITS data in {path}")
    # Astropy loads gzipped FITS transparently via filename.
    return np.asarray(data, dtype=np.int32)


def _render_path_for_canonical(
    gt_dir_root: Path, base_key: str, cfg_render: dict[str, Any]
) -> Path:
    """Resolve the gt_canonical render path: data/02_processed/renders/{condition}/{variant}/{base_key}/0000.png."""
    condition = cfg_render.get("condition", "current")
    variant = cfg_render.get("variant", "linear_magnitude")
    # gt_canonical reuses data/02_processed/renders/ NOT renders_eval
    renders_root = PROJECT_ROOT / "data" / "02_processed" / "renders"
    return renders_root / condition / variant / base_key / "0000.png"


def _render_path_for_eval_benchmark(
    benchmark: str, base_key: str, cfg_render: dict[str, Any]
) -> Path:
    """Resolve a fbox/dr1 render path under data/02_processed/renders_eval/{benchmark}/{condition}/{variant}/[profile/]{base_key}/0000.png."""
    condition = cfg_render.get("condition", "current")
    variant = cfg_render.get("variant", "linear_magnitude")
    noise_profile = cfg_render.get("noise_profile")
    root = _resolve(cfg_render.get("root", "data/02_processed/renders_eval"))
    if condition == "noisy":
        if not noise_profile:
            raise ValueError(
                "render.noise_profile is required when render.condition=='noisy'"
            )
        path = root / benchmark / "noisy" / variant / noise_profile / base_key / "0000.png"
    else:
        path = root / benchmark / "current" / variant / base_key / "0000.png"
    return path  # existence checked lazily when the sample is opened


def _load_fbox_gold_satellites(cfg: dict[str, Any]) -> list[Sample]:
    fbox_cfg = cfg["benchmark"]["fbox"]
    manifest_path = _resolve(fbox_cfg["manifest"])
    masks_root = _resolve(fbox_cfg["masks_root"])
    allow_drop = bool(cfg["benchmark"].get("allow_instance_drop", False))
    render_cfg = cfg["render"]

    with open(manifest_path) as f:
        manifest = json.load(f)

    samples: list[Sample] = []
    for entry in manifest["samples"]:
        base_key = entry["sample_id"]
        sample_dir = masks_root / base_key
        instance_map_npy = sample_dir / "instance_map.npy"
        instances_json = sample_dir / "instances.json"

        if not instance_map_npy.exists() or not instances_json.exists():
            raise FileNotFoundError(
                f"fbox_gold_satellites GT missing for {base_key}: "
                f"{instance_map_npy} or {instances_json}"
            )

        gt_native = np.load(instance_map_npy).astype(np.int32)
        gt_1024, dropped = _downsample_instance_map(gt_native, base_key, allow_drop)

        with open(instances_json) as f:
            instances = json.load(f)
        gt_type_of_id = {int(inst["id"]): inst["type"] for inst in instances}

        render_path = _render_path_for_eval_benchmark(
            "fbox_gold_satellites", base_key, render_cfg
        )
        if not render_path.exists():
            logger.warning(
                "fbox_gold_satellites: skipping %s — render not found: %s",
                base_key, render_path,
            )
            continue

        samples.append(Sample(
            base_key=base_key,
            galaxy_id=int(entry["galaxy_id"]),
            view=entry["view"],
            benchmark_mode="fbox_gold_satellites",
            render_1024_path=render_path,
            gt_instance_map_1024=gt_1024,
            gt_type_of_id=gt_type_of_id,
            roi_bbox_1024=FBOX_ROI_1024,
            dropped_instance_ids=dropped,
        ))
    return samples


def _load_firebox_dr1_streams(cfg: dict[str, Any]) -> list[Sample]:
    dr1_cfg = cfg["benchmark"]["dr1"]
    root = _resolve(dr1_cfg["root"])
    allow_drop = bool(cfg["benchmark"].get("allow_instance_drop", False))
    render_cfg = cfg["render"]

    # Glob mask files directly per plan §3b.
    mask_paths: list[tuple[int, str, Path]] = []
    for view_dir in ("MASKS_EO", "MASKS_FO"):
        for path in sorted((root / view_dir).glob("ark_features-*-SBlim31.5.fits.gz")):
            m = _DR1_MASK_REGEX.match(path.name)
            if not m:
                continue
            mask_paths.append((int(m.group(1)), m.group(2), path))
    if not mask_paths:
        raise FileNotFoundError(
            f"No DR1 SBlim31.5 masks found under {root}/MASKS_{{EO,FO}}/"
        )

    samples: list[Sample] = []
    for gid, view, mask_path in mask_paths:
        base_key = f"{gid:05d}_{view}"
        gt_native = _load_fits_int_map(mask_path)
        gt_1024, dropped = _downsample_instance_map(gt_native, base_key, allow_drop)

        pos_ids = [int(x) for x in np.unique(gt_1024) if x != 0]
        gt_type_of_id = {pid: "streams" for pid in pos_ids}

        render_path = _render_path_for_eval_benchmark(
            "firebox_dr1_streams", base_key, render_cfg
        )
        if not render_path.exists():
            logger.warning(
                "firebox_dr1_streams: skipping %s — render not found: %s",
                base_key, render_path,
            )
            continue

        samples.append(Sample(
            base_key=base_key,
            galaxy_id=gid,
            view=view,
            benchmark_mode="firebox_dr1_streams",
            render_1024_path=render_path,
            gt_instance_map_1024=gt_1024,
            gt_type_of_id=gt_type_of_id,
            roi_bbox_1024=None,
            dropped_instance_ids=dropped,
        ))
    return samples


def _load_gt_canonical(cfg: dict[str, Any]) -> list[Sample]:
    """Load `gt_canonical` samples.

    Two on-disk shapes are supported:

    - **Legacy** (``instance_map_uint8.png`` + ``instances.json``):
      packed int32 raster, 2-class types ``streams``/``satellites``.
      Eval reads exactly these files, exactly as before (F12).

    - **New tidal_v1** (per-class ``*_instance_map.npy`` +
      ``instances.json`` + ``sam3_predictions_post.json``): per-class
      RLE lists are constructed and stored on ``Sample.gt_rles_by_type``;
      ``Sample.gt_path_version`` is ``"tidal_v1"`` and downstream
      ``_typed_blocks`` branches accordingly. ``instance_map_uint8.png``
      is NOT read on this path.
    """
    canonical_cfg = cfg["benchmark"]["canonical"]
    gt_dir_root = _resolve(canonical_cfg["gt_dir"])
    render_cfg = cfg["render"]

    key_re = re.compile(r"^(\d+)_([^_]+)$")
    samples: list[Sample] = []

    for subdir in sorted(gt_dir_root.iterdir()):
        if not subdir.is_dir():
            continue
        m = key_re.match(subdir.name)
        if not m:
            continue

        instances_json = subdir / "instances.json"
        if not instances_json.exists():
            logger.warning(
                "gt_canonical: skipping %s — missing instances.json", subdir.name,
            )
            continue

        new_path_marker = subdir / "tidal_features_instance_map.npy"
        on_new_path = new_path_marker.exists()

        if on_new_path:
            sample = _build_tidal_v1_sample(subdir, m, render_cfg, gt_dir_root)
            if sample is not None:
                samples.append(sample)
            continue

        # ---- Legacy GT path (UNCHANGED; reads instance_map_uint8.png) ----
        gt_map_path = subdir / "instance_map_uint8.png"
        if not gt_map_path.exists():
            logger.warning(
                "gt_canonical: skipping %s — missing instance_map_uint8.png "
                "(legacy path)", subdir.name,
            )
            continue

        gt_map = np.array(Image.open(gt_map_path)).astype(np.int32)
        if gt_map.shape != (TARGET_H, TARGET_W):
            raise ValueError(
                f"gt_canonical/{subdir.name}: expected {TARGET_H}x{TARGET_W}, "
                f"got {gt_map.shape}"
            )
        with open(instances_json) as f:
            instances = json.load(f)
        gt_type_of_id = {int(inst["id"]): inst["type"] for inst in instances}

        map_ids = set(int(x) for x in np.unique(gt_map) if x != 0)
        json_ids = set(gt_type_of_id.keys())
        if map_ids != json_ids:
            logger.warning(
                "[%s] gt_canonical ID mismatch: map_only=%s json_only=%s",
                subdir.name, map_ids - json_ids, json_ids - map_ids,
            )
            gt_type_of_id = {i: t for i, t in gt_type_of_id.items() if i in map_ids}

        render_path = _render_path_for_canonical(gt_dir_root, subdir.name, render_cfg)
        if not render_path.exists():
            logger.warning(
                "gt_canonical: skipping %s — render not found: %s",
                subdir.name, render_path,
            )
            continue

        samples.append(Sample(
            base_key=subdir.name,
            galaxy_id=int(m.group(1)),
            view=m.group(2),
            benchmark_mode="gt_canonical",
            render_1024_path=render_path,
            gt_instance_map_1024=gt_map,
            gt_type_of_id=gt_type_of_id,
            roi_bbox_1024=None,
            dropped_instance_ids=[],
            gt_rles_by_type=None,
            gt_path_version="legacy",
        ))
    return samples


def _build_tidal_v1_sample(
    subdir: Path,
    key_match: re.Match[str],
    render_cfg: dict[str, Any],
    gt_dir_root: Path,
) -> Optional[Sample]:
    """Construct a Sample for the new (tidal_v1) GT layout.

    Builds per-class RLE lists by:
    - decoding tidal_features rows from ``tidal_features_instance_map.npy``
      (FITS-derived; no within-class overlap to preserve).
    - pulling satellites + inner_galaxy RLEs from
      ``sam3_predictions_post.json`` indexed by ``raw_index`` (preserves
      within-class overlap).

    Fail-closed (F20): a row carrying ``source: "sam3_post"`` whose
    ``raw_index`` is missing from the predictions JSON raises ``KeyError``
    rather than silently falling back to a per-class map decode.
    """
    instances = json.loads((subdir / "instances.json").read_text())

    pred_path = subdir / "sam3_predictions_post.json"
    pred_index: dict[int, dict[str, Any]] = {}
    if pred_path.exists():
        pred_doc = json.loads(pred_path.read_text())
        for p in pred_doc.get("predictions", []):
            if "raw_index" in p and "rle" in p:
                pred_index[int(p["raw_index"])] = p["rle"]

    per_class_map_cache: dict[str, np.ndarray] = {}

    def _per_class_map(filename: str) -> np.ndarray:
        if filename not in per_class_map_cache:
            per_class_map_cache[filename] = np.load(subdir / filename).astype(np.int32)
        return per_class_map_cache[filename]

    gt_rles_by_type: dict[str, list[dict[str, Any]]] = {
        TIDAL_FEATURES: [],
        SATELLITES: [],
        INNER_GALAXY: [],
    }
    gt_type_of_id: dict[int, str] = {}

    H, W = TARGET_H, TARGET_W

    for inst in instances:
        type_label = normalize_type_label(inst.get("type_label") or inst["type"])
        raw_idx = inst.get("raw_index")
        source = inst.get("source")

        rle: Optional[dict[str, Any]] = None
        if raw_idx is not None and int(raw_idx) in pred_index:
            rle = pred_index[int(raw_idx)]
        elif source == "sam3_post":
            raise KeyError(
                f"[{subdir.name}] sam3_post row {inst.get('candidate_id')!r} "
                f"(raw_index={raw_idx}) missing from sam3_predictions_post.json"
            )
        elif "map_file" in inst and "local_id" in inst:
            m_map = _per_class_map(inst["map_file"])
            if m_map.shape != (H, W):
                raise ValueError(
                    f"[{subdir.name}] {inst['map_file']} has shape {m_map.shape}, "
                    f"expected {(H, W)}"
                )
            binary = (m_map == int(inst["local_id"])).astype(np.uint8)
            if binary.sum() == 0:
                continue
            rle = mask_to_rle(binary)
        else:
            logger.warning(
                "[%s] tidal_v1 row has no raw_index and no map_file/local_id; skipping",
                subdir.name,
            )
            continue

        gt_rles_by_type[type_label].append(rle)

        gid = int(inst.get("global_id", inst.get("id", 0)))
        if gid:
            gt_type_of_id[gid] = type_label

    render_path = _render_path_for_canonical(gt_dir_root, subdir.name, render_cfg)
    if not render_path.exists():
        logger.warning(
            "gt_canonical: skipping %s — render not found: %s",
            subdir.name, render_path,
        )
        return None

    # Placeholder int32 map for the legacy attribute access; real metric
    # paths on this Sample go through gt_rles_by_type.
    placeholder_map = np.zeros((H, W), dtype=np.int32)

    return Sample(
        base_key=subdir.name,
        galaxy_id=int(key_match.group(1)),
        view=key_match.group(2),
        benchmark_mode="gt_canonical",
        render_1024_path=render_path,
        gt_instance_map_1024=placeholder_map,
        gt_type_of_id=gt_type_of_id,
        roi_bbox_1024=None,
        dropped_instance_ids=[],
        gt_rles_by_type=gt_rles_by_type,
        gt_path_version="tidal_v1",
    )


def load_benchmark(cfg: dict[str, Any]) -> list[Sample]:
    """Dispatch to a concrete benchmark loader per cfg.benchmark.mode."""
    mode = cfg["benchmark"]["mode"]
    if tuple(cfg.get("target_size", [TARGET_H, TARGET_W])) != (TARGET_H, TARGET_W):
        raise ValueError(
            f"checkpoint_eval is 1024-only; got target_size={cfg.get('target_size')}"
        )
    loaders = {
        "fbox_gold_satellites": _load_fbox_gold_satellites,
        "firebox_dr1_streams": _load_firebox_dr1_streams,
        "gt_canonical": _load_gt_canonical,
    }
    if mode not in loaders:
        raise ValueError(
            f"Unknown benchmark mode: {mode!r}. Expected: {list(loaders)}"
        )
    return loaders[mode](cfg)


# =========================================================================== #
#  SAM3 inference (thin pass-through)
# =========================================================================== #


def run_sam3_on_sample(
    runner: Any, sample: Sample, prompts: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], float]:
    """Run SAM3 on sample.render_1024_path; return (masks, time_ms)."""
    image_pil = Image.open(sample.render_1024_path).convert("RGB")
    masks, time_ms = runner.run(image_pil, prompts)
    if masks:
        append_metrics_to_masks(masks, TARGET_H, TARGET_W, compute_hull=True)
    return masks, time_ms


# =========================================================================== #
#  Post-processing (dual layer)
# =========================================================================== #


def _build_streams_sanity(cfg_stage: dict[str, Any]) -> StreamsSanityFilter:
    stats_json = cfg_stage.get("stats_json")
    gt_defaults = load_streams_cfg(_resolve(stats_json) if stats_json else None)
    return StreamsSanityFilter(
        min_area=cfg_stage.get("min_area", gt_defaults["min_area"]),
        max_area_frac=cfg_stage.get("max_area_frac", 0.5),
        edge_touch_frac=cfg_stage.get("edge_touch_frac", gt_defaults["edge_touch_frac"]),
        max_area_px=cfg_stage.get("max_area_px", gt_defaults["max_area_px"]),
    )


def _build_prior_filter(cfg_stage: dict[str, Any], stats_json: Optional[str]) -> SatellitePriorFilter:
    if stats_json:
        defaults = load_filter_cfg(_resolve(stats_json))
    else:
        defaults = load_filter_cfg()
    defaults.update({k: v for k, v in cfg_stage.items() if v is not None})
    if "hard_center_radius_frac" in cfg_stage:
        defaults["hard_center_radius_frac"] = cfg_stage["hard_center_radius_frac"]
    return SatellitePriorFilter(defaults)


def _score_gate_keep(mask: dict[str, Any], gate: SatelliteScoreGate) -> bool:
    area = int(mask.get("area_clean", 0))
    score = float(mask.get("score", 0.0))
    decision, _ = gate.decide(area, score)
    return decision == "pass"


def _core_policy_keep(mask: dict[str, Any], core: SatelliteCorePolicy, H: int, W: int) -> bool:
    dist_px = mask.get("dist_to_center")
    dist_frac = float("inf") if dist_px is None else float(dist_px) / float(min(H, W))
    area = int(mask.get("area_clean", 0))
    score = float(mask.get("score", 0.0))
    solidity = float(mask.get("solidity") or 0.0)
    aspect = float(mask.get("aspect_sym_moment") or mask.get("aspect_sym") or 0.0)
    decision, _ = core.decide(
        dist_to_center_frac=dist_frac,
        area_clean_px=area,
        score=score,
        solidity=solidity,
        aspect_sym_moment=aspect,
    )
    return decision in {"pass", "rescue"}


def apply_post_pred_only(
    stream_masks: list[dict[str, Any]],
    sat_masks: list[dict[str, Any]],
    H: int,
    W: int,
    cfg_pred_only: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Five-stage prediction-only post (no GT touched).

    Stages:
        1. streams_sanity  (StreamsSanityFilter on streams)
        2. score_gate      (SatelliteScoreGate on sats)
        3. prior_filter    (SatellitePriorFilter on sats)
        4. core_policy     (SatelliteCorePolicy on sats)
        5. cross_type_conflict  (StreamSatelliteConflictFilter; may trim/drop sats AND drop streams)

    Each stage gated by enable_* flag. Short-circuit cross-type stage when
    either channel is empty.
    """
    streams = list(stream_masks)
    sats = list(sat_masks)

    # --- stage 1: streams_sanity ---
    if cfg_pred_only.get("enable_streams_sanity", True) and streams:
        flt = _build_streams_sanity(cfg_pred_only.get("streams_sanity", {}))
        streams, _rej = flt.filter(streams, H, W)

    # --- stage 2: score_gate ---
    if cfg_pred_only.get("enable_score_gate", True) and sats:
        gate = SatelliteScoreGate(**cfg_pred_only.get("score_gate", {}))
        sats = [m for m in sats if _score_gate_keep(m, gate)]

    # --- stage 3: prior_filter ---
    if cfg_pred_only.get("enable_prior_filter", True) and sats:
        prior_cfg = cfg_pred_only.get("prior_filter", {}) or {}
        stats_json = prior_cfg.get("stats_json")
        prior = _build_prior_filter(prior_cfg, stats_json)
        kept, _rej, _ambig = prior.filter(sats)
        sats = list(kept)

    # --- stage 4: core_policy ---
    if cfg_pred_only.get("enable_core_policy", True) and sats:
        core = SatelliteCorePolicy(**cfg_pred_only.get("core_policy", {}))
        sats = [m for m in sats if _core_policy_keep(m, core, H, W)]

    # --- stage 5: cross_type_conflict ---
    if (
        cfg_pred_only.get("enable_cross_type_conflict", True)
        and streams and sats
    ):
        cross_cfg = cfg_pred_only.get("cross_type_conflict", {}) or {}
        flt = StreamSatelliteConflictFilter(
            policy=cross_cfg.get("policy", "stream_first"),
            keep_stream_aspect_min=cross_cfg.get("keep_stream_aspect_min", 1.9),
            keep_stream_curvature_min=cross_cfg.get("keep_stream_curvature_min", 1.2),
            keep_stream_area_ratio=cross_cfg.get("keep_stream_area_ratio", 1.5),
            drop_compact_stream_overlap=cross_cfg.get("drop_compact_stream_overlap", 0.75),
            satellite_solidity_min=cross_cfg.get("satellite_solidity_min", 0.83),
            satellite_aspect_max=cross_cfg.get("satellite_aspect_max", 1.75),
        )
        # Re-filter hooks intentionally disabled: plan §1, stream_satellite_conflict_filter.py:148-161.
        result = flt.filter(
            streams, sats, H, W,
            streams_filter=None,
            satellite_prior_filter=None,
            satellite_core_filter=None,
        )
        streams = result["streams"]
        sats = result["satellites"]

    return streams, sats


# =========================================================================== #
#  Satellite-only post with per-candidate trace (fbox_gold_satellites)
# =========================================================================== #

#: Ordered stages traced by apply_satellite_post_with_trace.
SATELLITE_POST_STAGES: tuple[str, ...] = ("score_gate", "prior_filter", "core_policy")


def apply_satellite_post_with_trace(
    sat_masks: list[dict[str, Any]],
    H: int,
    W: int,
    cfg_pred_only: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Run satellite-only 3-stage post and return (kept_sats, trace_records).

    Re-implements score_gate → prior_filter → core_policy inline to capture
    per-candidate stage decisions.  Streams stages (streams_sanity,
    cross_type_conflict) are NOT executed — this helper is designed for
    fbox_gold_satellites where no streams exist.

    Each record in trace_records has:
        raw_index, candidate_id, candidate_rle_sha1,
        final_status   ("kept" | "removed"),
        first_drop_stage  (str | None),
        first_drop_reason (str | None),
        stage_results: [
            {"stage": str, "outcome": str, "reason": str | None}, ...
        ]
    """
    # Pre-build filter instances once.
    enable_score_gate = cfg_pred_only.get("enable_score_gate", True)
    enable_prior_filter = cfg_pred_only.get("enable_prior_filter", True)
    enable_core_policy = cfg_pred_only.get("enable_core_policy", True)

    gate = (
        SatelliteScoreGate(**cfg_pred_only.get("score_gate", {}))
        if enable_score_gate else None
    )
    prior: SatellitePriorFilter | None = None
    if enable_prior_filter:
        prior_cfg = cfg_pred_only.get("prior_filter", {}) or {}
        stats_json = prior_cfg.get("stats_json")
        prior = _build_prior_filter(prior_cfg, stats_json)
    core = (
        SatelliteCorePolicy(**cfg_pred_only.get("core_policy", {}))
        if enable_core_policy else None
    )

    trace_records: list[dict[str, Any]] = []
    kept_sats: list[dict[str, Any]] = []

    for mask in sat_masks:
        raw_index = mask.get("raw_index")
        candidate_id = mask.get("candidate_id")
        candidate_rle_sha1 = mask.get("candidate_rle_sha1")

        stage_results: list[dict[str, Any]] = []
        dropped = False
        first_drop_stage: str | None = None
        first_drop_reason: str | None = None

        # --- stage 1: score_gate ---
        if dropped:
            stage_results.append({"stage": "score_gate", "outcome": "not_reached", "reason": None})
        elif not enable_score_gate:
            stage_results.append({"stage": "score_gate", "outcome": "disabled", "reason": None})
        else:
            area = int(mask.get("area_clean", 0))
            score = float(mask.get("score", 0.0))
            decision, reason = gate.decide(area, score)
            if decision == "pass":
                stage_results.append({"stage": "score_gate", "outcome": "pass", "reason": reason})
            else:
                stage_results.append({"stage": "score_gate", "outcome": "drop", "reason": reason})
                dropped = True
                first_drop_stage = "score_gate"
                first_drop_reason = reason

        # --- stage 2: prior_filter ---
        if dropped:
            stage_results.append({"stage": "prior_filter", "outcome": "not_reached", "reason": None})
        elif not enable_prior_filter:
            stage_results.append({"stage": "prior_filter", "outcome": "disabled", "reason": None})
        else:
            decision, reason = prior.decide(mask)
            if decision == "pass":
                stage_results.append({"stage": "prior_filter", "outcome": "pass", "reason": reason})
            else:
                stage_results.append({"stage": "prior_filter", "outcome": "drop", "reason": reason})
                dropped = True
                first_drop_stage = "prior_filter"
                first_drop_reason = reason

        # --- stage 3: core_policy ---
        if dropped:
            stage_results.append({"stage": "core_policy", "outcome": "not_reached", "reason": None})
        elif not enable_core_policy:
            stage_results.append({"stage": "core_policy", "outcome": "disabled", "reason": None})
        else:
            dist_px = mask.get("dist_to_center")
            dist_frac = float("inf") if dist_px is None else float(dist_px) / float(min(H, W))
            area = int(mask.get("area_clean", 0))
            score = float(mask.get("score", 0.0))
            solidity = float(mask.get("solidity") or 0.0)
            aspect = float(mask.get("aspect_sym_moment") or mask.get("aspect_sym") or 0.0)
            decision, reason = core.decide(
                dist_to_center_frac=dist_frac,
                area_clean_px=area,
                score=score,
                solidity=solidity,
                aspect_sym_moment=aspect,
            )
            if decision == "pass":
                stage_results.append({"stage": "core_policy", "outcome": "pass", "reason": reason})
            elif decision == "rescue":
                stage_results.append({"stage": "core_policy", "outcome": "rescue", "reason": reason})
            else:
                stage_results.append({"stage": "core_policy", "outcome": "drop", "reason": reason})
                dropped = True
                first_drop_stage = "core_policy"
                first_drop_reason = reason

        final_status = "removed" if dropped else "kept"
        if not dropped:
            kept_sats.append(mask)

        trace_records.append({
            "raw_index": raw_index,
            "candidate_id": candidate_id,
            "candidate_rle_sha1": candidate_rle_sha1,
            "final_status": final_status,
            "first_drop_stage": first_drop_stage,
            "first_drop_reason": first_drop_reason,
            "stage_results": stage_results,
        })

    return kept_sats, trace_records


def apply_post_gt_aware(
    stream_masks: list[dict[str, Any]],
    sat_masks: list[dict[str, Any]],
    streams_gt_map: np.ndarray,
    H: int,
    W: int,
    cfg_gt_aware: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """GT-aware layer: start from post_pred_only output and run the
    GT-stream conflict resolver on satellites only. Streams are passed
    through unchanged (the conflict-resolver only rules on sats vs GT).
    """
    if not cfg_gt_aware.get("enable_gt_stream_conflict", True):
        return list(stream_masks), list(sat_masks)

    resolver = SatelliteConflictResolver(**cfg_gt_aware.get("conflict_policy", {}))
    kept_sats: list[dict[str, Any]] = []
    for mask in sat_masks:
        seg = mask.get("segmentation")
        if seg is None:
            continue
        matched_id, _overlap_px, ratio_sat, _ratio_stream = resolver.match_stream(
            seg, streams_gt_map
        )
        area = int(mask.get("area_clean", 0))
        solidity = float(mask.get("solidity") or 0.0)
        aspect = float(mask.get("aspect_sym_moment") or mask.get("aspect_sym") or 0.0)
        decision, _reason, _extras = resolver.decide(
            matched_stream_id=matched_id,
            overlap_ratio_satellite=ratio_sat,
            area_clean_px=area,
            solidity=solidity,
            aspect_sym_moment=aspect,
        )
        # decide() returns 'pass' | 'win' | 'drop'. Keep everything except 'drop'.
        if decision != "drop":
            kept_sats.append(mask)
    return list(stream_masks), kept_sats


# =========================================================================== #
#  Matching + per-sample report
# =========================================================================== #


def _filter_gt_by_type(
    gt_map: np.ndarray, gt_type_of_id: dict[int, str], want_type: str
) -> np.ndarray:
    """Return a (H, W) int32 map with only instances of the requested type."""
    keep_ids = [i for i, t in gt_type_of_id.items() if t == want_type]
    if not keep_ids:
        return np.zeros_like(gt_map, dtype=np.int32)
    mask = np.isin(gt_map, keep_ids)
    return np.where(mask, gt_map, 0).astype(np.int32)


def _instance_centroid_in_roi(seg: np.ndarray, roi: tuple[int, int, int, int]) -> bool:
    ys, xs = np.nonzero(seg)
    if len(xs) == 0:
        return False
    cy = float(ys.mean())
    cx = float(xs.mean())
    y0, x0, y1, x1 = roi
    return bool(y0 <= cy < y1 and x0 <= cx < x1)


def _compute_slice_block(
    pred_masks: list[dict[str, Any]],
    gt_type_map: np.ndarray,
    iou_thresh: float,
) -> dict[str, Any]:
    """Compute the full per-slice block (detection, pixel, per-instance).

    Applies the frozen empty-sample conventions for detection metrics.
    """
    gt_ids = np.unique(gt_type_map)
    gt_ids = [int(g) for g in gt_ids if g != 0]
    num_gt = len(gt_ids)
    num_pred = len(pred_masks)

    # --- instance matching (Hungarian) ---
    inst = calculate_optimal_instance_metrics(pred_masks, gt_type_map, iou_thresh)

    # --- pixel metrics ---
    gt_bin = gt_type_map > 0
    if pred_masks:
        pred_bin = np.zeros_like(gt_bin, dtype=bool)
        for m in pred_masks:
            pred_bin |= m["segmentation"].astype(bool)
    else:
        pred_bin = np.zeros_like(gt_bin, dtype=bool)
    pixel = calculate_pixel_metrics(pred_bin, gt_bin)

    # --- detection metrics (empty-sample conventions) ---
    tp = int(inst["num_detected"])
    fp = num_pred - tp
    fn = num_gt - tp
    if num_gt == 0 and num_pred == 0:
        precision: Optional[float] = 1.0
        recall: Optional[float] = 1.0
        f1: Optional[float] = 1.0
        is_empty_trivial = True
    elif num_gt == 0 and num_pred > 0:
        precision, recall, f1 = 0.0, None, None
        is_empty_trivial = False
    elif num_gt > 0 and num_pred == 0:
        precision, recall, f1 = None, 0.0, None
        is_empty_trivial = False
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else 0.0
        )
        is_empty_trivial = False

    # matched IoU mean / median (from per-instance details)
    matched = [d for d in inst["per_instance_details"] if d["detected"]]
    matched_ious = [d["iou"] for d in matched]
    matched_iou_mean = float(np.mean(matched_ious)) if matched_ious else None
    matched_iou_median = float(np.median(matched_ious)) if matched_ious else None

    return {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detection": {"precision": precision, "recall": recall, "f1": f1},
        "matched_iou_mean": matched_iou_mean,
        "matched_iou_median": matched_iou_median,
        "pixel": {
            "dice": pixel["dice"],
            "precision": pixel["precision"],
            "recall": pixel["recall"],
            "capped_hausdorff95": pixel["capped_hausdorff95"],
        },
        "per_instance": [
            {
                "gt_id": d["gt_instance_id"],
                "matched_pred_idx": d["matched_pred_idx"],
                "iou": d["iou"],
                "detected": d["detected"],
                "gt_area": d["gt_area"],
            }
            for d in inst["per_instance_details"]
        ],
        "is_empty_trivial": is_empty_trivial,
    }


def _restrict_to_roi(
    pred_masks: list[dict[str, Any]],
    gt_type_map: np.ndarray,
    roi: tuple[int, int, int, int],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Keep only masks/GT whose mask centroid falls inside the ROI."""
    pred_in_roi = [
        m for m in pred_masks
        if _instance_centroid_in_roi(m["segmentation"], roi)
    ]

    gt_ids = np.unique(gt_type_map)
    gt_ids = [int(g) for g in gt_ids if g != 0]
    keep = [
        g for g in gt_ids
        if _instance_centroid_in_roi(gt_type_map == g, roi)
    ]
    if keep:
        roi_gt = np.where(np.isin(gt_type_map, keep), gt_type_map, 0).astype(np.int32)
    else:
        roi_gt = np.zeros_like(gt_type_map, dtype=np.int32)
    return pred_in_roi, roi_gt


def _restrict_gt_map_to_roi(
    gt_type_map: np.ndarray, roi: tuple[int, int, int, int]
) -> np.ndarray:
    """Return a (H, W) int32 map containing only GT instances whose centroid
    lies inside ``roi``. IDs outside the ROI are zeroed.
    """
    gt_ids = [int(g) for g in np.unique(gt_type_map) if g != 0]
    keep = [g for g in gt_ids if _instance_centroid_in_roi(gt_type_map == g, roi)]
    if not keep:
        return np.zeros_like(gt_type_map, dtype=np.int32)
    return np.where(np.isin(gt_type_map, keep), gt_type_map, 0).astype(np.int32)


def _pixel_block(
    sat_masks: list[dict[str, Any]], gt_type_map: np.ndarray
) -> dict[str, Optional[float]]:
    """OR-of-mask pixel metrics against a binary GT, shape unchanged."""
    gt_bin = gt_type_map > 0
    if sat_masks:
        pred_bin = np.zeros_like(gt_bin, dtype=bool)
        for m in sat_masks:
            pred_bin |= m["segmentation"].astype(bool)
    else:
        pred_bin = np.zeros_like(gt_bin, dtype=bool)
    pixel = calculate_pixel_metrics(pred_bin, gt_bin)
    return {
        "dice": pixel["dice"],
        "precision": pixel["precision"],
        "recall": pixel["recall"],
        "capped_hausdorff95": pixel["capped_hausdorff95"],
    }


def _empty_taxonomy_counts() -> dict[str, int]:
    return {label: 0 for label in TAXONOMY_LABELS}


def _taxonomy_block_from_entries(
    scope_sats: list[dict[str, Any]],
    scope_entries: list[TaxonomyEntry],
    scope_gt_map: np.ndarray,
) -> dict[str, Any]:
    """Build one satellite taxonomy block from pre-classified entries.

    Caller decides scope by pre-filtering ``scope_sats`` / ``scope_entries``
    and picking the correct ``scope_gt_map`` (full-frame or ROI-restricted).
    No rebucket on ``matched_gt_id`` — matched candidates outside the scope
    GT set still count as matched (mirrors diagnostics ``counts_by_label_roi``
    contract, see plan §"Deriving the official satellite block").
    """
    num_pred = len(scope_entries)
    num_gt = int(sum(1 for g in np.unique(scope_gt_map) if g != 0))

    counts_by_label = _empty_taxonomy_counts()
    matched_candidates = 0
    unmatched_candidates = 0
    covered_gt_ids: set[int] = set()
    for e in scope_entries:
        counts_by_label[e.taxonomy_label] += 1
        if e.taxonomy_label in MATCHED_LABELS:
            matched_candidates += 1
            if e.matched_gt_id is not None:
                covered_gt_ids.add(int(e.matched_gt_id))
        else:
            unmatched_candidates += 1
    unique_gt_covered = len(covered_gt_ids)
    gt_uncovered = max(0, num_gt - unique_gt_covered)

    # Empty-sample conventions (mirror _compute_slice_block).
    precision: Optional[float]
    recall: Optional[float]
    f1: Optional[float]
    if num_gt == 0 and num_pred == 0:
        precision, recall, f1 = 1.0, 1.0, 1.0
        is_empty_trivial = True
    elif num_gt == 0 and num_pred > 0:
        precision, recall, f1 = 0.0, None, None
        is_empty_trivial = False
    elif num_gt > 0 and num_pred == 0:
        precision, recall, f1 = None, 0.0, None
        is_empty_trivial = False
    else:
        precision = matched_candidates / num_pred
        recall = unique_gt_covered / num_gt
        denom = precision + recall
        f1 = (2 * precision * recall / denom) if denom > 0 else 0.0
        is_empty_trivial = False

    return {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "matched_candidates": matched_candidates,
        "unmatched_candidates": unmatched_candidates,
        "unique_gt_covered": unique_gt_covered,
        "gt_uncovered": gt_uncovered,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "counts_by_label": counts_by_label,
        "pixel": _pixel_block(scope_sats, scope_gt_map),
        "is_empty_trivial": is_empty_trivial,
    }


def _satellite_typed_entry(
    sat_masks: list[dict[str, Any]],
    entries: list[TaxonomyEntry],
    gt_sat_map_full: np.ndarray,
    roi_bbox: Optional[tuple[int, int, int, int]],
) -> dict[str, Optional[dict[str, Any]]]:
    """Derive ``{full_frame, roi}`` satellite taxonomy blocks from a single
    classification pass. ``roi`` is ``None`` when the benchmark has no ROI.

    The ROI slice is a pure subset of the full-frame entries (rows with
    ``intersects_roi=True``) plus a ROI-restricted GT map for ``num_gt`` and
    pixel metrics — no second matching pass, no rebucket on ``matched_gt_id``.
    """
    full_block = _taxonomy_block_from_entries(sat_masks, entries, gt_sat_map_full)
    if roi_bbox is None:
        return {"full_frame": full_block, "roi": None}

    roi_sats = [m for m, e in zip(sat_masks, entries) if e.intersects_roi]
    roi_entries = [e for e in entries if e.intersects_roi]
    roi_gt_map = _restrict_gt_map_to_roi(gt_sat_map_full, roi_bbox)
    roi_block = _taxonomy_block_from_entries(roi_sats, roi_entries, roi_gt_map)
    return {"full_frame": full_block, "roi": roi_block}


def _stream_typed_entry(
    stream_masks: list[dict[str, Any]],
    gt_str_map: np.ndarray,
    roi_bbox: Optional[tuple[int, int, int, int]],
    iou_thresh: float,
) -> dict[str, Optional[dict[str, Any]]]:
    """Detection (IoU/Hungarian) block for streams. Unchanged semantics."""
    full_block = _compute_slice_block(stream_masks, gt_str_map, iou_thresh)
    if roi_bbox is None:
        return {"full_frame": full_block, "roi": None}
    roi_preds, roi_gt = _restrict_to_roi(stream_masks, gt_str_map, roi_bbox)
    roi_block = _compute_slice_block(roi_preds, roi_gt, iou_thresh)
    return {"full_frame": full_block, "roi": roi_block}


def _compute_slice_block_rle(
    pred_masks: list[dict[str, Any]],
    gt_rles: list[dict[str, Any]],
    iou_thresh: float,
    H: int,
    W: int,
) -> dict[str, Any]:
    """RLE-aware variant of ``_compute_slice_block``.

    Mirrors the dict shape of the integer-raster variant so callers see a
    consistent block layout. Pixel metrics are still computed via union
    rasters (last-wins on overlap is fine for pixel metrics — it does not
    change the binary union).
    """
    pred_rles: list[dict[str, Any]] = []
    for m in pred_masks:
        seg = m.get("segmentation")
        if seg is None:
            continue
        pred_rles.append(mask_to_rle(np.asarray(seg).astype(np.uint8)))

    inst = calculate_optimal_instance_metrics_rle(pred_rles, gt_rles, iou_thresh)
    num_gt = inst["num_gt"]
    num_pred = inst["num_pred"]

    # --- pixel metrics: build union rasters ---
    gt_bin = np.zeros((H, W), dtype=bool)
    for r in gt_rles:
        from src.utils.coco_utils import decode_rle as _decode
        gt_bin |= _decode(r).astype(bool)
    if pred_masks:
        pred_bin = np.zeros((H, W), dtype=bool)
        for m in pred_masks:
            seg = m.get("segmentation")
            if seg is not None:
                pred_bin |= np.asarray(seg).astype(bool)
    else:
        pred_bin = np.zeros((H, W), dtype=bool)
    pixel = calculate_pixel_metrics(pred_bin, gt_bin)

    tp = int(inst["num_detected"])
    fp = num_pred - tp
    fn = num_gt - tp
    if num_gt == 0 and num_pred == 0:
        precision, recall, f1 = 1.0, 1.0, 1.0
        is_empty_trivial = True
    elif num_gt == 0 and num_pred > 0:
        precision, recall, f1 = 0.0, None, None
        is_empty_trivial = False
    elif num_gt > 0 and num_pred == 0:
        precision, recall, f1 = None, 0.0, None
        is_empty_trivial = False
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else 0.0
        )
        is_empty_trivial = False

    matched = [d for d in inst["per_instance_details"] if d["detected"]]
    matched_ious = [d["iou"] for d in matched]
    matched_iou_mean = float(np.mean(matched_ious)) if matched_ious else None
    matched_iou_median = float(np.median(matched_ious)) if matched_ious else None

    return {
        "num_gt": num_gt,
        "num_pred": num_pred,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "detection": {"precision": precision, "recall": recall, "f1": f1},
        "matched_iou_mean": matched_iou_mean,
        "matched_iou_median": matched_iou_median,
        "pixel": {
            "dice": pixel["dice"],
            "precision": pixel["precision"],
            "recall": pixel["recall"],
            "capped_hausdorff95": pixel["capped_hausdorff95"],
        },
        "per_instance": [
            {
                "gt_id": d["gt_instance_id"],
                "matched_pred_idx": d["matched_pred_idx"],
                "iou": d["iou"],
                "detected": d["detected"],
                "gt_area": d["gt_area"],
            }
            for d in inst["per_instance_details"]
        ],
        "is_empty_trivial": is_empty_trivial,
    }


def _rle_typed_entry(
    pred_masks: list[dict[str, Any]],
    gt_rles: list[dict[str, Any]],
    roi_bbox: Optional[tuple[int, int, int, int]],
    iou_thresh: float,
    H: int,
    W: int,
) -> dict[str, Optional[dict[str, Any]]]:
    """RLE-aware {full_frame, roi} block.

    ROI restriction uses the same rules as the integer-raster path:
    predictions filtered by mask-centroid-in-ROI; GT RLEs kept when any
    pixel intersects the ROI box.
    """
    full_block = _compute_slice_block_rle(pred_masks, gt_rles, iou_thresh, H, W)
    if roi_bbox is None:
        return {"full_frame": full_block, "roi": None}

    y0, x0, y1, x1 = roi_bbox
    roi_preds = [
        m for m in pred_masks
        if m.get("segmentation") is not None
        and _instance_centroid_in_roi(np.asarray(m["segmentation"]), roi_bbox)
    ]
    roi_gt: list[dict[str, Any]] = []
    from src.utils.coco_utils import decode_rle as _decode
    for r in gt_rles:
        bin_mask = _decode(r).astype(bool)
        if bin_mask[y0:y1, x0:x1].any():
            roi_gt.append(r)
    roi_block = _compute_slice_block_rle(roi_preds, roi_gt, iou_thresh, H, W)
    return {"full_frame": full_block, "roi": roi_block}


def _typed_blocks_tidal_v1(
    stream_masks: list[dict[str, Any]],
    sat_masks: list[dict[str, Any]],
    sample: Sample,
    iou_thresh: float,
    diag_cfg: DiagnosticCfg,
) -> dict[str, Any]:
    """3-class RLE-aware metric blocks for the new (tidal_v1) GT.

    The eval call signature still passes ``(stream_masks, sat_masks)`` for
    API compatibility. We split the satellite bucket further by
    ``type_label`` so any inner_galaxy predictions (rare on eval, since
    eval configs set ``hard_center_action: drop``) flow into their own
    block; legacy ``"streams"``/``"stellar stream"`` labels in
    stream_masks normalize to ``tidal_features``.
    """
    H, W = TARGET_H, TARGET_W
    out: dict[str, Any] = {}
    gt_rles_by_type = sample.gt_rles_by_type or {
        TIDAL_FEATURES: [], SATELLITES: [], INNER_GALAXY: [],
    }

    tidal_preds = list(stream_masks)
    sat_preds: list[dict[str, Any]] = []
    inner_preds: list[dict[str, Any]] = []
    for m in sat_masks:
        label = normalize_type_label(m.get("type_label", "satellites"))
        if label == INNER_GALAXY:
            inner_preds.append(m)
        elif label == TIDAL_FEATURES:
            tidal_preds.append(m)
        else:
            sat_preds.append(m)

    out[TIDAL_FEATURES] = _rle_typed_entry(
        tidal_preds, gt_rles_by_type.get(TIDAL_FEATURES, []),
        sample.roi_bbox_1024, iou_thresh, H, W,
    )

    # Satellites: hybrid block. Build a per-class GT raster from the RLE
    # list (last-wins on the rare within-class overlap pixel) so the
    # taxonomy / pixel-metric paths keep working unchanged. The new
    # instance_rle sub-block reports Hungarian metrics on the RLEs
    # directly so within-class overlap is preserved end-to-end.
    sat_gt_rles = gt_rles_by_type.get(SATELLITES, [])
    sat_gt_raster = rasterize_per_class_rles(sat_gt_rles, H, W)
    sat_entries = classify_candidates(
        sat_preds, sat_gt_raster, H, W, diag_cfg, roi_bbox=sample.roi_bbox_1024,
    )
    sat_block = _satellite_typed_entry(
        sat_preds, sat_entries, sat_gt_raster, sample.roi_bbox_1024,
    )
    sat_rle_block = _rle_typed_entry(
        sat_preds, sat_gt_rles, sample.roi_bbox_1024, iou_thresh, H, W,
    )
    for scope in ("full_frame", "roi"):
        if sat_block.get(scope) is not None and sat_rle_block.get(scope) is not None:
            sat_block[scope]["instance_rle"] = sat_rle_block[scope]
    out[SATELLITES] = sat_block

    out[INNER_GALAXY] = _rle_typed_entry(
        inner_preds, gt_rles_by_type.get(INNER_GALAXY, []),
        sample.roi_bbox_1024, iou_thresh, H, W,
    )

    if sample.benchmark_mode == "gt_canonical":
        combined: dict[str, Optional[dict[str, Any]]] = {}
        for scope in ("full_frame", "roi"):
            tidal_scope = out[TIDAL_FEATURES][scope]
            sat_scope = out[SATELLITES][scope]
            inner_scope = out[INNER_GALAXY][scope]
            if tidal_scope is None and sat_scope is None and inner_scope is None:
                combined[scope] = None
            else:
                combined[scope] = {
                    TIDAL_FEATURES: tidal_scope,
                    SATELLITES: sat_scope,
                    INNER_GALAXY: inner_scope,
                }
        out["combined"] = combined

    return out


def _typed_blocks(
    stream_masks: list[dict[str, Any]],
    sat_masks: list[dict[str, Any]],
    sample: Sample,
    iou_thresh: float,
    diag_cfg: DiagnosticCfg,
) -> dict[str, Any]:
    """Build benchmark-relevant type blocks × {full_frame, roi}.

    Two metric semantics, dispatched on ``sample.gt_path_version``:

    - ``"legacy"``: existing 2-class IoU/Hungarian + taxonomy + pixel metrics
      built from the packed int32 GT raster.
    - ``"tidal_v1"``: 3-class RLE-aware path (F11, F17). Tidal features +
      inner_galaxy use ``_rle_typed_entry``. Satellites keep today's
      taxonomy + pixel metrics (built from a per-class on-the-fly raster,
      last-wins for the rare within-class overlap pixel) AND get a new
      ``instance_rle`` sub-block that fully respects within-class overlap.
    """
    if sample.gt_path_version == "tidal_v1":
        return _typed_blocks_tidal_v1(stream_masks, sat_masks, sample, iou_thresh, diag_cfg)

    gt = sample.gt_instance_map_1024
    out: dict[str, Any] = {}

    has_satellites = sample.benchmark_mode in _SAT_BENCHMARKS
    has_streams = sample.benchmark_mode in {"firebox_dr1_streams", "gt_canonical"}

    if has_satellites:
        gt_sat = _filter_gt_by_type(gt, sample.gt_type_of_id, "satellites")
        H, W = int(gt_sat.shape[0]), int(gt_sat.shape[1])
        sat_entries = classify_candidates(
            sat_masks, gt_sat, H, W, diag_cfg,
            roi_bbox=sample.roi_bbox_1024,
        )
        out["satellites"] = _satellite_typed_entry(
            sat_masks, sat_entries, gt_sat, sample.roi_bbox_1024,
        )

    if has_streams:
        gt_str = _filter_gt_by_type(gt, sample.gt_type_of_id, "streams")
        out["streams"] = _stream_typed_entry(
            stream_masks, gt_str, sample.roi_bbox_1024, iou_thresh,
        )

    if sample.benchmark_mode == "gt_canonical":
        # combined is a pure composite container — no mixed detection score.
        combined: dict[str, Optional[dict[str, Any]]] = {}
        for scope in ("full_frame", "roi"):
            sat_scope = out["satellites"][scope]
            str_scope = out["streams"][scope]
            if sat_scope is None and str_scope is None:
                combined[scope] = None
            else:
                combined[scope] = {
                    "satellites": sat_scope,
                    "streams": str_scope,
                }
        out["combined"] = combined

    return out


_SAT_BENCHMARKS = frozenset({"fbox_gold_satellites", "gt_canonical"})


def _diag_cfg_from_dict(cfg: dict[str, Any]) -> DiagnosticCfg:
    """Build a DiagnosticCfg from the ``diagnostics.satellites`` block."""
    sat_cfg = (cfg.get("diagnostics") or {}).get("satellites") or {}
    return DiagnosticCfg(
        min_purity_for_match=float(sat_cfg.get("min_purity_for_match", 0.50)),
        completeness_complete=float(sat_cfg.get("completeness_complete", 0.50)),
        complete_one_to_one_min_completeness=float(
            sat_cfg.get("complete_one_to_one_min_completeness", 0.95)
        ),
        complete_one_to_one_max_seed_ratio=float(
            sat_cfg.get("complete_one_to_one_max_seed_ratio", 3.0)
        ),
        annulus_r_in_frac=float(sat_cfg.get("annulus_r_in_frac", 1.2)),
        annulus_r_out_frac=float(sat_cfg.get("annulus_r_out_frac", 2.0)),
        radial_n_rings=int(sat_cfg.get("radial_n_rings", 6)),
    )


def compute_sample_report(
    sample: Sample,
    raw_masks: list[dict[str, Any]],
    post_pred_only: tuple[list[dict[str, Any]], list[dict[str, Any]]],
    post_gt_aware: Optional[tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    cfg: dict[str, Any],
    render_signal: Optional[np.ndarray] = None,
) -> tuple[dict[str, Any], Optional[SatelliteDiagnosticReport]]:
    """Per-sample JSON report conforming to plan §3c.

    Returns:
        (report, diag_report):
            - ``report`` carries the typed-block layers plus, when the
              benchmark has satellites and ``render_signal`` is provided,
              a ``report["diagnostics"]["satellites_raw"]`` block that
              contains only the summary + a relative sidecar path. Per-
              candidate rows are NEVER embedded in ``report`` (to keep
              report.json small).
            - ``diag_report`` is the full ``SatelliteDiagnosticReport`` —
              the caller is responsible for writing it to
              ``sample_dir/diagnostics.json`` and feeding it into
              ``aggregate_diagnostics``. ``None`` when diagnostics were
              skipped (mode is streams-only, or render_signal is None,
              or ``diagnostics.enabled`` is False).
    """
    iou_thresh = float(cfg["matching"]["iou_threshold"])
    diag_cfg = _diag_cfg_from_dict(cfg)

    # Split raw by type_label.
    raw_streams = [m for m in raw_masks if m.get("type_label") == "streams"]
    raw_sats = [m for m in raw_masks if m.get("type_label") == "satellites"]

    po_streams, po_sats = post_pred_only

    # Count GT by type.
    num_gt_sat = sum(1 for t in sample.gt_type_of_id.values() if t == "satellites")
    num_gt_str = sum(1 for t in sample.gt_type_of_id.values() if t == "streams")

    report: dict[str, Any] = {
        "base_key": sample.base_key,
        "galaxy_id": sample.galaxy_id,
        "view": sample.view,
        "benchmark_mode": sample.benchmark_mode,
        "target_size": [TARGET_H, TARGET_W],
        "roi_bbox_1024": list(sample.roi_bbox_1024) if sample.roi_bbox_1024 else None,
        "roi_rule": ROI_RULE,
        "dropped_instance_ids": sample.dropped_instance_ids,
        "layers": {
            "raw": _typed_blocks(raw_streams, raw_sats, sample, iou_thresh, diag_cfg),
            "post_pred_only": _typed_blocks(
                po_streams, po_sats, sample, iou_thresh, diag_cfg,
            ),
            "post_gt_aware": None,
        },
        "layer_uses_gt": {
            "raw": False,
            "post_pred_only": False,
            "post_gt_aware": True,  # always True when populated
        },
        "diagnostics": None,
    }
    if sample.benchmark_mode == "fbox_gold_satellites":
        report["num_gt_satellites"] = num_gt_sat
    elif sample.benchmark_mode == "firebox_dr1_streams":
        report["num_gt_streams"] = num_gt_str
    else:
        report["num_gt_satellites"] = num_gt_sat
        report["num_gt_streams"] = num_gt_str

    if post_gt_aware is not None:
        ga_streams, ga_sats = post_gt_aware
        report["layers"]["post_gt_aware"] = _typed_blocks(
            ga_streams, ga_sats, sample, iou_thresh, diag_cfg,
        )

    # fbox invariant: curated GT lies entirely inside the ROI, so the official
    # ROI block must have the same num_gt as the full-frame block and as the
    # sample's overall num_gt_satellites. Fail-fast if this ever breaks.
    if sample.benchmark_mode == "fbox_gold_satellites":
        for layer_name, layer_blocks in report["layers"].items():
            if layer_blocks is None:
                continue
            sat_blocks = layer_blocks.get("satellites")
            if sat_blocks is None:
                continue
            full_num_gt = sat_blocks["full_frame"]["num_gt"]
            roi = sat_blocks.get("roi")
            roi_num_gt = roi["num_gt"] if roi is not None else None
            if full_num_gt != num_gt_sat or (
                roi_num_gt is not None and roi_num_gt != num_gt_sat
            ):
                raise RuntimeError(
                    f"[{sample.base_key}] fbox invariant broken on layer "
                    f"{layer_name!r}: num_gt_satellites={num_gt_sat}, "
                    f"full={full_num_gt}, roi={roi_num_gt}"
                )

    # --- pred-centric satellite diagnostics (Phase 1) ---
    diag_enabled = bool((cfg.get("diagnostics") or {}).get("enabled", False))
    diag_report: Optional[SatelliteDiagnosticReport] = None
    if (
        diag_enabled
        and render_signal is not None
        and sample.benchmark_mode in _SAT_BENCHMARKS
    ):
        gt_sat_map = _filter_gt_by_type(
            sample.gt_instance_map_1024, sample.gt_type_of_id, "satellites"
        )
        diag_report = build_candidate_table(
            raw_sats=raw_sats,
            gt_sat_map=gt_sat_map,
            render_signal=render_signal,
            H=TARGET_H,
            W=TARGET_W,
            cfg=_diag_cfg_from_dict(cfg),
            roi_bbox=sample.roi_bbox_1024,
            host_support=None,  # Phase 2+
            post_sats=po_sats,
        )
        report["diagnostics"] = {
            "satellites_raw": {
                "summary": {
                    "counts_by_label": diag_report["counts_by_label"],
                    "counts_by_label_roi": diag_report["counts_by_label_roi"],
                    "counts_post_by_label": diag_report["counts_post_by_label"],
                    "counts_post_by_label_roi": diag_report["counts_post_by_label_roi"],
                    "roi_matched_unmatched": matched_unmatched_counts(
                        diag_report["counts_by_label_roi"],
                        diag_report["counts_post_by_label_roi"],
                    ),
                    "thresholds_used": diag_report["thresholds_used"],
                    "host_support_available": diag_report["host_support_available"],
                    "n_candidates": len(diag_report["per_candidate"]),
                    "n_post_candidates": (
                        sum(diag_report["counts_post_by_label"].values())
                        if diag_report["counts_post_by_label"] is not None
                        else None
                    ),
                },
                "per_candidate_path": "diagnostics.json",
            }
        }

    return report, diag_report


# =========================================================================== #
#  Aggregation
# =========================================================================== #


def _micro_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Streams micro aggregate: sum tp/fp/fn across blocks, recompute P/R/F1."""
    tp = sum(b["tp"] for b in blocks)
    fp = sum(b["fp"] for b in blocks)
    fn = sum(b["fn"] for b in blocks)
    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall and (precision + recall) > 0
        else None
    )
    return {
        "tp": tp, "fp": fp, "fn": fn,
        "precision": precision, "recall": recall, "f1": f1,
    }


def _macro_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Streams macro aggregate: mean of per-sample detection p/r/f1."""
    def _collect(field: str) -> list[float]:
        vals = []
        for b in blocks:
            v = b["detection"].get(field)
            if v is not None:
                vals.append(float(v))
        return vals

    precs = _collect("precision")
    recs = _collect("recall")
    f1s = _collect("f1")
    return {
        "precision_mean": float(np.mean(precs)) if precs else None,
        "recall_mean": float(np.mean(recs)) if recs else None,
        "f1_mean": float(np.mean(f1s)) if f1s else None,
        "samples_used": {
            "precision": len(precs),
            "recall": len(recs),
            "f1": len(f1s),
        },
        "samples_total": len(blocks),
    }


def _taxonomy_micro_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Satellite taxonomy micro aggregate: sum matched_candidates / num_pred
    for precision and unique_gt_covered / num_gt for recall.
    """
    matched = sum(int(b["matched_candidates"]) for b in blocks)
    unmatched = sum(int(b["unmatched_candidates"]) for b in blocks)
    num_pred = sum(int(b["num_pred"]) for b in blocks)
    num_gt = sum(int(b["num_gt"]) for b in blocks)
    unique_gt_covered = sum(int(b["unique_gt_covered"]) for b in blocks)
    precision = (matched / num_pred) if num_pred > 0 else None
    recall = (unique_gt_covered / num_gt) if num_gt > 0 else None
    if precision is None or recall is None:
        f1: Optional[float] = None
    elif (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return {
        "matched_candidates": matched,
        "unmatched_candidates": unmatched,
        "num_pred": num_pred,
        "unique_gt_covered": unique_gt_covered,
        "num_gt": num_gt,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def _taxonomy_macro_from_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    """Satellite taxonomy macro aggregate: mean of per-sample p/r/f1 over
    samples where that field is defined (matches the streams macro shape).
    """
    def _collect(field: str) -> list[float]:
        vals = []
        for b in blocks:
            v = b.get(field)
            if v is not None:
                vals.append(float(v))
        return vals

    precs = _collect("precision")
    recs = _collect("recall")
    f1s = _collect("f1")
    return {
        "precision_mean": float(np.mean(precs)) if precs else None,
        "recall_mean": float(np.mean(recs)) if recs else None,
        "f1_mean": float(np.mean(f1s)) if f1s else None,
        "samples_used": {
            "precision": len(precs),
            "recall": len(recs),
            "f1": len(f1s),
        },
        "samples_total": len(blocks),
    }


def _aggregate_type(
    reports: list[dict[str, Any]],
    layer: str,
    type_key: str,
) -> dict[str, Any]:
    """Build ``{full_frame: {micro, macro}, roi: {micro, macro}}`` for one
    type under one layer. Dispatches to taxonomy aggregators for satellites
    and detection aggregators for streams.
    """
    micro_fn = (
        _taxonomy_micro_from_blocks if type_key == "satellites"
        else _micro_from_blocks
    )
    macro_fn = (
        _taxonomy_macro_from_blocks if type_key == "satellites"
        else _macro_from_blocks
    )
    type_out: dict[str, Any] = {}
    for scope in ("full_frame", "roi"):
        blocks = [
            r["layers"][layer][type_key][scope]
            for r in reports
            if (
                type_key in r["layers"][layer]
                and r["layers"][layer][type_key].get(scope) is not None
            )
        ]
        if not blocks:
            type_out[scope] = None
            continue
        type_out[scope] = {
            "micro": micro_fn(blocks),
            "macro": macro_fn(blocks),
        }
    return type_out


def aggregate(reports: list[dict[str, Any]]) -> dict[str, Any]:
    """Build the summary: per-layer × per-type × per-scope (full_frame, roi).

    Typed entries are ``satellites`` (taxonomy micro/macro) and ``streams``
    (IoU/Hungarian detection micro/macro). The per-sample ``combined``
    container in ``gt_canonical`` is deliberately NOT re-aggregated here —
    the summary exposes only the two underlying typed views.
    """
    out: dict[str, Any] = {"n_samples": len(reports)}
    type_order = ("satellites", "streams")
    for layer in ("raw", "post_pred_only", "post_gt_aware"):
        present = [r for r in reports if r["layers"].get(layer) is not None]
        if not present:
            out[layer] = None
            continue
        layer_out: dict[str, Any] = {}
        for type_key in type_order:
            if not any(type_key in r["layers"][layer] for r in present):
                continue
            layer_out[type_key] = _aggregate_type(present, layer, type_key)
        out[layer] = layer_out
    return out


# =========================================================================== #
#  JSON helper
# =========================================================================== #


def json_default(obj: Any) -> Any:
    """Default serialiser for numpy scalars / arrays / Path."""
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"Not JSON-serialisable: {type(obj)}")
