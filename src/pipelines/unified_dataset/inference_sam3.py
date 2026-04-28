"""
Phase 3 – SAM3 engine: text-prompt -> type-aware filter -> evaluate / pseudo_label.

DR1 keeps the legacy evaluate/pseudo-label behavior under gt_canonical.
PNbody pseudo-label mode adds:
    - authoritative clean pseudo GT
    - noisy clone GT roots (never native GT from noisy predictions)
    - per-condition diagnostics with pre-/post-conflict artifacts
"""
from __future__ import annotations

import json
import logging
import shutil
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent

from .artifacts import (
    assign_stable_ids,
    load_predictions_json,
    merge_instances,
    save_per_class_instance_maps,
    save_per_class_pseudo_gt,
    save_predictions_json,
    save_pseudo_gt,
)
from .taxonomy import INNER_GALAXY, SATELLITES, TIDAL_FEATURES, normalize_type_label
from .keys import BaseKey
from .paths import PathResolver
from src.visualization.overlay import (
    save_evaluation_overlay,
    save_instance_overlay,
    save_pseudo_label_overlay,
    save_raw_overlay,
)

_PSEUDO_LABEL_ARTIFACTS = [
    "instance_map_uint8.png",
    "instances.json",
    "manifest.json",
    "sam3_predictions_raw.json",
    "sam3_predictions_post.json",
    "sam3_pseudo_label_overlay.png",
]

_PNBODY_GT_ARTIFACTS = [
    "instance_map_uint8.png",
    "instances.json",
    "id_map.json",
    "manifest.json",
    "sam3_predictions_raw.json",
    "sam3_predictions_post.json",
    "sam3_pseudo_label_overlay.png",
    "overlay.png",
]

_PNBODY_DIAGNOSTIC_ARTIFACTS = [
    "streams_predictions_raw.json",
    "satellites_predictions_raw.json",
    "streams_predictions_post_filter.json",
    "satellites_predictions_post_filter.json",
    "dual_predictions_pre_conflict.json",
    "dual_overlay_pre_conflict.png",
    "conflict_resolution.json",
    "dual_predictions_post_conflict.json",
    "dual_overlay_post_conflict.png",
    "sam3_predictions_raw.json",
    "sam3_predictions_post.json",
    "manifest.json",
]

# Completion guard for the new (tidal_v1) GT path. Used by both DR1 evaluate
# and PNbody pseudo-label flows when paths.gt_subdir is non-default.
NEW_GT_REQUIRED_FILES = [
    "tidal_features_instance_map.npy",
    "satellites_instance_map.npy",
    "inner_galaxy_instance_map.npy",
    "instances.json",
    "id_map.json",
    "manifest.json",
    "sam3_predictions_post.json",
]


def _new_gt_complete(gt_dir: Path) -> bool:
    return all((gt_dir / f).exists() for f in NEW_GT_REQUIRED_FILES)


def _pnbody_diagnostic_artifacts_for_path(on_new_path: bool) -> list[str]:
    """Return the diagnostic artifacts expected for the active path.

    On the new path the cross-class conflict is OFF, so the artifacts
    that document the pre/post-conflict pair no longer exist.
    """
    if not on_new_path:
        return _PNBODY_DIAGNOSTIC_ARTIFACTS
    return [
        "streams_predictions_raw.json",
        "satellites_predictions_raw.json",
        "streams_predictions_post_filter.json",
        "satellites_predictions_post_filter.json",
        "sam3_predictions_raw.json",
        "sam3_predictions_post.json",
        "manifest.json",
    ]


def _pseudo_label_complete(gt_dir: Path) -> bool:
    return all((gt_dir / f).exists() for f in _PSEUDO_LABEL_ARTIFACTS)


def _evaluate_complete(gt_dir: Path) -> bool:
    return (
        (gt_dir / "sam3_predictions_raw.json").exists()
        and (gt_dir / "instance_map_uint8.png").exists()
        and (gt_dir / "overlay.png").exists()
        and (gt_dir / "sam3_raw_overlay.png").exists()
        and (gt_dir / "sam3_eval_overlay.png").exists()
        and (gt_dir / "sam3_satellite_diagnostics.json").exists()
    )


def _pnbody_gt_complete(gt_dir: Path) -> bool:
    return all((gt_dir / f).exists() for f in _PNBODY_GT_ARTIFACTS)


def _pnbody_diag_complete(diag_dir: Path) -> bool:
    return all((diag_dir / f).exists() for f in _PNBODY_DIAGNOSTIC_ARTIFACTS)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))


def _condition_sort_key(resolver: PathResolver, condition: str) -> tuple[int, str]:
    label_mode = resolver.get_label_mode(condition)
    return (0 if label_mode == "authoritative" else 1, 0 if condition == "clean" else 1, condition)


def _assert_no_runtime_overrides(config: dict[str, Any]) -> None:
    """Fail fast when legacy satellite override config is still wired into runtime.

    Reviewed exceptions now live as explicit human-adopted instances in the
    Shadow GT migration flow (see ``scripts/review/migrate_satellite_overrides.py``);
    no YAML override file may be consumed by inference anymore.
    """
    sam3_cfg = config.get("inference_phase", {}).get("sam3", {})
    if "satellite_overrides_path" in sam3_cfg:
        raise ValueError(
            "inference_phase.sam3.satellite_overrides_path is no longer supported; "
            "migrate reviewed exceptions through the Shadow GT flow "
            "(scripts/review/migrate_satellite_overrides.py) and remove this key "
            "from your config."
        )


def _prepare_sam3_context(config: dict[str, Any]) -> dict[str, Any]:
    _assert_no_runtime_overrides(config)

    from src.analysis.mask_metrics import append_metrics_to_masks
    from src.inference.sam3_prompt_runner import SAM3PromptRunner
    from src.postprocess.core_exclusion_filter import CoreExclusionFilter
    from src.postprocess.satellite_conflict_resolver import SatelliteConflictResolver
    from src.postprocess.satellite_core_policy import SatelliteCorePolicy
    from src.postprocess.satellite_pipeline import SatellitePipelineRunner
    from src.postprocess.satellite_prior_filter import SatellitePriorFilter, load_filter_cfg
    from src.postprocess.satellite_score_gate import SatelliteScoreGate
    from src.postprocess.stream_satellite_conflict_filter import StreamSatelliteConflictFilter
    from src.postprocess.streams_sanity_filter import StreamsSanityFilter, load_streams_cfg

    inf_cfg = config.get("inference_phase", {})
    sam3_cfg = inf_cfg.get("sam3", {})
    target_size = tuple(config["processing"]["target_size"])
    H_work, W_work = target_size

    runner = SAM3PromptRunner(
        checkpoint=sam3_cfg.get("checkpoint"),
        bpe_path=sam3_cfg.get("bpe_path"),
        confidence_threshold=sam3_cfg.get("confidence_threshold", 0.55),
        resolution=sam3_cfg.get("resolution", 1008),
        target_size=target_size,
    )

    sanity_cfg = sam3_cfg.get("streams_sanity", {})
    streams_stats_json = Path(
        sanity_cfg.get("stats_json", "outputs/mask_stats/mask_stats_summary.json")
    )
    if not streams_stats_json.is_absolute():
        streams_stats_json = PROJECT_ROOT / streams_stats_json
    streams_gt_cfg = load_streams_cfg(streams_stats_json)
    streams_flt = StreamsSanityFilter(
        min_area=sanity_cfg.get("min_area", streams_gt_cfg["min_area"]),
        max_area_frac=sanity_cfg.get("max_area_frac", 0.5),
        edge_touch_frac=sanity_cfg.get("edge_touch_frac", streams_gt_cfg["edge_touch_frac"]),
        max_area_px=sanity_cfg.get("max_area_px", streams_gt_cfg["max_area_px"]),
    )

    paths_cfg = config.get("paths", {})
    gt_subdir = paths_cfg.get("gt_subdir", "gt_canonical")
    pseudo_gt_subdir = paths_cfg.get("pseudo_gt_subdir", "pseudo_gt_canonical")
    on_new_path = (gt_subdir != "gt_canonical") or (pseudo_gt_subdir != "pseudo_gt_canonical")

    # Prior-filter config sourcing (F4, F18). New-path configs put the frozen
    # thresholds directly under inference_phase.sam3.prior_filter; legacy
    # configs continue to read from satellites.prior.stats_json.
    sat_cfg = config.get("satellites", {})
    prior_cfg_entry = sat_cfg.get("prior", {})
    core_cfg_legacy = sat_cfg.get("core_exclusion", {})

    if on_new_path:
        if "prior_filter" not in sam3_cfg:
            raise KeyError(
                "inference_phase.sam3.prior_filter is required when paths.gt_subdir "
                "or paths.pseudo_gt_subdir is non-default (tidal_v1 path active)."
            )
        prior_cfg = dict(sam3_cfg["prior_filter"])
    else:
        stats_json = Path(
            prior_cfg_entry.get("stats_json", "outputs/mask_stats/mask_stats_summary.json")
        )
        if not stats_json.is_absolute():
            stats_json = PROJECT_ROOT / stats_json
        prior_cfg = load_filter_cfg(stats_json)
        # Kept for backward-compatible pnbody callers; the slim prior filter ignores them.
        prior_cfg["ambiguous_factor"] = prior_cfg_entry.get("ambiguous_factor", 0.25)
        prior_cfg["core_radius_frac"] = core_cfg_legacy.get("radius_frac", 0.08)
        if "hard_center_radius_frac" in prior_cfg_entry:
            prior_cfg["hard_center_radius_frac"] = prior_cfg_entry["hard_center_radius_frac"]

    sat_prior_flt = SatellitePriorFilter(prior_cfg)
    sat_core_flt = CoreExclusionFilter(radius_frac=core_cfg_legacy.get("radius_frac", 0.08))

    stream_conflict_cfg = sam3_cfg.get("stream_conflict", {})
    conflict_flt = StreamSatelliteConflictFilter(
        policy=stream_conflict_cfg.get("policy", "stream_first"),
        keep_stream_aspect_min=stream_conflict_cfg.get("keep_stream_aspect_min", 1.9),
        keep_stream_curvature_min=stream_conflict_cfg.get("keep_stream_curvature_min", 1.2),
        keep_stream_area_ratio=stream_conflict_cfg.get("keep_stream_area_ratio", 1.5),
        drop_compact_stream_overlap=stream_conflict_cfg.get("drop_compact_stream_overlap", 0.75),
        satellite_solidity_min=stream_conflict_cfg.get("satellite_solidity_min", prior_cfg.get("solidity_min", 0.83)),
        satellite_aspect_max=stream_conflict_cfg.get("satellite_aspect_max", prior_cfg.get("aspect_sym_max", 1.75)),
    )

    # DR1 v4 satellite pipeline components (static, unit-safe configs)
    score_gate_cfg = sam3_cfg.get("score_gate", {})
    score_gate = SatelliteScoreGate(**score_gate_cfg)

    # Conflict cfg copy (no live-config mutation). The "enabled" flag is a
    # runner-level switch; the resolver itself never sees it.
    conflict_policy_cfg = dict(sam3_cfg.get("conflict_policy", {}))
    enable_conflict = conflict_policy_cfg.pop("enabled", True)
    conflict_resolver = SatelliteConflictResolver(**conflict_policy_cfg)

    # Per-path core-policy (F16). Legacy keeps running the stage; new path
    # passes core_policy=None and disables the stage so the relabel-to-
    # inner_galaxy decision in _stage_prior_filter is the only authority.
    if on_new_path:
        core_policy = None
        enable_core = False
    else:
        core_policy_cfg = sam3_cfg.get("core_policy", {})
        core_policy = SatelliteCorePolicy(**core_policy_cfg)
        enable_core = True

    satellite_runner = SatellitePipelineRunner(
        score_gate=score_gate,
        prior_filter=sat_prior_flt,
        core_policy=core_policy,
        conflict_resolver=conflict_resolver,
        enable_core_policy=enable_core,
        enable_conflict_resolution=enable_conflict,
    )

    logging.getLogger(__name__).info(
        "Satellite pipeline assembled (path=%s): score_gate=%s prior_filter=%s "
        "core_policy=%s conflict=%s",
        "tidal_v1" if on_new_path else "legacy",
        score_gate.threshold_version,
        sat_prior_flt.threshold_version,
        core_policy.threshold_version if core_policy is not None else "disabled",
        conflict_resolver.threshold_version if enable_conflict else "disabled",
    )

    return {
        "append_metrics_to_masks": append_metrics_to_masks,
        "runner": runner,
        "streams_filter": streams_flt,
        "satellite_prior_filter": sat_prior_flt,
        "satellite_core_filter": sat_core_flt,
        "conflict_filter": conflict_flt,
        "satellite_pipeline_runner": satellite_runner,
        "prompts": sam3_cfg.get("prompts", []),
        "H_work": H_work,
        "W_work": W_work,
        "on_new_path": on_new_path,
    }


def _clone_clean_gt_to_condition(
    resolver: PathResolver,
    key: BaseKey,
    condition: str,
    clean_gt_dir: Path,
    clean_final_masks: list[dict[str, Any]],
    image_np: np.ndarray,
) -> None:
    target_gt_dir = resolver.get_pseudo_gt_dir(key, dataset=resolver.dataset_name, condition=condition)
    target_gt_dir.mkdir(parents=True, exist_ok=True)

    # Detect whether the clean GT is on the new (tidal_v1) path. The marker
    # is a class-specific .npy file; if absent we fall back to the legacy
    # artifact set.
    on_new_path = (clean_gt_dir / "tidal_features_instance_map.npy").exists()

    if on_new_path:
        required = NEW_GT_REQUIRED_FILES + ["sam3_predictions_raw.json"]
        missing = [f for f in NEW_GT_REQUIRED_FILES if not (clean_gt_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"Clean GT at {clean_gt_dir} missing required tidal_v1 artifacts: {missing}"
            )
        for filename in required:
            src = clean_gt_dir / filename
            if src.exists():
                shutil.copy2(src, target_gt_dir / filename)
        # The QA preview and uint8 PNG live alongside the .npy files.
        for filename in ("instance_map_uint8.png",):
            src = clean_gt_dir / filename
            if src.exists():
                shutil.copy2(src, target_gt_dir / filename)
    else:
        for filename in (
            "instance_map_uint8.png",
            "instances.json",
            "id_map.json",
            "sam3_predictions_raw.json",
            "sam3_predictions_post.json",
        ):
            src = clean_gt_dir / filename
            if src.exists():
                shutil.copy2(src, target_gt_dir / filename)

    save_pseudo_label_overlay(
        target_gt_dir / "sam3_pseudo_label_overlay.png",
        image_np,
        clean_final_masks,
    )

    instance_map = np.array(Image.open(target_gt_dir / "instance_map_uint8.png"))
    save_instance_overlay(target_gt_dir / "overlay.png", image_np, instance_map)

    clean_manifest = {}
    clean_manifest_path = clean_gt_dir / "manifest.json"
    if clean_manifest_path.exists():
        clean_manifest = json.loads(clean_manifest_path.read_text())

    target_manifest = {
        **clean_manifest,
        "dataset": resolver.dataset_name,
        "condition": condition,
        "label_mode": "clone_from_clean",
        "source_dataset": resolver.dataset_name,
        "source_condition": "clean",
        "source_base_key": str(key),
        "cloned_at": datetime.now().isoformat(),
    }
    _write_json(target_gt_dir / "manifest.json", target_manifest)


def _run_standard_inference_sam3(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """Legacy DR1-oriented SAM3 flow."""
    inf_cfg = config.get("inference_phase", {})
    input_variant = inf_cfg.get(
        "input_image_variant",
        config.get("satellites", {}).get("input_image_variant", "linear_magnitude"),
    )
    target_size = tuple(config["processing"]["target_size"])
    run_mode = inf_cfg.get("run_mode", "evaluate")
    resolver = PathResolver(config)
    ctx = _prepare_sam3_context(config)
    runner = ctx["runner"]
    streams_flt = ctx["streams_filter"]
    sat_prior_flt = ctx["satellite_prior_filter"]
    sat_core_flt = ctx["satellite_core_filter"]
    satellite_pipeline_runner = ctx["satellite_pipeline_runner"]
    append_metrics_to_masks = ctx["append_metrics_to_masks"]
    prompts = ctx["prompts"]
    H_work, W_work = target_size

    is_pseudo = run_mode == "pseudo_label"
    on_new_path = ctx["on_new_path"]
    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_render": 0, "skipped_no_streams": 0}

    for key in base_keys:
        gt_dir = resolver.get_gt_dir(key)

        should_rebuild = force or (force_variants is not None)
        if on_new_path and not is_pseudo:
            already_done = _new_gt_complete(gt_dir)
        else:
            already_done = _pseudo_label_complete(gt_dir) if is_pseudo else _evaluate_complete(gt_dir)

        if already_done:
            if should_rebuild:
                eval_arts = [
                    "sam3_predictions_raw.json",
                    "sam3_predictions_post.json",
                    "sam3_eval_overlay.png",
                    "sam3_raw_overlay.png",
                    "sam3_satellite_diagnostics.json",
                    "instance_map_uint8.png",
                    "id_map.json",
                    "instances.json",
                    "overlay.png",
                ]
                for art in (_PSEUDO_LABEL_ARTIFACTS if is_pseudo else eval_arts):
                    (gt_dir / art).unlink(missing_ok=True)
                logger.info("Force rebuild SAM3 %s: %s", run_mode, key)
            else:
                stats["skipped_exists"] += 1
                continue

        render_path = resolver.get_render_dir(input_variant, key) / "0000.png"
        if not render_path.exists():
            logger.warning("Render not found: %s", render_path)
            stats["skipped_no_render"] += 1
            continue

        streams_map = None
        tidal_local_map = None
        if not is_pseudo:
            if on_new_path:
                tidal_npy = gt_dir / "tidal_features_instance_map.npy"
                if not tidal_npy.exists():
                    logger.warning("Tidal-features map not found: %s", tidal_npy)
                    stats["skipped_no_streams"] += 1
                    continue
                tidal_local_map = np.load(tidal_npy).astype(np.int32)
                streams_map = np.zeros_like(tidal_local_map, dtype=np.int32)
            else:
                streams_npy = gt_dir / "streams_instance_map.npy"
                if not streams_npy.exists():
                    logger.warning("Streams map not found: %s", streams_npy)
                    stats["skipped_no_streams"] += 1
                    continue
                streams_map = np.load(streams_npy)

        gt_dir.mkdir(parents=True, exist_ok=True)

        image_pil = Image.open(render_path).convert("RGB")
        image_np = np.array(image_pil)

        masks, time_ms = runner.run(image_pil, prompts)
        logger.info("%s: SAM3 returned %d masks in %.0fms", key, len(masks), time_ms)

        if masks:
            append_metrics_to_masks(masks, H_work, W_work, compute_hull=True)
            # Stamp stable IDs (raw_index, candidate_id, candidate_rle_sha1) on the
            # combined raw mask list before any filtering layer touches it. This
            # is the single source of identity for downstream stages and JSON
            # consumers; idempotent if already stamped.
            assign_stable_ids(masks)

        save_predictions_json(
            gt_dir / "sam3_predictions_raw.json",
            masks,
            H_work,
            W_work,
            engine="sam3",
            layer="raw",
        )

        stream_masks = [m for m in masks if m.get("type_label") == "streams"]
        sat_masks = [m for m in masks if m.get("type_label") == "satellites"]

        kept_streams, _ = streams_flt.filter(stream_masks, H_work, W_work)

        pipeline_result = None
        kept_inner_galaxy: list[dict[str, Any]] = []
        if is_pseudo:
            # DR1 pseudo_label still uses the legacy filter chain (out of scope for v4).
            if sat_masks:
                kept_sat_prior, _, _ = sat_prior_flt.filter(sat_masks)
                kept_sat_final, _, _ = sat_core_flt.filter(kept_sat_prior, H_work, W_work)
            else:
                kept_sat_final = []
        else:
            # DR1 evaluate: explicit satellite state-machine with GT-aware conflict resolution.
            pipeline_result = satellite_pipeline_runner.run(
                sat_masks,
                streams_gt_map=streams_map,
                H=H_work,
                W=W_work,
                base_key=str(key),
            )
            kept_sat_final = pipeline_result.final_sats
            kept_inner_galaxy = pipeline_result.final_inner_galaxy

        # On the new path, tidal_features come from the FITS map, not from
        # SAM3 prompts; the predictions JSON carries only SAM-derived masks
        # (satellites + inner_galaxy). On the legacy path, predictions JSON
        # also carries kept stream-prompt masks for backward compatibility.
        if on_new_path and not is_pseudo:
            post_masks = kept_sat_final + kept_inner_galaxy
        else:
            post_masks = kept_streams + kept_sat_final

        save_predictions_json(
            gt_dir / "sam3_predictions_post.json",
            post_masks,
            H_work,
            W_work,
            engine="sam3",
            layer="post",
        )

        if is_pseudo:
            save_pseudo_gt(gt_dir, post_masks, H_work, W_work)
            save_pseudo_label_overlay(gt_dir / "sam3_pseudo_label_overlay.png", image_np, post_masks)
        else:
            save_raw_overlay(
                gt_dir / "sam3_raw_overlay.png",
                image_np,
                stream_masks,
                sat_masks,
            )
            save_evaluation_overlay(
                gt_dir / "sam3_eval_overlay.png",
                image_np,
                gt_streams=streams_map,
                predictions=post_masks,
            )

            diagnostics_payload = {
                "image_summary": pipeline_result.image_summary if pipeline_result else {},
                "candidates": pipeline_result.candidates if pipeline_result else [],
            }
            _write_json(gt_dir / "sam3_satellite_diagnostics.json", diagnostics_payload)

            if on_new_path:
                # New (tidal_v1) path: tidal features come from the FITS map
                # written by gt.py; the runner contributes satellites and
                # inner_galaxy. The writer merges its manifest stamps with
                # the FITS-phase manifest (F6).
                tidal_features_masks: list[dict[str, Any]] = []
                for local_id in sorted(int(x) for x in np.unique(tidal_local_map) if x != 0):
                    seg = (tidal_local_map == local_id)
                    tidal_features_masks.append({
                        "segmentation": seg,
                        "type_label": TIDAL_FEATURES,
                        "source": "firebox_sb31.5_fits",
                        "source_instance_id": int(local_id),
                    })
                masks_by_type = {
                    TIDAL_FEATURES: tidal_features_masks,
                    SATELLITES: kept_sat_final,
                    INNER_GALAXY: kept_inner_galaxy,
                }
                save_per_class_instance_maps(gt_dir, masks_by_type, H_work, W_work)
                # Build the QA overlay from the writer's last-wins preview.
                instance_map_qa = np.array(Image.open(gt_dir / "instance_map_uint8.png"))
                save_instance_overlay(gt_dir / "overlay.png", image_np, instance_map_qa)
                # Bookkeeping for the manifest update below.
                max_stream_id = 0
                max_final_id = (
                    len(tidal_features_masks) + len(kept_sat_final) + len(kept_inner_galaxy)
                )
                overlap_policy = "per_class_rle"
                overlap_stats = {"overlap_px": 0, "overlap_rate": 0.0}
            else:
                max_stream_id = int(streams_map.max())
                overlap_policy = config.get("satellites", {}).get("overlap_policy", "keep_streams")
                sort_policy = config.get(
                    "satellites",
                    {},
                ).get("satellite_sort_policy", ["area_desc", "centroid_x_asc", "centroid_y_asc"])

                def _sort_key(mask: dict[str, Any]) -> tuple[Any, ...]:
                    keys = []
                    for policy in sort_policy:
                        if policy == "area_desc":
                            keys.append(-mask.get("area", 0))
                        elif policy == "area_asc":
                            keys.append(mask.get("area", 0))
                        elif policy == "centroid_x_asc":
                            keys.append(mask.get("centroid_x", 0))
                        elif policy == "centroid_y_asc":
                            keys.append(mask.get("centroid_y", 0))
                    return tuple(keys)

                sorted_sats = sorted(kept_sat_final, key=_sort_key)
                instance_map, instances_list, id_map, overlap_stats = merge_instances(
                    streams_map,
                    sorted_sats,
                    max_stream_id,
                    overlap_policy,
                )
                max_final_id = int(instance_map.max())

                Image.fromarray(instance_map).save(gt_dir / "instance_map_uint8.png")
                (gt_dir / "instances.json").write_text(json.dumps(instances_list, indent=2))
                (gt_dir / "id_map.json").write_text(json.dumps(id_map, indent=2))
                save_instance_overlay(gt_dir / "overlay.png", image_np, instance_map)

        manifest_path = gt_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
        if on_new_path and not is_pseudo:
            gt_source = "tidal_features_instance_map_npy+sam3_predictions_post"
            pipeline_version = "tidal_v1"
        elif is_pseudo:
            gt_source = "none"
            pipeline_version = "legacy"
        else:
            gt_source = "streams_instance_map"
            pipeline_version = "v4"
        manifest.update(
            {
                "engine": "sam3",
                "run_mode": run_mode,
                "gt_source": gt_source,
                "sam3_n_raw": len(masks),
                "sam3_n_post": len(post_masks),
                "sam3_n_streams_kept": len(kept_streams),
                "sam3_n_satellites_kept": len(kept_sat_final),
                "sam3_n_inner_galaxy": len(kept_inner_galaxy),
                "sam3_inference_time_ms": round(time_ms, 2),
                "sam3_updated_at": datetime.now().isoformat(),
                "sam3_satellite_pipeline_version": pipeline_version,
            }
        )
        if not is_pseudo:
            manifest.update(
                {
                    "overlap_policy": overlap_policy,
                    "max_stream_id": max_stream_id,
                    "max_final_id": max_final_id,
                    "n_satellites_kept": len(kept_sat_final),
                    "overlap_px": overlap_stats.get("overlap_px", 0),
                    "overlap_rate": overlap_stats.get("overlap_rate", 0.0),
                }
            )
        manifest_path.write_text(json.dumps(manifest, indent=2))

        stats["processed"] += 1
        if stats["processed"] % 10 == 0:
            logger.info("Progress: %d/%d", stats["processed"], len(base_keys))

    logger.info(
        "SAM3 %s: %d processed, %d skipped",
        run_mode,
        stats["processed"],
        stats["skipped_exists"],
    )


def _run_pnbody_clean_truth_noisy_clone(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """PNbody pseudo GT flow: clean authoritative labels, noisy clone labels, diagnostics for all."""
    inf_cfg = config.get("inference_phase", {})
    input_variant = inf_cfg.get(
        "input_image_variant",
        config.get("satellites", {}).get("input_image_variant", "linear_magnitude"),
    )
    resolver = PathResolver(config)
    ctx = _prepare_sam3_context(config)
    append_metrics_to_masks = ctx["append_metrics_to_masks"]
    runner = ctx["runner"]
    streams_flt = ctx["streams_filter"]
    sat_prior_flt = ctx["satellite_prior_filter"]
    sat_core_flt = ctx["satellite_core_filter"]
    conflict_flt = ctx["conflict_filter"]
    satellite_pipeline_runner = ctx["satellite_pipeline_runner"]
    prompts = ctx["prompts"]
    H_work = ctx["H_work"]
    W_work = ctx["W_work"]
    on_new_path = ctx["on_new_path"]

    conditions = sorted(resolver.get_active_conditions(), key=lambda c: _condition_sort_key(resolver, c))
    should_rebuild = force or (force_variants is not None)
    stats = {"processed": 0, "skipped_exists": 0, "skipped_no_render": 0, "cloned": 0}

    for key in base_keys:
        clean_gt_dir = resolver.get_pseudo_gt_dir(key, dataset=resolver.dataset_name, condition="clean")
        clean_final_masks: list[dict[str, Any]] | None = None

        if "clean" not in conditions and (clean_gt_dir / "sam3_predictions_post.json").exists():
            _, clean_final_masks = load_predictions_json(clean_gt_dir / "sam3_predictions_post.json")

        for condition in conditions:
            diag_dir = resolver.get_diagnostics_dir(key, dataset=resolver.dataset_name, condition=condition)
            gt_dir = resolver.get_pseudo_gt_dir(key, dataset=resolver.dataset_name, condition=condition)
            label_mode = resolver.get_label_mode(condition)

            if on_new_path:
                diag_complete = all(
                    (diag_dir / f).exists() for f in _pnbody_diagnostic_artifacts_for_path(True)
                )
                gt_complete = (
                    label_mode != "authoritative" or _new_gt_complete(gt_dir)
                )
                already_done = diag_complete and gt_complete
            else:
                already_done = _pnbody_diag_complete(diag_dir) and _pnbody_gt_complete(gt_dir)
            if already_done and not should_rebuild:
                stats["skipped_exists"] += 1
                continue

            render_path = resolver.get_render_dir(
                input_variant,
                key,
                dataset=resolver.dataset_name,
                condition=condition,
            ) / "0000.png"
            if not render_path.exists():
                logger.warning("Render not found for %s/%s: %s", condition, key, render_path)
                stats["skipped_no_render"] += 1
                continue

            diag_dir.mkdir(parents=True, exist_ok=True)
            image_pil = Image.open(render_path).convert("RGB")
            image_np = np.array(image_pil)

            masks, time_ms = runner.run(image_pil, prompts)
            if masks:
                append_metrics_to_masks(masks, H_work, W_work, compute_hull=True)
                # F19: stamp stable IDs once on the combined raw mask list so
                # raw_index is unique across BOTH stream-prompt and
                # satellite-prompt candidates. Required because PNbody runs
                # both prompts in a single runner.run; export/eval index by
                # raw_index and need a unique key.
                assign_stable_ids(masks)

            if on_new_path:
                # New-path PNbody: tidal_features = stream-prompt outputs (no
                # cross-class conflict, no core stage); satellites + inner_galaxy
                # come from the SatellitePipelineRunner with conflict OFF.
                tidal_raw = [
                    m for m in masks
                    if normalize_type_label(m.get("type_label", "satellites")) == TIDAL_FEATURES
                ]
                sat_raw = [
                    m for m in masks
                    if normalize_type_label(m.get("type_label", "satellites")) == SATELLITES
                ]
                # Force the canonical type_label so downstream writers see "tidal_features".
                for m in tidal_raw:
                    m["type_label"] = TIDAL_FEATURES
                for m in sat_raw:
                    m["type_label"] = SATELLITES

                save_predictions_json(
                    diag_dir / "sam3_predictions_raw.json",
                    masks, H_work, W_work, engine="sam3", layer="raw",
                )
                save_predictions_json(
                    diag_dir / "streams_predictions_raw.json",
                    tidal_raw, H_work, W_work, engine="sam3", layer="tidal_raw",
                )
                save_predictions_json(
                    diag_dir / "satellites_predictions_raw.json",
                    sat_raw, H_work, W_work, engine="sam3", layer="satellites_raw",
                )

                kept_tidal, _ = streams_flt.filter(tidal_raw, H_work, W_work)
                pipeline_result = satellite_pipeline_runner.run(
                    sat_raw,
                    streams_gt_map=np.zeros((H_work, W_work), dtype=np.int32),
                    H=H_work, W=W_work, base_key=str(key),
                )
                kept_sats = pipeline_result.final_sats
                kept_inner = pipeline_result.final_inner_galaxy

                save_predictions_json(
                    diag_dir / "streams_predictions_post_filter.json",
                    kept_tidal, H_work, W_work, engine="sam3", layer="tidal_post_filter",
                )
                save_predictions_json(
                    diag_dir / "satellites_predictions_post_filter.json",
                    kept_sats + kept_inner, H_work, W_work, engine="sam3",
                    layer="satellites_post_filter",
                )

                post_masks_new = kept_tidal + kept_sats + kept_inner
                save_predictions_json(
                    diag_dir / "sam3_predictions_post.json",
                    post_masks_new, H_work, W_work, engine="sam3", layer="post",
                )

                diag_manifest_new = {
                    "dataset": resolver.dataset_name,
                    "condition": condition,
                    "base_key": str(key),
                    "label_mode": label_mode,
                    "gt_path_version": "tidal_v1",
                    "sam3_n_raw": len(masks),
                    "sam3_n_tidal_raw": len(tidal_raw),
                    "sam3_n_satellites_raw": len(sat_raw),
                    "sam3_n_tidal_post_filter": len(kept_tidal),
                    "sam3_n_satellites_post_filter": len(kept_sats),
                    "sam3_n_inner_galaxy": len(kept_inner),
                    "sam3_inference_time_ms": round(time_ms, 2),
                    "satellite_pipeline_summary": pipeline_result.image_summary,
                    "updated_at": datetime.now().isoformat(),
                }
                _write_json(diag_dir / "manifest.json", diag_manifest_new)

                if label_mode == "authoritative":
                    gt_dir.mkdir(parents=True, exist_ok=True)
                    save_predictions_json(
                        gt_dir / "sam3_predictions_raw.json",
                        masks, H_work, W_work, engine="sam3", layer="raw",
                    )
                    save_predictions_json(
                        gt_dir / "sam3_predictions_post.json",
                        post_masks_new, H_work, W_work, engine="sam3", layer="post",
                    )
                    masks_by_type = {
                        TIDAL_FEATURES: kept_tidal,
                        SATELLITES: kept_sats,
                        INNER_GALAXY: kept_inner,
                    }
                    save_per_class_pseudo_gt(gt_dir, masks_by_type, H_work, W_work)
                    save_pseudo_label_overlay(
                        gt_dir / "sam3_pseudo_label_overlay.png",
                        image_np, post_masks_new,
                    )
                    instance_map_qa = np.array(Image.open(gt_dir / "instance_map_uint8.png"))
                    save_instance_overlay(gt_dir / "overlay.png", image_np, instance_map_qa)
                    # Augment the GT manifest with diag fields for parity with legacy.
                    gt_manifest_path = gt_dir / "manifest.json"
                    gt_manifest = (
                        json.loads(gt_manifest_path.read_text())
                        if gt_manifest_path.exists()
                        else {}
                    )
                    gt_manifest.update({
                        **diag_manifest_new,
                        "label_mode": "authoritative",
                        "gt_source": "sam3_clean_per_class_post",
                    })
                    gt_manifest_path.write_text(json.dumps(gt_manifest, indent=2))

                    if condition == "clean":
                        clean_final_masks = deepcopy(post_masks_new)
                else:
                    if clean_final_masks is None:
                        if not (clean_gt_dir / "sam3_predictions_post.json").exists():
                            raise FileNotFoundError(
                                f"Clean authoritative pseudo GT missing for {key}; "
                                f"cannot clone into {condition}"
                            )
                        _, clean_final_masks = load_predictions_json(
                            clean_gt_dir / "sam3_predictions_post.json"
                        )
                    _clone_clean_gt_to_condition(
                        resolver, key, condition, clean_gt_dir, clean_final_masks, image_np,
                    )
                    stats["cloned"] += 1

                stats["processed"] += 1
                if stats["processed"] % 10 == 0:
                    logger.info("PNbody SAM3 progress: %d condition-runs", stats["processed"])
                continue

            stream_masks = [m for m in masks if m.get("type_label") == "streams"]
            sat_masks = [m for m in masks if m.get("type_label") == "satellites"]

            save_predictions_json(
                diag_dir / "sam3_predictions_raw.json",
                masks,
                H_work,
                W_work,
                engine="sam3",
                layer="raw",
            )
            save_predictions_json(
                diag_dir / "streams_predictions_raw.json",
                stream_masks,
                H_work,
                W_work,
                engine="sam3",
                layer="streams_raw",
            )
            save_predictions_json(
                diag_dir / "satellites_predictions_raw.json",
                sat_masks,
                H_work,
                W_work,
                engine="sam3",
                layer="satellites_raw",
            )

            kept_streams, rej_streams = streams_flt.filter(stream_masks, H_work, W_work)
            if sat_masks:
                kept_sat_prior, rej_sat_prior, ambig_sat = sat_prior_flt.filter(sat_masks)
                kept_sat_post_filter, core_hits, sat_core_diag = sat_core_flt.filter(
                    kept_sat_prior,
                    H_work,
                    W_work,
                )
            else:
                kept_sat_post_filter = []
                rej_sat_prior = []
                ambig_sat = []
                core_hits = []
                sat_core_diag = {}

            save_predictions_json(
                diag_dir / "streams_predictions_post_filter.json",
                kept_streams,
                H_work,
                W_work,
                engine="sam3",
                layer="streams_post_filter",
            )
            save_predictions_json(
                diag_dir / "satellites_predictions_post_filter.json",
                kept_sat_post_filter,
                H_work,
                W_work,
                engine="sam3",
                layer="satellites_post_filter",
            )

            pre_conflict_masks = kept_streams + kept_sat_post_filter
            save_predictions_json(
                diag_dir / "dual_predictions_pre_conflict.json",
                pre_conflict_masks,
                H_work,
                W_work,
                engine="sam3",
                layer="pre_conflict",
            )
            save_pseudo_label_overlay(
                diag_dir / "dual_overlay_pre_conflict.png",
                image_np,
                pre_conflict_masks,
            )

            conflict_result = conflict_flt.filter(
                kept_streams,
                kept_sat_post_filter,
                H_work,
                W_work,
                streams_filter=streams_flt,
                satellite_prior_filter=sat_prior_flt,
                satellite_core_filter=sat_core_flt,
            )
            final_streams = conflict_result["streams"]
            final_sats = conflict_result["satellites"]
            post_masks = final_streams + final_sats

            save_predictions_json(
                diag_dir / "dual_predictions_post_conflict.json",
                post_masks,
                H_work,
                W_work,
                engine="sam3",
                layer="post_conflict",
            )
            save_predictions_json(
                diag_dir / "sam3_predictions_post.json",
                post_masks,
                H_work,
                W_work,
                engine="sam3",
                layer="post_conflict",
            )
            save_pseudo_label_overlay(
                diag_dir / "dual_overlay_post_conflict.png",
                image_np,
                post_masks,
            )

            diag_manifest = {
                "dataset": resolver.dataset_name,
                "condition": condition,
                "base_key": str(key),
                "label_mode": label_mode,
                "policy": conflict_result["report"]["policy"],
                "sam3_n_raw": len(masks),
                "sam3_n_streams_raw": len(stream_masks),
                "sam3_n_satellites_raw": len(sat_masks),
                "sam3_n_streams_post_filter": len(kept_streams),
                "sam3_n_satellites_post_filter": len(kept_sat_post_filter),
                "sam3_n_streams_final": len(final_streams),
                "sam3_n_satellites_final": len(final_sats),
                "sam3_inference_time_ms": round(time_ms, 2),
                "streams_rejected_pre_conflict": len(rej_streams),
                "satellites_rejected_prior_pre_conflict": len(rej_sat_prior),
                "satellites_ambiguous_pre_conflict": len(ambig_sat),
                "satellites_rejected_core_pre_conflict": len(core_hits),
                "satellite_core_diag_pre_conflict": sat_core_diag,
                "conflict_action_counts": conflict_result["report"].get("action_counts", {}),
                "updated_at": datetime.now().isoformat(),
            }
            _write_json(diag_dir / "conflict_resolution.json", conflict_result["report"])
            _write_json(diag_dir / "manifest.json", diag_manifest)

            if label_mode == "authoritative":
                gt_dir.mkdir(parents=True, exist_ok=True)
                save_predictions_json(
                    gt_dir / "sam3_predictions_raw.json",
                    masks,
                    H_work,
                    W_work,
                    engine="sam3",
                    layer="raw",
                )
                save_predictions_json(
                    gt_dir / "sam3_predictions_post.json",
                    post_masks,
                    H_work,
                    W_work,
                    engine="sam3",
                    layer="post_conflict",
                )
                instance_map, _ = save_pseudo_gt(
                    gt_dir,
                    post_masks,
                    H_work,
                    W_work,
                    overlap_policy="stream_first",
                    write_id_map=True,
                )
                save_pseudo_label_overlay(gt_dir / "sam3_pseudo_label_overlay.png", image_np, post_masks)
                save_instance_overlay(gt_dir / "overlay.png", image_np, instance_map)
                _write_json(
                    gt_dir / "manifest.json",
                    {
                        **diag_manifest,
                        "label_mode": "authoritative",
                        "gt_source": "sam3_clean_post_conflict",
                    },
                )
                if condition == "clean":
                    clean_final_masks = deepcopy(post_masks)
            else:
                if clean_final_masks is None:
                    if not (clean_gt_dir / "sam3_predictions_post.json").exists():
                        raise FileNotFoundError(
                            f"Clean authoritative pseudo GT missing for {key}; "
                            f"cannot clone into {condition}"
                        )
                    _, clean_final_masks = load_predictions_json(clean_gt_dir / "sam3_predictions_post.json")
                _clone_clean_gt_to_condition(
                    resolver,
                    key,
                    condition,
                    clean_gt_dir,
                    clean_final_masks,
                    image_np,
                )
                stats["cloned"] += 1

            stats["processed"] += 1
            if stats["processed"] % 10 == 0:
                logger.info("PNbody SAM3 progress: %d condition-runs", stats["processed"])

    logger.info(
        "PNbody SAM3 pseudo_label: %d processed, %d cloned, %d skipped",
        stats["processed"],
        stats["cloned"],
        stats["skipped_exists"],
    )


def run_inference_sam3(
    config: dict[str, Any],
    base_keys: list[BaseKey],
    logger: logging.Logger,
    force: bool = False,
    force_variants: set[str] | None = None,
) -> None:
    """SAM3 text-prompt -> type-aware filter -> evaluate or pseudo_label."""
    inf_cfg = config.get("inference_phase", {})
    run_mode = inf_cfg.get("run_mode", "evaluate")
    dataset_name = config.get("dataset_name", "dr1")

    if dataset_name == "pnbody" and run_mode == "pseudo_label":
        _run_pnbody_clean_truth_noisy_clone(
            config,
            base_keys,
            logger,
            force=force,
            force_variants=force_variants,
        )
        return

    _run_standard_inference_sam3(
        config,
        base_keys,
        logger,
        force=force,
        force_variants=force_variants,
    )
