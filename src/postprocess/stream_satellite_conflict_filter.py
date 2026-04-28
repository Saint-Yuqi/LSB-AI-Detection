"""
Cross-type conflict resolution between stream and satellite masks.

Designed for PNbody pseudo-label generation:
    - preserve dual-channel predictions pre-conflict for diagnostics
    - apply a final stream-first policy for authoritative clean pseudo GT
    - trim or drop overlapping satellites instead of silently overwriting pixels
"""
from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from src.analysis.mask_metrics import append_metrics_to_masks


class StreamSatelliteConflictFilter:
    """Resolve overlapping stream/satellite masks with a stream-first bias."""

    def __init__(
        self,
        policy: str = "stream_first",
        keep_stream_aspect_min: float = 1.9,
        keep_stream_curvature_min: float = 1.2,
        keep_stream_area_ratio: float = 1.5,
        drop_compact_stream_overlap: float = 0.75,
        satellite_solidity_min: float = 0.83,
        satellite_aspect_max: float = 1.75,
    ) -> None:
        self.policy = policy
        self.keep_stream_aspect_min = keep_stream_aspect_min
        self.keep_stream_curvature_min = keep_stream_curvature_min
        self.keep_stream_area_ratio = keep_stream_area_ratio
        self.drop_compact_stream_overlap = drop_compact_stream_overlap
        self.satellite_solidity_min = satellite_solidity_min
        self.satellite_aspect_max = satellite_aspect_max

    def filter(
        self,
        streams: list[dict[str, Any]],
        satellites: list[dict[str, Any]],
        H: int,
        W: int,
        streams_filter: Any | None = None,
        satellite_prior_filter: Any | None = None,
        satellite_core_filter: Any | None = None,
    ) -> dict[str, Any]:
        work_streams = [deepcopy(m) for m in streams]
        work_sats = [deepcopy(m) for m in satellites]
        actions: list[dict[str, Any]] = []

        for s_idx, stream in enumerate(work_streams):
            stream_seg = stream.get("segmentation")
            if stream_seg is None:
                continue
            stream_bool = stream_seg.astype(bool)
            if stream_bool.sum() == 0:
                continue

            for sat_idx, satellite in enumerate(work_sats):
                sat_seg = satellite.get("segmentation")
                if sat_seg is None:
                    continue
                sat_bool = sat_seg.astype(bool)
                if sat_bool.sum() == 0:
                    continue

                overlap = stream_bool & sat_bool
                overlap_px = int(overlap.sum())
                if overlap_px == 0:
                    continue

                stream_area = max(
                    int(stream.get("area_clean", stream.get("area", stream_bool.sum()))),
                    1,
                )
                sat_area = max(
                    int(satellite.get("area_clean", satellite.get("area", sat_bool.sum()))),
                    1,
                )
                overlap_ratio_stream = overlap_px / stream_area
                overlap_ratio_sat = overlap_px / sat_area

                stream_aspect = float(stream.get("aspect_sym_moment") or stream.get("aspect_sym") or 1.0)
                stream_curvature = float(stream.get("curvature_ratio") or 1.0)
                sat_solidity = float(satellite.get("solidity") or 1.0)
                sat_aspect = float(satellite.get("aspect_sym_moment") or satellite.get("aspect_sym") or 1.0)

                stream_like = (
                    stream_aspect >= self.keep_stream_aspect_min
                    or stream_curvature >= self.keep_stream_curvature_min
                    or stream_area >= sat_area * self.keep_stream_area_ratio
                )
                satellite_compact = (
                    sat_solidity >= self.satellite_solidity_min
                    and sat_aspect <= self.satellite_aspect_max
                )

                action = "keep_stream"
                sat_new = sat_bool
                stream_new = stream_bool

                if (
                    self.policy == "stream_first"
                    and not stream_like
                    and satellite_compact
                    and overlap_ratio_stream >= self.drop_compact_stream_overlap
                    and overlap_ratio_sat >= self.drop_compact_stream_overlap
                ):
                    action = "drop_stream"
                    stream_new = np.zeros_like(stream_bool, dtype=bool)
                    work_streams[s_idx]["segmentation"] = stream_new
                    stream_bool = stream_new
                else:
                    sat_new = sat_bool & ~stream_bool
                    if sat_new.sum() == 0:
                        action = "drop_satellite"
                    elif sat_new.sum() != sat_bool.sum():
                        action = "trim_satellite"
                    work_sats[sat_idx]["segmentation"] = sat_new

                actions.append({
                    "stream_index": s_idx,
                    "satellite_index": sat_idx,
                    "action": action,
                    "overlap_px": overlap_px,
                    "overlap_ratio_stream": round(overlap_ratio_stream, 4),
                    "overlap_ratio_satellite": round(overlap_ratio_sat, 4),
                    "stream_score": round(float(stream.get("score", 0.0)), 4),
                    "satellite_score": round(float(satellite.get("score", 0.0)), 4),
                    "stream_like": bool(stream_like),
                    "satellite_compact": bool(satellite_compact),
                })

                if action == "drop_stream":
                    break

        final_streams = [m for m in work_streams if m.get("segmentation") is not None and m["segmentation"].sum() > 0]
        final_sats = [m for m in work_sats if m.get("segmentation") is not None and m["segmentation"].sum() > 0]

        if final_streams:
            append_metrics_to_masks(final_streams, H, W, compute_hull=True)
        if final_sats:
            append_metrics_to_masks(final_sats, H, W, compute_hull=True)

        rejected_streams: list[dict[str, Any]] = []
        if streams_filter is not None and final_streams:
            final_streams, rejected_streams = streams_filter.filter(final_streams, H, W)

        rejected_sats_prior: list[dict[str, Any]] = []
        ambiguous_sats: list[dict[str, Any]] = []
        if satellite_prior_filter is not None and final_sats:
            final_sats, rejected_sats_prior, ambiguous_sats = satellite_prior_filter.filter(final_sats)

        rejected_sats_core: list[dict[str, Any]] = []
        sat_core_diag: dict[str, Any] = {}
        if satellite_core_filter is not None and final_sats:
            final_sats, rejected_sats_core, sat_core_diag = satellite_core_filter.filter(final_sats, H, W)

        action_counts: dict[str, int] = {}
        for entry in actions:
            action_counts[entry["action"]] = action_counts.get(entry["action"], 0) + 1

        return {
            "streams": final_streams,
            "satellites": final_sats,
            "rejected_streams": rejected_streams,
            "rejected_satellites_prior": rejected_sats_prior,
            "rejected_satellites_core": rejected_sats_core,
            "ambiguous_satellites": ambiguous_sats,
            "report": {
                "policy": self.policy,
                "n_conflicts": len(actions),
                "action_counts": action_counts,
                "entries": actions,
                "satellite_core_diag": sat_core_diag,
            },
        }
