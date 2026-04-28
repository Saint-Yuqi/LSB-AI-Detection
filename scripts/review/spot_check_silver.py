#!/usr/bin/env python3
"""Spot-check silver labels: distribution, signal audit, edge-case flags,
MV/EV cross-consistency, and physical sanity invariants.

Usage:
    python scripts/review/spot_check_silver.py \
        --mv   data/02_processed/review/silver_labels_satellite_mv.jsonl \
        --ev   data/02_processed/review/silver_labels_satellite_ev.jsonl \
        --sev  data/02_processed/review/silver_labels_stream_ev.jsonl \
        --policy configs/review/silver_policy_v1.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path) as fh:
        for line in fh:
            records.append(json.loads(line))
    return records


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    import statistics
    return {
        "count": len(values),
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
    }


def _print_header(title: str) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def check_satellite_mv(records: list[dict], policy: dict) -> int:
    """Check satellite_mv silver labels. Returns count of violations."""
    _print_header("SATELLITE_MV  (candidate-level)")
    violations = 0

    dist = Counter(r["decision_label"] for r in records)
    print(f"\n  Label distribution ({len(records)} total):")
    for label, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {label:20s}  {cnt:5d}  ({100*cnt/len(records):.1f}%)")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_label[r["decision_label"]].append(r)

    signal_keys = ["best_iou", "area_ratio", "boundary_f1", "confidence_score"]
    print("\n  Signal statistics per decision:")
    for label, recs in sorted(by_label.items()):
        print(f"\n    --- {label} ({len(recs)} records) ---")
        for sk in signal_keys:
            vals = [r["signals"][sk] for r in recs if r["signals"].get(sk) is not None]
            s = _stats(vals)
            if s["count"] > 0:
                print(f"      {sk:25s}  min={s['min']:.4f}  max={s['max']:.4f}  "
                      f"mean={s['mean']:.4f}  median={s['median']:.4f}")

    iou_thresh = policy.get("accept_iou_thresh", 0.7)
    conf_thresh = policy.get("confidence_thresh", 0.5)

    print(f"\n  Sanity invariants (accept_iou_thresh={iou_thresh}, "
          f"confidence_thresh={conf_thresh}):")

    bad_accept = [
        r for r in by_label.get("accept", [])
        if r["signals"]["best_iou"] < iou_thresh
        or (r["signals"].get("confidence_score") is not None
            and r["signals"]["confidence_score"] < conf_thresh)
    ]
    if bad_accept:
        violations += len(bad_accept)
        print(f"    FAIL: {len(bad_accept)} accepts with IoU<{iou_thresh} or "
              f"confidence<{conf_thresh}")
        for r in bad_accept[:5]:
            print(f"          {r['sample_id']} {r['candidate_key']}  "
                  f"IoU={r['signals']['best_iou']:.4f}  "
                  f"conf={r['signals'].get('confidence_score', 'N/A')}")
    else:
        print(f"    OK: all accepts have IoU>={iou_thresh} and confidence>={conf_thresh}")

    suspect_reject = [
        r for r in by_label.get("reject", [])
        if r["signals"]["best_iou"] > iou_thresh
        and r["signals"].get("area_ratio") is not None
        and 0.5 <= r["signals"]["area_ratio"] <= 2.0
        and r["signals"].get("filter_status") == "kept"
    ]
    if suspect_reject:
        print(f"    WARN: {len(suspect_reject)} rejects with IoU>{iou_thresh} "
              f"and normal area_ratio (may be legitimate filter_status rejects)")
        for r in suspect_reject[:5]:
            print(f"          {r['sample_id']} {r['candidate_key']}  "
                  f"IoU={r['signals']['best_iou']:.4f}  "
                  f"status={r['signals']['filter_status']}")
    else:
        print(f"    OK: no suspicious high-IoU rejects")

    edge_accept = [
        r for r in by_label.get("accept", [])
        if iou_thresh <= r["signals"]["best_iou"] < iou_thresh + 0.05
    ]
    edge_other = [
        r for r in records
        if r["decision_label"] != "accept"
        and iou_thresh - 0.05 <= r["signals"]["best_iou"] < iou_thresh
    ]
    if edge_accept or edge_other:
        print(f"\n  Edge cases near IoU threshold ({iou_thresh}):")
        print(f"    accepts with IoU in [{iou_thresh:.2f}, {iou_thresh+0.05:.2f}): "
              f"{len(edge_accept)}")
        print(f"    non-accepts with IoU in [{iou_thresh-0.05:.2f}, {iou_thresh:.2f}): "
              f"{len(edge_other)}")

    return violations


def check_ev(records: list[dict], family_name: str) -> int:
    """Check satellite_ev or stream_ev silver labels."""
    _print_header(f"{family_name.upper()}  (image-level)")
    violations = 0

    dist = Counter(r["decision_label"] for r in records)
    print(f"\n  Label distribution ({len(records)} total):")
    for label, cnt in sorted(dist.items(), key=lambda x: -x[1]):
        print(f"    {label:25s}  {cnt:5d}  ({100*cnt/len(records):.1f}%)")

    # Synthetic variant distribution
    variant_dist = Counter(
        r["signals"].get("synthetic_variant_id", "<legacy>") for r in records
    )
    print(f"\n  Synthetic variant distribution:")
    for vid, cnt in sorted(variant_dist.items(), key=lambda x: -x[1]):
        print(f"    {vid:25s}  {cnt:5d}")

    # Validate visible/hidden disjointness
    bad_overlap = 0
    for r in records:
        vis = set(r["signals"].get("visible_instance_ids") or [])
        hid = set(r["signals"].get("hidden_instance_ids") or [])
        if vis & hid:
            bad_overlap += 1
            if bad_overlap <= 3:
                print(f"    FAIL: {r['sample_id']} variant="
                      f"{r['signals'].get('synthetic_variant_id')} "
                      f"overlap={vis & hid}")
    if bad_overlap > 0:
        violations += bad_overlap
        print(f"  FAIL: {bad_overlap} records have visible/hidden overlap")
    else:
        print(f"\n    OK: visible_instance_ids ∩ hidden_instance_ids == ∅ for all")

    # confirm_empty must have zero instances
    bad_empty = [
        r for r in records
        if r["decision_label"] == "confirm_empty"
        and len(r["signals"].get("visible_instance_ids") or []) > 0
    ]
    if bad_empty:
        violations += len(bad_empty)
        print(f"    FAIL: {len(bad_empty)} confirm_empty with visible instances")
    else:
        print(f"    OK: all confirm_empty have empty visible set")

    return violations


def check_mv_ev_consistency(
    mv_records: list[dict], ev_records: list[dict],
) -> int:
    """Cross-check MV and EV label consistency.

    Only ``gt_complete`` satellite_ev records are used for the lookup
    to avoid multi-variant false positives.
    """
    _print_header("MV / EV CROSS-CONSISTENCY")
    violations = 0

    mv_by_sample: dict[str, list[dict]] = defaultdict(list)
    for r in mv_records:
        mv_by_sample[r["sample_id"]].append(r)

    # Only gt_complete variants for consistency check
    ev_by_sample = {
        r["sample_id"]: r for r in ev_records
        if r["signals"].get("synthetic_variant_id") in ("gt_complete", None)
    }

    all_accept_samples = set()
    for sid, mvs in mv_by_sample.items():
        if all(m["decision_label"] == "accept" for m in mvs):
            all_accept_samples.add(sid)

    consistent = 0
    inconsistent = 0
    missing_ev = 0
    for sid in all_accept_samples:
        ev = ev_by_sample.get(sid)
        if ev is None:
            missing_ev += 1
            continue
        if ev["decision_label"] == "confirm_complete":
            consistent += 1
        else:
            inconsistent += 1
            if inconsistent <= 5:
                print(f"  Inconsistency: {sid} has all-accept MV but "
                      f"EV={ev['decision_label']}  "
                      f"(variant={ev['signals'].get('synthetic_variant_id')})")

    total = len(all_accept_samples)
    print(f"\n  All-accept MV images: {total}")
    print(f"    EV=confirm_complete:  {consistent}")
    print(f"    EV=other:             {inconsistent}")
    print(f"    EV missing (abstain): {missing_ev}")

    if inconsistent > 0:
        print(f"\n  NOTE: Inconsistencies are expected when EV uses all preds "
              f"(satellites+streams) while MV only checks satellites.")

    mv_samples = set(mv_by_sample.keys())
    ev_samples = set(ev_by_sample.keys())
    ev_only = ev_samples - mv_samples
    mv_only = mv_samples - ev_samples
    print(f"\n  Coverage overlap:")
    print(f"    In both MV and EV:  {len(mv_samples & ev_samples)}")
    print(f"    MV only (no EV):    {len(mv_only)}")
    print(f"    EV only (no MV):    {len(ev_only)}")

    return violations


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mv", type=Path, required=True)
    parser.add_argument("--ev", type=Path, required=True)
    parser.add_argument("--sev", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    args = parser.parse_args(argv)

    import yaml
    with open(args.policy) as f:
        policy = yaml.safe_load(f).get("satellite_mv", {})

    mv_records = _load_jsonl(args.mv)
    ev_records = _load_jsonl(args.ev)
    sev_records = _load_jsonl(args.sev)

    total_violations = 0

    total_violations += check_satellite_mv(mv_records, policy)
    total_violations += check_ev(ev_records, "satellite_ev")
    total_violations += check_ev(sev_records, "stream_ev")
    total_violations += check_mv_ev_consistency(mv_records, ev_records)

    _print_header("SUMMARY")
    total = len(mv_records) + len(ev_records) + len(sev_records)
    print(f"\n  Total silver labels:  {total}")
    print(f"    satellite_mv:  {len(mv_records)}")
    print(f"    satellite_ev:  {len(ev_records)}")
    print(f"    stream_ev:     {len(sev_records)}")
    if total_violations > 0:
        print(f"\n  VIOLATIONS FOUND: {total_violations}")
        sys.exit(1)
    else:
        print(f"\n  All sanity checks PASSED")


if __name__ == "__main__":
    main()
