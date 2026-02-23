#!/usr/bin/env python3
"""
Analyze instance mask statistics from canonical GT masks (unified pipeline).

Usage:
    conda run -n sam2 --no-capture-output python scripts/analyze_mask_stats.py [--gt_root PATH] [--output_dir PATH]

Args:
    --gt_root: Root directory of canonical GT (default: data/02_processed/gt_canonical/current)
    --output_dir: Output directory for stats files (default: outputs/mask_stats)

Outputs:
    - mask_instance_stats.csv: Per-instance statistics
    - mask_stats_summary.json: Quantile summary by feature_type + SB threshold

Dependencies: numpy, PIL, scipy, opencv-python-headless, scikit-image

Aspect ratio fields:
    aspect_sym_moment:   cov-eigenvalue axis ratio (rotation-invariant, global shape)
    aspect_sym_boundary: cv2.fitEllipse axis ratio (boundary-aware)
    curvature_ratio:     skeleton_len / ellipse_major (>1 = curved/bent)
    aspect_sym:          alias for aspect_sym_moment (backward compat)
"""
import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as ndimage_label
from scipy.spatial import ConvexHull


def parse_base_key(folder_name: str) -> tuple | None:
    """Parse BaseKey folder: {galaxy_id:05d}_{orientation} e.g. '00011_eo'."""
    pattern = r"^(\d+)_(eo|fo)$"
    match = re.match(pattern, folder_name)
    if not match:
        return None
    return (match.group(1).lstrip("0") or "0", match.group(2))


def compute_perimeter(binary: np.ndarray) -> float:
    """Compute perimeter by counting edge pixels (4-connectivity)."""
    # Pad to handle boundaries
    padded = np.pad(binary, 1, mode="constant", constant_values=0)
    # Count transitions in both directions
    h_edges = np.sum(padded[:-1, :] != padded[1:, :])
    v_edges = np.sum(padded[:, :-1] != padded[:, 1:])
    return float(h_edges + v_edges)


def compute_convex_area(binary: np.ndarray) -> float:
    """Compute convex hull area from boundary points."""
    coords = np.argwhere(binary)
    if len(coords) < 3:
        return float(np.sum(binary))  # Too few points for convex hull
    try:
        hull = ConvexHull(coords)
        return float(hull.volume)  # In 2D, volume is area
    except Exception:
        return float(np.sum(binary))


def compute_instance_stats(mask: np.ndarray, instance_id: int) -> dict | None:
    """Compute region properties using pure numpy/scipy."""
    binary = (mask == instance_id).astype(np.uint8)

    # Connected components - take largest
    labeled, num_features = ndimage_label(binary)
    if num_features == 0:
        return None

    # Find largest component
    component_sizes = np.bincount(labeled.ravel())
    component_sizes[0] = 0  # Ignore background
    largest = component_sizes.argmax()
    binary = (labeled == largest).astype(np.uint8)

    area = int(np.sum(binary))
    if area == 0:
        return None

    # Bounding box (kept for extent computation)
    rows, cols = np.where(binary)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)
    bbox_area = bbox_w * bbox_h
    extent = area / bbox_area if bbox_area > 0 else 0.0

    # Ellipse axis ratio: 2nd-order central moments (rotation-invariant)
    coords = np.column_stack((rows.astype(np.float64), cols.astype(np.float64)))
    if len(coords) >= 3:
        cov = np.cov(coords, rowvar=False)
        eigvals = np.linalg.eigvalsh(cov)
        eigvals = np.clip(eigvals, 1e-12, None)
        aspect_sym_moment = float(np.sqrt(eigvals[1] / eigvals[0]))  # major/minor
    else:
        aspect_sym_moment = 1.0

    # Ellipse axis ratio: cv2.fitEllipse on contour (boundary-aware)
    aspect_sym_boundary = None
    try:
        import cv2
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if contours:
            cnt = max(contours, key=len)
            if len(cnt) >= 5:
                (_, (w_ell, h_ell), _) = cv2.fitEllipse(cnt)
                if min(w_ell, h_ell) > 1e-6:
                    aspect_sym_boundary = float(max(w_ell, h_ell) / min(w_ell, h_ell))
    except ImportError:
        pass

    # Curvature: skeleton_length / ellipse_major_axis
    curvature_ratio = None
    try:
        from skimage.morphology import skeletonize
        if area >= 20:
            skel = skeletonize(binary.astype(bool))
            skel_len = float(np.sum(skel))
            if skel_len >= 1 and len(coords) >= 3:
                major_axis = 2.0 * np.sqrt(max(eigvals.max(), 1e-12))
                if major_axis >= 1:
                    curvature_ratio = float(skel_len / major_axis)
    except ImportError:
        pass

    perimeter = compute_perimeter(binary)
    convex_area = compute_convex_area(binary)
    solidity = area / convex_area if convex_area > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "aspect_sym_moment": round(aspect_sym_moment, 4),
        "aspect_sym_boundary": round(aspect_sym_boundary, 4) if aspect_sym_boundary is not None else None,
        "curvature_ratio": round(curvature_ratio, 4) if curvature_ratio is not None else None,
        "aspect_sym": round(aspect_sym_moment, 4),  # backward compat
        "extent": round(extent, 4),
        "perimeter": round(perimeter, 2),
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
    }


def scan_masks(gt_root: Path) -> list[dict]:
    """Scan canonical GT dirs and extract per-instance statistics.

    Expects gt_root/{BaseKey}/instance_map_uint8.png + instances.json + manifest.json.
    """
    records = []
    # Find all BaseKey dirs that contain the canonical merged mask
    mask_paths = sorted(gt_root.glob("*/instance_map_uint8.png"))
    total = len(mask_paths)

    for i, mask_path in enumerate(mask_paths):
        if i % 20 == 0:
            print(f"Processing {i}/{total}...")
        base_dir = mask_path.parent
        parsed = parse_base_key(base_dir.name)
        if not parsed:
            continue

        galaxy_id, orientation = parsed

        # Read instances.json for feature_type lookup
        instances_path = base_dir / "instances.json"
        if not instances_path.exists():
            continue
        with open(instances_path) as f:
            instances_list = json.load(f)
        # Build inst_id → feature_type map
        id_to_type = {inst["id"]: inst["type"] for inst in instances_list}

        # Read manifest.json for sb_threshold
        manifest_path = base_dir / "manifest.json"
        sb_threshold = 32.0  # fallback
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            sb_threshold = manifest.get("sb_threshold_used", 32.0)

        mask = np.array(Image.open(mask_path))

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
            feature_type = id_to_type.get(int(inst_id))
            if feature_type is None:
                continue  # unknown instance, skip
            stats = compute_instance_stats(mask, int(inst_id))
            if stats:
                records.append({
                    "galaxy_id": galaxy_id,
                    "orientation": orientation,
                    "sb_threshold": sb_threshold,
                    "feature_type": feature_type,
                    "instance_id": int(inst_id),
                    **stats,
                })
    return records


def compute_summary(records: list[dict]) -> dict:
    """Compute quantile summary grouped by (feature_type, sb_threshold)."""
    quantiles = [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99]
    metrics = ["area", "bbox_w", "bbox_h",
               "aspect_sym_moment", "aspect_sym_boundary", "curvature_ratio",
               "extent", "perimeter", "solidity", "circularity"]

    # Group by (feature_type, sb_threshold)
    groups = {}
    for r in records:
        key = (r["feature_type"], r["sb_threshold"])
        groups.setdefault(key, []).append(r)

    summary = {}
    for (ft, sb), grp in groups.items():
        key = f"{ft}_SB{sb}"
        summary[key] = {"count": len(grp), "quantiles": {}}
        for metric in metrics:
            vals_raw = [r[metric] for r in grp]
            vals = np.array([v for v in vals_raw if v is not None])
            if len(vals) == 0:
                continue
            summary[key]["quantiles"][metric] = {
                f"p{int(q*100):02d}": round(float(np.percentile(vals, q * 100)), 4)
                for q in quantiles
            }

    # Global stats per feature_type
    for ft in ["streams", "satellites"]:
        ft_records = [r for r in records if r["feature_type"] == ft]
        if not ft_records:
            continue
        key = f"{ft}_global"
        summary[key] = {"count": len(ft_records), "quantiles": {}}
        for metric in metrics:
            vals_raw = [r[metric] for r in ft_records]
            vals = np.array([v for v in vals_raw if v is not None])
            if len(vals) == 0:
                continue
            summary[key]["quantiles"][metric] = {
                f"p{int(q*100):02d}": round(float(np.percentile(vals, q * 100)), 4)
                for q in quantiles
            }

    # Recommendations
    recommendations = {}
    for ft in ["streams", "satellites"]:
        key = f"{ft}_global"
        if key not in summary:
            continue
        q = summary[key]["quantiles"]
        recommendations[ft] = {
            "min_area": q["area"]["p01"],
            "max_area": q["area"]["p99"],
            "min_solidity": q["solidity"]["p01"],
            "aspect_sym_moment_max": q.get("aspect_sym_moment", {}).get("p99", 999.0),
            "aspect_sym_boundary_max": q.get("aspect_sym_boundary", {}).get("p99", 999.0),
            "curvature_ratio_p99": q.get("curvature_ratio", {}).get("p99", 999.0),
            # backward compat alias
            "aspect_sym_max": q.get("aspect_sym_moment", {}).get("p99", 999.0),
            "comment": f"Based on p01/p99 of {summary[key]['count']} instances",
        }
    summary["filter_recommendations"] = recommendations

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze instance mask statistics")
    parser.add_argument("--gt_root", type=Path, default=Path("data/02_processed/gt_canonical/current"))
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/mask_stats"))
    args = parser.parse_args()

    gt_root = args.gt_root
    if not gt_root.is_absolute():
        gt_root = Path(__file__).parent.parent / gt_root

    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = Path(__file__).parent.parent / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Scanning masks from: {gt_root}")
    records = scan_masks(gt_root)

    if not records:
        print("No instances found!")
        return

    # CSV output
    csv_path = output_dir / "mask_instance_stats.csv"
    fieldnames = ["galaxy_id", "orientation", "sb_threshold", "feature_type", "instance_id",
                  "area", "bbox_w", "bbox_h",
                  "aspect_sym_moment", "aspect_sym_boundary", "curvature_ratio", "aspect_sym",
                  "extent", "perimeter", "solidity", "circularity"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved {len(records)} instances to {csv_path}")

    # Summary JSON
    summary = compute_summary(records)
    json_path = output_dir / "mask_stats_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {json_path}")

    # Quick preview
    print("\n=== Filter Recommendations ===")
    for ft, rec in summary.get("filter_recommendations", {}).items():
        print(f"\n{ft.upper()}:")
        print(f"  min_area: {rec['min_area']}")
        print(f"  max_area: {rec['max_area']}")
        print(f"  min_solidity: {rec['min_solidity']}")
        print(f"  aspect_sym_moment_max: {rec['aspect_sym_moment_max']}")
        print(f"  aspect_sym_boundary_max: {rec['aspect_sym_boundary_max']}")
        print(f"  curvature_ratio_p99: {rec['curvature_ratio_p99']}")


if __name__ == "__main__":
    main()
