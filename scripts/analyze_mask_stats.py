#!/usr/bin/env python3
"""
Analyze instance mask statistics from SAM2 GT masks.

Usage:
    python scripts/analyze_mask_stats.py [--gt_root PATH] [--output_dir PATH]

Args:
    --gt_root: Root directory of GT masks (default: data/02_processed/sam2_prepared/gt_folder)
    --output_dir: Output directory for stats files (default: outputs/mask_stats)

Outputs:
    - mask_instance_stats.csv: Per-instance statistics
    - mask_stats_summary.json: Quantile summary by feature_type + SB threshold

Dependencies: numpy, PIL, scipy (for convex hull), tqdm
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


def parse_folder_name(folder_name: str) -> tuple | None:
    """Parse folder name: {galaxy_id}_{orientation}_SB{threshold}_{feature_type}"""
    pattern = r"^(\d+)_(eo|fo)_SB([\d.]+)_(streams|satellites)$"
    match = re.match(pattern, folder_name)
    if not match:
        return None
    return (match.group(1), match.group(2), float(match.group(3)), match.group(4))


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

    # Bounding box
    rows, cols = np.where(binary)
    y_min, y_max = rows.min(), rows.max()
    x_min, x_max = cols.min(), cols.max()
    bbox_h = int(y_max - y_min + 1)
    bbox_w = int(x_max - x_min + 1)

    aspect_ratio = bbox_w / bbox_h if bbox_h > 0 else 0.0
    aspect_sym = max(bbox_w / bbox_h, bbox_h / bbox_w) if min(bbox_w, bbox_h) > 0 else 1.0
    bbox_area = bbox_w * bbox_h
    extent = area / bbox_area if bbox_area > 0 else 0.0

    perimeter = compute_perimeter(binary)
    convex_area = compute_convex_area(binary)
    solidity = area / convex_area if convex_area > 0 else 0.0
    circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0.0

    return {
        "area": area,
        "bbox_w": bbox_w,
        "bbox_h": bbox_h,
        "aspect_ratio": round(aspect_ratio, 4),
        "aspect_sym": round(aspect_sym, 4),
        "extent": round(extent, 4),
        "perimeter": round(perimeter, 2),
        "solidity": round(solidity, 4),
        "circularity": round(circularity, 4),
    }


def scan_masks(gt_root: Path) -> list[dict]:
    """Scan all masks and extract per-instance statistics."""
    records = []
    mask_paths = sorted(gt_root.glob("*/0000.png"))
    total = len(mask_paths)

    for i, mask_path in enumerate(mask_paths):
        if i % 50 == 0:
            print(f"Processing {i}/{total}...")
        folder_name = mask_path.parent.name
        parsed = parse_folder_name(folder_name)
        if not parsed:
            continue

        galaxy_id, orientation, sb_threshold, feature_type = parsed
        mask = np.array(Image.open(mask_path))

        instance_ids = np.unique(mask)
        instance_ids = instance_ids[instance_ids > 0]

        for inst_id in instance_ids:
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
    metrics = ["area", "bbox_w", "bbox_h", "aspect_ratio", "aspect_sym", "extent", "perimeter", "solidity", "circularity"]

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
            vals = np.array([r[metric] for r in grp])
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
            vals = np.array([r[metric] for r in ft_records])
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
            "aspect_ratio_range": [q["aspect_ratio"]["p01"], q["aspect_ratio"]["p99"]],
            "aspect_sym_max": q["aspect_sym"]["p99"],
            "comment": f"Based on p01/p99 of {summary[key]['count']} instances",
        }
    summary["filter_recommendations"] = recommendations

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze instance mask statistics")
    parser.add_argument("--gt_root", type=Path, default=Path("data/02_processed/sam2_prepared/gt_folder"))
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
                  "area", "bbox_w", "bbox_h", "aspect_ratio", "aspect_sym", "extent", "perimeter", "solidity", "circularity"]
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
        print(f"  aspect_sym_max: {rec['aspect_sym_max']}")
        print(f"  aspect_ratio: {rec['aspect_ratio_range']}")


if __name__ == "__main__":
    main()
