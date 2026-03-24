#!/usr/bin/env python3
"""
plot_recall_curve.py
Generate Recall vs SB Threshold plots from SAM2 evaluation JSON.

Usage:
    python scripts/plot_recall_curve.py data/03_results/sam2_eval/iou_results_*.json
    python scripts/plot_recall_curve.py --dir data/03_results/sam2_eval/

Output: recall_curve_{timestamp}.png saved to same directory as input JSON.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_results(json_path: Path) -> dict:
    """Load evaluation JSON."""
    with open(json_path) as f:
        return json.load(f)


def aggregate_recall_by_threshold(results: list) -> dict:
    """
    Group recall by SB threshold and type.
    
    Returns:
        {type: {sb_threshold: [recall_values, ...]}}
    """
    grouped = defaultdict(lambda: defaultdict(list))
    for r in results:
        sb = r['sb_threshold']
        t = r['type']
        grouped[t][sb].append(r['recall'])
    return grouped


def plot_recall_curve(grouped: dict, output_path: Path, title: str = "Recall vs SB Threshold"):
    """Generate line plot with mean and std error bands."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)
    
    colors = {'streams': '#2ecc71', 'satellites': '#e74c3c', 'all': '#3498db'}
    markers = {'streams': 'o', 'satellites': 's', 'all': '^'}
    
    for type_name, sb_data in grouped.items():
        thresholds = sorted(sb_data.keys())
        means = [np.mean(sb_data[sb]) for sb in thresholds]
        stds = [np.std(sb_data[sb]) for sb in thresholds]
        
        ax.plot(thresholds, means, 
                color=colors.get(type_name, '#9b59b6'),
                marker=markers.get(type_name, 'o'),
                linewidth=2, markersize=8, label=type_name.capitalize())
        
        # Std error band
        ax.fill_between(thresholds,
                        np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds),
                        alpha=0.2, color=colors.get(type_name, '#9b59b6'))
    
    ax.set_xlabel('SB Threshold (mag/arcsec²)', fontsize=12)
    ax.set_ylabel('Recall', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.05)
    
    # Discrete x-axis: show only actual SB thresholds
    all_thresholds = sorted({sb for d in grouped.values() for sb in d.keys()})
    ax.set_xticks(all_thresholds)
    ax.set_xticklabels([f'{t:.1f}' if t % 1 else f'{int(t)}' for t in all_thresholds], rotation=45)
    
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Plot Recall vs SB Threshold curve')
    parser.add_argument('json_file', nargs='?', type=str, help='Path to iou_results_*.json')
    parser.add_argument('--dir', type=str, help='Directory containing JSON results (uses latest)')
    args = parser.parse_args()
    
    # Find JSON file
    if args.json_file:
        json_path = Path(args.json_file)
    elif args.dir:
        json_dir = Path(args.dir)
        json_files = sorted(json_dir.glob('iou_results_*.json'))
        if not json_files:
            print(f"No iou_results_*.json found in {json_dir}")
            sys.exit(1)
        json_path = json_files[-1]  # Latest
    else:
        print("Provide JSON file path or --dir")
        sys.exit(1)
    
    print(f"Loading: {json_path}")
    data = load_results(json_path)
    
    results = data.get('per_sample_results', [])
    if not results:
        print("No results in JSON")
        sys.exit(1)
    
    # Aggregate
    grouped = aggregate_recall_by_threshold(results)
    
    # Also compute "all" (combined)
    all_data = defaultdict(list)
    for t_data in grouped.values():
        for sb, recalls in t_data.items():
            all_data[sb].extend(recalls)
    grouped['all'] = all_data
    
    # Output path
    out_path = json_path.parent / f"recall_curve_{json_path.stem.split('_')[-1]}.png"
    
    # Extract config info for title
    cfg = data.get('config', {})
    ckpt = Path(cfg.get('checkpoint', 'unknown')).stem
    title = f"Recall vs SB Threshold\n(ckpt: {ckpt[:30]}...)"
    
    plot_recall_curve(grouped, out_path, title)
    
    # Print summary table
    print("\nRecall Summary:")
    print("-" * 50)
    print(f"{'SB':<8} {'Streams':<12} {'Satellites':<12} {'All':<12}")
    print("-" * 50)
    for sb in sorted(grouped['all'].keys()):
        s_recall = np.mean(grouped.get('streams', {}).get(sb, [0])) if grouped.get('streams', {}).get(sb) else 0
        sat_recall = np.mean(grouped.get('satellites', {}).get(sb, [0])) if grouped.get('satellites', {}).get(sb) else 0
        all_recall = np.mean(grouped['all'][sb])
        print(f"{sb:<8.1f} {s_recall:<12.4f} {sat_recall:<12.4f} {all_recall:<12.4f}")


if __name__ == "__main__":
    main()
