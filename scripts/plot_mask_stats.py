"""
Visualize mask statistics comparing Streams vs Satellites.
Generates:
1. mask_stats_distributions.png: Boxplots for Solidity, Aspect Sym (Moment), Aspect Sym (Boundary), Curvature Ratio.
2. mask_stats_trends.png: Trends of median metrics over SB thresholds.

Usage:
    python scripts/plot_mask_stats.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Configuration
INPUT_CSV = "outputs/mask_stats/mask_instance_stats.csv"
OUTPUT_DIR = "outputs/mask_stats"
THEME_BG = "#2b1641" # Dark purple
THEME_FG = "#ffffff" # White
# Custom palette
COLOR_STREAMS = "#00ffff" # Cyan
COLOR_SATELLITES = "#ff00ff" # Magenta
PALETTE = {
    'streams': COLOR_STREAMS,
    'satellites': COLOR_SATELLITES
}

def setup_theme():
    """Applies the requested dark theme."""
    plt.style.use('dark_background')
    plt.rcParams['figure.facecolor'] = THEME_BG
    plt.rcParams['axes.facecolor'] = THEME_BG
    plt.rcParams['savefig.facecolor'] = THEME_BG
    plt.rcParams['text.color'] = THEME_FG
    plt.rcParams['axes.labelcolor'] = THEME_FG
    plt.rcParams['xtick.color'] = THEME_FG
    plt.rcParams['ytick.color'] = THEME_FG
    plt.rcParams['grid.color'] = '#554466'
    plt.rcParams['grid.alpha'] = 0.3
    plt.rcParams['axes.edgecolor'] = THEME_FG
    plt.rcParams['patch.edgecolor'] = THEME_FG

def plot_distributions(df):
    """Generates side-by-side boxplots for shape metrics."""
    metric_cols = [
        ('solidity',            'Solidity'),
        ('aspect_sym_moment',   'Aspect Sym (Moment)'),
        ('aspect_sym_boundary', 'Aspect Sym (Boundary)'),
        ('curvature_ratio',     'Curvature Ratio'),
    ]
    fig, axes = plt.subplots(1, len(metric_cols), figsize=(7 * len(metric_cols), 6))
    features = ['streams', 'satellites']

    for ax, (col, label) in zip(axes, metric_cols):
        data = [df[df['feature_type'] == f][col].dropna().values for f in features]
        bplot = ax.boxplot(
            data, patch_artist=True, labels=[f.capitalize() for f in features],
            boxprops=dict(linewidth=1.5),
            medianprops=dict(color='white', linewidth=2),
            whiskerprops=dict(color=THEME_FG),
            capprops=dict(color=THEME_FG),
            flierprops=dict(marker='o', markerfacecolor=THEME_FG, markeredgecolor='none', alpha=0.5),
        )
        for patch, feature in zip(bplot['boxes'], features):
            patch.set_facecolor(PALETTE[feature])
            patch.set_alpha(0.8)
        ax.set_title(f'{label} Distribution', color=THEME_FG, fontsize=14, fontweight='bold')
        ax.set_ylabel(label, fontsize=12)
        ax.grid(True, axis='y')

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'mask_stats_distributions.png')
    plt.savefig(out_path, dpi=150)
    print(f"[Done] Saved distribution plot to {out_path}")
    plt.close()

def plot_trends(df):
    """Generates trend lines for metrics across SB thresholds."""
    metrics = ['area', 'solidity', 'aspect_sym_moment', 'aspect_sym_boundary', 'curvature_ratio', 'circularity']
    
    # Aggregate (skip NaN columns gracefully)
    agg_df = df.groupby(['feature_type', 'sb_threshold'])[metrics].median().reset_index()
    agg_df = agg_df.sort_values(by='sb_threshold')
    
    ncols = 3
    nrows = 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 5 * nrows))
    axes = axes.flatten()

    for idx, metric in enumerate(metrics):
        ax = axes[idx]
        
        for feature in ['streams', 'satellites']:
            subset = agg_df[agg_df['feature_type'] == feature]
            if subset.empty:
                continue
                
            ax.plot(subset['sb_threshold'], subset[metric], 
                   marker='o', linewidth=2.5, markersize=8,
                   color=PALETTE[feature], label=feature.capitalize())
        
        ax.set_title(f'Median {metric.capitalize()} vs SB Threshold', color=THEME_FG, fontsize=12, fontweight='bold')
        ax.set_xlabel('SB Threshold (mag/arcsec^2)')
        ax.set_ylabel(f'Median {metric.capitalize()}')
        ax.grid(True)
        ax.legend()
        
        # Ensure integer ticks for SB if few unique values
        unique_sb = agg_df['sb_threshold'].unique()
        if len(unique_sb) < 10:
            ax.set_xticks(unique_sb)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, 'mask_stats_id_trends.png')
    plt.savefig(out_path, dpi=150)
    print(f"[Done] Saved trends plot to {out_path}")
    plt.close()

def main():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: {INPUT_CSV} not found. Please run the stats generation script first.")
        return

    print(f"Loading data from {INPUT_CSV}...")
    df = pd.read_csv(INPUT_CSV)
    
    # Cap extreme outliers to prevent visualization collapse (aspect ratios > 100)
    clip_cols = [c for c in ['aspect_sym_boundary', 'aspect_sym_moment', 'aspect_sym'] if c in df.columns]
    if clip_cols:
        df[clip_cols] = df[clip_cols].where(df[clip_cols] <= 100, np.nan)
        
    print(f"Loaded {len(df)} mask instances.")
    
    setup_theme()
    plot_distributions(df)
    plot_trends(df)

if __name__ == "__main__":
    main()
