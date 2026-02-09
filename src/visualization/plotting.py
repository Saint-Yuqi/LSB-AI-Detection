"""
Visualization utilities for segmentation results.

Provides functions for visualizing predictions alongside ground truth.
"""

from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def save_visualization(
    image: np.ndarray,
    pred_masks: List[Dict],
    gt_mask: np.ndarray,
    output_path: Path,
    dpi: int = 100
) -> None:
    """
    Save a visualization comparing predictions to ground truth.
    
    Creates a 3-panel figure showing:
    1. Original input image
    2. SAM2 predicted masks (overlaid with distinct colors)
    3. Ground truth instances (overlaid with distinct colors)
    
    Args:
        image: Input RGB image array with shape (H, W, 3)
        pred_masks: List of predicted masks from SAM2.
            Each dict should have a 'segmentation' key with a binary mask.
        gt_mask: Ground truth mask with instance IDs (0=bg, 1,2,...=instances)
        output_path: Path to save the visualization image
        dpi: Output image resolution
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Panel 1: Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Panel 2: Predicted masks
    if len(pred_masks) > 0:
        pred_overlay = np.zeros((*image.shape[:2], 4))
        colors = plt.cm.tab20(np.linspace(0, 1, max(20, len(pred_masks))))
        
        for i, mask_data in enumerate(pred_masks):
            color = colors[i % len(colors)]
            mask = mask_data['segmentation']
            pred_overlay[mask, :3] = color[:3]
            pred_overlay[mask, 3] = 0.6
        
        axes[1].imshow(image)
        axes[1].imshow(pred_overlay)
    else:
        axes[1].imshow(image)
    
    axes[1].set_title(f'SAM2 Predictions ({len(pred_masks)} masks)')
    axes[1].axis('off')
    
    # Panel 3: Ground truth
    gt_overlay = np.zeros((*image.shape[:2], 4))
    unique_ids = np.unique(gt_mask)
    unique_ids = unique_ids[unique_ids > 0]  # Exclude background
    
    colors = plt.cm.Set1(np.linspace(0, 1, max(9, len(unique_ids))))
    
    for i, inst_id in enumerate(unique_ids):
        color = colors[i % len(colors)]
        inst_mask = (gt_mask == inst_id)
        gt_overlay[inst_mask, :3] = color[:3]
        gt_overlay[inst_mask, 3] = 0.6
    
    axes[2].imshow(image)
    axes[2].imshow(gt_overlay)
    axes[2].set_title(f'Ground Truth ({len(unique_ids)} instances)')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
