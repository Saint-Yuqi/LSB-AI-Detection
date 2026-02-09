#!/usr/bin/env python3
"""
SAM2 Inference and IOU Evaluation Script

Runs SAM2 model inference on the sam2_prepared dataset and calculates
IOU (Intersection over Union) metrics against ground truth masks.

Usage:
    python sam2_inference_iou.py \
        --checkpoint /path/to/checkpoint.pt \
        --data_dir /home/yuqyan/Yuqi/scripts/sam2_prepared \
        --output_dir /home/yuqyan/Yuqi/scripts/sam2_eval_results

Author: Yuqi
Date: 2026-01-31
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from collections import defaultdict

import numpy as np
from PIL import Image
import torch

# Add sam2 to path if needed
SAM2_PATH = Path("/home/yuqyan/Yuqi/sam2")
if SAM2_PATH.exists():
    sys.path.insert(0, str(SAM2_PATH))

from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator


def parse_sample_name(name: str) -> dict:
    """
    Parse sample folder name like '00011_eo_SB27.5_streams'.
    Returns: {'galaxy_id': 11, 'orientation': 'eo', 'sb_threshold': 27.5, 'type': 'streams'}
    """
    pattern = r'^(\d+)_([ef]o)_SB([\d.]+)_(streams|satellites)$'
    match = re.match(pattern, name)
    if match:
        return {
            'galaxy_id': int(match.group(1)),
            'orientation': match.group(2),
            'sb_threshold': float(match.group(3)),
            'type': match.group(4)
        }
    return None


def calculate_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """
    Calculate IoU between predicted and ground truth masks.
    Both masks are binary (0 = background, >0 = foreground).
    """
    pred_binary = (pred_mask > 0).astype(bool)
    gt_binary = (gt_mask > 0).astype(bool)
    
    intersection = np.logical_and(pred_binary, gt_binary).sum()
    union = np.logical_or(pred_binary, gt_binary).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection) / float(union)


def calculate_instance_iou(pred_masks: list, gt_mask: np.ndarray) -> dict:
    """
    Calculate IOU metrics for instance segmentation.
    
    Args:
        pred_masks: List of predicted masks from SAM2 (each is a dict with 'segmentation')
        gt_mask: Ground truth mask with instance IDs (0=bg, 1,2,3...=instances)
    
    Returns:
        Dict with various IOU metrics
    """
    # Combine all predicted masks into one binary mask
    if len(pred_masks) == 0:
        combined_pred = np.zeros_like(gt_mask, dtype=bool)
    else:
        combined_pred = np.zeros_like(gt_mask, dtype=bool)
        for mask_data in pred_masks:
            combined_pred = np.logical_or(combined_pred, mask_data['segmentation'])
    
    # Binary IOU (overall coverage)
    binary_iou = calculate_iou(combined_pred.astype(np.uint8), gt_mask)
    
    # Instance-level analysis
    gt_instances = np.unique(gt_mask)
    gt_instances = gt_instances[gt_instances > 0]  # Remove background
    
    instance_ious = []
    for inst_id in gt_instances:
        inst_mask = (gt_mask == inst_id)
        inst_iou = calculate_iou(combined_pred.astype(np.uint8), inst_mask.astype(np.uint8))
        instance_ious.append(inst_iou)
    
    return {
        'binary_iou': binary_iou,
        'mean_instance_iou': np.mean(instance_ious) if instance_ious else 0.0,
        'num_gt_instances': len(gt_instances),
        'num_pred_masks': len(pred_masks),
        'instance_ious': instance_ious
    }


def load_image(path: Path) -> np.ndarray:
    """Load image as RGB numpy array."""
    img = Image.open(path).convert('RGB')
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    """Load mask as grayscale numpy array."""
    mask = Image.open(path)
    mask_arr = np.array(mask)
    # Handle multi-channel masks
    if len(mask_arr.shape) == 3:
        mask_arr = mask_arr[:, :, 0]
    return mask_arr


def collect_samples(data_dir: Path, sb_threshold: float = None, 
                    sample_type: str = None, max_samples: int = None) -> list:
    """
    Collect samples from the dataset directory.
    
    Args:
        data_dir: Path to sam2_prepared directory
        sb_threshold: Optional filter by SB threshold
        sample_type: Optional filter by 'streams' or 'satellites'
        max_samples: Optional limit on number of samples
    
    Returns:
        List of sample dicts with paths and metadata
    """
    img_folder = data_dir / 'img_folder'
    gt_folder = data_dir / 'gt_folder'
    
    samples = []
    
    for sample_dir in sorted(img_folder.iterdir()):
        if not sample_dir.is_dir():
            continue
        
        meta = parse_sample_name(sample_dir.name)
        if meta is None:
            continue
        
        # Apply filters
        if sb_threshold is not None and meta['sb_threshold'] != sb_threshold:
            continue
        if sample_type is not None and meta['type'] != sample_type:
            continue
        
        img_path = sample_dir / '0000.png'
        gt_path = gt_folder / sample_dir.name / '0000.png'
        
        if img_path.exists() and gt_path.exists():
            samples.append({
                'name': sample_dir.name,
                'img_path': img_path,
                'gt_path': gt_path,
                **meta
            })
        
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples


def save_visualization(image: np.ndarray, pred_masks: list, gt_mask: np.ndarray,
                       output_path: Path):
    """Save visualization of image, predictions, and ground truth."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    axes[0].axis('off')
    
    # Predicted masks
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
    
    # Ground truth
    gt_overlay = np.zeros((*image.shape[:2], 4))
    unique_ids = np.unique(gt_mask)
    unique_ids = unique_ids[unique_ids > 0]
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
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='SAM2 Inference and IOU Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SAM2 checkpoint file')
    parser.add_argument('--model_cfg', type=str, default='configs/sam2.1/sam2.1_hiera_l.yaml',
                        help='SAM2 model config (default: sam2.1_hiera_l.yaml)')
    parser.add_argument('--data_dir', type=str, 
                        default='/home/yuqyan/Yuqi/scripts/sam2_prepared',
                        help='Path to sam2_prepared directory')
    parser.add_argument('--output_dir', type=str,
                        default='/home/yuqyan/Yuqi/scripts/sam2_eval_results',
                        help='Output directory for results')
    parser.add_argument('--sb_threshold', type=float, default=None,
                        help='Filter by SB threshold (e.g., 27, 30, 32)')
    parser.add_argument('--type', type=str, choices=['streams', 'satellites'], default=None,
                        help='Filter by type (streams or satellites)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images')
    parser.add_argument('--points_per_side', type=int, default=32,
                        help='Points per side for automatic mask generation')
    parser.add_argument('--pred_iou_thresh', type=float, default=0.7,
                        help='Predicted IOU threshold for filtering masks')
    parser.add_argument('--stability_score_thresh', type=float, default=0.92,
                        help='Stability score threshold for filtering masks')
    args = parser.parse_args()
    
    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.save_vis:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"ERROR: Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Collect samples
    print(f"Collecting samples from {data_dir}...")
    samples = collect_samples(
        data_dir, 
        sb_threshold=args.sb_threshold,
        sample_type=args.type,
        max_samples=args.max_samples
    )
    print(f"Found {len(samples)} samples")
    
    if len(samples) == 0:
        print("No samples found. Check your filters and data directory.")
        sys.exit(1)
    
    # Load SAM2 model
    print(f"\nLoading SAM2 model...")
    print(f"  Config: {args.model_cfg}")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Device: {args.device}")
    
    sam2 = build_sam2(args.model_cfg, str(checkpoint_path), device=args.device)
    mask_generator = SAM2AutomaticMaskGenerator(
        sam2,
        points_per_side=args.points_per_side,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
    )
    
    # Process samples
    print(f"\nProcessing {len(samples)} samples...")
    results = []
    
    for i, sample in enumerate(samples):
        if (i + 1) % 10 == 0 or i == 0:
            print(f"  [{i+1}/{len(samples)}] {sample['name']}")
        
        # Load image and ground truth
        image = load_image(sample['img_path'])
        gt_mask = load_mask(sample['gt_path'])
        
        # Run SAM2 inference
        with torch.inference_mode():
            pred_masks = mask_generator.generate(image)
        
        # Calculate IOU metrics
        iou_metrics = calculate_instance_iou(pred_masks, gt_mask)
        
        # Store result
        result = {
            'sample_name': sample['name'],
            'galaxy_id': sample['galaxy_id'],
            'orientation': sample['orientation'],
            'sb_threshold': sample['sb_threshold'],
            'type': sample['type'],
            **{k: v for k, v in iou_metrics.items() if k != 'instance_ious'}
        }
        results.append(result)
        
        # Save visualization
        if args.save_vis:
            vis_path = vis_dir / f"{sample['name']}.png"
            save_visualization(image, pred_masks, gt_mask, vis_path)
    
    # Compute summary statistics
    print("\n" + "=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)
    
    binary_ious = [r['binary_iou'] for r in results]
    mean_inst_ious = [r['mean_instance_iou'] for r in results]
    
    print(f"\nOverall Statistics (n={len(results)}):")
    print(f"  Binary IOU:         mean={np.mean(binary_ious):.4f}, std={np.std(binary_ious):.4f}")
    print(f"  Mean Instance IOU:  mean={np.mean(mean_inst_ious):.4f}, std={np.std(mean_inst_ious):.4f}")
    
    # Group by type
    for sample_type in ['streams', 'satellites']:
        type_results = [r for r in results if r['type'] == sample_type]
        if type_results:
            type_binary = [r['binary_iou'] for r in type_results]
            type_inst = [r['mean_instance_iou'] for r in type_results]
            print(f"\n{sample_type.capitalize()} (n={len(type_results)}):")
            print(f"  Binary IOU:         mean={np.mean(type_binary):.4f}, std={np.std(type_binary):.4f}")
            print(f"  Mean Instance IOU:  mean={np.mean(type_inst):.4f}, std={np.std(type_inst):.4f}")
    
    # Group by SB threshold
    sb_groups = defaultdict(list)
    for r in results:
        sb_groups[r['sb_threshold']].append(r)
    
    print("\nBy SB Threshold:")
    print(f"  {'SB':<8} {'Count':<8} {'Binary IOU':<15} {'Instance IOU':<15}")
    print(f"  {'-'*8} {'-'*8} {'-'*15} {'-'*15}")
    for sb in sorted(sb_groups.keys()):
        group = sb_groups[sb]
        b_iou = np.mean([r['binary_iou'] for r in group])
        i_iou = np.mean([r['mean_instance_iou'] for r in group])
        print(f"  {sb:<8} {len(group):<8} {b_iou:<15.4f} {i_iou:<15.4f}")
    
    # Save results to JSON
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"iou_results_{timestamp}.json"
    
    summary = {
        'config': {
            'checkpoint': str(checkpoint_path),
            'model_cfg': args.model_cfg,
            'data_dir': str(data_dir),
            'sb_threshold_filter': args.sb_threshold,
            'type_filter': args.type,
            'points_per_side': args.points_per_side,
            'pred_iou_thresh': args.pred_iou_thresh,
            'stability_score_thresh': args.stability_score_thresh,
        },
        'summary': {
            'total_samples': len(results),
            'mean_binary_iou': float(np.mean(binary_ious)),
            'std_binary_iou': float(np.std(binary_ious)),
            'mean_instance_iou': float(np.mean(mean_inst_ious)),
            'std_instance_iou': float(np.std(mean_inst_ious)),
        },
        'per_sample_results': results
    }
    
    with open(results_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    if args.save_vis:
        print(f"Visualizations saved to: {vis_dir}")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
