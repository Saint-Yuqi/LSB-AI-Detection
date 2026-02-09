# eval_model.py
# SAM2 LSB Inference Evaluation
# Efficiency: Single inference per unique image, multiple GT comparisons.
# PRECISION: bf16 (Banned: fp32)
#
# Usage: python scripts/eval_model.py --config configs/eval_sam2.yaml

import json
import pickle
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.io import load_image, load_mask, parse_sample_name
from src.evaluation.metrics import calculate_instance_iou, calculate_matched_metrics
from src.utils.logger import setup_logger
from src.visualization.plotting import save_visualization


def collect_samples_grouped(
    data_dir: Path,
    sb_threshold: Optional[float] = None,
    sample_type: Optional[str] = None,
    max_samples: Optional[int] = None
) -> Dict[tuple, List[dict]]:
    """
    Collect samples grouped by (galaxy_id, orientation, type).
    
    Returns:
        Dict mapping (galaxy_id, orientation, type) -> List of GT masks for that image.
        This enables single inference per unique image (streams and satellites use different filters).
    """
    img_folder = data_dir / 'img_folder'
    gt_folder = data_dir / 'gt_folder'
    
    # Group samples by unique image
    grouped_samples = defaultdict(list)
    
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
            # Group by unique image identifier (including type since filters differ)
            group_key = (meta['galaxy_id'], meta['orientation'], meta['type'])
            grouped_samples[group_key].append({
                'name': sample_dir.name,
                'img_path': img_path,  # Same for all in group
                'gt_path': gt_path,
                **meta
            })
    
    # Apply max_samples limit at group level
    if max_samples:
        limited_groups = dict(list(grouped_samples.items())[:max_samples])
        return limited_groups
    
    return dict(grouped_samples)


def merge_config_with_args(config: dict, args) -> dict:
    """Merge YAML config with CLI arguments (CLI takes precedence)."""
    # Model config
    if args.checkpoint:
        config['model']['checkpoint'] = args.checkpoint
    if args.model_cfg:
        config['model']['model_cfg'] = args.model_cfg
    if args.device:
        config['model']['device'] = args.device
    
    # Data config
    if args.data_dir:
        config['data']['data_dir'] = args.data_dir
    if args.output_dir:
        config['data']['output_dir'] = args.output_dir
    
    # Generation config
    if args.points_per_side:
        config['generation']['points_per_side'] = args.points_per_side
    if args.pred_iou_thresh:
        config['generation']['pred_iou_thresh'] = args.pred_iou_thresh
    if args.stability_score_thresh:
        config['generation']['stability_score_thresh'] = args.stability_score_thresh
    
    # Evaluation config
    if args.sb_threshold:
        config['evaluation']['sb_threshold'] = args.sb_threshold
    if args.type:
        config['evaluation']['sample_type'] = args.type
    if args.max_samples:
        config['evaluation']['max_samples'] = args.max_samples
    if args.save_vis:
        config['evaluation']['save_visualizations'] = True
    
    return config


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description='SAM2 Inference and IoU Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file
    parser.add_argument('--config', '-c', type=str, default='configs/eval_sam2.yaml',
                        help='Path to configuration YAML file')
    
    # Model overrides
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to SAM2 checkpoint (overrides config)')
    parser.add_argument('--model_cfg', type=str, default=None,
                        help='SAM2 model config (overrides config)')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (overrides config)')
    
    # Data overrides
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to dataset directory (overrides config)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for results (overrides config)')
    
    # Generation overrides
    parser.add_argument('--points_per_side', type=int, default=None,
                        help='Grid density for mask generation (overrides config)')
    parser.add_argument('--pred_iou_thresh', type=float, default=None,
                        help='Predicted IoU threshold (overrides config)')
    parser.add_argument('--stability_score_thresh', type=float, default=None,
                        help='Stability score threshold (overrides config)')
    
    # Evaluation overrides
    parser.add_argument('--sb_threshold', type=float, default=None,
                        help='Filter by SB threshold (overrides config)')
    parser.add_argument('--type', type=str, choices=['streams', 'satellites'],
                        default=None, help='Filter by type (overrides config)')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='Maximum samples to process (overrides config)')
    parser.add_argument('--save_vis', action='store_true',
                        help='Save visualization images')
    
    args = parser.parse_args()
    
    # Load and merge configuration
    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    config = merge_config_with_args(config, args)
    
    # Setup logging
    log_dir = PROJECT_ROOT / "logs"
    logger = setup_logger("eval_model", log_dir)
    logger.info(f"Configuration loaded from {config_path}")
    
    # Validate checkpoint
    checkpoint = config['model']['checkpoint']
    if not checkpoint:
        logger.error("No checkpoint specified! Use --checkpoint or set in config.")
        sys.exit(1)
    
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)
    
    # Setup paths
    data_dir = Path(config['data']['data_dir'])
    output_dir = Path(config['data']['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if config['evaluation'].get('save_visualizations'):
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect samples (grouped by unique image)
    logger.info(f"Collecting samples from {data_dir}...")
    grouped_samples = collect_samples_grouped(
        data_dir,
        sb_threshold=config['evaluation'].get('sb_threshold'),
        sample_type=config['evaluation'].get('sample_type'),
        max_samples=config['evaluation'].get('max_samples')
    )
    total_groups = len(grouped_samples)
    total_samples = sum(len(group) for group in grouped_samples.values())
    logger.info(f"Found {total_groups} unique images with {total_samples} GT masks")
    
    if total_groups == 0:
        logger.error("No samples found. Check your data directory and filters.")
        sys.exit(1)
    
    # Load SAM2 model
    logger.info("Loading SAM2 model...")
    logger.info(f"  Config: {config['model']['model_cfg']}")
    logger.info(f"  Checkpoint: {checkpoint_path}")
    logger.info(f"  Device: {config['model']['device']}")
    
    # Import SAM2 (path may vary)
    SAM2_PATH = Path("/home/yuqyan/Yuqi/sam2")
    if SAM2_PATH.exists():
        sys.path.insert(0, str(SAM2_PATH))
    
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    
    sam2 = build_sam2(
        config['model']['model_cfg'],
        str(checkpoint_path),
        device=config['model']['device']
    )
    
    # Helper function to create mask generator from config
    def create_mask_generator(gen_cfg: dict) -> SAM2AutomaticMaskGenerator:
        return SAM2AutomaticMaskGenerator(
            model=sam2,
            points_per_side=gen_cfg['points_per_side'],
            points_per_batch=gen_cfg.get('points_per_batch', 128),
            pred_iou_thresh=gen_cfg['pred_iou_thresh'],
            stability_score_thresh=gen_cfg['stability_score_thresh'],
            stability_score_offset=gen_cfg.get('stability_score_offset', 0.7),
            crop_n_layers=gen_cfg.get('crop_n_layers', 1),
            box_nms_thresh=gen_cfg.get('box_nms_thresh', 0.7),
            crop_n_points_downscale_factor=gen_cfg.get('crop_n_points_downscale_factor', 2),
            min_mask_region_area=gen_cfg.get('min_mask_region_area', 25.0),
            use_m2m=gen_cfg.get('use_m2m', True),
        )
    
    # Create type-specific mask generators
    mask_generators = {
        'streams': create_mask_generator(config['generation_streams']),
        'satellites': create_mask_generator(config['generation_satellites'])
    }
    logger.info("Created type-specific mask generators:")
    logger.info(f"  Streams: points_per_side={config['generation_streams']['points_per_side']}, pred_iou_thresh={config['generation_streams']['pred_iou_thresh']}")
    logger.info(f"  Satellites: points_per_side={config['generation_satellites']['points_per_side']}, pred_iou_thresh={config['generation_satellites']['pred_iou_thresh']}")
    
    # Process samples: SINGLE INFERENCE PER UNIQUE IMAGE + CACHING
    logger.info(f"Processing {len(grouped_samples)} unique images...")
    results = []
    total_evals = sum(len(group) for group in grouped_samples.values())
    logger.info(f"  ({total_evals} total GT mask evaluations)")
    
    # Setup incremental saving
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"iou_results_{timestamp}.json"
    
    # Inference cache: avoid re-running SAM2 for already-processed (galaxy, ori, type)
    cache_file = output_dir / "inference_cache.pkl"
    pred_cache: Dict[tuple, Any] = {}
    if cache_file.exists():
        try:
            with open(cache_file, 'rb') as f:
                pred_cache = pickle.load(f)
            logger.info(f"Loaded {len(pred_cache)} cached predictions from {cache_file}")
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}. Starting fresh.")
            pred_cache = {}
    
    cache_hits = 0
    cache_misses = 0
    
    try:
        group_idx = 0
        for group_key, group_samples in grouped_samples.items():
            group_idx += 1
            galaxy_id, orientation, s_type = group_key
            
            # Check cache first
            if group_key in pred_cache:
                pred_masks = pred_cache[group_key]
                cache_hits += 1
                if group_idx % 50 == 0:
                    logger.info(f"  [{group_idx}/{len(grouped_samples)}] {galaxy_id}_{orientation}_{s_type} (cached, {len(group_samples)} GT masks)")
            else:
                cache_misses += 1
                if group_idx % 10 == 0 or group_idx == 1:
                    logger.info(f"  [{group_idx}/{len(grouped_samples)}] {galaxy_id}_{orientation}_{s_type} (inference, {len(group_samples)} GT masks)")
                
                # Load image once (streams and satellites use different filters)
                image = load_image(group_samples[0]['img_path'])
                
                # Run appropriate generator for this type
                mask_generator = mask_generators[s_type]
                pred_masks = None
                try:
                    # bf16 precision (Ampere+)
                    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        pred_masks = mask_generator.generate(image)
                    # Cache the result (only segmentation masks, not full dict)
                    pred_cache[group_key] = pred_masks
                except Exception as e:
                    logger.warning(f"  ⚠️  SAM2 {s_type} failed for {galaxy_id}_{orientation}: {e}")
                    pred_cache[group_key] = None  # Cache failure too

            # Visualization: Use max SB threshold (darkest, most complete structures)
            if config['evaluation'].get('save_visualizations') and pred_masks is not None:
                # Select sample with maximum SB threshold
                max_sb_sample = max(group_samples, key=lambda s: s['sb_threshold'])
                vis_path = vis_dir / f"{galaxy_id}_{orientation}_{s_type}_SB{max_sb_sample['sb_threshold']}.png"
                gt_vis = load_mask(max_sb_sample['gt_path'])
                save_visualization(image, pred_masks, gt_vis, vis_path)
            
            # Multiple Evaluations: compare single prediction against all SB thresholds
            for sample in group_samples:
                # All samples in group have same type, so directly use pred_masks
                
                if pred_masks is None:
                    # Record failure if prediction failed
                    result = {
                        'sample_name': sample['name'],
                        'galaxy_id': sample['galaxy_id'],
                        'orientation': sample['orientation'],
                        'sb_threshold': sample['sb_threshold'],
                        'type': sample['type'],
                        'binary_iou': 0.0,
                        'mean_instance_iou': 0.0,
                        'num_gt_instances': 0,
                        'num_pred_masks': 0,
                        'recall': 0.0,
                        'matched_iou': 0.0,
                        'num_detected': 0,
                        'instance_details': [],
                        'error': "Generator failed"
                    }
                    results.append(result)
                    continue

                gt_mask = load_mask(sample['gt_path'])
                iou_metrics = calculate_instance_iou(pred_masks, gt_mask)
                
                match_thresh = config['evaluation'].get('match_iou_thresh', 0.5)
                matched_metrics = calculate_matched_metrics(pred_masks, gt_mask, match_thresh)
                
                result = {
                    'sample_name': sample['name'],
                    'galaxy_id': sample['galaxy_id'],
                    'orientation': sample['orientation'],
                    'sb_threshold': sample['sb_threshold'],
                    'type': sample['type'],
                    **{k: v for k, v in iou_metrics.items() if k != 'instance_ious'},
                    'recall': matched_metrics['recall'],
                    'matched_iou': matched_metrics['matched_iou'],
                    'num_detected': matched_metrics['num_detected'],
                    'instance_details': matched_metrics['per_instance_details']
                }
                results.append(result)
            
            # Incremental save every 5 images
            if group_idx % 5 == 0:
                summary = {
                    'status': 'partial',
                    'processed_images': group_idx,
                    'total_images': len(grouped_samples),
                    'per_sample_results': results
                }
                with open(results_file, 'w') as f:
                    json.dump(summary, f, indent=2)

    except Exception as e:
        logger.error(f"FATAL ERROR during evaluation loop: {e}", exc_info=True)
        # Final attempt to save whatever we have
        if results:
            with open(results_file, 'w') as f:
                json.dump({'status': 'crashed', 'per_sample_results': results}, f, indent=2)
        # Save cache even on crash
        with open(cache_file, 'wb') as f:
            pickle.dump(pred_cache, f)
        sys.exit(1)
    
    # Save updated inference cache
    with open(cache_file, 'wb') as f:
        pickle.dump(pred_cache, f)
    logger.info(f"Saved inference cache: {len(pred_cache)} predictions to {cache_file}")
    logger.info(f"Cache stats: {cache_hits} hits, {cache_misses} new inferences")
    
    # Compute summary statistics
    binary_ious = [r['binary_iou'] for r in results]
    mean_inst_ious = [r['mean_instance_iou'] for r in results]
    
    logger.info("=" * 70)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 70)
    logger.info(f"Overall (n={len(results)}):")
    logger.info(f"  Binary IoU:        mean={np.mean(binary_ious):.4f}, std={np.std(binary_ious):.4f}")
    logger.info(f"  Mean Instance IoU: mean={np.mean(mean_inst_ious):.4f}, std={np.std(mean_inst_ious):.4f}")
    
    # Group by type
    for sample_type in ['streams', 'satellites']:
        type_results = [r for r in results if r['type'] == sample_type]
        if type_results:
            type_binary = [r['binary_iou'] for r in type_results]
            type_inst = [r['mean_instance_iou'] for r in type_results]
            type_recall = [r['recall'] for r in type_results]
            logger.info(f"{sample_type.capitalize()} (n={len(type_results)}):")
            logger.info(f"  Binary IoU:        mean={np.mean(type_binary):.4f}")
            logger.info(f"  Mean Instance IoU: mean={np.mean(type_inst):.4f}")
            logger.info(f"  Recall:            mean={np.mean(type_recall):.4f}")

    
    # Group by SB threshold
    sb_groups = defaultdict(list)
    for r in results:
        sb_groups[r['sb_threshold']].append(r)
    
    logger.info("By SB Threshold:")
    for sb in sorted(sb_groups.keys()):
        group = sb_groups[sb]
        b_iou = np.mean([r['binary_iou'] for r in group])
        i_iou = np.mean([r['mean_instance_iou'] for r in group])
        logger.info(f"  SB{sb}: n={len(group)}, Binary={b_iou:.4f}, Instance={i_iou:.4f}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"iou_results_{timestamp}.json"
    
    summary = {
        'config': {
            'checkpoint': str(checkpoint_path),
            'model_cfg': config['model']['model_cfg'],
            'data_dir': str(data_dir),
            'generation_streams': config['generation_streams'],
            'generation_satellites': config['generation_satellites']
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
    
    logger.info(f"Results saved to: {results_file}")
    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()
