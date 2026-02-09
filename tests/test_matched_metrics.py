#!/usr/bin/env python3
"""
Quick validation test for calculate_matched_metrics.
Tests vectorized IoU computation and shape extraction.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.metrics import calculate_matched_metrics


def test_basic_matching():
    """Test basic GT-to-pred matching logic."""
    # Create synthetic GT mask: 2 instances
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[10:30, 10:30] = 1  # Instance 1: 20x20 = 400 pixels
    gt_mask[50:80, 50:80] = 2  # Instance 2: 30x30 = 900 pixels
    
    # Create synthetic predictions
    pred_masks = [
        {'segmentation': np.zeros((100, 100), dtype=bool)},  # Background noise
        {'segmentation': np.zeros((100, 100), dtype=bool)},  # Match inst 1
    ]
    pred_masks[1]['segmentation'][12:28, 12:28] = True  # Overlap with inst 1
    
    # Run matching
    result = calculate_matched_metrics(pred_masks, gt_mask, iou_threshold=0.3)
    
    # Validate
    assert result['num_gt_instances'] == 2, f"Expected 2 GT instances, got {result['num_gt_instances']}"
    assert len(result['per_instance_details']) == 2, "Should have 2 instance details"
    
    # Check instance 1 (should be detected)
    inst1 = [d for d in result['per_instance_details'] if d['gt_instance_id'] == 1][0]
    assert inst1['detected'], "Instance 1 should be detected"
    assert inst1['gt_area'] == 400, f"Instance 1 area should be 400, got {inst1['gt_area']}"
    assert inst1['best_iou'] > 0.3, f"Instance 1 IoU should be > 0.3, got {inst1['best_iou']}"
    
    # Check instance 2 (should NOT be detected - no match)
    inst2 = [d for d in result['per_instance_details'] if d['gt_instance_id'] == 2][0]
    assert not inst2['detected'], "Instance 2 should not be detected"
    assert inst2['gt_area'] == 900, f"Instance 2 area should be 900, got {inst2['gt_area']}"
    
    # Check overall Recall (1 detected out of 2)
    assert result['recall'] == 0.5, f"Recall should be 0.5, got {result['recall']}"
    assert result['num_detected'] == 1, f"Should have 1 detected, got {result['num_detected']}"
    
    print("✓ Basic matching test passed")
    print(f"  Recall: {result['recall']:.2f}")
    print(f"  Matched IoU: {result['matched_iou']:.2f}")
    print(f"  Detected: {result['num_detected']}/{result['num_gt_instances']}")


def test_empty_cases():
    """Test edge cases: no GT, no predictions."""
    # No GT instances
    gt_empty = np.zeros((100, 100), dtype=np.uint8)
    pred_masks = [{'segmentation': np.ones((100, 100), dtype=bool)}]
    result = calculate_matched_metrics(pred_masks, gt_empty)
    assert result['recall'] == 0.0
    assert result['num_gt_instances'] == 0
    print("✓ Empty GT test passed")
    
    # No predictions
    gt_mask = np.zeros((100, 100), dtype=np.uint8)
    gt_mask[10:30, 10:30] = 1
    result = calculate_matched_metrics([], gt_mask)
    assert result['recall'] == 0.0
    assert result['num_detected'] == 0
    print("✓ Empty predictions test passed")


if __name__ == "__main__":
    print("Running calculate_matched_metrics validation tests...\n")
    test_basic_matching()
    test_empty_cases()
    print("\n✅ All validation tests passed!")
