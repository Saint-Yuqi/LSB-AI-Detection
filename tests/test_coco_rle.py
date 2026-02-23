#!/usr/bin/env python3
"""
Round-trip validation for COCO RLE encode/decode.

Usage:
    conda run -n sam2 --no-capture-output python tests/test_coco_rle.py
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.coco_utils import mask_to_rle, decode_rle


def test_round_trip_simple():
    """Single contiguous blob."""
    mask = np.zeros((100, 100), dtype=np.uint8)
    mask[20:50, 30:70] = 1

    rle = mask_to_rle(mask)
    assert isinstance(rle['counts'], str), f"Expected str counts, got {type(rle['counts'])}"
    assert rle['size'] == [100, 100]

    recovered = decode_rle(rle)
    assert np.array_equal(mask, recovered), "Pixel mismatch after round-trip"
    print("✓ Simple blob round-trip passed")


def test_round_trip_disconnected():
    """Disconnected components (stream-like)."""
    mask = np.zeros((256, 256), dtype=np.uint8)
    mask[10:30, 10:30] = 1
    mask[200:220, 200:220] = 1

    rle = mask_to_rle(mask)
    recovered = decode_rle(rle)
    assert np.array_equal(mask, recovered), "Disconnected mask mismatch"
    print("✓ Disconnected components round-trip passed")


def test_area_consistency():
    """Area field matches pixel count."""
    mask = np.zeros((512, 512), dtype=np.uint8)
    mask[100:200, 100:200] = 1
    expected_area = int(mask.sum())

    rle = mask_to_rle(mask)
    recovered = decode_rle(rle)
    assert recovered.sum() == expected_area, f"Area: {recovered.sum()} != {expected_area}"
    print(f"✓ Area consistency passed (area={expected_area})")


def test_empty_mask():
    """All-zero mask."""
    mask = np.zeros((64, 64), dtype=np.uint8)
    rle = mask_to_rle(mask)
    recovered = decode_rle(rle)
    assert recovered.sum() == 0, "Empty mask should decode to all zeros"
    print("✓ Empty mask round-trip passed")


if __name__ == "__main__":
    print("Running COCO RLE round-trip tests...\n")
    test_round_trip_simple()
    test_round_trip_disconnected()
    test_area_consistency()
    test_empty_mask()
    print("\n✅ All RLE tests passed!")
