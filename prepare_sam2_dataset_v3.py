#!/usr/bin/env python3
"""
Prepare SAM2 Training Dataset with Multi-Granularity Masks v3

Keeps streams and satellites as SEPARATE samples with suffixes:
- {galaxy_id}_{orient}_SB{threshold}_streams
- {galaxy_id}_{orient}_SB{threshold}_satellites

Mask Processing:
- Streams: PRESERVE instance IDs (1, 2, 3... for different stream instances)
- Satellites: Convert to same value (instance from 1 to  for all satellites)

Output Structure:
sam2_dataset_v3/
├── img_folder/
│   ├── {galaxy_id}_{orient}_SB{threshold}_streams/
│   │   └── 0000.png
│   ├── {galaxy_id}_{orient}_SB{threshold}_satellites/
│   │   └── 0000.png
│   └── ...
├── gt_folder/
│   └── ... (same structure)

Author: Yuqi
Date: 2026-01-26 (Fixed)
"""

import re
import shutil
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from scipy import ndimage
# from tqdm import tqdm  # Optional, removed to avoid dependency

# === Configuration ===
SCRIPT_DIR = Path(__file__).parent

# Input directories
STREAMS_DIR = SCRIPT_DIR / "sam3_prepared"
SATELLITES_DIR = SCRIPT_DIR / "sam3_prepared_satelites"

# Output directory
OUTPUT_DIR = SCRIPT_DIR / "sam2_prepared"

# Target image size (use streams size as standard)
TARGET_SIZE = (1072, 1072)  # (width, height)


def parse_mask_filename(filename: str):
    """
    Parse mask filename like 'Fbox-11-eo_SB27.5.png'
    Returns: (galaxy_id, orientation, sb_threshold) or None
    """
    m = re.match(r'Fbox-(\d+)-([ef]o)_SB([\d.]+)\.png', filename)
    if m:
        return int(m.group(1)), m.group(2), m.group(3)
    return None


def parse_image_filename(filename: str):
    """
    Parse image filename like 'Fbox-11-eo.png'
    Returns: (galaxy_id, orientation) or None
    """
    m = re.match(r'Fbox-(\d+)-([ef]o)\.png', filename)
    if m:
        return int(m.group(1)), m.group(2)
    return None


def load_and_resize_image(path: Path, target_size: tuple) -> np.ndarray:
    """
    Enhanced Asinh Stretch: Preserves details but boosts visibility.
    """
    img = Image.open(path)
    img_array = np.array(img, dtype=np.float32)

    # 1. 如果是 RGB，先转灰度处理再堆叠，或者逐通道处理
    # 这里假设输入是单通道 16-bit
    if len(img_array.shape) == 3:
        # 简单起见，取平均变成单通道处理，最后再变回 RGB
        # 或者你可以对每个通道单独做，但天文图通常只是单波段
        img_array = img_array.mean(axis=2)

    # 2. 去底噪 (Background Subtraction)
    bg = np.median(img_array)
    img_sub = np.maximum(img_array - bg, 0)

    # 3. 归一化 (Normalization)
    # 这里依然使用 99.5% 作为参考最大值，防止极亮星捣乱
    # 但我们不截断它，只是用它做缩放基准
    vmax = np.percentile(img_sub, 99.5)
    if vmax <= 0: vmax = img_sub.max()
    
    # 归一化到 0~1 附近 (允许超过 1)
    img_norm = img_sub / (vmax + 1e-5)

    # 4. Asinh 变换 (关键步骤)
    # nonlinearity 越大，暗部（Stream）提亮越明显，越接近 Log 的效果
    # 你之前觉得暗，是因为这个数可能太小了 (比如 10)
    # 建议尝试 30 到 50，这样主体会变强
    nonlinearity = 10.0  
    img_stretched = np.arcsinh(img_norm * nonlinearity) / np.arcsinh(nonlinearity)

    # 5. 映射到 0-255
    img_8bit = (img_stretched * 255).clip(0, 255).astype(np.uint8)

    # 6. 转为 RGB
    img_rgb = np.stack([img_8bit] * 3, axis=-1)

    # 7. Resize
    h, w = img_rgb.shape[:2]
    if (w, h) != target_size:
        img_pil = Image.fromarray(img_rgb)
        img_pil = img_pil.resize(target_size, Image.Resampling.BILINEAR)
        img_rgb = np.array(img_pil)

    return img_rgb


def _apply_log_stretch(img: np.ndarray, a: float = 1000.0) -> np.ndarray:
    """
    Apply logarithmic stretching to a single channel.
    
    Args:
        img: Input image array (float)
        a: Scaling parameter for log transformation (default: 1000)
    
    Returns:
        Stretched image normalized to 0-255
    """
    # Calculate background using median
    bg = np.median(img)
    
    # Subtract background and clip to non-negative
    img_sub = np.maximum(img - bg, 0)
    
    # Apply log transformation
    img_max = img_sub.max()
    if img_max > 0:
        # log1p(a * img) / log1p(a * max)
        img_log = np.log1p(a * img_sub) / np.log1p(a * img_max)
    else:
        # Empty or constant image
        img_log = img_sub
    
    # Normalize to 0-255
    img_normalized = (img_log * 255).clip(0, 255)
    
    return img_normalized


def load_and_resize_mask(path: Path, target_size: tuple, preserve_instances: bool = False) -> np.ndarray:
    """
    Load mask and resize if needed (nearest neighbor for masks).
    
    Args:
        preserve_instances: If True, preserve instance IDs (for streams).
                           If False, use fixed value 255 (for satellites).
    """
    mask = np.array(Image.open(path))
    
    # Handle multi-channel masks
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]
    
    h, w = mask.shape
    if (w, h) != target_size:
        mask_pil = Image.fromarray(mask)
        mask_pil = mask_pil.resize(target_size, Image.Resampling.NEAREST)
        mask = np.array(mask_pil)
    
    if preserve_instances:
        # For streams: PRESERVE instance IDs as-is
        # The source masks already have instance IDs (0=bg, 1=instance1, 2=instance2, etc.)
        # Just ensure they fit in uint8
        unique_vals = np.unique(mask)
        unique_vals = unique_vals[unique_vals > 0]
        
        if len(unique_vals) == 0:
            # Empty mask
            pass
        elif len(unique_vals) == 1 and unique_vals[0] == 255:
            # Binary mask with 255 - use connected components to get instance IDs
            mask_binary = (mask > 0).astype(np.uint8)
            labeled, num_features = ndimage.label(mask_binary)
            mask = labeled.astype(np.uint8)
        elif mask.max() > 254:
            # Remap large values to fit in uint8 (shouldn't normally happen)
            mask = np.clip(mask, 0, 254).astype(np.uint8)
        else:
            # Already has proper instance IDs, keep as-is
            mask = mask.astype(np.uint8)
    else:
        # For satellites: use connected components to get instance IDs starting from 1
        mask_binary = (mask > 0).astype(np.uint8)
        labeled, num_features = ndimage.label(mask_binary)
        mask = labeled.astype(np.uint8)
    
    return mask


def collect_data_info(images_dir: Path, masks_dir: Path) -> dict:
    """
    Collect information about available images and masks.
    Returns: {(galaxy_id, orientation, sb_threshold): {'image': Path, 'mask': Path}}
    """
    data = {}
    
    # Build image lookup: (galaxy_id, orientation) -> image_path
    image_lookup = {}
    for img_path in images_dir.glob("*.png"):
        parsed = parse_image_filename(img_path.name)
        if parsed:
            gid, orient = parsed
            image_lookup[(gid, orient)] = img_path
    
    # Collect masks and pair with images
    for mask_path in masks_dir.glob("*.png"):
        parsed = parse_mask_filename(mask_path.name)
        if parsed:
            gid, orient, sb_thresh = parsed
            img_path = image_lookup.get((gid, orient))
            if img_path:
                key = (gid, orient, sb_thresh)
                data[key] = {'image': img_path, 'mask': mask_path}
    
    return data


def process_dataset(name: str, suffix: str, images_dir: Path, masks_dir: Path, 
                    img_out_dir: Path, gt_out_dir: Path,
                    target_size: tuple, stats: dict):
    """Process one dataset (streams or satellites) with suffix."""
    print(f"\nProcessing {name}...")
    
    data = collect_data_info(images_dir, masks_dir)
    
    items = sorted(data.items())
    total = len(items)
    for i, ((gid, orient, sb_thresh), info) in enumerate(items):
        if (i + 1) % 50 == 0 or i == 0:
            print(f"  {name}: {i+1}/{total}")
        # Create folder name with suffix: {galaxy_id}_{orient}_SB{threshold}_{suffix}
        folder_name = f"{gid:05d}_{orient}_SB{sb_thresh}_{suffix}"
        
        # Create output directories
        img_sample_dir = img_out_dir / folder_name
        gt_sample_dir = gt_out_dir / folder_name
        img_sample_dir.mkdir(parents=True, exist_ok=True)
        gt_sample_dir.mkdir(parents=True, exist_ok=True)
        
        # Process image
        try:
            img = load_and_resize_image(info['image'], target_size)
            img_pil = Image.fromarray(img)
            img_out_path = img_sample_dir / "0000.png"
            img_pil.save(str(img_out_path))
        except Exception as e:
            print(f"  Error processing image {info['image'].name}: {e}")
            continue
        
        # Process mask
        try:
            # Streams: preserve instances; Satellites: fixed value 255
            preserve_inst = (suffix == "streams")
            mask = load_and_resize_mask(info['mask'], target_size, preserve_instances=preserve_inst)
            mask_pil = Image.fromarray(mask)
            gt_out_path = gt_sample_dir / "0000.png"
            mask_pil.save(str(gt_out_path))
            stats['samples'] += 1
            stats[suffix] += 1
        except Exception as e:
            print(f"  Error processing mask {info['mask'].name}: {e}")


def main():
    print("=" * 80)
    print("SAM2 Dataset Preparation v3 - Streams & Satellites Separate")
    print("=" * 80)
    
    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing output directory: {OUTPUT_DIR}")
        shutil.rmtree(OUTPUT_DIR)
    
    # Create output directories
    img_out_dir = OUTPUT_DIR / "img_folder"
    gt_out_dir = OUTPUT_DIR / "gt_folder"
    img_out_dir.mkdir(parents=True, exist_ok=True)
    gt_out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nConfiguration:")
    print(f"  Target size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Structure: Separate folders for streams and satellites")
    
    print(f"\nInput datasets:")
    print(f"  Streams: {STREAMS_DIR}")
    print(f"  Satellites: {SATELLITES_DIR}")
    
    stats = {'samples': 0, 'streams': 0, 'satellites': 0}
    
    # Process stellar streams (already 1072×1072)
    process_dataset(
        "Stellar Streams", "streams",
        STREAMS_DIR / "images",
        STREAMS_DIR / "masks",
        img_out_dir, gt_out_dir,
        TARGET_SIZE, stats
    )
    
    # Process satellites (resize from 2051×2051 to 1072×1072)
    process_dataset(
        "Satellites", "satellites",
        SATELLITES_DIR / "images",
        SATELLITES_DIR / "masks",
        img_out_dir, gt_out_dir,
        TARGET_SIZE, stats
    )
    
    # Count statistics
    num_samples = len(list(img_out_dir.glob("*")))
    
    # Analyze by type
    streams_count = len([f for f in img_out_dir.glob("*_streams")])
    satellites_count = len([f for f in img_out_dir.glob("*_satellites")])
    
    # Analyze threshold distribution
    threshold_counts = defaultdict(lambda: {'streams': 0, 'satellites': 0})
    for folder in img_out_dir.glob("*"):
        m = re.search(r'_SB([\d.]+)_(streams|satellites)$', folder.name)
        if m:
            threshold_counts[m.group(1)][m.group(2)] += 1
    
    print("\n" + "=" * 80)
    print("Done!")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_DIR}")
    print(f"  Total samples: {num_samples}")
    print(f"    - Streams: {streams_count}")
    print(f"    - Satellites: {satellites_count}")
    
    print(f"\nSamples per SB threshold:")
    print(f"  {'Threshold':<12} {'Streams':>10} {'Satellites':>12} {'Total':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*12} {'-'*10}")
    for thresh in sorted(threshold_counts.keys(), key=float):
        s = threshold_counts[thresh]['streams']
        sat = threshold_counts[thresh]['satellites']
        print(f"  SB{thresh:<9} {s:>10} {sat:>12} {s+sat:>10}")
    
    print(f"\nStructure:")
    print(f"  {img_out_dir}/")
    print(f"  {gt_out_dir}/")
    
    # Create README
    readme_content = f"""================================================================================
SAM2 Dataset v3 - Streams & Satellites SEPARATE
================================================================================

Each (galaxy, orientation, SB_threshold, type) combination is a separate sample.
Streams and satellites are distinguished by suffix.

Generated from:
- sam3_prepared (stellar streams) → suffix: _streams
- sam3_prepared_satelites (satellites) → suffix: _satellites

Structure:
--------------------------------------------------------------------------------
{OUTPUT_DIR.name}/
├── img_folder/
│   ├── 00011_eo_SB27_streams/
│   │   └── 0000.png      # Stellar stream features
│   ├── 00011_eo_SB27_satellites/
│   │   └── 0000.png      # Satellite galaxy features  
│   ├── 00011_eo_SB26_satellites/
│   │   └── 0000.png      # Only satellites (no streams at SB26)
│   └── ...
│
├── gt_folder/
│   ├── 00011_eo_SB27_streams/
│   │   └── 0000.png      # Mask for stellar streams
│   ├── 00011_eo_SB27_satellites/
│   │   └── 0000.png      # Mask for satellite galaxies
│   └── ...
│
└── README.txt

Statistics:
--------------------------------------------------------------------------------
- Total samples: {num_samples}
- Streams: {streams_count}
- Satellites: {satellites_count}
- Image size: {TARGET_SIZE[0]}×{TARGET_SIZE[1]}

Feature Types:
--------------------------------------------------------------------------------
- Stellar Streams (_streams): Tidal tails, elongated diffuse structures
- Satellites (_satellites): Satellite galaxies, more compact objects

Folder Naming Convention:
--------------------------------------------------------------------------------
{{galaxy_id}}_{{orientation}}_SB{{threshold}}_{{type}}
- galaxy_id: 5-digit zero-padded (e.g., 00011)
- orientation: eo (edge-on) or fo (face-on)
- threshold: Surface brightness threshold (e.g., 27, 27.5, 28, ...)
- type: streams or satellites

Mask Format:
--------------------------------------------------------------------------------
- Stellar Streams: Instance-based (0=background, 1, 2, 3...=different instances)
- Satellites: Instance-based (0=background, 1, 2, 3...=different instances)
- 8-bit grayscale

Image Processing:
--------------------------------------------------------------------------------
- 16-bit astronomical images with logarithmic stretching
- Background subtraction using median
- Log transform: log1p(a * img) / log1p(a * max), a=1000
- Normalized to 8-bit RGB (0-255)

================================================================================
"""
    
    with open(OUTPUT_DIR / "README.txt", 'w') as f:
        f.write(readme_content)
    
    print(f"\nREADME saved to {OUTPUT_DIR / 'README.txt'}")


if __name__ == "__main__":
    main()
