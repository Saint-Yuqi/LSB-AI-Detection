# Dataset Summary

## Overview

This document summarizes two astronomical image segmentation datasets prepared for SAM (Segment Anything Model) training:
- `sam2_prepared`: Video-frame format dataset for SAM2
- `sam3_prepared_unified_v2`: COCO-format dataset for SAM3

---

## Dataset 1: sam2_prepared

### Purpose
SAM2 training dataset with separate samples for stellar streams and satellite galaxies. Each (galaxy, orientation, SB_threshold, type) combination is a distinct sample.

### Statistics
- **Total samples**: 1,738 folders
- **Image files**: 1,468 PNG files
- **Mask files**: 1,705 PNG files
- **Image size**: 1072×1072 pixels
- **Format**: 8-bit RGB PNG

### Directory Structure
```
sam2_prepared/
├── img_folder/          # Input images (1,468 files)
│   ├── {galaxy_id}_{orient}_SB{threshold}_streams/
│   │   └── 0000.png
│   ├── {galaxy_id}_{orient}_SB{threshold}_satellites/
│   │   └── 0000.png
│   └── ...
├── gt_folder/           # Ground truth masks (1,705 files)
│   ├── {galaxy_id}_{orient}_SB{threshold}_streams/
│   │   └── 0000.png
│   ├── {galaxy_id}_{orient}_SB{threshold}_satellites/
│   │   └── 0000.png
│   └── ...
├── visualizations/      # Visualization outputs (125 files)
├── visualizations_backup/
├── _trash_bin/         # Removed empty samples (670 files)
├── clean_empty_samples.py  # Script to remove empty samples
└── README.txt
```

### Naming Convention
**Folder name format**: `{galaxy_id}_{orientation}_SB{threshold}_{type}`
- `galaxy_id`: 5-digit zero-padded (e.g., `00011`)
- `orientation`: `eo` (edge-on) or `fo` (face-on)
- `threshold`: Surface brightness threshold (e.g., `27`, `27.5`, `28`, `28.5`, `29`, `29.5`, `30`, `30.5`, `31`, `31.5`, `32`)
- `type`: `streams` or `satellites`

**File name**: `0000.png` (single frame per sample)

### Data Sources
- **Streams**: Generated from `sam3_prepared` (stellar streams)
- **Satellites**: Generated from `sam3_prepared_satelites` (satellite galaxies)
- Original streams images: 1072×1072 (no resize)
- Original satellites images: 2051×2051 → resized to 1072×1072

### Mask Format
- **Stellar Streams**: Instance-based masks (0=background, 1, 2, 3...=different stream instances)
- **Satellites**: Instance-based masks (0=background, 1, 2, 3...=different satellite instances)
- **Format**: 8-bit grayscale PNG

### Image Processing
- 16-bit astronomical images with logarithmic stretching
- Background subtraction using median
- Log transform: `log1p(a * img) / log1p(a * max)`, where `a=1000`
- Normalized to 8-bit RGB (0-255)

### Feature Types
- **Stellar Streams** (`_streams`): Tidal tails, elongated diffuse structures
- **Satellites** (`_satellites`): Satellite galaxies, more compact objects

### Maintenance
- `clean_empty_samples.py`: Removes samples with no objects (all-zero masks)
- Empty samples are moved to `_trash_bin/` for backup

---

## Dataset 2: sam3_prepared_unified_v2

### Purpose
Unified SAM3 training dataset in COCO format, combining stellar streams and satellite galaxies with proper image-mask alignment.

### Statistics
- **Total images**: 138 files
  - Streams: 68 images
  - Satellites: 70 images
- **Total masks**: 1,025 files
- **Image size**: 1024×1024 pixels
- **Format**: 8-bit RGB PNG
- **Annotation format**: COCO JSON

### Directory Structure
```
sam3_prepared_unified_v2/
├── images/              # Input images (138 files)
│   ├── Fbox-{id}-{orient}_streams.png
│   ├── Fbox-{id}-{orient}_satellites.png
│   └── ...
├── masks/               # Ground truth masks (1,025 files)
│   ├── Fbox-{id}-{orient}_SB{threshold}_streams.png
│   ├── Fbox-{id}-{orient}_SB{threshold}_satellites.png
│   └── ...
├── annotations.json     # COCO-format annotations
├── visualizations/      # Visualization outputs
│   ├── Fbox-11-eo_all_thresholds.jpg
│   └── Fbox-11-fo_all_thresholds.jpg
├── verification/        # Verification images (5 files)
├── visualize_sam3.py    # Visualization script
└── debug_path.py
```

### Naming Convention
**Image files**: `Fbox-{galaxy_id}-{orientation}_{type}.png`
- `galaxy_id`: Galaxy ID (e.g., `11`, `13`, `19`, ...)
- `orientation`: `eo` (edge-on) or `fo` (face-on)
- `type`: `streams` or `satellites`

**Mask files**: `Fbox-{galaxy_id}-{orientation}_SB{threshold}_{type}.png`
- `threshold`: Surface brightness threshold (e.g., `27.0`, `27.5`, `28.0`, `28.5`, `29.0`, `29.5`, `30.0`, `30.5`, `31.0`, `31.5`, `32.0`)

### Data Sources
- **Streams images**: VIS2 images from `FIREbox-DR1/SB_maps`
- **Satellites images**: VIS images from `fbox/sb_maps`
- **Streams masks**: From `FIREbox-DR1/MASKS_EO` and `FIREbox-DR1/MASKS_FO`
- **Satellites masks**: Generated from satellite properties pickle file

### Image Processing
- **Global magnitude normalization**: 
  - Brightest (saturation): 20.0 mag/arcsec²
  - Faintest (black point): 35.0 mag/arcsec²
- Linear magnitude scaling with global normalization
- Target resolution: 1024×1024 pixels

### Surface Brightness Thresholds
11 thresholds: 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32 (mag/arcsec²)

### Galaxy Coverage
35 galaxies: 11, 13, 19, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 53, 56, 63, 64, 66, 68, 72

### Annotation Format
- **COCO JSON**: Standard COCO format with instance segmentation masks
- Proper image-mask correspondence via `image_id`
- Instance-based masks encoded as RLE (Run-Length Encoding)

### Key Differences from sam2_prepared
1. **Format**: COCO JSON vs. folder-based structure
2. **Image size**: 1024×1024 vs. 1072×1072
3. **Organization**: Flat file structure vs. nested folders
4. **Image sources**: Separate VIS2/VIS images for streams/satellites vs. unified processing
5. **Mask-to-image ratio**: Multiple masks per image (multiple SB thresholds) vs. one mask per sample

---

## Comparison Summary

| Feature | sam2_prepared | sam3_prepared_unified_v2 |
|---------|---------------|--------------------------|
| **Format** | Folder-based (SAM2 video) | COCO JSON (SAM3) |
| **Image size** | 1072×1072 | 1024×1024 |
| **Total samples** | 1,738 folders | 138 images |
| **Masks per sample** | 1 mask per sample | Multiple masks per image (11 SB thresholds) |
| **Total masks** | ~1,705 | 1,025 |
| **Organization** | Nested folders | Flat structure |
| **Use case** | SAM2 video segmentation | SAM3 instance segmentation |
| **Image processing** | Log transform + median background | Linear magnitude + global normalization |
| **Streams source** | sam3_prepared | FIREbox-DR1 VIS2 |
| **Satellites source** | sam3_prepared_satelites | fbox VIS |

---

## Usage Notes

### sam2_prepared
- Designed for SAM2 video segmentation training
- Each folder represents a single frame sample
- Use `clean_empty_samples.py` to remove empty samples before training
- Folder structure: `{galaxy_id}_{orient}_SB{threshold}_{type}/0000.png`

### sam3_prepared_unified_v2
- Designed for SAM3 instance segmentation training
- Load via COCO API: `from pycocotools.coco import COCO`
- Multiple masks per image (different SB thresholds) enable multi-threshold training
- Image-mask alignment: streams use VIS2 images, satellites use VIS images

---

## File Size Reference
- Typical image size: ~700KB per PNG (1024×1024 or 1072×1072 RGB)
- Dataset total size: ~1-2 GB (estimated)
