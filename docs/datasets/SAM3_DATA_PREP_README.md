# SAM3 Data Preparation for FIREbox-DR1

This guide explains how to prepare the FIREbox-DR1 galaxy dataset for SAM3 training.

## Overview

For each galaxy and each surface brightness (SB) threshold, we create:
- **1 threshold-specific image**: The galaxy rendered at that brightness level
- **1 corresponding mask**: The detected features at that threshold

**Total**: ~770 image-mask pairs (35 galaxies × 2 orientations × 11 thresholds)

## Quick Start

```bash
# 1. Test on a single galaxy first
cd /home/yuqyan/Yuqi/scripts/LSB_and_Satellites/FIREbox-DR1
python test_preparation.py

# 2. Run full preparation
python prepare_sam3_data.py
```

## Input Data

```
FIREbox-DR1/
├── SB_maps/                    # Source brightness maps (~3.7MB each)
│   └── magnitudes-Fbox-{ID}-{eo/fo}-VIS2.fits.gz
├── MASKS_EO/                   # Edge-on masks (~10-17KB each)
│   └── ark_features-{ID}-eo-SBlim{threshold}.fits.gz
└── MASKS_FO/                   # Face-on masks
    └── ark_features-{ID}-fo-SBlim{threshold}.fits.gz
```

**SB Thresholds**: 27, 27.5, 28, 28.5, 29, 29.5, 30, 30.5, 31, 31.5, 32 mag/arcsec²

## Output Structure

```
sam3_prepared/
├── images/
│   ├── magnitudes-Fbox-11-eo-SBlim27.png
│   ├── magnitudes-Fbox-11-eo-SBlim27.5.png
│   └── ...
└── masks/
    ├── ark_features-11-eo-SBlim27.png
    ├── ark_features-11-eo-SBlim27.5.png
    └── ...
```

**Naming Convention** (preserves original file names):
- Image: `magnitudes-Fbox-{ID}-{orientation}-SBlim{threshold}.png`
- Mask: `ark_features-{ID}-{orientation}-SBlim{threshold}.png`

## Dependencies

```bash
pip install numpy astropy pillow tqdm
```

## How It Works

1. **Load SB Map**: Read the galaxy's surface brightness FITS file
2. **Apply Threshold**: For each SB threshold, create an image showing only pixels brighter than the threshold
3. **Save Mask**: Convert the corresponding FITS mask to PNG format
4. **Output**: Paired images and masks ready for SAM3

## Next Steps

After preparation, configure SAM3 to use these image-mask pairs:
- Images: `/home/yuqyan/Yuqi/scripts/LSB_and_Satellites/FIREbox-DR1/sam3_prepared/images/`
- Masks: `/home/yuqyan/Yuqi/scripts/LSB_and_Satellites/FIREbox-DR1/sam3_prepared/masks/`

SAM3 will handle bbox extraction and annotation generation internally.
