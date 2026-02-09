# FBox Dataset Analysis

**Path:** `/home/yuqyan/Yuqi/scripts/LSB_and_Satellites/fbox`

## Overview
This directory contains the output of an automated source detection pipeline (likely **NoiseChisel**) applied to the FIREbox galaxy simulations. Unlike `FIREbox-DR1`, which contains manually curated or catalogue-matched instance masks, `fbox` appears to contain **raw detection outputs** swept across multiple surface brightness limits (`sblim`).

## File Structure & Naming Convention

The files are organized broadly into three types for various Galaxy IDs and Orientations (`eo`, `fo`):

### 1. Images (`counts`)
*   **Pattern:** `counts-Fbox-{ID}-{ori}-SDSS_r.fits`
*   **Format:** FITS (uncompressed), HDU 1.
*   **Content:** Flux / Photon count maps.
*   **Dtype:** `float64`
*   **Description:** These are the primary **input images** for the galaxies.

### 2. Detection Masks (`det`)
*   **Pattern:** `det-{ID}-{ori}-SDSS_r-sblim{THRESHOLD}.fits.gz`
*   **Format:** Multi-extension FITS (gzipped).
*   **Key HDU:** `HDU 2 ("DETECTIONS")` contains the mask.
*   **Content:** **Binary Detection Map** (0 = Background, 1 = Detected Object).
    *   **Crucial Note:** These are **NOT** instance masks. All objects have value `1`.
    *   **Preprocessing for SAM3:** To use these for SAM3 training (which requires instance discrimination), you must run a **Connected Components** algorithm (e.g., `scipy.ndimage.label`) to separate discontinuous regions into distinct instance IDs (1, 2, 3...).
*   **Thresholds:** Available for `sblim = 26, 26.5, ..., 31.5`. This provides a "stack" of masks ranging from conservative (bright objects only) to aggressive (faint diffuse emission included).

### 3. Preprocessed Images (`prep`)
*   **Pattern:** `prep-{ID}-{ori}-SDSS_r-sblim{THRESHOLD}.fits.gz`
*   **Format:** FITS (gzipped), HDU 1.
*   **Content:** Image data corresponding to the detection run.
*   **Observation:** These appear nearly identical to the `counts` images in terms of pixel values, possibly serving as the specific input fed to the detection algorithm for that run.

## Strategy for SAM3 Training

This dataset is highly valuable for **Weakly Supervised** or **Heuristic-based** training of SAM3 because it provides a massive amount of "automatically annotated" data at varying sensitivity levels.

### Proposed Workflow:

1.  **Input Images**: Use `counts-Fbox-{ID}-{ori}-SDSS_r.fits`.
    *   *Action*: Convert to 16-bit PNG (similar to `FIREbox-DR1` prep).

2.  **Generate Instance Masks**:
    *   Iterate through `det` files at selected `sblim` levels (e.g., 27, 29, 31).
    *   Load HDU 2 (`DETECTIONS`).
    *   **Process**: Apply Connected Components (`scipy.ndimage.label`) to convert the binary `0/1` mask into an instance mask `0, 1, 2, 3...`.
    *   *Rationale*: SAM3 learns to separate objects. Since NoiseChisel detections are mostly spatially separated, connected components will serve as a good proxy for instance labels.

3.  **Construct Dataset**:
    *   Create a similar `annotations.json` COCO structure.
    *   **Categories**: You can define categories either by `sblim` (e.g., "detection_sb27", "detection_sb30") OR just treat them all as generic "galaxy" objects but populate the training set with masks from different depths to satisfy "multi-granularity" capabilities.

### Comparison with FIREbox-DR1
*   **FIREbox-DR1**: Clean, likely catalog-matched instance masks. High quality, potentially fewer artifacts. Good for "Ground Truth".
*   **FBox**: Noisier, automated detections. **Binary-only** (requires post-processing). Massive coverage of sensitivity limits. Good for teaching the model "detection at limit X".

**Recommendation**: Start by generating a visualization script for `fbox` similar to the one for `FIREbox-DR1`, applying the connected components trick to see if the instances look reasonable.
