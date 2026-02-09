# ARRAKIHS FIREbox Catalogue - Dataset Summary 
**Date:** 2026-01-08  
**Purpose:** Understanding what the hell each file does for stellar stream identification with SAM2

---

## Core Philosophy: What This Dataset Actually Is

**The Real Problem:** You need mock galaxy observations to train SAM2 for detecting stellar streams—faint structures from tidally disrupted dwarf galaxies.

**The Data:** 51 simulated galaxies (00000-00050) from FIREbox cosmological simulation, post-processed with SKIRT dust radiative transfer. Stellar mass range: 8×10⁹ to 8×10¹⁰ M☉. All central galaxies at z=0, placed at 40 Mpc distance.

**Total Size:** ~330 GB (6-13 GB per galaxy)

---

## Directory Structure - The Good Parts

```
ARRAKIHS_FIREbox_catalogue/
├── 00000/ to 00050/          # 51 galaxy directories (the actual data)
├── Catalogue.pdf              # 151 MB visual reference (face-on/edge-on previews)
├── FIREbox_haloID.csv         # Mapping: ARRAKIHS ID → FIREbox halo ID
├── wavelengths_sed.dat        # 350 wavelength bins (0.03-1000 μm)
├── README.txt                 # Documentation (read this, it's good)
├── parameter_scan/            # Parameter convergence tests (for nerds)
└── scripts/                   # Processing scripts (may be useful)
```

---

## What Each Galaxy Directory Contains

### 1. **SED Cubes (FITS files)** - The Raw Physics

**File Pattern:** `smooth-2h_UHR_wd01mw_sb99_dz4_{orientation}_{rt_type}.fits.gz`

**Shape:** (350, 2051, 2051)
- **Dimension 1:** 350 wavelength bins (0.03-1000 μm)
- **Dimension 2-3:** 2051×2051 pixels = 1.31 Mpc × 1.31 Mpc field of view
- **Pixel scale:** 640 pc physical size per pixel
- **Units:** W/m²/arcsec² (surface spectral flux density)

**Orientations (9 total):**
- `i0` - Face-on (z-axis view)
- `i90` - Edge-on (x-axis view)  
- `i48a0`, `i48a180` - Intermediate inclinations
- `i90a45`, `i90a135` - Edge-on with azimuthal rotation
- (3 more intermediate orientations)

**RT Types:**
- `total` - Includes dust extinction & emission (realistic)
- `transparent` - No dust (stars only, for comparison)

**What this is for:** Full spectral information if you need custom filter integration or want to understand dust effects.

---

### 2. **Surface Brightness Maps (NPZ files)** - The Useful Stuff

**File Pattern:** `mag_map_{orientation}_{rt_type}.npz`

**Loading:**
```python
data = np.load('mag_map_i0_total.npz')['flux']
```

**Shape:** (4, 2051, 2051)
- **Dimension 1:** 4 ARRAKIHS filters
  - `[0]` HST-F475X (optical, ~475 nm, blue)
  - `[1]` Euclid-VIS (optical, ~700 nm, red)
  - `[2]` Euclid-J (NIR, ~1.25 μm)
  - `[3]` Euclid-Y (NIR, ~1.0 μm)
- **Dimension 2-3:** 2051×2051 pixels
- **Units:** mag/arcsec² (surface brightness)

**Why you care:** This is your SAM2 training data. Pre-integrated, ready to visualize, shows stellar streams clearly in surface brightness space.

**ARRAKIHS FOV:** 1.4 deg² circle overlaid in the catalog PDF.

---

### 3. **Catalogue.pdf** - The Preview

**Size:** 151 MB

**Content:** Visual catalog showing each galaxy in face-on and edge-on views with three colormaps:
1. Discretized colormap (feature detection)
2. Grayscale 19-29.5 mag/arcsec² (standard depth)
3. Grayscale 19-30.5 mag/arcsec² (deep, for faint streams)

**Use case:** Quick visual inspection to select interesting galaxies with prominent stellar streams.

---

## Supporting Files

### wavelengths_sed.dat
**Format:** 4 columns, 350 rows
- Column 1: Characteristic wavelength (μm)
- Column 2: Effective bin width (μm)  
- Column 3: Left border (μm)
- Column 4: Right border (μm)

**Range:** 0.03 μm (UV) to 1000 μm (far-IR)

**Why it exists:** SKIRT uses logarithmic wavelength sampling. You need this to map wavelength indices to actual λ values.

---

### FIREbox_haloID.csv
**Format:** 2 columns
```
ARRAKIHS_FIREbox_ID, FIREbox_halo_ID
0, 11
1, 13
...
```

**Use case:** If you need to cross-reference with original FIREbox simulation properties (masses, dark matter profiles, merger histories, etc.)

---

### parameter_scan/ Directory
**Content:** Convergence tests for:
- **Smoothing length:** 4 values (12 pc, h₃₂, 2h₃₂, 4h₃₂)  
  - Fiducial = 2h₃₂ (distance to 32nd nearest neighbor)
- **Photon count:** 3 values (10⁸, 4×10⁸, 1.6×10⁹)
  - Main catalog uses 10⁹ photons (between VHR and UHR)

**Coverage:** Galaxies 00000-00004 (full), plus partial coverage for 6 others.

**File types:**
- Surface brightness maps: `mag_map_{smoothing}_{resolution}.npz`
- Stellar surface density: `{smoothing}_stellar_sigma.fits`

**Why you probably don't care:** The main catalog already uses converged parameters. Only relevant if you're debugging numerical artifacts or writing a methods paper.

---

## Technical Details That Matter

### Radiative Transfer Setup
- **Code:** SKIRT v9 (state-of-the-art dust RT)
- **Stellar synthesis:** Starburst99 (SB99)
- **Photon count:** 10⁹ photons per galaxy
- **Smoothing:** 2h₃₂ = 2 × distance to 32nd nearest stellar neighbor
- **Dust model:** Full extinction + thermal emission

### Spatial Resolution
- **Field of view:** 1.31 Mpc × 1.31 Mpc (comoving)
- **Pixel count:** 2051 × 2051 = 4.2 million pixels
- **Pixel size:** 640 pc physical = 3.2 arcsec at 40 Mpc
- **Angular resolution:** Well-resolved for ARRAKIHS (0.1 arcsec PSF)

### Wavelength Coverage
- **UV:** 0.03-0.4 μm (stellar light)
- **Optical:** 0.4-1.0 μm (ARRAKIHS filters here)
- **NIR:** 1.0-5 μm (includes Euclid-Y, J)
- **MIR-FIR:** 5-1000 μm (dust emission, probably irrelevant for streams)

---

## What You Should Actually Use

### For SAM2 Stellar Stream Training:

1. **Input Images:**  
   Load `mag_map_i0_total.npz` or `mag_map_i90_total.npz`  
   Use Euclid-VIS filter `[1]` or Euclid-J `[2]`  
   → These show stellar streams clearly in surface brightness

2. **Multi-orientation augmentation:**  
   Use all 9 orientations per galaxy  
   → Robust to viewing angle

3. **Color information (optional):**  
   Combine HST-F475X, VIS, Y, J into RGB composite  
   → May help distinguish streams from disk/halo light

4. **Catalog preview:**  
   Use `Catalogue.pdf` to select the ~20 best galaxies with obvious streams  
   → Don't waste time on boring galaxies

---

## Data Quality Notes

### Good:
✓ High spatial resolution (640 pc/pixel)  
✓ Realistic dust effects  
✓ Multiple orientations  
✓ Converged numerical parameters  
✓ Well-documented  

### Limitations:
✗ Only 51 galaxies (small sample)  
✗ Stellar mass range limited (no dwarfs, no massive ellipticals)  
✗ All central galaxies (no satellites)  
✗ Single snapshot (z=0 only, no evolution)  
✗ No observational noise/PSF (you'll need to add this)

---

## File Naming Convention Decoded

```
smooth-2h_UHR_wd01mw_sb99_dz4_i48a180_transparent.fits.gz
│         │    │       │    │   │       │          
│         │    │       │    │   │       └─ RT type (total/transparent)
│         │    │       │    │   └─────── Orientation (i=inclination, a=azimuth)
│         │    │       │    └─────────── Redshift z=4 lookback (ignore, data is z=0)
│         │    │       └──────────────── Stellar library (Starburst99)
│         │    └──────────────────────── Wind model (wd01mw = Weingartner & Draine 2001 Milky Way)
│         └───────────────────────────── Resolution (UHR ≈ 10⁹ photons)
└─────────────────────────────────────── Smoothing (2h = 2×h₃₂)
```

---

## Workflow Recommendation

### Step 1: Visual inspection
```bash
# Open Catalogue.pdf, identify galaxies with clear stellar streams
# Examples: 00000, 00010, 00007, etc.
```

### Step 2: Load surface brightness maps
```python
from arrahkis_visualization import ARRAKIHSVisualizer

viz = ARRAKIHSVisualizer()
sb_maps = viz.load_surface_brightness('00000', 'i0', 'total')

# Use Euclid-VIS for stellar streams
vis_map = sb_maps['VIS']  # Shape: (2051, 2051), units: mag/arcsec²
```

### Step 3: Create training images
```python
# Convert to log scale for visualization
import numpy as np
log_sb = np.log10(vis_map + 1e-10)

# Clip to dynamic range (19-30.5 mag/arcsec²)
# Stellar streams typically 26-30 mag/arcsec²
```

### Step 4: Manual annotation
- Use your demo_visualization.py to generate PNGs
- Annotate stellar streams manually (Napari, CVAT, LabelMe)
- Create binary masks for SAM2 training

### Step 5: Augmentation
- Use all 9 orientations per galaxy
- Add realistic noise (Poisson + Gaussian)
- Simulate ARRAKIHS PSF (0.1 arcsec Gaussian)
- Random crops/rotations

---

## Key Insight: What Makes This Dataset Valuable

**The Problem:** Real stellar stream observations are rare, heterogeneous, and unlabeled.

**The Solution:** This dataset gives you:
1. **Ground truth:** You know exactly where stars came from (simulation particles)
2. **Controlled variations:** Same galaxy, 9 viewing angles
3. **Physical realism:** Includes dust, proper stellar populations
4. **Scale:** 51 galaxies × 9 orientations = 459 training examples

**The Limitation:** It's synthetic. Your trained SAM2 model must generalize to real Euclid/HST data. Validate on real observations ASAP.

---

## Questions You Might Have

**Q: Why 350 wavelength bins? That's overkill.**  
A: SKIRT uses logarithmic sampling to properly handle dust emission across UV to far-IR. You only care about 4 bins (the filters), but the full SED lets you derive any custom filter.

**Q: What's the difference between total and transparent?**  
A: `transparent` = no dust (stars only). `total` = includes dust extinction and emission. Stellar streams are faint—dust matters. Use `total`.

**Q: Why 2051 pixels instead of 2048?**  
A: Because someone at SKIRT hates power-of-2. Deal with it.

**Q: Do I need the SED cubes or just the surface brightness maps?**  
A: For SAM2 training, surface brightness maps are sufficient. SED cubes are for detailed spectral analysis or custom filter integration.

**Q: Which filter is best for stellar streams?**  
A: Euclid-VIS (700 nm, red optical) or Euclid-J (1.25 μm, NIR). Old stars dominate streams, they're redder. NIR has less dust extinction.

**Q: How do I convert mag/arcsec² to flux?**  
A: You don't need to. Surface brightness in mag/arcsec² is already normalized and ready for visualization. Lower magnitude = brighter (astronomy is backwards).

**Q: Can I ignore the parameter_scan directory?**  
A: Yes, unless you're debugging or writing a methods paper. The main catalog uses converged parameters.

---

## Summary: The Simplest Possible Explanation

**What this dataset is:**  
Pictures of 51 fake galaxies in 9 different angles, with realistic dust, at 4 different colors (filters), ready for machine learning.

**What you should use:**  
The `mag_map_*.npz` files. They're pre-processed surface brightness maps in the ARRAKIHS filters.

**What you should ignore:**  
The giant FITS cubes (unless you need custom filters), the parameter scan (already converged), and most of the metadata (nice to have, not essential).

**What to do next:**  
1. Look at `Catalogue.pdf` to pick interesting galaxies  
2. Run `demo_visualization.py` to make pretty pictures  
3. Annotate stellar streams manually  
4. Train SAM2  
5. Test on real Euclid data

---

## Linus's Final Judgment

**✅ Worth using:**  
- Data structures are clean (FITS + NPZ, standard formats)  
- No unnecessary abstraction layers  
- Documentation is competent (rare in astronomy)  
- Physical parameters are reasonable  

**🟡 Could be better:**  
- Only 51 galaxies (small sample size)  
- File naming convention is verbose but decodable  
- Missing observational realism (noise, PSF)  

**🔴 Watch out for:**  
- Generalization gap: synthetic → real data is non-trivial  
- No labeled stream masks (you have to create them)  
- All galaxies are Milky Way mass (narrow dynamic range)  

**The core insight:**  
This dataset solves one real problem: getting ground truth for stellar stream morphology. But it doesn't solve the problem of labeling the streams or bridging to real data. That's your job.

**Recommendation:**  
Use this for initial SAM2 training, but budget time for creating labels and validating on real observations. The dataset is good enough to start, not good enough to finish.

---

## File Count Quick Reference

Per galaxy directory (typical):
- 18 FITS files (9 orientations × 2 RT types) @ ~700 MB each → ~12 GB
- 18 NPZ files (9 orientations × 2 RT types) @ ~200 MB each → ~3.6 GB
- **Total per galaxy:** ~15 GB
- **Total dataset:** ~800 GB (51 galaxies)

---

*"Bad programmers worry about code. Good programmers worry about data structures and their relationships."*  
— Some Finnish guy who knows what he's talking about

