# PNbody VIS2 Notes And Integration Plan

This note summarizes what the locally installed `mockimgs_sb_compute_images` / `pNbody` can do, what the existing `FIREbox-DR1/SB_maps/*.fits.gz` were actually built with, and how to prepare new PNbody-generated data so it plugs into our current `04_noise` and render pipeline with minimal friction.

## Executive Summary

Short answers to the main questions:

1. Can we switch to the same filter as the existing FITS (`BPASS230_ARK_VIS2`)?
   Yes. In the current `sam3` environment, the built-in instrument `arrakihs_vis_VIS2` already uses `BPASS230_ARK_VIS2`.

2. Does that automatically make the output match the existing FITS?
   No. The filter only controls the stellar-population synthesis bandpass. The current built-in VIS instrument geometry is:
   - `4096 x 4096`
   - `10 micron` pixels
   - `1.3751 arcsec/pixel`

   The existing `FIREbox-DR1/SB_maps` VIS2 FITS use a different custom instrument:
   - `1072 x 1072`
   - `24 micron` pixels
   - `3.3002 arcsec/pixel`

3. What is `--rsp_val`?
   It is only used when `--rsp_mode const`. In that mode every particle gets the same smoothing length before projection. It is not what the existing FITS used.

4. What did the existing FITS use for smoothing?
   The existing FITS headers show:
   - `RSPMODE = None`
   - `RSPFAC = 0.6`

   In the installed code, that means:
   - keep each particle's original `rsp`
   - multiply it by `0.6`

5. Why is `--los 1 0 0` used?
   In pNbody, `--los x y z` is a direction vector for the observer position looking at the origin, not Euler angles. So `--los 1 0 0` means "observer on +x, looking toward the origin". The default is along +z, i.e. `[0, 0, 1]`.

6. What is the safest path for new data?
   If the goal is compatibility with the existing `FIREbox-DR1` VIS2 images and masks, use a custom legacy instrument matching the old geometry, keep the exact same FITS basename pattern, and store noisy outputs under a parallel sub-root like `data/04_noise/pnbody_legacy/` to avoid filename collisions.

## What The Current Installed pNbody Supports

Local source used:
- `sam3` env executable: `/shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/sam3/bin/mockimgs_sb_compute_images`
- package source: `/shares/feldmann.ics.mnf.uzh/Yuqi/conda/envs/sam3/lib/python3.12/site-packages/pNbody`

### Built-in Instruments In The Current `sam3` Environment

All ARRAKIHS VIS instruments share the same geometry:
- telescope: `iSIM-170`
- focal: `1500 mm`
- CCD: `CIS-300`
- shape: `4096 x 4096`
- pixel size: `10 micron`
- pixel scale: `1.375098708293604 arcsec/pixel`
- FOV: `1.5641680537665499 deg`

ARRAKIHS VIS defaults:

| Instrument | Default filter |
| --- | --- |
| `arrakihs_vis_GAIA_VIS` | `GAEA_VIS` |
| `arrakihs_vis_G` | `BastI_GAIA_G` |
| `arrakihs_vis_G_RP` | `BastI_GAIA_G_RP` |
| `arrakihs_vis_G_BP` | `BastI_GAIA_G_BP` |
| `arrakihs_vis_SDSSu` | `BastI_SDSS_u` |
| `arrakihs_vis_SDSSg` | `BastI_SDSS_g` |
| `arrakihs_vis_SDSSr` | `BastI_SDSS_r` |
| `arrakihs_vis_SDSSi` | `BastI_SDSS_i` |
| `arrakihs_vis_SDSSz` | `BastI_SDSS_z` |
| `arrakihs_vis_F475X` | `BastI_HST_F475X` |
| `arrakihs_vis_VIS` | `BastI_Euclid_VIS` |
| `arrakihs_vis_VIS1` | `BPASS230_ARK_VIS1` |
| `arrakihs_vis_VIS2` | `BPASS230_ARK_VIS2` |

All ARRAKIHS NIR instruments share the same geometry:
- telescope: `iSIM-170`
- focal: `1500 mm`
- CCD: `Hawaii-2RG`
- shape: `2048 x 2048`
- pixel size: `18 micron`
- pixel scale: `2.475177674846348 arcsec/pixel`
- FOV: `1.4078176918108052 deg`

ARRAKIHS NIR defaults:

| Instrument | Default filter |
| --- | --- |
| `arrakihs_nir_GAIA_Y` | `GAEA_Y` |
| `arrakihs_nir_GAIA_J` | `GAEA_J` |
| `arrakihs_nir_J` | `BastI_Euclid_J` |
| `arrakihs_nir_Y` | `BastI_Euclid_Y` |
| `arrakihs_nir_H` | `BastI_Euclid_H` |
| `arrakihs_nir_NIR1` | `BPASS230_ARK_NIR1` |
| `arrakihs_nir_NIR2` | `BPASS230_ARK_NIR2` |

All DES/DECam instruments share the same geometry:
- telescope: `CTIO4.0m`
- focal: `11.81 m`
- CCD: `DECam`
- shape: `2290 x 2290`
- pixel size: `15 micron`
- pixel scale: `0.26197900878109914 arcsec/pixel`
- FOV: `0.1666472884373729 deg`

DES/DECam defaults:

| Instrument | Default filter |
| --- | --- |
| `DES_DECam_SDSSg` | `BastI_SDSS_g` |
| `DES_DECam_SDSSr` | `BastI_SDSS_r` |
| `DES_DECam_SDSSi` | `BastI_SDSS_i` |
| `DES_DECam_SDSSz` | `BastI_SDSS_z` |

### Available Filters

The installed `--filters_list` contains these families:

- `GAEA_*`: `GAEA_B`, `GAEA_VIS`, `GAEA_Y`, `GAEA_J`
- `CMD_*`: `CMD_F475X`, `CMD_VIS`, `CMD_VISb`, `CMD_Y`, `CMD_J`
- `BastI_*`: Gaia, SDSS, HST, Euclid, JKC
- `SB99_*`: SDSS, ARK VIS/NIR
- `BPASS221_*`: SDSS
- `BPASS230_*`: ARK VIS/NIR, SDSS, Euclid, JKC

The exact local list is also copied in:
- `/home/yuqyan/Yuqi/LSB-AI-Detection/references/filters_PNbody.txt`

Important behavior:
- `--instrument` sets geometry plus a default filter.
- `--filter` can override the filter without changing the CCD/telescope geometry.
- So `arrakihs_vis_G --filter BPASS230_ARK_VIS2` still keeps the `4096 / 10 micron` VIS geometry.

## What The Existing `FIREbox-DR1/SB_maps` VIS2 FITS Actually Use

Directly inspected from:
- `data/01_raw/LSB_and_Satellites/FIREbox-DR1/SB_maps/magnitudes-Fbox-11-eo-VIS2.fits.gz`

Observed header values:

| Header key | Existing value |
| --- | --- |
| `FILTER` | `BPASS230_ARK_VIS2` |
| `NAME` | `arrakihs_vis` |
| `NAXIS1`, `NAXIS2` | `1072`, `1072` |
| `PIXSIZEX`, `PIXSIZEY` | `24.0`, `24.0` micron |
| `PIXFOVX`, `PIXFOVY` | `3.300236899671922`, `3.300236899671922` arcsec |
| `FOVX`, `FOVY` | `0.9826408563674109`, `0.9826408563674109` deg |
| `OBJ_LOS` | `[1.000,0.000,0.000]` for `eo` |
| `RSPMODE` | `None` |
| `RSPFAC` | `0.6` |

The FITS header comment also stores the original generation command. It shows that the existing data were not produced with a built-in instrument name, but with a custom python instrument file:

- `--instrument=/.../ARK_VIS2.py`
- `--distance=35`
- `--los 1 0 0`
- `--rsp_mode=None`
- `--rsp_fac=0.6`

For `Galaxy 11`, we also checked the two existing orientations:

| Filename orientation | Existing `OBJ_LOS` |
| --- | --- |
| `eo` | `[1,0,0]` |
| `fo` | `[0,0,1]` |

### Inferred Legacy Instrument Definition

We did not recover the original `ARK_VIS2.py` file itself, but from the existing FITS headers its behavior is straightforward to reconstruct:

```python
from astropy import units as u
from pNbody.Mockimgs import instrument, telescope, filters, ccd

instrument = instrument.Instrument(
    name="arrakihs_vis",
    telescope=telescope.Telescope(name="iSIM-170", focal=1500 * u.mm),
    ccd=ccd.CCD(
        name="arrakihs_vis_legacy",
        shape=[1072, 1072],
        pixel_size=[24 * u.micron, 24 * u.micron],
    ),
    filter_type=filters.Filter("BPASS230_ARK_VIS2"),
)
```

That exact geometry reproduces the observed header values:
- `1072 x 1072`
- `24 micron`
- `3.3002369 arcsec/pixel`
- `0.982640856 deg` FOV

## Detailed Meaning Of The Main CLI Options

### `--instrument`

Can be either:
- the name of a built-in instrument, e.g. `arrakihs_vis_VIS2`
- a python file path defining an `instrument` object, e.g. `configs/pnbody/ARK_VIS2_legacy.py`

This controls:
- telescope focal length
- detector shape
- pixel size
- field of view
- default filter

This is the main knob for geometry.

### `--filter`

Overrides only the filter object after the instrument is loaded.

Use this when:
- you want the same CCD/telescope geometry
- but a different SPS filter

Do not use this if your real goal is to match the legacy `1072 x 1072 / 24 micron` data. That requires a custom legacy instrument, not just a filter change.

### `--fov`

In the current code, `instrument.change_fov()` rescales the CCD shape to keep pixel size fixed.

Implication:
- `--fov` changes the number of pixels
- but it does not change physical pixel size

So `--fov` is not enough to turn the built-in `4096 / 10 micron` ARRAKIHS VIS instrument into the legacy `1072 / 24 micron` one.

### `--rsp_mode`, `--rsp_val`, `--rsp_max`, `--rsp_sca`, `--rsp_fac`

In the local `pNbody.Mockimgs.obs_object.Object.ScaleSmoothingLength()` code:

- if `mode == "const"`:
  - `rsp = val`
- if `mode == "arctan"`:
  - `rsp = max / (pi/2) * arctan(x / sca)`
- if `mode == "ln"`:
  - `rsp = sca * ln(x / sca + 1)`
- otherwise:
  - `rsp = x`

After that, all modes are multiplied by `fac`.

Interpretation:

| Setting | Meaning |
| --- | --- |
| `--rsp_mode const --rsp_val 1.5` | force all particles to the same smoothing length, here `1.5` before projection |
| `--rsp_mode None --rsp_fac 0.6` | keep each particle's original `rsp`, then shrink by `0.6` |
| `--rsp_fac` | multiplicative factor applied at the end in all modes |

For compatibility with the existing `FIREbox-DR1/SB_maps` VIS2 FITS:
- do not use `--rsp_mode const`
- do use `--rsp_mode None --rsp_fac 0.6`

### `--los`

From local source `pNbody/Mockimgs/los.py`:
- default LOS is `[0, 0, 1]`
- pNbody interprets `[1,0,0]` as "observer at +x looking toward the origin"

This is not an inclination angle. It is a direction vector.

Practical mapping we observed in the existing dataset:

| Orientation token | Recommended LOS |
| --- | --- |
| `eo` | `1 0 0` |
| `fo` | `0 0 1` |

### `--nlos`, `--random_los`, `--random_seed`, `-p params.yml`

Supported ways to generate multiple sightlines:

- single explicit LOS:
  - `--los 0 0 1`
- random LOS set:
  - `--nlos 9 --random_los --random_seed 42`
- deterministic hemisphere sampling:
  - `--nlos 9`
- YAML-driven LOS grid:

```yaml
LineOfSights:
  los: [0, 1, 0]
  grid:
    n_phi: 3
    n_theta: 3
    d_phi: 10
    d_theta: 10
```

This creates `3 x 3 = 9` LOS directions centered on `[0,1,0]` and perturbed within the specified angular box.

### `--mapping_method`

Supported local values:
- `gauss`
- `spline`
- `gauss_old` is present in help but intentionally rejected in code

Current default in the local instrument class is `gauss`.

### Other Useful Options

| Option | Meaning |
| --- | --- |
| `--magFromField FIELD` | take magnitudes from a field already stored in the HDF5 |
| `--psf FILE` | convolve the flux map with a PSF FITS |
| `--output-flux` | save flux instead of mag/arcsec^2 |
| `--IDsMap file.npy` | save particle ID map instead of image |
| `--remove_identical_coords` | remove duplicated 3D coordinates |
| `--distance 35` | object distance in Mpc |

### CLI Gotcha In The Installed Version

The help text suggests:

- `mockimgs_sb_compute_images --instrument arrakihs_vis_G --info`

But the installed script accesses `opt.file[0]` before checking `opt.info`, so this crashes if no input file is given. Use `--instruments_list` or inspect the source directly instead.

## Recommended Naming And Directory Scheme

### FITS Basename

Keep the exact existing basename pattern:

`magnitudes-Fbox-{galaxy_id}-{orientation}-VIS2.fits.gz`

Reasons:
- current `generate_noisy_fits.py` and `render_noisy_fits.py` already assume it
- current regex in `render_noisy_fits.py` matches only this pattern
- downstream base keys are derived from the same `{galaxy_id}` and `{orientation}`
- keeping the same galaxy IDs preserves mask alignment with the existing `FIREbox-DR1` masks

Do not add extra suffixes like:
- `_pnbody`
- `_legacy`
- `_bpass`

Those would require code changes in multiple places.

### Where To Put New Clean FITS

Recommended staging root:

`data/01_raw/PNbody_FIREbox_VIS2_legacy/SB_maps/`

Example:

`data/01_raw/PNbody_FIREbox_VIS2_legacy/SB_maps/magnitudes-Fbox-11-eo-VIS2.fits.gz`

Why:
- mirrors the existing `FIREbox-DR1/SB_maps` layout
- can be consumed by the current noise script with only a config change
- keeps the clean generated images separate from consortium-delivered originals

### Where To Put New Noisy FITS

Recommended root:

`data/04_noise/pnbody_legacy/{snr_profile}/`

Examples:

- `data/04_noise/pnbody_legacy/snr05/magnitudes-Fbox-11-eo-VIS2.fits.gz`
- `data/04_noise/pnbody_legacy/snr20/magnitudes-Fbox-11-fo-VIS2.fits.gz`

Why this is better than writing directly into `data/04_noise/snr05/`:
- no collision with the existing noise set
- same basename pattern, so downstream logic still works
- only the root path changes in config

If you write directly into:
- `data/04_noise/snr05/`
- `data/04_noise/snr10/`

you will overwrite or mix with the current noisy FIREbox files for the same galaxy IDs.

## Practical Plan For Preparing A New Compatible Dataset

### Plan A: Compatibility-First Legacy VIS2 Set

Goal:
- preserve compatibility with existing masks and current evaluation logic
- minimize domain gap with the current `FIREbox-DR1/SB_maps`

Steps:

1. Create a custom legacy instrument file matching the old geometry.
   - suggested file: `configs/pnbody/ARK_VIS2_legacy.py`
   - geometry: `1072 x 1072`, `24 micron`, `BPASS230_ARK_VIS2`

2. Generate clean PNbody FITS into a staging root.
   - root: `data/01_raw/PNbody_FIREbox_VIS2_legacy/SB_maps`
   - naming: `magnitudes-Fbox-{id}-{eo|fo}-VIS2.fits.gz`

3. Use the legacy LOS convention.
   - `eo -> --los 1 0 0`
   - `fo -> --los 0 0 1`

4. Use the legacy smoothing convention.
   - `--rsp_mode None --rsp_fac 0.6`

5. Validate a few galaxies against the original VIS2 FITS.
   - header geometry
   - visual morphology
   - surface-brightness range
   - pixel scale

6. Create a dedicated noise config for the new clean set.
   - input root: `data/01_raw/PNbody_FIREbox_VIS2_legacy`
   - output root: `data/04_noise/pnbody_legacy`

7. Run the existing noise pipeline unchanged except for config path.

8. Run the existing render pipeline on the new noise config.
   - no filename changes needed
   - rendered keys will still be `{gid:05d}_{orient}`

### Plan B: Modern High-Resolution PNbody Set

Goal:
- exploit the current built-in pNbody VIS2 instrument
- use `4096 x 4096` outputs as an additional augmentation domain

Settings:
- instrument: `arrakihs_vis_VIS2`
- filter: default `BPASS230_ARK_VIS2`
- geometry: `4096 / 10 micron`

This is useful for experimentation, but it is not the best first choice if the priority is close compatibility with the existing `FIREbox-DR1` VIS2 data.

Recommended root if you do this later:

`data/04_noise/pnbody_modern/{snr_profile}/`

## Concrete Command Template

Compatibility-first generation:

```bash
conda run -n sam3 mockimgs_sb_compute_images \
  /path/to/pnbody_Fbox_halo_11.hdf5 \
  --instrument /home/yuqyan/Yuqi/LSB-AI-Detection/configs/pnbody/ARK_VIS2_legacy.py \
  --distance 35 \
  --los 1 0 0 \
  --rsp_mode None \
  --rsp_fac 0.6 \
  -o /tmp/magnitudes-Fbox-11-eo-VIS2.fits
```

Modern built-in VIS2 generation:

```bash
conda run -n sam3 mockimgs_sb_compute_images \
  /path/to/pnbody_Fbox_halo_11.hdf5 \
  --instrument arrakihs_vis_VIS2 \
  --distance 35 \
  --los 1 0 0 \
  --rsp_mode None \
  --rsp_fac 0.6 \
  -o /tmp/magnitudes-Fbox-11-eo-VIS2.fits
```

The second command matches the filter but still does not match the legacy geometry.

## Recommended Next Files To Add

If we move from planning to implementation, the next useful repo files are:

1. `configs/pnbody/ARK_VIS2_legacy.py`
   - inferred old VIS2 instrument geometry

2. `configs/noise_profiles_pnbody_legacy.yaml`
   - `firebox_root: data/01_raw/PNbody_FIREbox_VIS2_legacy`
   - `output_root: data/04_noise/pnbody_legacy`

3. `scripts/generate_pnbody_vis2_legacy.py`
   - batch wrapper around `mockimgs_sb_compute_images`
   - maps halo IDs to HDF5 files and `eo/fo` LOS vectors

4. `data/01_raw/PNbody_FIREbox_VIS2_legacy/manifest.csv`
   - columns:
   - `galaxy_id`
   - `orientation`
   - `los_x`
   - `los_y`
   - `los_z`
   - `source_hdf5`
   - `output_fits`
   - `rsp_mode`
   - `rsp_fac`

## Final Recommendation

If the immediate goal is "new data that drop into the current logic with the least pain", the best route is:

- use the same basename pattern as the existing VIS2 FITS
- keep the same galaxy IDs and `eo` / `fo` tokens
- reproduce the legacy geometry with a custom `ARK_VIS2_legacy.py`
- use `--rsp_mode None --rsp_fac 0.6`
- use `eo -> [1,0,0]`, `fo -> [0,0,1]`
- store noisy outputs under `data/04_noise/pnbody_legacy/` instead of overwriting `data/04_noise/snrXX/`

That gives us compatibility, avoids collisions, and keeps almost all current code unchanged.
