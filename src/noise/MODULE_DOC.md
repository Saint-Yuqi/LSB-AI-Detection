# Module: noise

## Responsibilities
- Model forward-observation noise for LSB surface-brightness maps.
- Inject physically motivated Poisson + read-noise perturbations in count space.
- Provide SNR estimation and target-SNR calibration utilities.

## Non-goals
- **No Rendering:** Does not convert arrays to PNG/JPEG assets.
- **No Dataset Traversal:** Does not iterate over folders or orchestrate batch jobs.

## Inputs / Outputs

### `ForwardObservationModel.__init__(zeropoint, signal_scale, sky_level, read_noise, signal_quantile, background_quantile, seed)`
- **Input:**
  - `zeropoint: float`
  - `signal_scale: float`
  - `sky_level: float`
  - `read_noise: float`
  - `signal_quantile: float`
  - `background_quantile: float`
  - `seed: Optional[int]`
- **Output:** Noise model instance with isolated RNG.

### `ForwardObservationModel.inject(sb_map)`
- **Input:** `sb_map: np.ndarray` — shape `(H, W)`, dtype float (mag/arcsec²)
- **Output:** `np.ndarray` — shape `(H, W)`, dtype `np.float32` (noisy mag map, invalid flux mapped to `NaN`)

### `ForwardObservationModel.compute_snr(sb_map)`
- **Input:** `sb_map: np.ndarray` — shape `(H, W)`, dtype float
- **Output:** `float` — empirical SNR from one stochastic realization

### `ForwardObservationModel.expected_snr(sb_map)`
- **Input:** `sb_map: np.ndarray` — shape `(H, W)`, dtype float
- **Output:** `float` — analytic expected SNR (deterministic)

### `ForwardObservationModel.from_target_snr(target_snr, sb_map, zeropoint, sky_level, read_noise, signal_quantile, background_quantile, seed, tol, max_iter, scale_lo, scale_hi)`
- **Input:** Target SNR and calibration/reference parameters.
- **Output:** `ForwardObservationModel` tuned so `expected_snr(sb_map) ≈ target_snr`.

## Invariants
- Operations are fully vectorized over pixels (`numpy` array math).
- RNG state is instance-local (`np.random.Generator`) for reproducibility and multi-process safety.
- Negative sky-subtracted flux is represented as `NaN` in magnitude space.

## Produced Artifacts
- In-memory noisy SB maps and calibrated model instances.

## Failure Modes
- `ValueError`: Raised by NumPy quantile/stat operations when input arrays are empty.
- `FloatingPointError`: Potentially raised if runtime is configured to treat invalid operations (e.g. `log10` on non-positive values) as exceptions.
