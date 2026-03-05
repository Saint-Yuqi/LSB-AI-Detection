"""
Forward Observation Noise Model for LSB Surface Brightness Maps.

Physics Chain:
    SB(mag/arcsec²) → flux → counts → +sky → Poisson(total) → +N(0,σ_read) → −sky → mag

Usage:
    from src.noise import ForwardObservationModel

    model = ForwardObservationModel.from_target_snr(
        target_snr=10, sb_map=sb_map, zeropoint=22.5,
        sky_level=200.0, read_noise=5.0,
        signal_quantile=0.90, background_quantile=0.20, seed=42,
    )
    noisy_mag = model.inject(sb_map)

Args (constructor):
    zeropoint      : float  – Mag zero-point (must match preprocessing), default 22.5
    signal_scale   : float  – Effective gain: counts = flux × signal_scale
    sky_level      : float  – Constant sky background (counts/pixel)
    read_noise     : float  – Gaussian σ CCD read noise (e⁻/pixel)
    signal_quantile: float  – Top-p% of counts_signal used as "signal region"
    background_quantile: float – Bottom-q% used as "background region"
    seed           : int|None – Per-instance RNG seed (isolated Generator)

Invariants:
    - All pixel ops are vectorised numpy (no Python for-loops on data).
    - Negative flux after sky subtraction → NaN in mag output
      (LSBPreprocessor already maps NaN → 35.0 mag via nan_to_num).
    - Each instance owns an isolated np.random.Generator.
"""

from __future__ import annotations

from typing import Optional

import numpy as np


class ForwardObservationModel:
    """Inject realistic photon-counting noise into clean SB magnitude maps."""

    def __init__(
        self,
        zeropoint: float = 22.5,
        signal_scale: float = 1e4,
        sky_level: float = 200.0,
        read_noise: float = 5.0,
        signal_quantile: float = 0.90,
        background_quantile: float = 0.20,
        seed: Optional[int] = None,
    ) -> None:
        self.zeropoint = zeropoint
        self.signal_scale = signal_scale
        self.sky_level = sky_level
        self.read_noise = read_noise
        self.signal_quantile = signal_quantile
        self.background_quantile = background_quantile
        # Isolated RNG — safe for multi-process batch generation
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Core pipeline
    # ------------------------------------------------------------------

    def inject(self, sb_map: np.ndarray) -> np.ndarray:
        """
        Apply full forward observation model to a clean SB magnitude map.

        Returns mag/arcsec² float32 array (same shape).
        Pixels where sky-subtracted flux ≤ 0 are set to NaN.
        """
        # 1. Mag → Flux (linear)
        flux = np.power(10.0, (self.zeropoint - sb_map) / 2.5)

        # 2. Flux → Counts (effective gain)
        counts_signal = flux * self.signal_scale

        # 3. Add sky background
        counts_total = counts_signal + self.sky_level

        # 4. Poisson shot noise on TOTAL counts (signal + sky)
        #    np.random.Generator.poisson accepts full arrays — vectorised
        counts_poisson = self._rng.poisson(counts_total).astype(np.float64)

        # 5. CCD read noise (Gaussian)
        counts_final = counts_poisson + self._rng.normal(
            0.0, self.read_noise, size=counts_poisson.shape
        )

        # 6. Sky subtraction (calibration)
        counts_sky_sub = counts_final - self.sky_level

        # 7. Counts → Flux
        flux_noisy = counts_sky_sub / self.signal_scale

        # 8. Flux → Mag  (negative flux → NaN, not clipped)
        mag_noisy = np.full_like(flux_noisy, np.nan, dtype=np.float32)
        valid = flux_noisy > 0
        mag_noisy[valid] = (
            self.zeropoint - 2.5 * np.log10(flux_noisy[valid])
        ).astype(np.float32)

        return mag_noisy

    # ------------------------------------------------------------------
    # SNR measurement (quantile-based, GT-free)
    # ------------------------------------------------------------------

    def compute_snr(self, sb_map: np.ndarray) -> float:
        """
        Empirical SNR from one realisation.

        signal  = mean of counts_signal in top-p% brightest pixels
        noise   = std  of (counts_final − sky) in bottom-q% faintest pixels
        """
        flux = np.power(10.0, (self.zeropoint - sb_map) / 2.5)
        counts_signal = flux * self.signal_scale
        counts_total = counts_signal + self.sky_level

        counts_poisson = self._rng.poisson(counts_total).astype(np.float64)
        counts_final = counts_poisson + self._rng.normal(
            0.0, self.read_noise, size=counts_poisson.shape
        )
        counts_sky_sub = counts_final - self.sky_level

        # Quantile partitioning on CLEAN counts_signal (deterministic regions)
        flat_signal = counts_signal.ravel()
        flat_noisy = counts_sky_sub.ravel()

        sig_thresh = np.quantile(flat_signal, self.signal_quantile)
        bkg_thresh = np.quantile(flat_signal, self.background_quantile)

        sig_mask = flat_signal >= sig_thresh
        bkg_mask = flat_signal <= bkg_thresh

        mean_sig = float(np.mean(flat_noisy[sig_mask]))
        std_bkg = float(np.std(flat_noisy[bkg_mask]))

        if std_bkg < 1e-12:
            return np.inf
        return mean_sig / std_bkg

    # ------------------------------------------------------------------
    # Analytic expected SNR (no random sampling needed)
    # ------------------------------------------------------------------

    def expected_snr(self, sb_map: np.ndarray) -> float:
        """
        Analytic expected SNR using the variance decomposition:

            Var[background_pixel] ≈ (counts_signal_bkg + sky_level) + read_noise²
            E[signal_pixel]       = counts_signal_top

        where counts_signal_bkg ~ bottom-q% and counts_signal_top ~ top-p%.
        Deterministic — no random draws.
        """
        flux = np.power(10.0, (self.zeropoint - sb_map) / 2.5)
        counts_signal = flux * self.signal_scale

        flat = counts_signal.ravel()
        sig_thresh = np.quantile(flat, self.signal_quantile)
        bkg_thresh = np.quantile(flat, self.background_quantile)

        sig_mask = flat >= sig_thresh
        bkg_mask = flat <= bkg_thresh

        # Expected signal: mean of counts_signal in top-p%
        mean_signal = float(np.mean(flat[sig_mask]))

        # Expected background variance: Poisson + read noise
        bkg_counts_total = flat[bkg_mask] + self.sky_level  # counts_signal + sky
        mean_var = float(np.mean(bkg_counts_total)) + self.read_noise ** 2
        std_bkg = np.sqrt(mean_var)

        if std_bkg < 1e-12:
            return np.inf
        return mean_signal / std_bkg

    # ------------------------------------------------------------------
    # Factory: auto-tune signal_scale to hit target SNR
    # ------------------------------------------------------------------

    @staticmethod
    def from_target_snr(
        target_snr: float,
        sb_map: np.ndarray,
        zeropoint: float = 22.5,
        sky_level: float = 200.0,
        read_noise: float = 5.0,
        signal_quantile: float = 0.90,
        background_quantile: float = 0.20,
        seed: Optional[int] = None,
        *,
        tol: float = 0.5,
        max_iter: int = 40,
        scale_lo: float = 1.0,
        scale_hi: float = 1e8,
    ) -> "ForwardObservationModel":
        """
        Find signal_scale that yields expected_snr ≈ target_snr.

        Uses bisection on the analytic expected_snr (deterministic,
        monotone in signal_scale). ~15 iterations typical.

        Args:
            target_snr: desired SNR value
            sb_map    : reference SB map to calibrate against
            tol       : acceptable |snr - target| tolerance
            max_iter  : bisection ceiling
            scale_lo/hi: search bracket for signal_scale
        """
        # Pre-compute flux once (reused every iteration)
        flux = np.power(10.0, (zeropoint - sb_map) / 2.5)
        flat = (flux).ravel()

        sig_thresh = np.quantile(flat, signal_quantile)
        bkg_thresh = np.quantile(flat, background_quantile)
        sig_mask = flat >= sig_thresh
        bkg_mask = flat <= bkg_thresh

        # Mean flux in signal/background regions (scale-independent)
        mean_flux_sig = float(np.mean(flat[sig_mask]))
        mean_flux_bkg = float(np.mean(flat[bkg_mask]))

        def _analytic_snr(scale: float) -> float:
            """SNR as function of signal_scale — closed-form."""
            mean_signal = mean_flux_sig * scale
            mean_var = (mean_flux_bkg * scale + sky_level) + read_noise ** 2
            return mean_signal / np.sqrt(mean_var)

        # Bisection (SNR is monotone increasing in signal_scale)
        lo, hi = scale_lo, scale_hi
        for _ in range(max_iter):
            mid = np.sqrt(lo * hi)  # geometric midpoint — better for log-scale
            snr_mid = _analytic_snr(mid)
            if abs(snr_mid - target_snr) < tol:
                break
            if snr_mid < target_snr:
                lo = mid
            else:
                hi = mid

        return ForwardObservationModel(
            zeropoint=zeropoint,
            signal_scale=float(mid),
            sky_level=sky_level,
            read_noise=read_noise,
            signal_quantile=signal_quantile,
            background_quantile=background_quantile,
            seed=seed,
        )
