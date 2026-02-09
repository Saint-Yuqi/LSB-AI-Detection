"""
Astrophysics-standard preprocessing for Low Surface Brightness (LSB) data.

Implements the proper astronomical pipeline for FITS magnitude maps:
    Raw FITS (mag/arcsec²) → Flux Conversion → Asinh Stretch → 8-bit RGB

This approach preserves dynamic range much better than linear magnitude scaling,
allowing both faint streams and bright galaxy cores to be visible simultaneously.

Pipeline Details:
1. Magnitude → Flux: flux = 10^((zeropoint - mag) / 2.5)
2. Background subtraction (median)
3. Percentile normalization
4. Asinh stretch: arcsinh(x * nonlinearity) / arcsinh(nonlinearity)
5. Convert to 8-bit RGB
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np


class LSBPreprocessor:
    """
    Astrophysics-standard preprocessor for Low Surface Brightness data.
    
    Implements the proper pipeline: Magnitude → Flux → Asinh Stretch
    
    This is standard in astrophysics visualization because:
    - Asinh is linear for small values (preserves faint features)
    - Asinh is logarithmic for large values (compresses bright cores)
    - Together they maximize visible dynamic range
    
    Attributes:
        zeropoint: Magnitude zeropoint for flux conversion (default 22.5)
        nonlinearity: Asinh stretch parameter (higher = more contrast)
        clip_percentile: Percentile for normalization reference
        target_size: Output image dimensions (width, height)
    """
    
    def __init__(
        self,
        zeropoint: float = 22.5,
        nonlinearity: float = 10.0,
        clip_percentile: float = 99.5,
        target_size: Tuple[int, int] = (1024, 1024)
    ):
        """
        Initialize the LSB preprocessor.
        
        Args:
            zeropoint: Magnitude zeropoint for flux conversion (default 22.5)
            nonlinearity: Asinh stretch parameter (200-300, higher = more contrast)
            clip_percentile: Percentile for vmax reference (default 99.5)
            target_size: Target output size as (width, height)
        """
        self.zeropoint = zeropoint
        self.nonlinearity = nonlinearity
        self.clip_percentile = clip_percentile
        self.target_size = target_size
    
    def mag_to_flux(self, mag: np.ndarray) -> np.ndarray:
        """
        Convert magnitude (mag/arcsec²) to flux.
        
        Formula: flux = 10^((zeropoint - mag) / 2.5)
        
        This inverts the magnitude scale:
        - Lower magnitude (brighter) → Higher flux
        - Higher magnitude (fainter) → Lower flux
        
        Args:
            mag: Magnitude array (lower = brighter)
            
        Returns:
            Flux array (higher = brighter)
        """
        return np.power(10.0, (self.zeropoint - mag) / 2.5)
    
    def asinh_stretch(self, flux: np.ndarray) -> np.ndarray:
        """
        Apply Asinh stretch to flux data.
        
        Formula: arcsinh(x * nonlinearity) / arcsinh(nonlinearity)
        
        Properties:
        - Linear for small x (preserves faint features like streams)
        - Logarithmic for large x (compresses bright cores)
        - Smooth transition between the two
        
        Args:
            flux: Normalized flux array (0 to ~1)
            
        Returns:
            Stretched array (0 to 1)
        """
        return np.arcsinh(flux * self.nonlinearity) / np.arcsinh(self.nonlinearity)
    
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """
        Process a surface brightness MAGNITUDE map to 8-bit RGB.
        
        Complete pipeline: Mag → Flux → Background Sub → Normalize → Asinh → 8-bit
        
        Args:
            sb_map: Surface brightness map in mag/arcsec² (LOWER = BRIGHTER)
            
        Returns:
            8-bit RGB image with shape (H, W, 3)
        """
        # 1. Clean Data (Handle NaNs and Infinities)
        # Set invalid values to very faint (high magnitude = low flux after conversion)
        sb_clean = np.nan_to_num(sb_map, nan=35.0, posinf=35.0, neginf=20.0)
        
        # 2. Convert Magnitude → Flux
        # This inverts the scale: lower mag → higher flux
        flux = self.mag_to_flux(sb_clean)
        
        # 3. Background Subtraction (using median)
        bg = np.median(flux)
        flux_sub = np.maximum(flux - bg, 0)
        
        # 4. Normalize by percentile (use as scaling reference)
        vmax = np.percentile(flux_sub, self.clip_percentile)
        if vmax <= 0:
            vmax = flux_sub.max()
        if vmax <= 0:
            vmax = 1.0  # Avoid division by zero for empty images
        
        # Normalize to 0~1 range (can exceed 1 for very bright cores)
        flux_norm = flux_sub / (vmax + 1e-10)
        
        # 5. Asinh Stretch (THE KEY STEP)
        # This preserves both faint streams and bright cores
        flux_stretched = self.asinh_stretch(flux_norm)
        
        # 6. Convert to 8-bit
        img_8bit = (flux_stretched * 255).clip(0, 255).astype(np.uint8)
        
        # 7. Convert to 3-channel RGB (grayscale duplicated)
        img_rgb = np.stack([img_8bit] * 3, axis=-1)
        
        # 8. Resize if needed (Bilinear for smooth interpolation)
        h, w = img_rgb.shape[:2]
        if (w, h) != self.target_size:
            img_rgb = cv2.resize(
                img_rgb,
                self.target_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        return img_rgb
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize a mask to target size using nearest neighbor interpolation.
        
        Nearest neighbor is used to preserve integer instance IDs.
        
        Args:
            mask: Input mask array (can be binary or instance mask)
            
        Returns:
            Resized mask with same dtype as input
        """
        h, w = mask.shape[:2]
        if (w, h) != self.target_size:
            mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST
            )
        return mask


class LinearMagnitudePreprocessor:
    """
    Direct linear magnitude normalization using global physical limits.
    
    Maps magnitude [global_mag_min, global_mag_max] → pixel [255, 0].
    This preserves cross-galaxy photometric consistency (same magnitude = same pixel value).
    
    Key differences from LSBPreprocessor:
    - NO flux conversion (works directly on magnitude scale)
    - NO asinh stretch (simple linear mapping)
    - NO background subtraction or percentile normalization
    - Uses global fixed range for consistency across ALL galaxies
    
    Physical interpretation:
    - global_mag_min (20.0): Saturation point → White (255)
    - global_mag_max (35.0): Background noise → Black (0)
    - Linear mapping: pixel = 255 * (max - mag) / (max - min)
    
    Use cases:
    - Cross-galaxy photometric comparisons
    - Consistent brightness scaling across datasets
    - Simple preprocessing without adaptive transformations
    
    Attributes:
        global_mag_min: Brightest magnitude (saturation), default 20.0 mag/arcsec²
        global_mag_max: Faintest magnitude (background), default 35.0 mag/arcsec²
        target_size: Output image dimensions (width, height)
        interpolation: cv2 interpolation mode for resizing
    """
    
    def __init__(
        self,
        global_mag_min: float = 20.0,
        global_mag_max: float = 35.0,
        target_size: Tuple[int, int] = (1024, 1024),
        interpolation: int = cv2.INTER_CUBIC
    ):
        """
        Initialize the linear magnitude preprocessor.
        
        Args:
            global_mag_min: Brightest magnitude (saturation point), default 20.0
            global_mag_max: Faintest magnitude (background point), default 35.0
            target_size: Target output size as (width, height)
            interpolation: cv2 interpolation mode (INTER_CUBIC for sharpness)
        """
        self.global_mag_min = global_mag_min
        self.global_mag_max = global_mag_max
        self.target_size = target_size
        self.interpolation = interpolation
    
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """
        Process surface brightness MAGNITUDE map using global linear normalization.
        
        Pipeline: Clean → Clip → Linear Normalize → 8-bit → RGB → Resize
        
        Args:
            sb_map: Surface brightness map in mag/arcsec² (LOWER = BRIGHTER)
            
        Returns:
            8-bit RGB image with shape (H, W, 3)
        """
        # 1. Clean Data (Handle NaNs and Infinities)
        # Replace invalid values with faintest magnitude (→ black after normalization)
        sb_clean = np.nan_to_num(
            sb_map,
            nan=self.global_mag_max,
            posinf=self.global_mag_max,
            neginf=self.global_mag_min
        )
        
        # 2. Clip to Global Range [global_mag_min, global_mag_max]
        # Ensures consistency: same magnitude → same pixel value across ALL galaxies
        sb_clipped = np.clip(sb_clean, self.global_mag_min, self.global_mag_max)
        
        # 3. Linear Normalization to [0, 1]
        # Formula: (Max - Magnitude) / (Max - Min)
        # Lower magnitude (brighter) → Higher value → Whiter pixel
        # Higher magnitude (fainter) → Lower value → Darker pixel
        img_norm = (self.global_mag_max - sb_clipped) / (
            self.global_mag_max - self.global_mag_min
        )
        
        # 4. Convert to 8-bit [0, 255]
        img_8bit = (img_norm * 255).astype(np.uint8)
        
        # 5. Convert to 3-channel RGB (grayscale duplicated)
        img_rgb = np.stack([img_8bit] * 3, axis=-1)
        
        # 6. Resize (using specified interpolation for sharpness)
        h, w = img_rgb.shape[:2]
        if (w, h) != self.target_size:
            img_rgb = cv2.resize(
                img_rgb,
                self.target_size,
                interpolation=self.interpolation
            )
        
        return img_rgb
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Resize a mask to target size using nearest neighbor interpolation.
        
        Nearest neighbor is used to preserve integer instance IDs.
        
        Args:
            mask: Input mask array (can be binary or instance mask)
            
        Returns:
            Resized mask with same dtype as input
        """
        h, w = mask.shape[:2]
        if (w, h) != self.target_size:
            mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST
            )
        return mask


class MultiExposurePreprocessor:
    """
    3-channel multi-exposure rendering for LSB visualization.
    
    Encodes different luminosity transformations into RGB channels:
        R: Linear magnitude mapping (global_mag_min → 255, global_mag_max → 0)
        G: Asinh stretch (mag → flux → asinh, preserves faint + bright)
        B: Configurable via b_mode parameter
    
    B Channel Modes:
        - "gamma": B = G^gamma (default, gamma < 1 boosts faint)
        - "zscale": ZScaleInterval on flux_sub (astronomy display standard)
        - "zscale_asinh": ZScale + our asinh formula (aligned with G channel)
        - "none": B = G (debug mode)
    
    Color Balance:
        r_gain/b_gain allow quick color tuning without recomputing channels.
        Final output: R = clip(R * r_gain), B = clip(B * b_gain)
    
    Attributes:
        last_stats: Dict with per-image computed values for metadata logging.
    """
    
    # Valid B channel modes
    VALID_B_MODES = ("gamma", "zscale", "zscale_asinh", "none")
    
    def __init__(
        self,
        global_mag_min: float = 20.0,
        global_mag_max: float = 35.0,
        zeropoint: float = 22.5,
        nonlinearity: float = 300.0,
        clip_percentile: float = 99.5,
        gamma: float = 0.5,
        b_mode: str = "gamma",
        zscale_contrast: float = 0.25,
        r_gain: float = 1.0,
        b_gain: float = 1.0,
        target_size: Tuple[int, int] = None,  # Required from config
    ):
        """
        Args:
            global_mag_min: Brightest magnitude (saturation) for R channel
            global_mag_max: Faintest magnitude (background) for R channel
            zeropoint: Magnitude zeropoint for flux conversion (G/B channels)
            nonlinearity: Asinh stretch parameter (used by G and zscale_asinh B)
            clip_percentile: Percentile for vmax in G/B channel normalization
            gamma: Exponent for B channel when b_mode="gamma" (< 1 boosts faint)
            b_mode: B channel strategy - "gamma", "zscale", "zscale_asinh", or "none"
            zscale_contrast: Contrast parameter for ZScaleInterval (0.25 is astropy default)
            r_gain: Multiplier for R channel (color balance, default 1.0)
            b_gain: Multiplier for B channel (color balance, default 1.0)
            target_size: Output (width, height) - REQUIRED, no default
        """
        if b_mode not in self.VALID_B_MODES:
            raise ValueError(
                f"Invalid b_mode: '{b_mode}'. "
                f"Expected one of: {self.VALID_B_MODES}"
            )
        if target_size is None:
            raise ValueError("target_size is required (no default)")
        
        self.global_mag_min = global_mag_min
        self.global_mag_max = global_mag_max
        self.zeropoint = zeropoint
        self.nonlinearity = nonlinearity
        self.clip_percentile = clip_percentile
        self.gamma = gamma
        self.b_mode = b_mode
        self.zscale_contrast = zscale_contrast
        self.r_gain = r_gain
        self.b_gain = b_gain
        self.target_size = target_size
        
        # Per-image stats updated by process() — for metadata logging
        self.last_stats: Dict[str, Any] = {}
        
        # Lazy-load astropy ZScaleInterval (only for zscale modes)
        self._zscale_interval = None
    
    def _compute_r_channel(self, sb_clean: np.ndarray) -> np.ndarray:
        """R channel: Linear magnitude mapping [global_mag_min, global_mag_max] → [255, 0]."""
        sb_clipped = np.clip(sb_clean, self.global_mag_min, self.global_mag_max)
        r_norm = (self.global_mag_max - sb_clipped) / (
            self.global_mag_max - self.global_mag_min
        )
        return (r_norm * 255).astype(np.uint8)
    
    def _compute_g_channel(self, sb_clean: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        G channel: Asinh stretch (mag → flux → background sub → asinh).
        
        Returns:
            g_8bit: uint8 G channel
            flux_sub: Background-subtracted flux (for B channel reuse)
            vmax_ref_flux: The computed vmax used for normalization (for metadata)
        """
        # Mag → Flux
        flux = np.power(10.0, (self.zeropoint - sb_clean) / 2.5)
        
        # Background subtraction
        bg = np.median(flux)
        flux_sub = np.maximum(flux - bg, 0)
        
        # Percentile normalization
        vmax = np.percentile(flux_sub, self.clip_percentile)
        if vmax <= 0:
            vmax = flux_sub.max()
        if vmax <= 0:
            vmax = 1.0
        
        flux_norm = flux_sub / (vmax + 1e-10)
        
        # Asinh stretch (our formula, aligned with nonlinearity param)
        g_stretched = np.arcsinh(flux_norm * self.nonlinearity) / np.arcsinh(self.nonlinearity)
        g_8bit = (g_stretched * 255).clip(0, 255).astype(np.uint8)
        
        return g_8bit, flux_sub, float(vmax)
    
    def _compute_b_channel(
        self, 
        g_channel: np.ndarray, 
        flux_sub: np.ndarray
    ) -> np.ndarray:
        """
        B channel: Computed based on b_mode.
        
        Modes:
            - "gamma": B = G^gamma (faint boost)
            - "zscale": ZScaleInterval on flux_sub (physically meaningful)
            - "zscale_asinh": ZScale + our asinh (same nonlinearity as G)
            - "none": B = G (debug)
        """
        if self.b_mode == "gamma":
            g_norm = g_channel.astype(np.float32) / 255.0
            g_norm = np.clip(g_norm, 0.0, 1.0)  # Defensive clip
            b_norm = np.power(g_norm, self.gamma)
            return (b_norm * 255).astype(np.uint8)
        
        elif self.b_mode == "none":
            return g_channel.copy()
        
        elif self.b_mode in ("zscale", "zscale_asinh"):
            # Lazy import astropy ZScaleInterval only
            from astropy.visualization import ZScaleInterval
            
            if self._zscale_interval is None:
                self._zscale_interval = ZScaleInterval(contrast=self.zscale_contrast)
            
            # Defensive nan_to_num on flux_sub (should be non-negative, but be safe)
            flux_safe = np.nan_to_num(
                flux_sub, nan=0.0, posinf=np.nanmax(flux_sub), neginf=0.0
            )
            
            # ZScale on flux_sub (physically meaningful, already background-subtracted)
            vmin, vmax = self._zscale_interval.get_limits(flux_safe)
            b_norm = (flux_safe - vmin) / (vmax - vmin + 1e-10)
            b_norm = np.clip(b_norm, 0, 1)
            
            if self.b_mode == "zscale_asinh":
                # Use OUR asinh formula (same nonlinearity as G channel)
                # This ensures consistent stretch behavior across channels
                b_norm = np.arcsinh(b_norm * self.nonlinearity) / np.arcsinh(self.nonlinearity)
            
            return (b_norm * 255).astype(np.uint8)
        
        else:
            # Fallback (should not reach due to validation)
            return g_channel.copy()
    
    def process(self, sb_map: np.ndarray) -> np.ndarray:
        """
        Process surface brightness map to 3-channel multi-exposure RGB.
        
        Args:
            sb_map: Surface brightness in mag/arcsec² (LOWER = BRIGHTER)
            
        Returns:
            (H, W, 3) uint8 RGB: [R=linear, G=asinh, B=mode-dependent]
        """
        # Clean data
        sb_clean = np.nan_to_num(sb_map, nan=35.0, posinf=35.0, neginf=20.0)
        
        # Track finite pixel ratio
        finite_mask = np.isfinite(sb_map)
        finite_ratio = float(finite_mask.sum()) / sb_map.size
        
        # Compute channels (G also returns flux_sub for B channel reuse)
        r_channel = self._compute_r_channel(sb_clean)
        g_channel, flux_sub, vmax_ref_flux = self._compute_g_channel(sb_clean)
        b_channel = self._compute_b_channel(g_channel, flux_sub)
        
        # Apply gain (color balance)
        if self.r_gain != 1.0:
            r_channel = np.clip(r_channel.astype(np.float32) * self.r_gain, 0, 255).astype(np.uint8)
        if self.b_gain != 1.0:
            b_channel = np.clip(b_channel.astype(np.float32) * self.b_gain, 0, 255).astype(np.uint8)
        
        # Stack RGB
        img_rgb = np.stack([r_channel, g_channel, b_channel], axis=-1)
        
        # Resize if needed
        h, w = img_rgb.shape[:2]
        if (w, h) != self.target_size:
            img_rgb = cv2.resize(
                img_rgb,
                self.target_size,
                interpolation=cv2.INTER_LINEAR
            )
        
        # Store per-image stats for metadata
        self.last_stats = {
            'vmax_ref_flux': vmax_ref_flux,
            'raw_finite_pixel_ratio': finite_ratio,  # Reflects original input, not cleaned
            'r_min': int(r_channel.min()),
            'r_max': int(r_channel.max()),
            'r_mean': float(r_channel.mean()),
            'g_min': int(g_channel.min()),
            'g_max': int(g_channel.max()),
            'g_mean': float(g_channel.mean()),
            'b_min': int(b_channel.min()),
            'b_max': int(b_channel.max()),
            'b_mean': float(b_channel.mean()),
        }
        
        return img_rgb
    
    def get_params_dict(self) -> Dict[str, Any]:
        """
        Get all rendering parameters for metadata logging.
        
        Returns dict with:
            - Config params (global_mag_min/max, zeropoint, nonlinearity, etc.)
            - Per-image computed values from last_stats (vmax_ref_flux, finite_pixel_ratio, etc.)
        """
        return {
            'global_mag_min': self.global_mag_min,
            'global_mag_max': self.global_mag_max,
            'zeropoint': self.zeropoint,
            'nonlinearity': self.nonlinearity,
            'clip_percentile': self.clip_percentile,
            'gamma': self.gamma,
            'b_mode': self.b_mode,
            'zscale_contrast': self.zscale_contrast,
            'r_gain': self.r_gain,
            'b_gain': self.b_gain,
            'target_size': list(self.target_size),
            **self.last_stats,
        }
    
    def resize_mask(self, mask: np.ndarray) -> np.ndarray:
        """Resize mask with nearest-neighbor interpolation."""
        h, w = mask.shape[:2]
        if (w, h) != self.target_size:
            mask = cv2.resize(
                mask,
                self.target_size,
                interpolation=cv2.INTER_NEAREST
            )
        return mask
