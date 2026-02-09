"""
File I/O utilities for astronomical data.

Provides functions for loading FITS files, images, masks, and parsing
sample naming conventions.
"""

import gzip
import re
from pathlib import Path
from typing import Optional

import numpy as np
from astropy.io import fits
from PIL import Image


def load_fits_gz(filepath: Path) -> np.ndarray:
    """
    Load a gzipped FITS file.
    
    Args:
        filepath: Path to the .fits.gz file
        
    Returns:
        numpy array containing the FITS data
        
    Raises:
        FileNotFoundError: If file doesn't exist or symlink is broken
        ValueError: If FITS data cannot be read
    """
    filepath = Path(filepath)
    
    # Resolve symlinks and check existence
    resolved_path = filepath.resolve()
    if not resolved_path.exists():
        if filepath.is_symlink():
            raise FileNotFoundError(
                f"Broken symlink detected: '{filepath}' -> '{resolved_path}'\n"
                f"The target file does not exist. Please check the symlink or "
                f"update the path in your configuration."
            )
        raise FileNotFoundError(
            f"File not found: '{filepath}'\n"
            f"Please verify the path exists and is accessible."
        )
    
    try:
        with gzip.open(resolved_path, 'rb') as f:
            with fits.open(f) as hdul:
                data = hdul[0].data
        
        if data is None:
            raise ValueError(f"No data found in primary HDU of '{filepath}'")
            
        return data
        
    except gzip.BadGzipFile:
        raise ValueError(f"Invalid gzip file: '{filepath}'")


def load_image(path: Path) -> np.ndarray:
    """
    Load image as RGB numpy array.
    
    Args:
        path: Path to the image file
        
    Returns:
        RGB numpy array with shape (H, W, 3)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    resolved_path = path.resolve()
    
    if not resolved_path.exists():
        if path.is_symlink():
            raise FileNotFoundError(
                f"Broken symlink: '{path}' -> '{resolved_path}'"
            )
        raise FileNotFoundError(f"Image not found: '{path}'")
    
    img = Image.open(resolved_path).convert('RGB')
    return np.array(img)


def load_mask(path: Path) -> np.ndarray:
    """
    Load mask as grayscale numpy array.
    
    Args:
        path: Path to the mask image file
        
    Returns:
        Grayscale numpy array with shape (H, W)
        
    Raises:
        FileNotFoundError: If file doesn't exist
    """
    path = Path(path)
    resolved_path = path.resolve()
    
    if not resolved_path.exists():
        if path.is_symlink():
            raise FileNotFoundError(
                f"Broken symlink: '{path}' -> '{resolved_path}'"
            )
        raise FileNotFoundError(f"Mask not found: '{path}'")
    
    mask = Image.open(resolved_path)
    mask_arr = np.array(mask)
    
    # Handle multi-channel masks (take first channel)
    if len(mask_arr.shape) == 3:
        mask_arr = mask_arr[:, :, 0]
        
    return mask_arr


def parse_sample_name(name: str) -> Optional[dict]:
    """
    Parse sample folder name like '00011_eo_SB27.5_streams'.
    
    Args:
        name: Sample folder name string
        
    Returns:
        Dictionary with parsed metadata:
        - galaxy_id: int
        - orientation: str ('eo' or 'fo')
        - sb_threshold: float
        - type: str ('streams' or 'satellites')
        
        Returns None if name doesn't match expected pattern.
    """
    pattern = r'^(\d+)_([ef]o)_SB([\d.]+)_(streams|satellites)$'
    match = re.match(pattern, name)
    
    if match:
        return {
            'galaxy_id': int(match.group(1)),
            'orientation': match.group(2),
            'sb_threshold': float(match.group(3)),
            'type': match.group(4)
        }
    
    return None


# =============================================================================
# SATELLITE DATA STRUCTURES (from props_gals_Fbox_new.pkl)
# =============================================================================

from dataclasses import dataclass, field
from typing import Dict, List, Any
import pickle


@dataclass
class SatelliteInstance:
    """
    Single satellite galaxy instance from the PKL file.
    
    Attributes:
        id: Instance ID (e.g., 'id1', 'id2')
        x: Centroid X coordinate
        y: Centroid Y coordinate
        geo_x: Geometric centroid X
        geo_y: Geometric centroid Y
        seg_map: Full segmentation map (2051×2051), or None if not needed
        seg_ids: Pixel coordinates [y, x] of segmented region
        area: Pixel count
        axis_ratio: Axis ratio
        orientation_angle: Orientation angle in degrees
        mag_r: r-band magnitude
        sb_fltr: Surface brightness (filter)
        gini: Gini coefficient
    """
    id: str
    x: float
    y: float
    geo_x: float
    geo_y: float
    seg_map: Optional[np.ndarray]  # Shape (2051, 2051), uint8
    seg_ids: Optional[np.ndarray]  # Shape (N, 2), [y, x] coords
    area: int
    axis_ratio: float = 0.0
    orientation_angle: float = 0.0
    mag_r: float = 0.0
    sb_fltr: float = 0.0
    gini: float = 0.0
    
    @classmethod
    def from_dict(cls, sat_id: str, data: Dict[str, Any], load_seg_map: bool = False) -> 'SatelliteInstance':
        """
        Create SatelliteInstance from PKL dict entry.
        
        Args:
            sat_id: Instance ID (e.g., 'id1')
            data: Dictionary from PKL file
            load_seg_map: If False, don't load the large seg_map (saves memory)
        """
        return cls(
            id=sat_id,
            x=float(data.get('x', 0)),
            y=float(data.get('y', 0)),
            geo_x=float(data.get('geo-x', 0)),
            geo_y=float(data.get('geo-y', 0)),
            seg_map=data.get('seg_map') if load_seg_map else None,
            seg_ids=data.get('seg_ids'),
            area=int(data.get('area', 0)),
            axis_ratio=float(data.get('axis_ratio', 0)),
            orientation_angle=float(data.get('orientation_angle', 0)),
            mag_r=float(data.get('mag_r', 0)),
            sb_fltr=float(data.get('sb_fltr', 0)),
            gini=float(data.get('gini', 0)),
        )
    
    def get_binary_mask(self, shape: tuple = (2051, 2051)) -> np.ndarray:
        """
        Generate binary mask from seg_ids coordinates.
        
        Args:
            shape: Output mask shape (height, width)
            
        Returns:
            Binary mask with 1s at satellite pixel locations
        """
        mask = np.zeros(shape, dtype=np.uint8)
        if self.seg_ids is not None and len(self.seg_ids) > 0:
            y_coords = np.clip(self.seg_ids[:, 0], 0, shape[0] - 1).astype(int)
            x_coords = np.clip(self.seg_ids[:, 1], 0, shape[1] - 1).astype(int)
            mask[y_coords, x_coords] = 1
        return mask


@dataclass
class GalaxySatellites:
    """All satellites for one galaxy at one SB threshold."""
    galaxy_id: int
    orientation: str  # 'eo' or 'fo'
    sb_threshold: float
    satellites: List[SatelliteInstance] = field(default_factory=list)
    
    @property
    def count(self) -> int:
        return len(self.satellites)
    
    def get_combined_mask(self, shape: tuple = (2051, 2051)) -> np.ndarray:
        """Get combined instance mask with unique IDs for each satellite."""
        mask = np.zeros(shape, dtype=np.uint8)
        for i, sat in enumerate(self.satellites, start=1):
            sat_mask = sat.get_binary_mask(shape)
            mask[sat_mask > 0] = i
        return mask


class SatelliteDataLoader:
    """
    Loader for props_gals_Fbox_new.pkl satellite data.
    
    PKL Structure:
    ```
    dict['{galaxy_id}, {orientation}']  # e.g., '11, eo'
        -> dict['SBlim{threshold}']     # e.g., 'SBlim27'
            -> dict['id{N}']            # e.g., 'id1', 'id2'
                -> {satellite properties}
    ```
    
    Example:
        >>> loader = SatelliteDataLoader('/path/to/props_gals_Fbox_new.pkl')
        >>> sats = loader.get_satellites(11, 'eo', 27.0)
        >>> print(f"Found {sats.count} satellites")
        >>> mask = sats.get_combined_mask()
    """
    
    def __init__(self, pkl_path: Path, lazy_load: bool = True):
        """
        Initialize the satellite data loader.
        
        Args:
            pkl_path: Path to props_gals_Fbox_new.pkl
            lazy_load: If True, don't load until first access (saves startup time)
        """
        self.pkl_path = Path(pkl_path)
        self._data: Optional[Dict] = None
        self._lazy_load = lazy_load
        
        if not lazy_load:
            self._load()
    
    def _load(self) -> None:
        """Load the PKL file."""
        resolved = self.pkl_path.resolve()
        if not resolved.exists():
            raise FileNotFoundError(
                f"Satellite PKL not found: {self.pkl_path}\n"
                f"Resolved path: {resolved}"
            )
        
        with open(resolved, 'rb') as f:
            self._data = pickle.load(f)
    
    @property
    def data(self) -> Dict:
        """Lazy-load and return the raw data."""
        if self._data is None:
            self._load()
        return self._data
    
    def get_galaxy_keys(self) -> List[str]:
        """Get all available galaxy keys ('{id}, {orient}')."""
        return list(self.data.keys())
    
    def parse_galaxy_key(self, key: str) -> tuple:
        """Parse '{id}, {orient}' to (galaxy_id: int, orientation: str)."""
        parts = key.replace(' ', '').split(',')
        return int(parts[0]), parts[1]
    
    def get_sb_thresholds(self, galaxy_id: int, orientation: str) -> List[float]:
        """Get available SB thresholds for a galaxy."""
        key = f"{galaxy_id}, {orientation}"
        if key not in self.data:
            return []
        
        thresholds = []
        for sb_key in self.data[key].keys():
            # Parse 'SBlim27.5' -> 27.5
            match = re.search(r'SBlim([\d.]+)', sb_key)
            if match:
                thresholds.append(float(match.group(1)))
        return sorted(thresholds)
    
    def get_satellites(
        self, 
        galaxy_id: int, 
        orientation: str, 
        sb_threshold: float,
        load_seg_map: bool = False
    ) -> GalaxySatellites:
        """
        Get all satellites for a specific galaxy and threshold.
        
        Args:
            galaxy_id: Galaxy ID (e.g., 11, 13)
            orientation: 'eo' or 'fo'
            sb_threshold: Surface brightness threshold (e.g., 27.0)
            load_seg_map: If True, load full seg_map arrays (memory intensive)
            
        Returns:
            GalaxySatellites object with list of SatelliteInstance
        """
        result = GalaxySatellites(
            galaxy_id=galaxy_id,
            orientation=orientation,
            sb_threshold=sb_threshold
        )
        
        gal_key = f"{galaxy_id}, {orientation}"
        if gal_key not in self.data:
            return result
        
        # Format threshold key
        if sb_threshold == int(sb_threshold):
            sb_key = f"SBlim{int(sb_threshold)}"
        else:
            sb_key = f"SBlim{sb_threshold}"
        
        if sb_key not in self.data[gal_key]:
            return result
        
        sat_dict = self.data[gal_key][sb_key]
        if not isinstance(sat_dict, dict):
            return result
        
        for sat_id, sat_data in sat_dict.items():
            if isinstance(sat_data, dict):
                instance = SatelliteInstance.from_dict(sat_id, sat_data, load_seg_map)
                result.satellites.append(instance)
        
        return result
