"""
Forward observation noise model for LSB surface brightness maps.

Injects physically-motivated photon noise into clean SB FITS:
    SB(mag) → flux → counts → +sky → Poisson → +read_noise → −sky → mag(noisy)
"""

from .forward_observation import ForwardObservationModel

__all__ = ["ForwardObservationModel"]
