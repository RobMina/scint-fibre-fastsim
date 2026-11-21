"""Experiment photosensor backends over the shared fibre/WLS core.

The core transport scores arrivals at fibre-end faces. Backends turn those
arrivals into experiment digi (Mu2e SiPM pixel response, NOvA APD + FEB). Offline
table formats stay in the experiment adapters that depend on this package.
"""

from .base import Arrival, PhotosensorBackend
from .mu2e_sipm import (
    Mu2eSiPMBackend,
    SiPMConfigView,
    bind_crvfastsim_sipm,
    bind_crvfastsim_sipm_synthetic,
    uniform_photon_map,
)
from .nova_apd import NOVA_APD, NOVA_ND_READOUT, APDParams, NovaAPDBackend, Readout

__all__ = [
    "Arrival",
    "PhotosensorBackend",
    "Mu2eSiPMBackend",
    "SiPMConfigView",
    "bind_crvfastsim_sipm",
    "bind_crvfastsim_sipm_synthetic",
    "uniform_photon_map",
    "NovaAPDBackend",
    "APDParams",
    "Readout",
    "NOVA_APD",
    "NOVA_ND_READOUT",
]
