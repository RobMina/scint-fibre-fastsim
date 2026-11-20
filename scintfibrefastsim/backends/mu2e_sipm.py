"""Mu2e CRV SiPM backend stub.

Full pixel model remains in ``crvfastsim.response`` until a later extract binds
it here. This stub defines the interface contract only.
"""

from __future__ import annotations

from .base import Arrival


class Mu2eSiPMBackend:
    """SiPM pixel response for Mu2e CRV (stub)."""

    name = "mu2e_sipm"
    sensor_kind = "sipm"

    def __init__(self, *, pde_peak: float = 0.35, n_pixels: int = 1584):
        self.pde_peak = float(pde_peak)
        self.n_pixels = int(n_pixels)

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        raise NotImplementedError(
            "Mu2e SiPM pixel model is still in crvfastsim.response; "
            "bind it in a Phase-1 extract PR without changing flag-off parity"
        )
