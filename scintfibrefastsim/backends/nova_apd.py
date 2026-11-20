"""NOvA ND APD backend stub.

Cell geometry and APD+FEB response land in Phase 2, informed by the public NOvA
TDR (arXiv:1807.02665: 39×66 mm internal cell, 0.7 mm looped fibre). Do not wait
on a private GDML owner for the public caricature.
"""

from __future__ import annotations

from .base import Arrival


class NovaAPDBackend:
    """APD + FEB response for NOvA ND cells (stub)."""

    name = "nova_apd"
    sensor_kind = "apd"

    def __init__(self, *, gain: float = 100.0):
        self.gain = float(gain)

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        raise NotImplementedError(
            "NOvA APD + FEB response is Phase 2; use the public TDR cell "
            "caricature (39x66 mm, 0.7 mm looped fibre) when implementing"
        )
