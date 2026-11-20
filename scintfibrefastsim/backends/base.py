"""Common photosensor interface (hardware-agnostic relative to the optical core).

Backends do not own fibre transport. They consume scored fibre-end arrivals from
``scintfibrefastsim.core`` and produce experiment-specific PE / waveform / digi
summaries. Implementations may raise ``NotImplementedError`` until Phase 1/2.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class Arrival:
    """One scored photon arrival at a fibre-end photosensor face.

    Units match the core transport: time in ns, position in mm in the strip/cell
    frame. ``channel`` is the backend's channel index (SiPM or APD).
    """

    time_ns: float
    channel: int
    energy_eV: float
    n_emissions: int = 0


@runtime_checkable
class PhotosensorBackend(Protocol):
    """Experiment photosensor model over shared fibre-end arrivals."""

    name: str
    sensor_kind: str  # "sipm" | "apd" | ...

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        """Map optical arrivals to a backend-specific detection summary.

        Returns a dict with at least ``n_pe`` (float or array) and ``backend``.
        Concrete backends may add waveforms, saturation flags, or digi words.
        """
        ...
