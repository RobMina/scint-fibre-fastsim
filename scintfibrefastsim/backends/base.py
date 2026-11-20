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
    """One scored optical arrival at a fibre-end photosensor face (pre-PDE).

    Units match the core transport: time in ns. ``channel`` is the backend's
    channel index (SiPM or APD). ``energy_eV`` is provenance for the photon
    energy at the face; stock Mu2e SiPM ``simulate`` does not use it (pixel
    firing is drawn inside ``simulate``). Arrivals are *after* fibre transport
    and *before* photosensor detection efficiency.
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
