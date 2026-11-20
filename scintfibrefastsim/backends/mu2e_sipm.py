"""Mu2e CRV SiPM backend — bind path to ``crvfastsim.response.sipm``.

Full pixel model stays in mu2e-fastsim until a later extract moves it. This module
accepts an injected ``simulate`` callable with that API so the public package
never imports Mu2e-internal code by default.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from .base import Arrival

# Signature compatible with crvfastsim.response.sipm.simulate
SimulateFn = Callable[..., dict]


@dataclass
class SiPMConfigView:
    """Subset of SiPM knobs visible to this package (mirrors crvfastsim.response.sipm.SiPMConfig defaults)."""

    n_pixels_x: int = 40
    n_pixels_y: int = 40
    overvoltage: float = 3.0
    time_constant: float = 13.3
    capacitance: float = 8.84e-14
    p1: float = 0.607
    p2: float = 2.7
    cross_talk: float = 0.04
    thermal_rate: float = 0.0
    pixel_id: str = "lrint"


class Mu2eSiPMBackend:
    """SiPM pixel response for Mu2e CRV.

    Construct with ``simulate`` / ``photon_map`` from ``crvfastsim.response.sipm``
    (see ``bind_crvfastsim_sipm``), or call ``detect`` only after binding.
    """

    name = "mu2e_sipm"
    sensor_kind = "sipm"

    def __init__(
        self,
        *,
        pde_peak: float = 0.35,
        n_pixels: int = 1584,
        simulate: SimulateFn | None = None,
        photon_map: Any = None,
        config: Any = None,
        t_end: float = 1e5,
    ):
        self.pde_peak = float(pde_peak)
        self.n_pixels = int(n_pixels)
        self._simulate = simulate
        self._photon_map = photon_map
        self._config = config if config is not None else SiPMConfigView()
        self.t_end = float(t_end)

    @property
    def bound(self) -> bool:
        return self._simulate is not None and self._photon_map is not None

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        if not self.bound:
            raise NotImplementedError(
                "Mu2e SiPM backend is unbound: pass simulate= and photon_map= from "
                "crvfastsim.response.sipm, or call bind_crvfastsim_sipm()"
            )
        if not arrivals:
            return {"n_pe": 0.0, "backend": self.name, "n_charges": 0}
        chan = np.asarray([a.channel for a in arrivals], dtype=np.int64)
        time = np.asarray([a.time_ns for a in arrivals], dtype=float)
        photon = np.arange(len(arrivals), dtype=np.int64)
        rng = np.random.default_rng(int(seed))
        out = self._simulate(
            chan, time, photon, self._photon_map, self._config, rng, t_end=self.t_end,
        )
        pe = np.asarray(out["pe"], dtype=float)
        return {
            "n_pe": float(pe.sum()),
            "backend": self.name,
            "n_charges": int(pe.size),
            "chan": out["chan"],
            "time": out["time"],
            "pe": pe,
            "charge": out["charge"],
        }


def bind_crvfastsim_sipm(
    *,
    sipm_module: Any = None,
    photon_map: Any = None,
    config: Any = None,
    **kwargs: Any,
) -> Mu2eSiPMBackend:
    """Bind ``crvfastsim.response.sipm`` into a ``Mu2eSiPMBackend``.

    Imports crvfastsim only when ``sipm_module`` is omitted. ``photon_map`` defaults
    to ``sipm_module.photon_map()`` (needs ``$CRVFS_LOOKUP_DIR`` / Mu2e inputs).
    """
    if sipm_module is None:
        from crvfastsim.response import sipm as sipm_module  # type: ignore
    if config is None:
        config = sipm_module.SiPMConfig()
    if photon_map is None:
        photon_map = sipm_module.photon_map()
    return Mu2eSiPMBackend(
        simulate=sipm_module.simulate,
        photon_map=photon_map,
        config=config,
        **kwargs,
    )
