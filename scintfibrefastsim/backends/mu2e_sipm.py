"""Mu2e CRV SiPM backend — bind path to ``crvfastsim.response.sipm``.

Full pixel model stays in mu2e-fastsim until a later extract moves it. This module
accepts an injected ``simulate`` callable with that API so the public package
never imports Mu2e-internal code by default.

Core fibre-end ``Arrival`` values are *optical* arrivals (after transport, before
photosensor PDE). ``detect`` does not apply a separate ``pde_peak`` scale: the
injected ``simulate`` (stock MakeCrvSiPMCharges) already draws per-pixel firing.
``Arrival.energy_eV`` is carried for provenance only; stock simulate does not use it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .base import Arrival

# Signature compatible with crvfastsim.response.sipm.simulate
SimulateFn = Callable[..., dict]


@dataclass
class SiPMConfigView:
    """Stand-in with every field ``crvfastsim.response.sipm.simulate`` reads.

    Defaults match ``sipm.SiPMConfig``. Prefer passing a real ``SiPMConfig`` when
    binding from mu2e-fastsim.
    """

    n_pixels_x: int = 40
    n_pixels_y: int = 40
    overvoltage: float = 3.0
    time_constant: float = 13.3
    capacitance: float = 8.84e-14
    p1: float = 0.607
    p2: float = 2.7
    cross_talk: float = 0.04
    thermal_rate: float = 0.0
    trap0_prob: float = 0.0
    trap1_prob: float = 0.0
    trap0_lifetime: float = 5.0
    trap1_lifetime: float = 50.0
    inactive: tuple = field(
        default_factory=lambda: tuple((x, y) for x in range(18, 22) for y in range(18, 22))
    )
    pixel_id: str = "lrint"

    def p_fire(self, v):
        return self.p1 * (1.0 - np.exp(-np.asarray(v, float) / self.p2))


class Mu2eSiPMBackend:
    """SiPM pixel response for Mu2e CRV.

    Construct with ``simulate`` / ``photon_map`` from ``crvfastsim.response.sipm``
    (see ``bind_crvfastsim_sipm``), or call ``detect`` only after binding.

    When ``simulate`` is injected, ``config`` must be a full config (or omit it to
    use ``SiPMConfigView``, which supplies every attribute stock ``simulate`` reads).
    """

    name = "mu2e_sipm"
    sensor_kind = "sipm"

    def __init__(
        self,
        *,
        simulate: SimulateFn | None = None,
        photon_map: Any = None,
        config: Any = None,
        t_end: float = 1e5,
    ):
        self._simulate = simulate
        self._photon_map = photon_map
        if config is None:
            if simulate is not None:
                config = SiPMConfigView()
            else:
                config = SiPMConfigView()
        self._config = config
        self.t_end = float(t_end)
        self._require_simulate_config_fields()

    def _require_simulate_config_fields(self) -> None:
        """Fail closed if an injected simulate would see a truncated config."""
        if self._simulate is None:
            return
        required = (
            "inactive", "trap0_prob", "trap1_prob", "trap0_lifetime", "trap1_lifetime",
            "overvoltage", "time_constant", "capacitance", "cross_talk", "thermal_rate",
            "n_pixels_x", "n_pixels_y", "pixel_id", "p_fire",
        )
        missing = [k for k in required if not hasattr(self._config, k)]
        if missing:
            raise TypeError(
                "SiPM config is missing fields required by crvfastsim.response.sipm.simulate: "
                + ", ".join(missing)
            )

    @property
    def bound(self) -> bool:
        return self._simulate is not None and self._photon_map is not None

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        if not self.bound:
            raise NotImplementedError(
                "Mu2e SiPM backend is unbound: pass simulate= and photon_map= from "
                "crvfastsim.response.sipm, or call bind_crvfastsim_sipm() / "
                "bind_crvfastsim_sipm_synthetic()"
            )
        if not arrivals:
            return {"n_pe": 0.0, "backend": self.name, "n_charges": 0}
        chan = np.asarray([a.channel for a in arrivals], dtype=np.int64)
        time = np.asarray([a.time_ns for a in arrivals], dtype=float)
        # carried id into sipm.simulate's photon column (not a physics index)
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
    Pass ``photon_map=uniform_photon_map()`` for a public synthetic map (no Mu2e inputs).
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


def uniform_photon_map(n: int = 40) -> np.ndarray:
    """Public synthetic fibre photon map (flat); for bind smokes without Mu2e inputs."""
    return np.ones((int(n), int(n)), dtype=float)


def bind_crvfastsim_sipm_synthetic(
    *,
    sipm_module: Any = None,
    n: int = 40,
    **kwargs: Any,
) -> Mu2eSiPMBackend:
    """Bind real ``sipm.simulate`` with a uniform public photon map (no CRVFS_LOOKUP_DIR)."""
    if sipm_module is None:
        from crvfastsim.response import sipm as sipm_module  # type: ignore
    cfg = kwargs.pop("config", None)
    if cfg is None:
        cfg = sipm_module.SiPMConfig(
            pixel_id="truncate", thermal_rate=0.0, trap0_prob=0.0, trap1_prob=0.0,
        )
    return bind_crvfastsim_sipm(
        sipm_module=sipm_module,
        photon_map=uniform_photon_map(n),
        config=cfg,
        **kwargs,
    )
