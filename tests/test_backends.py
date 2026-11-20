"""Photosensor backend contract."""

from __future__ import annotations

import numpy as np
import pytest

from scintfibrefastsim.backends import (
    Arrival,
    Mu2eSiPMBackend,
    NovaAPDBackend,
    PhotosensorBackend,
    SiPMConfigView,
)


def test_backends_satisfy_protocol() -> None:
    assert isinstance(Mu2eSiPMBackend(), PhotosensorBackend)
    assert isinstance(NovaAPDBackend(), PhotosensorBackend)


def test_sensor_kinds() -> None:
    assert Mu2eSiPMBackend().sensor_kind == "sipm"
    assert NovaAPDBackend().sensor_kind == "apd"


def test_unbound_sipm_raises() -> None:
    arrivals = [Arrival(time_ns=1.0, channel=0, energy_eV=2.5)]
    with pytest.raises(NotImplementedError, match="unbound"):
        Mu2eSiPMBackend().detect(arrivals)


def test_nova_apd_stub_raises() -> None:
    with pytest.raises(NotImplementedError):
        NovaAPDBackend().detect([Arrival(time_ns=1.0, channel=0, energy_eV=2.5)])


def test_uniform_photon_map() -> None:
    from scintfibrefastsim.backends import uniform_photon_map

    m = uniform_photon_map(40)
    assert m.shape == (40, 40) and float(m.sum()) == 1600.0


def test_sipm_bind_with_injected_simulate() -> None:
    """Bind path without importing crvfastsim: inject a tiny fake simulate + map."""

    def fake_simulate(chan, time, photon, pmap, cfg, rng, t_end=1e5):
        n = len(chan)
        return {
            "chan": np.asarray(chan),
            "time": np.asarray(time, float),
            "charge": np.full(n, cfg.capacitance),
            "pe": np.ones(n),
            "photon": np.asarray(photon),
        }

    pmap = np.ones((40, 40))
    backend = Mu2eSiPMBackend(
        simulate=fake_simulate, photon_map=pmap, config=SiPMConfigView(),
    )
    assert backend.bound
    out = backend.detect(
        [Arrival(time_ns=10.0, channel=3, energy_eV=2.5),
         Arrival(time_ns=12.0, channel=3, energy_eV=2.5)],
        seed=7,
    )
    assert out["backend"] == "mu2e_sipm"
    assert out["n_pe"] == 2.0
    assert out["n_charges"] == 2
