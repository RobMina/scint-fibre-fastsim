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
    uniform_photon_map,
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
    m = uniform_photon_map(40)
    assert m.shape == (40, 40) and float(m.sum()) == 1600.0


def test_sipm_config_view_has_simulate_fields() -> None:
    cfg = SiPMConfigView()
    for name in (
        "inactive", "trap0_prob", "trap1_prob", "trap0_lifetime", "trap1_lifetime",
        "overvoltage", "time_constant", "capacitance", "cross_talk", "thermal_rate",
        "n_pixels_x", "n_pixels_y", "pixel_id",
    ):
        assert hasattr(cfg, name)
    assert float(cfg.p_fire(cfg.overvoltage)) > 0


def test_truncated_config_rejected_when_simulate_injected() -> None:
    class Truncated:
        capacitance = 1e-13

    with pytest.raises(TypeError, match="missing fields"):
        Mu2eSiPMBackend(simulate=lambda *a, **k: {}, photon_map=np.ones((40, 40)), config=Truncated())


def test_sipm_bind_with_injected_simulate_reads_full_config() -> None:
    """Contract: a stand-in that touches every field stock simulate uses must succeed."""

    seen = set()

    def fake_simulate(chan, time, photon, pmap, cfg, rng, t_end=1e5):
        for name in (
            "inactive", "trap0_prob", "trap1_prob", "trap0_lifetime", "trap1_lifetime",
            "overvoltage", "time_constant", "capacitance", "cross_talk", "thermal_rate",
            "n_pixels_x", "n_pixels_y", "pixel_id",
        ):
            getattr(cfg, name)
            seen.add(name)
        cfg.p_fire(cfg.overvoltage)
        seen.add("p_fire")
        n = len(chan)
        return {
            "chan": np.asarray(chan),
            "time": np.asarray(time, float),
            "charge": np.full(n, cfg.capacitance),
            "pe": np.ones(n),
            "photon": np.asarray(photon),
        }

    backend = Mu2eSiPMBackend(
        simulate=fake_simulate, photon_map=np.ones((40, 40)),
    )  # default SiPMConfigView
    out = backend.detect(
        [Arrival(time_ns=10.0, channel=3, energy_eV=2.5),
         Arrival(time_ns=12.0, channel=3, energy_eV=2.5)],
        seed=7,
    )
    assert out["n_pe"] == 2.0
    assert "inactive" in seen and "p_fire" in seen
