"""Photosensor backend contract."""

from __future__ import annotations

import pytest

from scintfibrefastsim.backends import (
    Arrival,
    Mu2eSiPMBackend,
    NovaAPDBackend,
    PhotosensorBackend,
)


def test_backends_satisfy_protocol() -> None:
    assert isinstance(Mu2eSiPMBackend(), PhotosensorBackend)
    assert isinstance(NovaAPDBackend(), PhotosensorBackend)


def test_sensor_kinds() -> None:
    assert Mu2eSiPMBackend().sensor_kind == "sipm"
    assert NovaAPDBackend().sensor_kind == "apd"


def test_stubs_raise() -> None:
    arrivals = [Arrival(time_ns=1.0, channel=0, energy_eV=2.5)]
    with pytest.raises(NotImplementedError):
        Mu2eSiPMBackend().detect(arrivals)
    with pytest.raises(NotImplementedError):
        NovaAPDBackend().detect(arrivals)
