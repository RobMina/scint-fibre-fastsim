"""Geant4 validation hook stubs."""

from __future__ import annotations

import pytest

from scintfibrefastsim.validation import compare_plan_crvstandalone, compare_plan_toy
from scintfibrefastsim.validation.g4_hooks import run_compare


def test_crvstandalone_plan() -> None:
    p = compare_plan_crvstandalone()
    assert p.oracle == "Mu2e/CRVStandalone"
    assert 0 in p.simtypes


def test_toy_plan() -> None:
    p = compare_plan_toy()
    assert p.nphot_per_bin == 100


def test_run_compare_stub() -> None:
    with pytest.raises(NotImplementedError):
        run_compare(compare_plan_toy())
