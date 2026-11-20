"""Geometry families and toy strip."""

from __future__ import annotations

from scintfibrefastsim.geometry.families import (
    cell_tdr_caricature,
    strip_family,
    toy_strip,
)


def test_strip_family() -> None:
    f = strip_family()
    assert f.kind == "strip"
    assert "fibre" in f.description.lower() or "fiber" in f.description.lower()


def test_cell_tdr_caricature() -> None:
    f = cell_tdr_caricature()
    assert f.kind == "cell"
    assert "39" in f.description and "66" in f.description
    assert "0.7" in f.description


def test_toy_strip_invariants() -> None:
    c = toy_strip(length=200.0)
    assert c.length == 200.0
    assert c.r_core > 0
    assert c.r_clad2 < c.hole_rx
    assert c.r_clad2 < c.hole_ry
