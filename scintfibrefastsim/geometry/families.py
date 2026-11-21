"""Named geometry families over the core ``Counter`` (strip) and ``Cell`` (looped-fibre cell)."""

from __future__ import annotations

from dataclasses import dataclass

from scintfibrefastsim.core.counter import Counter


@dataclass(frozen=True)
class GeometryFamily:
    """One public geometry family name and a short description."""

    name: str
    kind: str  # "strip" | "cell"
    description: str


def strip_family() -> GeometryFamily:
    """Mu2e-style extruded strip with two WLS fibres (core ``Counter``)."""
    return GeometryFamily(
        name="strip_two_fibre",
        kind="strip",
        description="Rectangular scintillator extrusion with two double-clad WLS fibres",
    )


def cell_tdr_caricature() -> GeometryFamily:
    """Public NOvA ND cell caricature from the TDR (arXiv:1807.02665).

    Internal cross-section 39×66 mm, 0.7 mm looped fibre. Not a private GDML. The concrete
    parameters are :func:`scintfibrefastsim.geometry.cell.nova_cell_tdr`.
    """
    return GeometryFamily(
        name="nova_cell_tdr",
        kind="cell",
        description="NOvA ND cell caricature: 39x66 mm internal, 0.7 mm looped fibre (TDR)",
    )


def toy_strip(*, length: float = 300.0) -> Counter:
    """A small public toy strip for tutorials and CI (not a production counter).

    Dimensions are illustrative; fibre and hole clearances follow the core
    Counter invariants (fibre inside hole, holes inside scintillator).
    """
    fiber_diameter = 1.4
    clad_fraction = 0.03
    hole_rx = fiber_diameter / 2 + 0.3
    hole_ry = fiber_diameter / 2 + 0.3
    return Counter(
        length=float(length),
        thickness=20.0,
        width=50.0,
        fiber_separation=26.0,
        fiber_diameter=fiber_diameter,
        clad_fraction=clad_fraction,
        hole_rx=hole_rx,
        hole_ry=hole_ry,
        coating=0.25,
        corner_radius=1.0,
        guide_bar=5.0,
        sipm_width=2.0,
        sipm_window=0.1,
        sipm_length=0.3,
        air_gap=0.001,
        reflector=0,
    )
