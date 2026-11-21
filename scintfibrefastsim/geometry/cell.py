"""Liquid-scintillator cell read out by one looped wavelength-shifting fibre (lengths in mm).

A rectangular extruded cell (interior ``width`` across, ``depth`` along the beam, ``length`` along the extrusion)
filled with liquid scintillator; one fibre runs down the cell, loops at the far end and comes back, and both ends sit
on the same photodetector pixel at the readout end. Defaults are the public NOvA cell caricature (arXiv:1807.02665:
39 x 66 mm interior, 0.7 mm looped fibre); an experiment adapter supplies its own values.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    width: float = 39.0
    depth: float = 66.0
    length: float = 3900.0
    wall: float = 3.3             # extrusion wall between neighbouring cells
    fibre_diameter: float = 0.7
    looped: bool = True           # False: one fibre end at the readout, the far end dark
    loop_extra: float = 0.0       # fibre length in the loop beyond 2 * length (mm)

    def __post_init__(self):
        if min(self.width, self.depth, self.length, self.fibre_diameter) <= 0 or self.wall < 0:
            raise ValueError("cell dimensions must be positive")
        if self.fibre_diameter >= min(self.width, self.depth) / 2:
            raise ValueError("the fibre must fit in the cell")

    @property
    def pitch(self) -> float:
        return self.width + self.wall

    @property
    def fibre_length(self) -> float:
        """Total fibre length from one pixel end to the other."""
        return (2.0 * self.length + self.loop_extra) if self.looped else self.length


def nova_cell_tdr(length: float = 3900.0) -> Cell:
    """Public NOvA cell caricature (TDR-cited dimensions); ``length`` is the detector's cell length in mm."""
    return Cell(length=float(length))
