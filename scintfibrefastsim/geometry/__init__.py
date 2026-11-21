"""Geometry families: scintillator strip (Mu2e CRV) and liquid-scintillator cell (NOvA ND).

Parametric strip/module knobs fold into Mu2eBO Candidate G
(CRV-module-geometry-optimization). See docs/explanation/candidate-g.md.
"""

from .cell import Cell, nova_cell_tdr
from .families import GeometryFamily, cell_tdr_caricature, strip_family

__all__ = ["Cell", "GeometryFamily", "cell_tdr_caricature", "nova_cell_tdr", "strip_family"]
