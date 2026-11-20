"""Geometry families: scintillator strip (Mu2e CRV) and cell (NOvA ND).

Parametric strip/module knobs fold into Mu2eBO Candidate G
(CRV-module-geometry-optimization). See docs/explanation/candidate-g.md.
"""

from .families import GeometryFamily, cell_tdr_caricature, strip_family

__all__ = ["GeometryFamily", "strip_family", "cell_tdr_caricature"]
