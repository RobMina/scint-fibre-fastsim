"""Expected (deterministic, differentiable) optical response for fast simulation: from energy deposits to photons
at the photodetector. Photosensors are in ``scintfibrefastsim.backends``.
"""

from .cell import (
    NOVA_CELL_OPTICS,
    CellOptics,
    CellTable,
    attenuation,
    birks,
    fibre_transport,
    light,
    lookup,
    tabulate,
)

__all__ = [
    "NOVA_CELL_OPTICS", "CellOptics", "CellTable", "attenuation", "birks", "fibre_transport", "light", "lookup",
    "tabulate",
]
