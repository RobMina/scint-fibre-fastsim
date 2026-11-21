"""Expected optical response of a looped-fibre cell: energy deposit → photons at the pixel, and their delay.

Deterministic and backend-generic: every function takes NumPy or JAX arrays and returns the same kind, so under JAX
it is differentiable in the deposit, its position along the cell and every ``CellOptics`` field. Two forms of the
fibre transport share one interface (:func:`light`):

- **analytic**: two-exponential attenuation; a looped fibre sends half the captured light each way, ``d`` and
  ``fibre_length - d`` from the emission point to the pixel;
- **tabulated** (``CellTable``): collection and mean delay on a grid of distances, linearly interpolated (gradients
  in the distance and the table values). Tables made by :func:`tabulate` reproduce the analytic form; tables from
  the photon Monte Carlo of ``scintfibrefastsim.core`` use the same container.

Units: mm, ns, MeV. Birks constants in mm/MeV, dE/dx in MeV/mm.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

from ..geometry.cell import Cell


def _xp(*xs):
    for x in xs:
        if type(x).__module__.split(".")[0] in ("jax", "jaxlib"):
            import jax.numpy as jnp
            return jnp
    return np


class CellOptics(NamedTuple):
    """Scintillator and fibre constants of one cell type (a pytree under JAX)."""

    yield_per_mev: float = 11.76   # photons reaching the pixel per visible MeV at zero fibre distance (both ends)
    birks_kb: float = 0.125        # mm / MeV
    birks_kc: float = 0.0          # (mm / MeV)^2
    frac_short: float = 0.35       # short-component fraction of the fibre attenuation
    atten_short: float = 1000.0    # mm
    atten_long: float = 7000.0     # mm
    speed: float = 170.0           # mm / ns, signal speed in the fibre
    decay: float = 10.0            # ns, mean scintillation + re-emission delay


NOVA_CELL_OPTICS = CellOptics()


class CellTable(NamedTuple):
    """Tabulated fibre transport: ``distance [n]`` (mm, increasing), ``collection [n]`` (fraction of the
    zero-distance light reaching the pixel), ``delay [n]`` (ns, mean fibre delay)."""

    distance: object
    collection: object
    delay: object


def birks(edep, dedx, kb, kc):
    """Visible energy after Birks-Chou quenching."""
    return edep / (1.0 + kb * dedx + kc * dedx * dedx)


def attenuation(d, optics: CellOptics):
    """Two-exponential fibre attenuation after a distance ``d``."""
    xp = _xp(d, optics.atten_long)
    f = optics.frac_short
    return f * xp.exp(-d / optics.atten_short) + (1.0 - f) * xp.exp(-d / optics.atten_long)


def fibre_transport(distance, cell: Cell, optics: CellOptics):
    """Analytic ``(collection, mean delay)`` for emission at ``distance`` from the readout end."""
    a1 = attenuation(distance, optics)
    if not cell.looped:
        return a1, distance / optics.speed
    d2 = cell.fibre_length - distance
    a2 = attenuation(d2, optics)
    s = a1 + a2
    return 0.5 * s, (a1 * distance + a2 * d2) / s / optics.speed


def tabulate(cell: Cell, optics: CellOptics = NOVA_CELL_OPTICS, n: int = 257) -> CellTable:
    """A ``CellTable`` of the analytic transport on ``n`` distances from 0 to the cell length (NumPy)."""
    d = np.linspace(0.0, cell.length, n)
    o = CellOptics(*(float(v) for v in optics))
    c, t = fibre_transport(d, cell, o)
    return CellTable(d, np.asarray(c), np.asarray(t))


def lookup(table: CellTable, distance):
    """Interpolated ``(collection, delay)``; the end values hold outside the table."""
    xp = _xp(distance, table.collection)
    return xp.interp(distance, table.distance, table.collection), xp.interp(distance, table.distance, table.delay)


def light(edep, dedx, distance, cell: Cell, optics: CellOptics, table: CellTable | None = None):
    """Expected photons at the pixel and their mean arrival delay (ns) for deposits ``edep`` (MeV) with stopping
    power ``dedx`` (MeV/mm) at ``distance`` (mm) from the readout end."""
    vis = birks(edep, dedx, optics.birks_kb, optics.birks_kc)
    coll, delay = fibre_transport(distance, cell, optics) if table is None else lookup(table, distance)
    return vis * optics.yield_per_mev * coll, delay + optics.decay
