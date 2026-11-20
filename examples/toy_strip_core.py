"""Exercise the copied core: Counter + Binning + optical model registry.

Does not run the full device transport (that needs a non-trivial bin set and
lanes). Safe for CI and laptops.
"""

from __future__ import annotations

import numpy as np

from scintfibrefastsim.core import materials as mat
from scintfibrefastsim.core.binning import Binning
from scintfibrefastsim.core.optics import DEFAULT_MODEL, get
from scintfibrefastsim.geometry.families import toy_strip


def main() -> None:
    c = toy_strip()
    edges = {
        "x": np.linspace(0.0, c.hx, 3),
        "y": np.linspace(-c.hy, c.hy, 3),
        "z": np.linspace(-c.half_length, c.half_length, 3),
        "beta": np.array([0.9, 1.0]),
        "r": np.linspace(0.0, c.r_core, 3),
        "phi": np.linspace(0.0, 2 * np.pi, 3),
        "theta": np.linspace(0.0, np.pi, 3),
    }
    order = {
        0: ["x", "y", "z"],
        1: ["x", "y", "z", "beta"],
        2: ["z", "r", "phi", "theta", "beta"],
    }
    b = Binning(edges=edges, order=order)
    model = get(DEFAULT_MODEL)
    # importing materials registers "current"
    assert mat.CURRENT.name == "current"
    print(f"bins sim0={b.n_bins(0)} model={model.name} fiber_end={model.fiber_end}")


if __name__ == "__main__":
    main()
