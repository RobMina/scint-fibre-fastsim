"""Toy transport: same seed, float32 and float64 both complete (flag-off style smoke).

Bit-identity across dtypes is not required; both must produce hits for the toy strip.
A stronger mu2e-fastsim extract test checks the shimmed import path against itself.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_toy_generate_completes(dtype: str) -> None:
    pytest.importorskip("jax")
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.run import generate
    from scintfibrefastsim.geometry.families import toy_strip

    c = toy_strip(length=200.0)
    edges = {
        "x": np.linspace(0.0, float(c.hx), 3),
        "y": np.linspace(-float(c.hy), float(c.hy), 3),
        "z": np.linspace(-float(c.half_length), float(c.half_length), 3),
    }
    b = Binning(edges=edges, order={0: ["x", "y", "z"]})
    r = generate(c, b, 0, np.arange(b.n_bins(0)), 40, dtype=dtype, seed=2,
                 lanes=256, chunk_iters=400)
    assert int(r["gen"].sum()) > 0
    assert np.all(r["ht"].sum(1) == r["he"].sum(1))


def test_same_seed_float64_is_deterministic() -> None:
    pytest.importorskip("jax")
    from scintfibrefastsim.validation import run_toy_selfcheck

    a = run_toy_selfcheck(dtype="float64", seed=11, nphot=30)
    b = run_toy_selfcheck(dtype="float64", seed=11, nphot=30)
    assert a.metrics["gen"] == b.metrics["gen"]
    assert a.metrics["hits"] == b.metrics["hits"]
