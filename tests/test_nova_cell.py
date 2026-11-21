"""Looped-fibre cell optics: analytic transport, tables, quenching, backends of NumPy and JAX."""

from __future__ import annotations

import numpy as np
import pytest

from scintfibrefastsim.geometry import Cell, nova_cell_tdr
from scintfibrefastsim.response import NOVA_CELL_OPTICS as O
from scintfibrefastsim.response import attenuation, birks, fibre_transport, light, lookup, tabulate

CELL = nova_cell_tdr(3900.0)


def test_cell_validation_and_fibre_length() -> None:
    assert CELL.fibre_length == 7800.0
    assert Cell(looped=False).fibre_length == Cell().length
    with pytest.raises(ValueError):
        Cell(fibre_diameter=30.0)


def test_looped_transport_values() -> None:
    d = np.linspace(0.0, CELL.length, 11)
    c, t = fibre_transport(d, CELL, O)
    assert c[0] == pytest.approx(0.5 * (1.0 + attenuation(2 * CELL.length, O)))
    assert c[-1] == pytest.approx(attenuation(CELL.length, O))
    assert np.all(np.diff(c) < 0)
    assert t[-1] == pytest.approx(CELL.length / O.speed)
    c1, t1 = fibre_transport(d, Cell(looped=False), O)
    np.testing.assert_allclose(c1, attenuation(d, O))
    np.testing.assert_allclose(t1, d / O.speed)


def test_table_reproduces_analytic() -> None:
    tab = tabulate(CELL, O, n=513)
    d = np.random.default_rng(1).uniform(0, CELL.length, 1000)
    ca, ta = fibre_transport(d, CELL, O)
    ct, tt = lookup(tab, d)
    np.testing.assert_allclose(ct, ca, rtol=2e-5)
    np.testing.assert_allclose(tt, ta, rtol=2e-5)


def test_birks_and_light() -> None:
    assert birks(10.0, 0.2, 0.0, 0.0) == 10.0
    assert birks(10.0, 0.2, O.birks_kb, 0.0) == pytest.approx(10.0 / (1 + 0.2 * O.birks_kb))
    ph, delay = light(np.array([10.0]), np.array([0.2]), np.array([0.0]), CELL, O)
    c0, _ = fibre_transport(0.0, CELL, O)
    assert ph[0] == pytest.approx(birks(10.0, 0.2, O.birks_kb, 0.0) * O.yield_per_mev * c0)
    assert delay[0] == pytest.approx(O.decay + fibre_transport(0.0, CELL, O)[1])


def test_jax_gradients_match_finite_differences() -> None:
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    def f(theta):
        o = O._replace(yield_per_mev=theta[0], atten_long=theta[1], birks_kb=theta[2], frac_short=theta[3])
        ph, dl = light(jnp.array([3.0, 5.0]), jnp.array([0.2, 0.4]), theta[4] * jnp.array([1.0, 0.4]), CELL, o)
        return jnp.sum(ph) + jnp.sum(dl)

    x = jnp.array([O.yield_per_mev, O.atten_long, O.birks_kb, O.frac_short, 1500.0])
    g = jax.grad(f)(x)
    for i in range(len(x)):
        h = 1e-6 * max(abs(float(x[i])), 1.0)
        e = jnp.zeros_like(x).at[i].set(h)
        fd = (f(x + e) - f(x - e)) / (2 * h)
        assert float(g[i]) == pytest.approx(float(fd), rel=1e-6, abs=1e-9)


def test_float32_matches_float64() -> None:
    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    d64 = np.linspace(0, CELL.length, 50)
    c64, _ = fibre_transport(d64, CELL, O)
    o32 = type(O)(*(jnp.float32(v) for v in O))
    c32, _ = fibre_transport(jnp.asarray(d64, jnp.float32), CELL, o32)
    assert c32.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(c32), c64, rtol=1e-5)
