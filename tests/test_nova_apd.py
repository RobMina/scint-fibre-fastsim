"""NOvA APD + front end: expected form, sampled form, backend over arrivals."""

from __future__ import annotations

import numpy as np
import pytest

from scintfibrefastsim.backends import NOVA_APD as P
from scintfibrefastsim.backends import NOVA_ND_READOUT as RO
from scintfibrefastsim.backends import Arrival, NovaAPDBackend
from scintfibrefastsim.backends import nova_apd as apd


def test_pulse_has_unit_peak_and_is_causal() -> None:
    t = np.linspace(-500, 5000, 20001)
    s = apd.pulse(t, P)
    assert s.max() == pytest.approx(1.0, rel=1e-6)
    assert np.all(s[t <= 0] == 0)


def test_expected_numpy_threshold_behaviour() -> None:
    pe = np.array([0.0, 5.0, 10.0, 40.0, 200.0])
    adc, ph = apd.expected(pe, np.full(5, 300.0), P, RO)
    assert np.all(np.diff(adc) > 0) and np.all(np.diff(ph) > 0)
    assert ph[0] < 1e-5 and ph[-1] > 0.999


@pytest.fixture
def jx():
    jax = pytest.importorskip("jax")
    jax.config.update("jax_enable_x64", True)
    return jax


def test_sampled_mean_and_hit_rate_match_expected(jx) -> None:
    import jax.numpy as jnp
    pe = jnp.array([3.0, 8.0, 12.0, 25.0, 80.0])
    time = jnp.array([40.0, 130.0, 260.0, 300.0, 410.0])
    keys = jx.random.split(jx.random.PRNGKey(2), 4000)
    q, adc, t, hit = jx.vmap(lambda k: apd.sample(k, pe, time, P, RO))(keys)
    np.testing.assert_allclose(np.asarray(q.mean(0)), np.asarray(pe), rtol=0.03, atol=0.05)
    _, ph = apd.expected(pe, time, P, RO)
    np.testing.assert_allclose(np.asarray(hit.mean(0)), np.asarray(ph), atol=0.03)  # gamma-shaped charge vs Poisson x Gaussian gain


def test_expected_gradients(jx) -> None:
    import jax.numpy as jnp

    def f(theta):
        p = P._replace(threshold_pe=theta[0], noise_pe=theta[1], tau_rise=theta[2], adc_per_pe=theta[3])
        adc, ph = apd.expected(jnp.array([6.0, 14.0, 30.0]), theta[4] + jnp.array([0.0, 50.0, 90.0]), p, RO)
        return jnp.sum(adc) + 100.0 * jnp.sum(ph)

    x = jnp.array([10.0, 1.5, 140.0, 1.0, 37.0])
    g = jx.grad(f)(x)
    for i in range(len(x)):
        h = 1e-6 * max(abs(float(x[i])), 1.0)
        e = jnp.zeros_like(x).at[i].set(h)
        assert float(g[i]) == pytest.approx(float((f(x + e) - f(x - e)) / (2 * h)), rel=1e-5, abs=1e-8)


def test_backend_over_arrivals_matches_expected_hit_rate() -> None:
    rng = np.random.default_rng(0)
    n_photons = 14.0
    hits, n_pe = [], []
    for s in range(400):
        n = rng.poisson(n_photons)
        arr = [Arrival(time_ns=200.0 + rng.exponential(10.0), channel=0, energy_eV=2.4) for _ in range(n)]
        out = NovaAPDBackend(n_channels=1).detect(arr, seed=s)
        hits.append(out["hit"][0])
        n_pe.append(out["n_pe"][0])
    assert np.mean(n_pe) == pytest.approx(n_photons * P.qe, rel=0.05)
    _, ph = apd.expected(np.array([n_photons * P.qe]), np.array([205.0]), P, RO)
    assert np.mean(hits) == pytest.approx(float(ph[0]), abs=0.06)


# ---------------------------------------------------------------------------------------------- ADC saturation

def test_no_ceiling_by_default(jx) -> None:
    import jax.numpy as jnp
    assert P.adc_max == float("inf") and P.sat_recovery == 0.0
    pe = jnp.array([1e4, 1e5])
    q, adc, _, _ = apd.sample(jx.random.PRNGKey(0), pe, jnp.array([100.0, 200.0]), P, RO)
    assert float(adc[1]) > 5e4 and float(q[1]) > 9e4
    a, _ = apd.expected(pe, jnp.array([100.0, 200.0]), P, RO)
    assert float(a[1]) > 5e4


def test_hard_ceiling_caps_adc_and_scales_charge(jx) -> None:
    import jax.numpy as jnp
    p0 = P._replace(adc_per_pe=2.0, adc_max=3500.0)
    pe = jnp.array([200.0, 4000.0, 20000.0])
    time = jnp.array([100.0, 150.0, 300.0])
    key = jx.random.PRNGKey(4)
    q_free, adc_free, _, _ = apd.sample(key, pe, time, P._replace(adc_per_pe=2.0), RO)
    q, adc, _, hit = apd.sample(key, pe, time, p0, RO)
    assert float(adc[0]) == float(adc_free[0]) and float(q[0]) == float(q_free[0])     # below the ceiling
    assert np.all(np.asarray(adc[1:]) == 3500.0)
    np.testing.assert_allclose(np.asarray(q[1:]), np.asarray(q_free[1:] * 3500.0 / adc_free[1:]), rtol=1e-3)
    q_r, _, _, _ = apd.sample(key, pe, time, p0._replace(sat_recovery=0.25), RO)
    assert np.all(np.asarray(q_r[1:]) > np.asarray(q[1:])) and np.all(np.asarray(q_r[1:]) < np.asarray(q_free[1:]))
    assert float(q_r[2]) > float(q_r[1])                                           # still increasing with charge
    assert bool(hit.all())


def test_expected_adc_matches_monte_carlo_of_a_spread_ceiling() -> None:
    p = P._replace(adc_max=3500.0, adc_max_sigma=80.0)
    x = np.array([1000.0, 3300.0, 3450.0, 3500.0, 3600.0, 5000.0])
    m = np.random.default_rng(1).normal(3500.0, 80.0, 400000)
    mc = np.minimum(x[:, None], m[None, :]).mean(1)
    np.testing.assert_allclose(apd.expected_adc(x, p), mc, rtol=2e-4)
    np.testing.assert_allclose(apd.expected_adc(x, p._replace(adc_max_sigma=0.0)), np.minimum(x, 3500.0), rtol=1e-12)
    np.testing.assert_allclose(apd.expected_adc(x, P), x, rtol=0)


def test_sampled_adc_mean_matches_expected_near_the_ceiling(jx) -> None:
    import jax.numpy as jnp
    p = P._replace(adc_per_pe=2.0, adc_max=3500.0, adc_max_sigma=80.0)
    pe = jnp.array([1500.0, 1900.0, 2100.0, 3000.0])
    time = jnp.array([60.0, 60.0, 60.0, 60.0])
    keys = jx.random.split(jx.random.PRNGKey(9), 4000)
    _, adc, _, _ = jx.vmap(lambda k: apd.sample(k, pe, time, p, RO))(keys)
    a, _ = apd.expected(pe, time, p, RO)
    np.testing.assert_allclose(np.asarray(adc.mean(0)), np.asarray(a), rtol=0.01)


def test_ceiling_gradients(jx) -> None:
    import jax.numpy as jnp

    def f(theta):
        p = P._replace(adc_per_pe=2.0, adc_max=theta[0], adc_max_sigma=theta[1])
        adc, _ = apd.expected(jnp.array([1500.0, 1800.0, 2400.0]), jnp.array([60.0, 90.0, 120.0]), p, RO)
        return jnp.sum(adc)

    x = jnp.array([3500.0, 80.0])
    g = jx.grad(f)(x)
    for i in range(len(x)):
        h = 1e-6 * float(x[i])
        e = jnp.zeros_like(x).at[i].set(h)
        assert float(g[i]) == pytest.approx(float((f(x + e) - f(x - e)) / (2 * h)), rel=1e-5, abs=1e-8)


def test_backend_caps_adc() -> None:
    p = P._replace(adc_per_pe=2.0, adc_max=300.0)
    arr = [Arrival(time_ns=100.0 + 0.1 * i, channel=0, energy_eV=2.4)
           for i in range(2000)]
    out = NovaAPDBackend(p, RO, n_channels=1).detect(arr, seed=3)
    assert out["adc"][0] == 300.0 and out["n_pe"][0] > 1000


# ---------------------------------------------------------------------------------------------- group crosstalk

def test_group_others_and_coupling(jx) -> None:
    import jax.numpy as jnp
    q = np.array([[10.0, 200.0, 300.0, 5.0], [40.0, 0.0, 0.0, 0.0]])
    g = np.array([[0, 0, 0, 1], [2, 2, 3, 3]])
    others = apd.group_others(q, g)
    np.testing.assert_allclose(others, [[500.0, 310.0, 210.0, 0.0], [0.0, 40.0, 0.0, 0.0]])
    np.testing.assert_allclose(np.asarray(apd.group_others(jnp.asarray(q), jnp.asarray(g))), others)
    p = P._replace(crosstalk=0.02)
    np.testing.assert_allclose(apd.coupled(q, g, p), np.maximum(q - 0.02 * others, 0.0))
    assert apd.coupled(q, None, p) is q
    np.testing.assert_array_equal(apd.coupled(q, g, P), q)                     # crosstalk 0 by default


def test_crosstalk_suppresses_small_hits_next_to_a_busy_board(jx) -> None:
    import jax.numpy as jnp
    p = P._replace(crosstalk=0.02, threshold_pe=20.0)
    pe = jnp.array([40.0, 1500.0, 1500.0, 40.0])
    time = jnp.full(4, 120.0)
    groups = jnp.array([0, 0, 0, 1])                                        # channel 3 is alone on its board
    keys = jx.random.split(jx.random.PRNGKey(1), 2000)
    q, _, _, hit = jx.vmap(lambda k: apd.sample(k, pe, time, p, RO, groups))(keys)
    h = np.asarray(hit.mean(0))
    assert h[0] < 0.05 and h[3] > 0.95 and h[1] > 0.999
    assert float(q[:, 1].mean()) == pytest.approx(1500.0 - 0.02 * 1540.0, rel=0.02)
    _, ph = apd.expected(pe, time, p, RO, groups)
    np.testing.assert_allclose(np.asarray(ph), h, atol=0.03)


def test_crosstalk_gradient(jx) -> None:
    import jax.numpy as jnp
    groups = jnp.array([0, 0, 0, 1])

    def f(theta):
        p = P._replace(crosstalk=theta[0], adc_per_pe=2.0)
        adc, ph = apd.expected(jnp.array([60.0, 900.0, 700.0, 50.0]), jnp.full(4, 80.0), p, RO, groups)
        return jnp.sum(adc) + 100.0 * jnp.sum(ph)

    x = jnp.array([0.015])
    g = float(jx.grad(f)(x)[0])
    h = 1e-7
    assert g == pytest.approx(float((f(x + h) - f(x - h)) / (2 * h)), rel=1e-5)
