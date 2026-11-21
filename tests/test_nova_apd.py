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
