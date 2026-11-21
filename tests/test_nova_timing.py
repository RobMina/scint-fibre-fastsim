"""The front-end timing bound (nova_apd.timing_crb) and the NOvA cell validation inputs."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from scintfibrefastsim.backends import nova_apd
from scintfibrefastsim.backends.nova_apd import APDParams, Readout

ROOT = Path(__file__).resolve().parents[1]
VAL = ROOT / "validation" / "nova_cell"


def test_crb_scales_as_one_over_charge():
    s = nova_apd.timing_crb(np.array([10.0, 100.0, 1000.0]))
    np.testing.assert_allclose(s[0] / s, [1.0, 10.0, 100.0], rtol=1e-12)


def test_crb_numpy_and_jax_agree():
    jnp = pytest.importorskip("jax.numpy")
    pe = np.array([12.0, 60.0, 400.0])
    a = nova_apd.timing_crb(pe)
    b = np.asarray(nova_apd.timing_crb(jnp.asarray(pe, jnp.float32)))
    np.testing.assert_allclose(b, a, rtol=1e-4)


def test_pulse_slope_is_the_derivative():
    p, t, h = APDParams(), np.linspace(-100.0, 3000.0, 400), 1e-3
    np.testing.assert_allclose(nova_apd.pulse_slope(t[t > 1], p),
                               (nova_apd.pulse(t[t > 1] + h, p) - nova_apd.pulse(t[t > 1] - h, p)) / (2 * h),
                               rtol=1e-6, atol=1e-12)


def test_crb_is_reached_by_a_template_fit():
    """Least-squares (charge, time) fits to noisy DCS samples have the bound's spread (high charge, so the fit is
    in its linear regime), averaged over the clock phase."""
    rng = np.random.default_rng(3)
    p, ro, q0 = APDParams(), Readout(), 300.0
    n, lag = ro.n_samples, ro.dcs_lag
    cov = 2.0 * np.eye(n) - np.eye(n, k=lag) - np.eye(n, k=-lag)
    W = np.linalg.inv(cov)

    def model(q, t, phase):
        tau = nova_apd.sample_times(np.asarray(phase), p, ro) - (t - phase)
        return q * nova_apd.dcs(nova_apd.pulse(tau, p), lag)

    err = []
    for _ in range(3000):
        phase = rng.uniform(0.0, ro.tick)
        raw = p.noise_pe * rng.normal(size=n + lag)
        y = model(q0, phase, phase) + nova_apd.dcs(raw, lag)
        q, t = q0, phase
        for _ in range(8):                                          # Gauss-Newton from the truth
            r = y - model(q, t, phase)
            J = np.stack([(model(q + 1e-3, t, phase) - model(q - 1e-3, t, phase)) / 2e-3,
                          (model(q, t + 1e-2, phase) - model(q, t - 1e-2, phase)) / 2e-2], axis=1)
            dq, dt = np.linalg.solve(J.T @ W @ J, J.T @ W @ r)
            q, t = q + dq, t + dt
        err.append(t - phase)
    rms = float(np.sqrt(np.mean(np.square(err))))
    assert rms == pytest.approx(float(nova_apd.timing_crb(np.array(q0))), rel=0.06)


def test_validation_data_are_complete():
    want = {"r1_testbeam_cell": 94, "r2_fd_cell": 32, "r4_fibre": 26, "r5_timing_nd": 64, "r5_timing_fd": 64}
    src = json.loads((VAL / "sources.json").read_text())["sources"]
    for name, n in want.items():
        d = json.loads((VAL / "data" / f"{name}.json").read_text())
        x = d.get("x_cm") or d.get("x_m") or d.get("pe")
        assert len(x) == n, name
        assert d["source_sha256"] == src[d["source"]]["sha256"]


def test_validation_runs_and_matches_the_record(tmp_path):
    pytest.importorskip("hepsimbench")
    pytest.importorskip("jax")
    sys.path.insert(0, str(VAL))
    import compare
    rec = tmp_path / "record.json"
    assert compare.main([str(tmp_path / "run"), "--record", str(rec)]) == 0
    new, old = json.loads(rec.read_text()), json.loads((VAL / "record.json").read_text())
    assert new["criteria_sha256"] == old["criteria_sha256"]
    assert new["verdicts"] == old["verdicts"]
    for cn, c in old["configurations"].items():
        for w, r in c["workflows"].items():
            for dt in ("float32", "float64"):
                assert new["configurations"][cn]["workflows"][w][dt]["validity"] == r[dt]["validity"], (cn, w, dt)


def test_time_resolution_limits():
    p = APDParams()
    pe = np.array([1.0, 12.5, 100.0, 1e6])
    s = nova_apd.time_resolution(pe, p)
    assert np.all(np.diff(s) < 0)
    assert s[-1] == pytest.approx(p.time_floor, rel=1e-3)
    crb = nova_apd.timing_crb(pe[1:2], p)
    assert s[1] ** 2 == pytest.approx(crb[0] ** 2 + (p.time_kappa * p.tau_rise) ** 2 / 12.5 + p.time_floor ** 2)


def test_sampler_time_spread_is_time_resolution():
    jax = pytest.importorskip("jax")
    jnp = jax.numpy
    p = APDParams()
    pe = jnp.full((20000,), 200.0, jnp.float32)
    q, _, t, _ = nova_apd.sample(jax.random.PRNGKey(5), pe, jnp.zeros_like(pe), p)
    want = float(np.sqrt(np.mean(np.asarray(nova_apd.time_resolution(np.asarray(q, np.float64), p)) ** 2)))
    assert float(jnp.std(t)) == pytest.approx(want, rel=0.03)
