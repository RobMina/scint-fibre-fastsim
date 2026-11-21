"""NOvA APD + front-end backend: photoelectrons → shaped, sampled, dual-correlated-sampled digits.

Model (public-paper scales; an experiment adapter overrides every value):

- the APD converts photons with quantum efficiency ``qe``; its gain fluctuates with excess noise factor ``F``
  (charge variance ``F * n_pe`` in photoelectron units);
- the front end shapes each photoelectron with a unit-peak CR-RC pulse ``exp(-t/tau_fall) - exp(-t/tau_rise)``,
  samples every ``tick`` ns with Gaussian noise ``noise_pe`` per sample, and triggers on dual-correlated sampling
  (``v[j] - v[j - dcs_lag]``) above ``threshold_pe``;
- the recorded charge is the DCS peak in ADC (``adc_per_pe`` per photoelectron).

Three forms share these parameters:

- :func:`expected` — deterministic expected DCS peak and hit probability per channel from expected photoelectrons
  and a channel time (NumPy or JAX; differentiable under JAX);
- :func:`sample` — one JAX draw per channel around the same model (its mean is the expected form);
- :class:`NovaAPDBackend` — the ``PhotosensorBackend`` over scored fibre-end ``Arrival`` lists (NumPy), with each
  photoelectron's own arrival time in the waveform.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from .base import Arrival


def _xp(*xs):
    for x in xs:
        if type(x).__module__.split(".")[0] in ("jax", "jaxlib"):
            import jax.numpy as jnp
            return jnp
    return np


class APDParams(NamedTuple):
    """APD and front-end constants (a pytree under JAX)."""

    qe: float = 0.85
    adc_per_pe: float = 1.0
    excess_noise: float = 2.0      # F
    noise_pe: float = 1.5          # per-sample noise, photoelectron equivalent
    threshold_pe: float = 10.0     # DCS threshold, photoelectron equivalent
    tau_rise: float = 140.0        # ns
    tau_fall: float = 4500.0       # ns
    clock_phase: float = 0.0       # ns
    time_kappa: float = 0.449      # extra hit-time spread kappa x tau_rise / sqrt(n_pe) (time_resolution); ND fit
    time_floor: float = 6.09       # ns, constant hit-time term (calibration, clock); ND fit


@dataclass(frozen=True)
class Readout:
    """Sampling of the front end (static: it fixes array shapes)."""

    n_samples: int = 8             # samples after the hit
    tick: float = 125.0            # ns
    dcs_lag: int = 3


NOVA_APD = APDParams()
NOVA_ND_READOUT = Readout()


def photoelectrons(photons, p: APDParams):
    return photons * p.qe


def pulse(tau, p: APDParams):
    """Unit-peak CR-RC pulse at ``tau`` ns after the photoelectron (0 for ``tau <= 0``)."""
    xp = _xp(tau, p.tau_rise)
    tr, tf = p.tau_rise, p.tau_fall
    tp = tr * tf / (tf - tr) * xp.log(tf / tr)
    norm = 1.0 / (xp.exp(-tp / tf) - xp.exp(-tp / tr))
    tc = xp.maximum(tau, 0.0)
    return xp.where(tau > 0, norm * (xp.exp(-tc / tf) - xp.exp(-tc / tr)), 0.0)


def sample_times(time, p: APDParams, ro: Readout = NOVA_ND_READOUT):
    """``[..., dcs_lag + n_samples]`` pulse arguments of the samples from ``dcs_lag`` before the first one after
    ``time``."""
    xp = _xp(time, p.clock_phase)
    t = time + p.clock_phase
    k0 = xp.floor(t / ro.tick)
    if xp is not np:
        import jax
        k0 = jax.lax.stop_gradient(k0)
    j = xp.arange(-ro.dcs_lag, ro.n_samples, dtype=t.dtype)
    return (k0[..., None] + 1.0 + j) * ro.tick - t[..., None]


def dcs(wave, lag):
    """Dual-correlated samples ``v[j] - v[j - lag]`` for the samples after the hit."""
    return wave[..., lag:] - wave[..., :-lag]


def pulse_slope(tau, p: APDParams):
    """d pulse / d tau (per ns) of the unit-peak CR-RC pulse (0 for ``tau <= 0``)."""
    xp = _xp(tau, p.tau_rise)
    tr, tf = p.tau_rise, p.tau_fall
    tp = tr * tf / (tf - tr) * xp.log(tf / tr)
    norm = 1.0 / (xp.exp(-tp / tf) - xp.exp(-tp / tr))
    tc = xp.maximum(tau, 0.0)
    return xp.where(tau > 0, norm * (xp.exp(-tc / tr) / tr - xp.exp(-tc / tf) / tf), 0.0)


def timing_crb(pe, p: APDParams = NOVA_APD, ro: Readout = NOVA_ND_READOUT, n_phase: int = 64):
    """Cramer-Rao bound (ns) on the hit time from a joint (charge, time) fit to the ``n_samples`` DCS samples.

    The samples are ``q s_j(t) + n_j`` with ``s`` the DCS of the unit-peak pulse and white noise ``noise_pe`` per raw
    sample, so the DCS noise covariance is ``noise_pe^2 (2 delta_jk - delta_|j-k|,dcs_lag)``. With ``g = ds/dt`` and
    ``W`` the inverse covariance, ``var t = 1 / (q^2 (g'Wg - (s'Wg)^2 / s'Ws))``; it is averaged over a uniform clock
    phase (``n_phase`` points). Charge fluctuations do not enter (the charge is the nuisance parameter). This is the
    best a template fit can do: an actual fit, calibration offsets and the photon arrival spread add to it."""
    xp = _xp(pe)
    dt = pe.dtype if hasattr(pe, "dtype") else np.float64
    n, lag = ro.n_samples, ro.dcs_lag
    cov = 2.0 * np.eye(n) - np.eye(n, k=lag) - np.eye(n, k=-lag)
    W = xp.asarray(np.linalg.inv(cov) / p.noise_pe ** 2, dt)
    phase = xp.asarray((np.arange(n_phase) + 0.5) / n_phase * ro.tick, dt)
    tau = sample_times(phase, p._replace(clock_phase=0.0), ro)                    # [n_phase, lag + n]
    s = dcs(pulse(tau, p), lag)
    g = -dcs(pulse_slope(tau, p), lag)                                         # d/dt of the samples
    sws = xp.einsum("pi,ij,pj->p", s, W, s)
    gwg = xp.einsum("pi,ij,pj->p", g, W, g)
    swg = xp.einsum("pi,ij,pj->p", s, W, g)
    info = gwg - swg ** 2 / sws                                                 # per unit charge^2, per phase
    return xp.sqrt(xp.mean(1.0 / info)) / xp.maximum(pe, 1e-12)


def time_resolution(pe, p: APDParams = NOVA_APD, ro: Readout = NOVA_ND_READOUT):
    """Hit-time resolution (ns) at charge ``pe``: ``sqrt(timing_crb^2 + (time_kappa tau_rise)^2 / pe + time_floor^2)``.

    The bound is the front end's own (no free parameter); the 1/sqrt(pe) term scales with the rise time so one
    ``time_kappa`` serves front ends with other shaping; the floor is constant. The defaults are fitted to public
    Near Detector data (validation/nova_cell, study T2). There it is within about 25 % of the measured resolution from
    12 to 400 PE, but the data fall faster than this form between 12 and 100 PE, and the transfer to the Far Detector
    front end fails the declared test (the form, not only the constants, is incomplete)."""
    xp = _xp(pe)
    q = xp.maximum(pe, 1.0)
    return xp.sqrt(timing_crb(q, p, ro) ** 2 + (p.time_kappa * p.tau_rise) ** 2 / q + p.time_floor ** 2)


def _ndtr(x):
    xp = _xp(x)
    if xp is np:
        return 0.5 * np.vectorize(math.erfc)(-np.asarray(x, np.float64) / math.sqrt(2.0))
    from jax.scipy.special import ndtr
    return ndtr(x)


_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(12)
_GH_W = _GH_W / _GH_W.sum()


def expected(pe, time, p: APDParams = NOVA_APD, ro: Readout = NOVA_ND_READOUT):
    """Expected DCS peak (ADC) and hit probability for expected photoelectrons ``pe`` arriving at ``time``.

    The hit probability averages, over the charge fluctuation, the probability that any DCS sample passes the
    threshold with independent noise ``sqrt(2) noise_pe`` per sample. The charge (mean ``pe``, variance
    ``F * pe``) is a gamma variable through the Wilson-Hilferty transform of Gauss-Hermite nodes, which keeps the
    right skew of the compound-Poisson charge at low ``pe``."""
    xp = _xp(pe, time)
    shape = dcs(pulse(sample_times(time, p, ro), p), ro.dcs_lag)               # [..., n_samples]
    peak = xp.max(shape, axis=-1)
    adc = pe * p.adc_per_pe * peak
    sig = xp.sqrt(2.0) * p.noise_pe
    k = xp.maximum(pe, 1e-12) / p.excess_noise                                   # gamma shape
    c = 1.0 / (9.0 * k[..., None])
    base = xp.maximum(1.0 - c + xp.asarray(_GH_X, pe.dtype) * xp.sqrt(c), 0.0)
    q = pe[..., None] * base ** 3                                                # [..., K]
    z = (p.threshold_pe - q[..., :, None] * shape[..., None, :]) / sig                # [..., K, n_samples]
    miss = xp.prod(_ndtr(z), axis=-1)
    return adc, xp.sum(xp.asarray(_GH_W, pe.dtype) * (1.0 - miss), axis=-1)


def sample(key, pe, time, p: APDParams = NOVA_APD, ro: Readout = NOVA_ND_READOUT):
    """One draw per channel (JAX): Poisson photoelectrons, APD gain fluctuation, per-sample noise, DCS threshold and
    time jitter (:func:`time_resolution` at the drawn charge). Returns ``(charge_pe, adc, time, hit)``; ADC rounded."""
    import jax
    import jax.numpy as jnp

    k1, k2, k3, k4 = jax.random.split(key, 4)
    n = jax.random.poisson(k1, pe).astype(pe.dtype)
    q = jnp.maximum(n + jnp.sqrt(n * (p.excess_noise - 1.0)) * jax.random.normal(k2, pe.shape, pe.dtype), 0.0)
    g = p.adc_per_pe
    shape = pulse(sample_times(time, p, ro), p)
    wave = q[..., None] * g * shape + p.noise_pe * g * jax.random.normal(k3, shape.shape, pe.dtype)
    peak = jnp.max(dcs(wave, ro.dcs_lag), axis=-1)
    t = time + time_resolution(q, p, ro) * jax.random.normal(k4, pe.shape, pe.dtype)
    return q, jnp.round(peak), t, peak > p.threshold_pe * g


class NovaAPDBackend:
    """APD + front-end response for NOvA cells over scored fibre-end arrivals."""

    name = "nova_apd"
    sensor_kind = "apd"

    def __init__(self, params: APDParams = NOVA_APD, readout: Readout = NOVA_ND_READOUT,
                 n_channels: int | None = None):
        self.params = params
        self.readout = readout
        self.n_channels = n_channels

    def detect(self, arrivals: list[Arrival], *, seed: int = 1) -> dict:
        """Per channel: photoelectrons drawn with ``qe``, each with its own arrival time and gain (mean 1, variance
        ``F - 1``), summed into the sampled waveform with noise; DCS peak, threshold, and the mean photoelectron time
        with jitter. Returns ``n_photons``, ``n_pe``, ``adc``, ``time``, ``hit`` per channel."""
        p, ro = self.params, self.readout
        rng = np.random.default_rng(seed)
        ch = np.array([a.channel for a in arrivals], np.int64)
        t = np.array([a.time_ns for a in arrivals], np.float64)
        C = self.n_channels if self.n_channels is not None else (int(ch.max()) + 1 if len(ch) else 0)
        n_ph = np.bincount(ch, minlength=C)[:C] if len(ch) else np.zeros(C, np.int64)
        det = rng.random(len(t)) < p.qe
        k = p.excess_noise - 1.0
        gain = rng.gamma(1.0 / k, k, len(t)) if k > 0 else np.ones(len(t))
        n_pe, adc, tm, hit = np.zeros(C), np.zeros(C), np.zeros(C), np.zeros(C, bool)
        for c in range(C):
            sel = det & (ch == c)
            n_pe[c] = sel.sum()
            t0 = t[sel].min() if sel.any() else 0.0
            taus = sample_times(np.array([t0]), p, ro)[0]
            wave = (gain[sel, None] * pulse(taus[None, :] - (t[sel, None] - t0), p)).sum(0) * p.adc_per_pe
            wave = wave + p.noise_pe * p.adc_per_pe * rng.normal(size=wave.shape)
            peak = dcs(wave, ro.dcs_lag).max()
            adc[c] = np.round(peak)
            hit[c] = peak > p.threshold_pe * p.adc_per_pe
            if sel.any():
                tm[c] = t[sel].mean() + float(time_resolution(np.array(float(sel.sum())), p, ro)) * rng.normal()
        return dict(backend=self.name, n_photons=n_ph, n_pe=n_pe, adc=adc, time=tm, hit=hit)
