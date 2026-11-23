# Add or use a photosensor backend

1. Implement `PhotosensorBackend` from `scintfibrefastsim.backends.base`
   (`detect(arrivals, *, seed=1) -> dict` with at least `n_pe` and `backend`).
2. Keep fibre transport in `scintfibrefastsim.core`; do not fork the photon loop
   per sensor.
3. Register experiment table formats in the experiment package, not here.

## Mu2e SiPM

Core `Arrival` values are optical (pre-PDE). Stock `sipm.simulate` draws pixel
firing itself; do not apply a separate PDE scale in `detect`.

```python
from scintfibrefastsim.backends import Mu2eSiPMBackend, bind_crvfastsim_sipm_synthetic

# Public smoke (no CRVFS_LOOKUP_DIR): real simulate + uniform map
backend = bind_crvfastsim_sipm_synthetic()

# Or inject explicitly — config must expose every field simulate reads
# (SiPMConfigView defaults do; truncated configs raise TypeError):
# from crvfastsim.response import sipm
# backend = Mu2eSiPMBackend(
#     simulate=sipm.simulate, photon_map=pmap, config=sipm.SiPMConfig(),
# )
```

## NOvA APD + front end

Three forms share `APDParams` (QE, ADC per photoelectron, excess noise, noise, threshold, shaping, clock phase,
time resolution) and a static `Readout` (samples, tick, DCS lag):

```python
from scintfibrefastsim.backends import NOVA_APD, NOVA_ND_READOUT, NovaAPDBackend, nova_apd

adc, p_hit = nova_apd.expected(pe, time, NOVA_APD, NOVA_ND_READOUT)     # NumPy or JAX; differentiable in JAX
q, adc, t, hit = nova_apd.sample(key, pe, time, NOVA_APD, NOVA_ND_READOUT)   # JAX draw around it
out = NovaAPDBackend(n_channels=32).detect(arrivals, seed=1)            # PhotosensorBackend over Arrival lists
```

`nova_apd.timing_crb(pe, p, ro)` is the Cramér–Rao bound on the hit time of a joint charge-and-time fit to the
DCS samples of this front end, with no free parameter.
- Against public NOvA data ([validation/nova_cell](../../validation/nova_cell/README.md)), the bound reproduces
  the measured resolution at 12.5 PE in both detectors, and is about a factor 2 low between 60 and 200 PE.

The sampler and the backend draw hit times with `nova_apd.time_resolution(pe, p, ro)`:
sqrt(timing_crb^2 + (`time_kappa` x `tau_rise`)^2 / pe + `time_floor`^2).
- The defaults (`time_kappa` 0.449, `time_floor` 6.09 ns) are fitted to public Near Detector data, within about
  25 % of it from 12 to 400 PE.
- The same form does not transfer to the Far Detector front end (study T2 in the validation): it is an ND
  description, not a validated front-end model.
- This replaces `time_res`, which is removed.

The recorded DCS peak can saturate (no ceiling by default):
- `adc_max` is the ADC ceiling of the DCS peak (digitiser range less the pedestal) and `adc_max_sigma` its
  hit-to-hit spread. `sample` and the backend cap the ADC; the threshold decision uses the uncapped peak.
- A saturated pulse's charge estimate is `q_ceiling (1 + sat_recovery ln(peak / ceiling))`, where `q_ceiling` is the
  charge at which the drawn pulse reaches the ceiling. `sat_recovery = 0` is a hard cap; a positive value is what a
  pulse fit to the unsaturated samples recovers. `sample` returns this estimate as its charge.
- `expected` gives `E[min(adc, M)]` with M normal (`nova_apd.expected_adc`), differentiable in both ceiling
  parameters.
- The values are an experiment's: they are not fitted to public data here.

Channels that share an APD and front-end board can couple (off by default):
- pass `groups` (an integer group id per channel, same shape as `pe`) to `expected` or `sample`; each channel's
  charge is then reduced by `crosstalk` times the summed charge of the other channels in its group, floored at 0
  (`nova_apd.coupled`, `nova_apd.group_others`). Small hits next to a busy board fall below threshold and the
  surviving ones read less charge.
- Pulses in a group are treated as simultaneous; `expected` takes the charge fluctuation around the coupled charge.
- The backend over arrivals does not group channels.

Upstream of the APD, `scintfibrefastsim.response.light(edep, dedx, distance, cell, optics, table=None)` gives
expected photons at the pixel and their delay for a `Cell` (`geometry.nova_cell_tdr()`), analytic or from a
`CellTable`. A detector simulation multiplies by `apd.qe` and hands the per-channel sums to `nova_apd`.
