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
- The sampler does not use it: it draws times with `time_res / sqrt(pe)`.
- Against public NOvA data ([validation/nova_cell](../../validation/nova_cell/README.md)), the bound reproduces
  the measured resolution at 12.5 PE in both detectors, and is about a factor 2 low between 60 and 200 PE.
- `time_res / sqrt(pe)` is far too optimistic at low charge.

Upstream of the APD, `scintfibrefastsim.response.light(edep, dedx, distance, cell, optics, table=None)` gives
expected photons at the pixel and their delay for a `Cell` (`geometry.nova_cell_tdr()`), analytic or from a
`CellTable`. A detector simulation multiplies by `apd.qe` and hands the per-channel sums to `nova_apd`.
