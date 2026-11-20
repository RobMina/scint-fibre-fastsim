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

`NovaAPDBackend` remains a Phase-2 stub.
