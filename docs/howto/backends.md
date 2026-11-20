# Add or use a photosensor backend

1. Implement `PhotosensorBackend` from `scintfibrefastsim.backends.base`
   (`detect(arrivals, *, seed=1) -> dict` with at least `n_pe` and `backend`).
2. Keep fibre transport in `scintfibrefastsim.core`; do not fork the photon loop
   per sensor.
3. Register experiment table formats and hep-simlib manifests in the experiment
   package (`mu2e-fastsim`, `nova-fastsim`), not here.

## Mu2e SiPM (bind start)

```python
from scintfibrefastsim.backends import Mu2eSiPMBackend, bind_crvfastsim_sipm

# In a mu2e-fastsim environment (needs CRVFS_LOOKUP_DIR for the photon map):
backend = bind_crvfastsim_sipm()
# Or inject: Mu2eSiPMBackend(simulate=sipm.simulate, photon_map=pmap, config=cfg)
```

`NovaAPDBackend` remains a Phase-2 stub.
