# Add or use a photosensor backend

1. Implement `PhotosensorBackend` from `scintfibrefastsim.backends.base`
   (`detect(arrivals, *, seed=1) -> dict` with at least `n_pe` and `backend`).
2. Keep fibre transport in `scintfibrefastsim.core`; do not fork the photon loop
   per sensor.
3. Register experiment table formats and hep-simlib manifests in the experiment
   package (`mu2e-fastsim`, `nova-fastsim`), not here.

Shipped stubs: `Mu2eSiPMBackend` (SiPM), `NovaAPDBackend` (APD). Both raise
`NotImplementedError` until Phase 1/2 binds real models.
