# Package boundaries

| Owns | Does not own |
|---|---|
| Strip/cell + WLS fibre optical MC → tables; toy geometries; G4 compare drivers | Offline / NOvASoft digi; Mu2e-internal v6 binaries; private full GDML worlds |
| Common photosensor **interface** and SiPM bind helpers | Production SiPM pixel model source (lives in `crvfastsim.response` until extracted) |
| Public NOvA TDR cell caricature (`geometry.Cell`, `nova_cell_tdr`) and its expected optics (`response`) | Private NOvA GDML |

`mu2e-fastsim` re-exports `crvfastsim.lookupgen.core` from this package
([PR #326](https://github.com/RobMina/mu2e-fastsim/pull/326)): the module objects are
shared. Experiment adapters keep offline table formats and Mu2e-tuned optical models.

## Downstream packages

```text
scint-fibre-fastsim
  core → tables / toy geometries / G4 compare hooks
        │
mu2e-fastsim (SiPM; re-exports core)    nova-fastsim (APD; optional pin)
```

| Backend | Status |
|---|---|
| Mu2e SiPM (`Mu2eSiPMBackend`) | bind by injecting `simulate` + full config, or `bind_crvfastsim_sipm_synthetic()` ([how-to](../howto/backends.md)) |
| NOvA APD (`NovaAPDBackend`, `nova_apd.expected` / `sample`) | APD + front end (QE, excess noise, CR-RC shaping, DCS threshold): expected form (differentiable), JAX sampler, and `detect` over arrivals; public-scale defaults |

The Geant4 reference used for the optical core is the public
[Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone) table mode, run through
`examples/crvstandalone_oracle_wrapper.sh` (not the raw `wls` binary).

## Public hygiene

This repository is public. Do **not** put private wiki paths, home-directory paths, or
site-specific scratch trees in docs, examples, commits, or PR text. Configure sites
through environment variables (`$SCINTFIBRE_*`, `$CRVFS_*`, kit `site.env` / endpoint
env) only.
