# API sketch (0.1.0)

## `scintfibrefastsim.core`

| Module | Role |
|---|---|
| `counter.Counter` | Parametric strip + two WLS fibres |
| `binning.Binning` | Photon-source bins |
| `optics` / `materials` | Geant4-matched optical building blocks; `current` model |
| `transport` / `run` | JAX device transport and host driver |

## `scintfibrefastsim.backends`

- `Arrival`, `PhotosensorBackend`
- `Mu2eSiPMBackend`, `NovaAPDBackend` (stubs)

## `scintfibrefastsim.geometry`

- `strip_family()`, `cell_tdr_caricature()`, `toy_strip()`

## `scintfibrefastsim.validation`

- `G4ComparePlan`, `compare_plan_crvstandalone()`, `compare_plan_toy()`
- `run_compare(plan)` — stub
