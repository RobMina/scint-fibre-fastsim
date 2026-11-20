# API sketch (0.1.3)

## `scintfibrefastsim.core`

| Module | Role |
|---|---|
| `counter.Counter` | Parametric strip + two WLS fibres |
| `binning.Binning` | Photon-source bins |
| `optics` / `materials` | Geant4-matched optical building blocks; `current` model |
| `transport` / `run` | JAX device transport and host driver (`stats` includes `package_version`) |

## `scintfibrefastsim.backends`

- `Arrival` (pre-PDE optical arrival), `PhotosensorBackend`
- `Mu2eSiPMBackend`, `SiPMConfigView` (full stand-in for stock `simulate`)
- `bind_crvfastsim_sipm`, `bind_crvfastsim_sipm_synthetic`, `uniform_photon_map`
- `NovaAPDBackend` (stub)

## `scintfibrefastsim.geometry`

- `strip_family()`, `cell_tdr_caricature()`, `toy_strip()`

## `scintfibrefastsim.validation`

- `G4ComparePlan`, `CompareReceipt`
- `compare_plan_crvstandalone()`, `compare_plan_toy()`
- `run_toy_selfcheck(...)`, `run_compare(..., mode=...)` — modes `auto` / `selfcheck` / `oracle`
