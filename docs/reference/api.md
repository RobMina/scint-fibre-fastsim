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
- `NovaAPDBackend`, `APDParams`, `Readout`, `NOVA_APD`, `NOVA_ND_READOUT`; `nova_apd.expected`, `nova_apd.sample`, `nova_apd.pulse`

## `scintfibrefastsim.geometry`

- `strip_family()`, `cell_tdr_caricature()`, `toy_strip()`
- `Cell`, `nova_cell_tdr()` (looped-fibre liquid-scintillator cell, mm)

## `scintfibrefastsim.response`

- `CellOptics` (yield, Birks, two-exponential attenuation, fibre speed, decay), `NOVA_CELL_OPTICS`
- `birks`, `attenuation`, `fibre_transport`, `light`; `CellTable`, `tabulate`, `lookup` (NumPy or JAX)

## `scintfibrefastsim.validation`

- `G4ComparePlan`, `CompareReceipt`
- `compare_plan_crvstandalone()`, `compare_plan_toy()`
- `run_toy_selfcheck(...)`, `run_compare(..., mode=...)` — modes `auto` / `selfcheck` / `oracle`
