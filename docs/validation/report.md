# Validation report

| Plan | Oracle | Status |
|---|---|---|
| `strip_vs_crvstandalone` | [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone) HEAD, table mode, its default counter | **passes** in float64 and float32 (2026-11-21): header geometry identical; per-bin probability, time and emission pulls consistent with zero over 61 bins of the three photon sources ([report](../../validation/crvstandalone/README.md)) |
| `toy_optical_smoke` | toy self-check (no Geant4) | runs in CI |

The CRVStandalone check runs on CRVStandalone's own default counter (`scintfibrefastsim.validation.crvstandalone`),
because CRVStandalone fixes its geometry at compile time; `toy_strip` has other dimensions and is not checked against
Geant4. Criteria were declared before the runs (`validation/crvstandalone/criteria.json`).
