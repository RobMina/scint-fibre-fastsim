# Validation report

| Plan | Oracle | Status |
|---|---|---|
| `strip_vs_crvstandalone` | [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone) HEAD, table mode, its default counter | **passes** in float64 and float32 (2026-11-21): header geometry identical; per-bin probability, time and emission pulls consistent with zero over 61 bins of the three photon sources ([report](../../validation/crvstandalone/README.md)) |
| `toy_optical_smoke` | toy self-check (no Geant4) | runs in CI |

The CRVStandalone check runs on CRVStandalone's own default counter (`scintfibrefastsim.validation.crvstandalone`),
because CRVStandalone fixes its geometry at compile time; `toy_strip` has other dimensions and is not checked against
Geant4. Criteria were declared before the runs (`validation/crvstandalone/criteria.json`).

## Earlier, independent evidence: the mu2e-fastsim CRVStandalone check

Before the core was extracted, the same code (then `crvfastsim.lookupgen.core`, model `current`) was compared with a
separate CRVStandalone build of its current optics (Geant4 11), in table mode, one bin per job, float64 only. It used
other counters than the check above: the Mu2e stock strip and a 24.71 x 25 mm cross-section (3200 mm, built with
those dimensions at compile time). Both programs wrote identical table
headers (every constant and bin edge) for both cross-sections. Per bin the arrival probability is compared with
binomial errors on both sides; the ratio is generated / CRVStandalone arrivals summed over the compared bins. Full
record: [mu2e-fastsim `crv/docs/lookupgen.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/crv/docs/lookupgen.md#agreement-with-crvstandalone-current-model).

| Source | cross-section, 3200 mm | bins (photons per bin) | generated / CRVStandalone | probability pull mean (rms) |
|---|---|---|---|---|
| scintillation | 24.71 x 25 mm | 30 (1e5) | 1.018 +- 0.020 | 0.12 +- 0.15 (0.82) |
| scintillation | stock | 10 (1e5) | 1.026 +- 0.042 | 0.18 +- 0.34 (1.01) |
| Cerenkov, scintillator | 24.71 x 25 mm | 30 (1e5) | 0.990 +- 0.026 | -0.15 +- 0.15 (0.82) |
| Cerenkov, fibre | 24.71 x 25 mm | 30 (1e4) | 1.31 +- 0.13 | 0.39 +- 0.26 (1.01) |

- Time and emission pulls average within +-0.6 (largest: scintillation time, 0.52 +- 0.24).
- The fibre-Cerenkov set is small (a few hundred arrivals in all): its per-bin pulls agree, but the summed ratio is
  2.4 sigma high.
- This check had **no criteria declared before it**, so it supports the result above rather than replacing it.

Read together, the two checks are consistent; they use different counters, so the ratios need not be identical: for scintillation 1.018 +- 0.020 here
against 1.040 +- 0.057 / 1.010 +- 0.058 (float64 / float32) above; for Cerenkov light in the scintillator 0.990 +-
0.026 against 1.107 +- 0.073 / 1.136 +- 0.072 (1.5 to 1.9 sigma apart, the largest difference); for Cerenkov light in
the fibre 1.31 +- 0.13 against 1.04 +- 0.37 / 1.25 +- 0.33. The earlier check has the tighter scintillation and
scintillator-Cerenkov ratios; the check above is the one with declared criteria, both
floats and a public, reproducible driver.
