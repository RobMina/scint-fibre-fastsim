# Validation report

No public compare campaign has been merged into this repository yet.

| Plan | Oracle | Status |
|---|---|---|
| `strip_vs_crvstandalone` | Mu2e/CRVStandalone | public run with declared criteria in review ([PR #12](https://github.com/RobMina/scint-fibre-fastsim/pull/12)); earlier check without declared criteria [below](#prior-evidence-crvstandalone-check-made-in-mu2e-fastsim) |
| `toy_optical_smoke` | toy G4 optical | declared; driver Phase 1 |

When filled, this page should be generated from compare receipts (criteria hashed
before the held-out read), matching the hep-simlib fidelity vocabulary.

## Prior evidence: CRVStandalone check made in mu2e-fastsim

Before the core was extracted, the same code (then `crvfastsim.lookupgen.core`) was
compared with a CRVStandalone build of its current optics (Geant4 11), run in table mode
one bin per job, on the same bins in float64. Both programs write identical table headers
(every constant and bin edge) for the stock and the 24.71 x 25 mm strip cross-sections.
Per bin, the arrival probability is compared with binomial errors on both sides; the
ratio is generated / CRVStandalone arrivals summed over the compared bins. Full record:
[mu2e-fastsim `crv/docs/lookupgen.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/crv/docs/lookupgen.md#agreement-with-crvstandalone-current-model).

| photon source | cross-section, 3200 mm strip | bins (photons per bin) | generated / CRVStandalone | probability pull mean (rms) |
|---|---|---|---|---|
| scintillation | 24.71 x 25 mm | 30 (1e5) | 1.018 +- 0.020 | 0.12 +- 0.15 (0.82) |
| scintillation | stock | 10 (1e5) | 1.026 +- 0.042 | 0.18 +- 0.34 (1.01) |
| Cerenkov in the scintillator | 24.71 x 25 mm | 30 (1e5) | 0.990 +- 0.026 | -0.15 +- 0.15 (0.82) |
| Cerenkov in the fibre | 24.71 x 25 mm | 30 (1e4) | 1.31 +- 0.13 | 0.39 +- 0.26 (1.01) |

- Scintillation and scintillator-Cerenkov light agree within 1 sigma.
- Time and emission pulls average within +-0.6 (largest: scintillation time,
  0.52 +- 0.24).
- The fibre-Cerenkov set is small (a few hundred arrivals in all). Its per-bin pulls
  agree (0.39 +- 0.26), but the summed ratio is 2.4 sigma high.

This check had **no criteria declared before it**, so it counts as "tested, not closed";
the declared-criteria run is the one in PR #12.
