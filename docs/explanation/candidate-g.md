# Parametric CRV module geometry

Public optical tables from this package support design campaigns that vary
scintillator-strip / module parameters (cross-section, fibre layout, tilt) and
need a Geant4-checked optical model without experiment-internal inputs.

| Piece | Where |
|---|---|
| Design knobs (geometry, cost proxies, miss probability) | Experiment design tooling (outside this repo) |
| Public optical generator | **this package** |
| G4 oracle | [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone) (via a wrapper; see howto/g4-compare) |
| Second geometry family | NOvA ND cell caricature (TDR; Phase 2) |

The first consumer is the Mu2e CRV fast-sim / design stack; a second geometry
family targets NOvA ND cells.
