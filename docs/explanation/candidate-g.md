# Candidate G — parametric CRV module geometry

**Mu2eBO CRV-module-geometry-optimization** (sprint G / Candidate G) needs a
**public** optical model. G1 already uses `crvfastsim.design` (parametric module,
tilt, scaled-optics yield, P(miss), deadtime, cost). This package supplies the
optical tables and the parametric strip family those campaigns can cite without
Mu2e-internal inputs.

| Piece | Where |
|---|---|
| Design knobs (tilt, cost, deadtime) | Mu2eBO / `crvfastsim.design` |
| Public optical generator | **this package** |
| G4 oracle | Mu2e/CRVStandalone |
| Second geometry family | NOvA ND cell (TDR caricature) |

Wiki: `~/ai-config/expdesign/wiki/synthesis/extraction-candidates.md` (row G) and
`scint-fibre-fastsim.md`.
