# scint-fibre-fastsim

Scintillator + wavelength-shifting fibre optical Monte Carlo and lookup-table
generation for HEP fast simulation. **Apache-2.0.** Common fibre/WLS interface;
separate experiment backends (Mu2e SiPM, NOvA APD). Hardware-agnostic via plain
JAX/XLA (NVIDIA, AMD, Intel — no NVIDIA-only photon-loop hard dependency).

| Layer | Role |
|---|---|
| `scintfibrefastsim.core` | Strip geometry, bins, Geant4-matched optics, device transport, host driver (from `crvfastsim.lookupgen.core`; see NOTICE) |
| `scintfibrefastsim.backends` | Photosensor contract + Mu2e SiPM / NOvA APD stubs |
| `scintfibrefastsim.geometry` | Geometry families (strip; NOvA TDR cell caricature) |
| `scintfibrefastsim.validation` | Geant4 / CRVStandalone compare-plan hooks (stubs) |

Not a fork of [lumen](https://github.com/OmarAlterkait/lumen): production tables
come from this lineage (CRVStandalone + `lookupgen.core`). Docs, examples and
test depth follow lumen's Diátaxis layout.

```text
scint-fibre-fastsim          THIS PACKAGE
  core → tables / toy geometries / G4 compare hooks
        │
        ▼ hep-simlib manifests (when wired)
mu2e-fastsim (SiPM)          nova-fastsim (APD)   [depend later]
```

Parametric strip/module geometry folds into **Mu2eBO Candidate G**
(CRV-module-geometry-optimization). Plan:
[`scint-fibre-fastsim` wiki](https://github.com/RobMina/ai-config-expdesign)
(local: `~/ai-config/expdesign/wiki/synthesis/scint-fibre-fastsim.md`).

## Status (0.1.1)

- Core synced with `crvfastsim.lookupgen.core` (copy-with-NOTICE); mu2e-fastsim
  extract PR shims/re-exports onto this package.
- Toy G4 self-check (`run_toy_selfcheck`) and CRVStandalone compare-plan hooks.
- Mu2e SiPM bind start: inject `crvfastsim.response.sipm.simulate` or
  `bind_crvfastsim_sipm()`; NOvA APD still a stub.
- No experiment-internal inputs, conditions or released tables.

## Install and test

```bash
python -m pip install -e '.[test]'   # NumPy + JAX (CPU)
python -m pytest -q
python examples/first_hour.py
```

## Licence

Apache-2.0. Cite CRVStandalone / Offline ports in NOTICE when publishing derived tables.
