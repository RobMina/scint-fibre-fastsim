# scint-fibre-fastsim

Scintillator + wavelength-shifting fibre optical Monte Carlo and lookup-table
generation for HEP fast simulation. **Apache-2.0.** Common fibre/WLS interface;
separate experiment backends (Mu2e SiPM, NOvA APD).

| Layer | Role |
|---|---|
| `scintfibrefastsim.core` | Strip geometry, bins, Geant4-matched optics, device transport, host driver |
| `scintfibrefastsim.backends` | Photosensor contract + Mu2e SiPM / NOvA APD |
| `scintfibrefastsim.geometry` | Geometry families (strip; NOvA TDR cell caricature) |
| `scintfibrefastsim.validation` | Toy self-check + CRVStandalone compare-plan hooks (wrapper required) |

Not a fork of [lumen](https://github.com/OmarAlterkait/lumen): production tables
come from this lineage (CRVStandalone + `lookupgen.core`). Docs, examples and
test depth follow a Diátaxis layout.

```text
scint-fibre-fastsim
  core → tables / toy geometries / G4 compare hooks
        │
mu2e-fastsim (SiPM; re-exports core)    nova-fastsim (APD; optional pin)
```

Parametric strip geometry is intended for public CRV-module design studies that
need optical tables without experiment-internal inputs.

## Status (0.1.3)

- Core used by mu2e-fastsim via re-export ([PR #326](https://github.com/RobMina/mu2e-fastsim/pull/326)).
- Toy self-check (`run_toy_selfcheck`); CRVStandalone path needs a **wrapper**
  (`examples/crvstandalone_oracle_wrapper.sh`), not raw `wls`.
- Mu2e SiPM bind: inject `simulate` + full config, or `bind_crvfastsim_sipm_synthetic()`.
- Golden regression tests + a CI leg with jax/numpy pinned as in mu2e-fastsim.
- Device support: **CPU JAX tested in CI**; NVIDIA/AMD/Intel via plain JAX/XLA are
  intended but not yet recorded in this repository's CI.

## Install and test

```bash
python -m pip install -e '.[test]'   # NumPy + JAX
python -m pytest -q
python examples/first_hour.py
```

## Licence

Apache-2.0. Copyright 2026 Rob Mina and contributors. Cite CRVStandalone / Offline
ports in NOTICE when publishing derived tables.
