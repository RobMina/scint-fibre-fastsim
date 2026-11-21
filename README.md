# scint-fibre-fastsim

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24418804.svg)](https://doi.org/10.5281/zenodo.24418804)

Scintillator + wavelength-shifting (WLS) fibre optical Monte Carlo and lookup-table
generation for fast detector simulation. Experiment-neutral: one optical core, a
common photosensor interface, and parametric geometry. **Apache-2.0.**

| Layer | Role |
|---|---|
| `scintfibrefastsim.core` | Strip geometry, bins, Geant4-matched optics, device transport, host driver |
| `scintfibrefastsim.response` | Expected (deterministic, differentiable) cell response: quenching, fibre transport, tabulated transport |
| `scintfibrefastsim.backends` | Photosensor contract (`PhotosensorBackend`) and the backends built on it |
| `scintfibrefastsim.geometry` | Parametric geometry families (strip, cell) |
| `scintfibrefastsim.validation` | Toy self-check and Geant4 compare-plan hooks |

```text
scintfibrefastsim.core  → photon arrivals → tables / toy geometries / G4 compare hooks
        │
backends (photosensor response)      downstream fast-simulation packages
```

The parametric strip geometry serves detector-design studies that need optical
tables without experiment-internal inputs. Documentation and examples follow the
[Diátaxis](https://diataxis.fr/) layout under [`docs/`](docs/index.md); specific
backends, geometry families and downstream users are described there.

## Status

- Optical core with golden regression tests guarding transport outputs for fixed seeds.
- Toy self-check (`run_toy_selfcheck`); an external Geant4 reference needs a wrapper
  script ([how-to](docs/howto/g4-compare.md)).
- Photosensor backends: see [docs/howto/backends.md](docs/howto/backends.md).
- Expected cell response (`response`) and photosensor forms that run on NumPy or JAX: differentiable under
  JAX for fast detector simulations that fit or optimise through the response.
- NOvA cell model against public NOvA data ([validation/nova_cell](validation/nova_cell/README.md)), with a
  Genesis/MLCommons-format benchmark ([benchmarks/nova_cell](benchmarks/nova_cell/README.md)).
  - Passes: the fibre curve and the 2.6 m Test Beam cell.
  - Fails, declared before running: transferring those optics to a 15 m Far Detector cell, and the front-end timing
    between 30 and 400 PE.
- Device support: **CPU JAX tested in CI**, including a leg with pinned jax/numpy;
  NVIDIA/AMD/Intel via plain JAX/XLA are intended but not yet recorded in CI.

## Install and test

```bash
python -m pip install -e '.[test]'   # NumPy + JAX
python -m pytest -q
python examples/first_hour.py
```

## Citing

Please cite the release you used: `CITATION.cff` holds the metadata (GitHub's "Cite this repository" renders it), and
each tagged release is archived on Zenodo with its own DOI. The concept DOI
[10.5281/zenodo.24418804](https://doi.org/10.5281/zenodo.24418804) covers every version and resolves to the latest one.

## Licence

Apache-2.0. Copyright 2026 Rob Mina and contributors. See [NOTICE](NOTICE) for the
provenance of ported code and what to cite when publishing derived tables.
