# The optical core against CRVStandalone (Geant4)

The public Geant4 oracle of `scintfibrefastsim.core`: [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone)
(Apache-2.0) in table mode, on its own default counter (`scintfibrefastsim.validation.crvstandalone`: the 19.78 x
51.34 mm two-fibre extrusion of `wls/src/WLSDetectorConstruction.cc`, 3200 mm, reflector 0). The core is a JAX port of
that table mode, so this checks the port bin by bin against the program it ports.

## Design

- `plan.py`: 20 bins per photon source drawn with a fixed seed (spread over every variable), plus bin 0 of
  scintillation, whose piece carries the table header (the geometry check): 61 oracle runs.
- Oracle: one CRVStandalone HEAD (`c56260a`) run per bin, `wls -c -t SIMTYPE -l 3200 -R 0 -m BIN -M BIN -n PHOTONS -r
  SEED`, 1e5 photons per bin for scintillation and Cerenkov in the scintillator, 1e4 for Cerenkov in the fibre. The
  binary is HEAD built against Geant4 11.3.2 with compile fixes only (Geant4 11 API changes; no physics change).
- Generator: `gen.py`, model `current`, CPU JAX, in float64 and float32, **3e4 photons per bin (3e3 in the fibre),
  three times fewer than the oracle**.
- `compare.py`: per bin, the arrival probability (binomial errors on both sides), the mean time delay and mean number
  of fibre emissions (from the uint8 distributions CRVStandalone writes; the generator side is encoded the same way),
  each as a pull; per source the probability-weighted ratio. Criteria: `criteria.json`, declared before any run.

## Result (2026-11-21, `6a61b43`): passes every criterion in both floats

The header: the piece's geometry constants equal the module's counter and its seven bin-edge arrays are identical.

| Source | float | bins with arrivals | probability pull mean (rms) | weighted ratio | time-mean pull (bins) | emission-mean pull (bins) |
|---|---|---|---|---|---|---|
| scintillation | float64 | 20 | +0.10 +- 0.21 (0.95) | 1.040 +- 0.057 | -0.72 +- 0.27 (14) | -0.01 +- 0.19 (14) |
| scintillation | float32 | 20 | -0.08 +- 0.20 (0.89) | 1.010 +- 0.058 | -0.57 +- 0.20 (12) | +0.44 +- 0.22 (12) |
| Cerenkov, scintillator | float64 | 19 | +0.27 +- 0.24 (1.02) | 1.107 +- 0.073 | -0.03 +- 0.35 (8) | +0.83 +- 0.43 (8) |
| Cerenkov, scintillator | float32 | 19 | +0.33 +- 0.22 (0.94) | 1.136 +- 0.072 | +0.54 +- 0.42 (9) | +0.83 +- 0.48 (9) |
| Cerenkov, fibre | float64 | 10 | -0.20 +- 0.30 (0.93) | 1.04 +- 0.37 | - | - |
| Cerenkov, fibre | float32 | 10 | -0.06 +- 0.27 (0.84) | 1.25 +- 0.33 | - | - |

float32 and float64 agree within errors for every source. `record.json` holds the summary numbers.

**Sensitivity.** The bins are spread over the whole counter, so most see few arrivals (detection probabilities up to
3e-3 per photon), and the generator ran three times fewer photons than the oracle. With 3e4 against 1e5 photons the
per-bin variance is 1/3e4 + 1/1e5, 2.2 times that of equal 1e5 samples, so every error above is about 1.5 times larger
than an equal-statistics run would give. As run, this checks the weighted probability to about 6 % for scintillation, 7 %
for Cerenkov light in the scintillator and 35 % for Cerenkov light in the fibre. Cerenkov light in the scintillator
sits 1.5-1.9 errors high in both floats. The time and emission moments are tested in 8-14 bins for the scintillator
sources and not at all for the fibre (no bin reaches 20 arrivals). A tighter check needs more photons per bin, mostly
on the generator side.

## Running

```bash
python validation/crvstandalone/plan.py PLAN_DIR
# one oracle run per line of PLAN_DIR/commands.txt (SIMTYPE BIN PHOTONS SEED), each in PIECES/s<SIMTYPE>/b<BIN>/ with
# CRVStandalone's photonMap.root and singlePEWaveform_v3.txt: wls -c -t SIMTYPE -l 3200 -R 0 -m BIN -M BIN -n PHOTONS -r SEED
python validation/crvstandalone/gen.py PLAN_DIR --simtype 0 --dtype float64 --photons 30000 --out GEN/gen_s0_float64.npz
python validation/crvstandalone/compare.py PLAN_DIR PIECES GEN --output record.json
```

Measured cost: oracle 13 s-1:20 per single-bin run on one core (0.65 core-h for the 61); generator on 4 CPU cores
25 / 21 / 16 min per source in float64 (13.9 / 12.7 / 9.4 min in float32), MaxRSS 3.1 GB.
