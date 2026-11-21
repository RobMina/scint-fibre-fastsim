# NOVA-CELL: NOvA cell response against public data

How well does a fast model of one NOvA detector cell reproduce the collaboration's published measurements? A
submission predicts five observables from scintillation to read-out time:
- the light along a 2.6 m cell and along a 15 m cell;
- the transmission of the wavelength-shifting fibre;
- the hit-time resolution of two front ends.

It is scored against the public data with a fixed flag rule, in float32 and float64.

**Scientific motif:** High-Energy Physics · **AI/ML motif:** Surrogate Modeling · **Computing motif:** Throughput
Bound (the model runs inside a detector simulation; throughput is reported, not scored).

This card follows the MLCommons Science Benchmarks Ontology (arXiv:2511.05614). The validation behind it, with the
extraction of every point and the criteria declared before any prediction, is
[`validation/nova_cell`](../../validation/nova_cell/README.md).

## Quick start

```bash
pip install -e '.[validation]'                     # jax + hep-simbench (pinned)
python benchmarks/nova_cell/build.py OUT           # reference, three configurations, scores, leaderboard, bundle
python -m hepsimbench.score MY_SUBMISSION OUT/reference --out scores.json
```

To score your own model, write `MY_SUBMISSION/<W>/<dtype>/hists.json` for every workflow W below and both dtypes,
with the stock arm's bin edges and histogram names (`OUT/reference/<W>/stock/hists.json`) and your predictions as
`values` (errors 0, or your own model errors). `OUT/codabench/solution_submission.zip` is a complete example.

## 1. Problem specification and constraints

**Task.**
- Inputs: a public description of the cell and the read-out:
  - the cell length and where the readout is;
  - the energy loss of a cosmic-ray muon;
  - the fibre length;
  - the shaping and sampling of the front end;
  - the hit charge.
- Outputs: predictions of the mean PE per cm of track along the cell, the fibre's ln transmission, and the hit-time
  resolution against charge.

**Workflows** (hep-simbench layout, one directory each):

| W | Prediction | Bins | Split |
|---|---|---|---|
| NC-TB | mean PE/cm vs distance from the centre, 2.6 m Test Beam cell, readout at +130 cm; plus the mean at \|w\| < 15 cm | 66 x 3 cm, \|w\| <= 99 cm | train |
| NC-FIBRE | ln T(x) of the fibre, any constant offset | x = 1, 2, ..., 26 m | train |
| NC-FD | mean PE/cm vs distance, 15 m Far Detector cell, readout at +750 cm; plus the far-end mean (w = -725, -675 cm) and the near/far ratio | 30 x 50 cm, \|w\| <= 725 cm | test |
| NC-TND | multi-point hit-time resolution (ns) vs charge (PE), ND front end (tick 125 ns, CR-RC 140 / 4500 ns) | 25 PE bins: `sigma_low` 0-400 PE (test), `sigma_high` 800-1600 PE (train) | test / train |
| NC-TFD | the same, FD front end (tick 500 ns, CR-RC 460 / 7000 ns) | as NC-TND | test / train |

**Constraints.**
- Both float32 and float64 predictions are required for every workflow.
- A submission may use the train data to set parameters, never the test data.
- The scorer needs NumPy only and runs in seconds on one CPU core.
- There is no throughput constraint, and throughput is not scored.

## 2. Dataset

**Summary.** 280 points read from five figures of three public NOvA documents:
- the Test Beam paper, arXiv:2609.10804v2, Fig. 22 (94 points, vector-exact);
- the simulation-chain paper, J. Phys. Conf. Ser. 664 (2015) 072002, Fig. 8 (26 points);
- a public PhD thesis, Indiana University (2015), Figs. 3.16, 4.8(b) and 4.9(b) (32 + 64 + 64 points).

`validation/nova_cell/extract.py` reproduces every point from the PDFs, which are pinned by sha256. Errors:
- the published error bars where there are any (NC-TB);
- otherwise empirical: the scatter of the data about the source's own fit (NC-FD, NC-TND, NC-TFD);
- for the fibre curve, half the drawn line width (NC-FIBRE).

**Splits.** The splits do not overlap. The test split is a different detector (15 m Far Detector against the
2.6 m Test Beam and a fibre-spool scan), and a different charge range for timing.

| Split | Content | Scored points |
|---|---|---|
| Train | NC-TB, NC-FIBRE, timing at >= 800 PE (both detectors) | 66 + 26 + 2 x 32 |
| Test | NC-FD, timing below 400 PE (both detectors) | 30 + 2 x 16 |

There is no separate validation split: the benchmark has no training in the machine-learning sense, only
parameter fits. Points outside the bins above are kept in `data/*.json` and not scored. These are the cell ends,
and 400-800 PE, which keeps the two timing ranges apart.

**Schema.** `reference/<W>/stock/hists.json` holds `{"hists": {name: {"edges", "values", "errors"}}, "yields":
{name: {"value", "error"}}}`. `data/*.json` holds each figure's points with its source, method, calibration and
source hash.

**Access and licence.** Everything is in this repository (Apache-2.0 code). The numbers are read from public papers,
which keep their own licences; cite them (`validation/nova_cell/sources.json`).

**FAIR.**
- Findable: every point is identified by figure and bin.
- Accessible: public sources and a public repository.
- Interoperable: JSON in the hep-simbench layout.
- Reusable: the extraction is versioned with the data and checked by hash.

**Bounded.** The dataset is fixed; submissions may not add data.

## 3. Performance metrics

| Metric | Definition | Computed on |
|---|---|---|
| chi2/ndf per histogram | sum over bins of (v_data - v_arm)^2 / (e_data^2 + e_arm^2), over the bins | every histogram |
| flag | chi2/ndf > max(2, 3 x floor) and p < 1e-3, where the floor is the same statistic for the source's own fit curve (none for NC-FIBRE: threshold 2) | every histogram |
| yield pass | \|r - 1\| - 3 se <= tol with r = arm / data: centre 5 %, far_end 25 %, near_over_far 10 % | NC-TB, NC-FD |
| validity per workflow and dtype | (1 - flagged fraction) x (fraction of yields passing) | each W, each dtype |
| **validity score** (headline) | mean of the validities over the 5 workflows x 2 dtypes | all |
| min validity | the smallest of those 10 | all |

The scorer is `hepsimbench.score` (hep-simbench `f966330`), copied unchanged into the Codabench bundle. It
fails closed: a missing dtype or workflow, mismatched edges, or negative errors are errors. The metrics cover both
the shape (every bin) and the absolute light level (yields). They do not cover event-by-event fluctuations or hit
efficiency.

## 4. Reference solution

**Configuration C** of the validation, made from scint-fibre-fastsim's NOvA cell model:
- **Optics:** `response.fibre_transport` on `nova_cell_tdr(length)` with Birks quenching. The attenuation (short
  fraction 0.552, 2.65 m, 8.65 m) is fitted to NC-FIBRE. The yield (28.3 photons/MeV, QE 0.85, dE/dx 1.8 MeV/cm) is
  fitted to NC-TB.
- **Timing:** `nova_apd.timing_crb`, the Cramér–Rao bound of a joint charge-and-time fit to the front end's DCS
  samples, plus a floor added in quadrature and fitted to `sigma_high` (5.3 ns ND, 9.1 ns FD). The bound itself has
  no free parameter.

Two baselines run beside it:
- **A:** the package defaults, nothing fitted.
- **B:** the optics of C, with timing `time_res / sqrt(PE)` fitted to `sigma_high`.

Results (CPU, 2026-11-21; [reference_results/LEADERBOARD.md](reference_results/LEADERBOARD.md)):

| Solution | Validity score | NC-TB | NC-FIBRE | NC-FD | NC-TND | NC-TFD |
|---|---|---|---|---|---|---|
| A defaults (weak baseline) | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B train fit (baseline) | 0.50 | 1.00 | 1.00 | 0.00 | 0.50 | 0.00 |
| **C reference** | **0.60** | 1.00 | 1.00 | 0.00 | 0.50 | 0.50 |

float32 and float64 give the same validity for every arm. How to read this:
- **Train split:** the reference solution passes everything.
- **Test split:**
  - It fails the 15 m cell. The fibre-spool attenuation over-attenuates over 15 m: the near/far ratio comes out
    1.48 times the data.
  - It fails low-charge timing. The bound matches the 12.5 PE bins but falls faster than the data up to 400 PE.
- The headroom (0.60 to 1.00) is the benchmark's open problem. The validation README gives the post-hoc diagnosis.

**Verification.** `tests/test_nova_timing.py` checks:
- that a template fit reaches `timing_crb` within 6 %;
- that the data files are complete and hash-checked;
- that a fresh run reproduces the committed record's validities and verdicts.

**Requirements.** Python >= 3.10, NumPy, JAX (CPU), hep-simbench; a few seconds on one core.

## 5. Documentation and reproducible protocol

**Reproduction.**
1. `pip install -e '.[validation]'`.
2. `python benchmarks/nova_cell/build.py OUT`. It writes `OUT/record.json`, `reference_results/LEADERBOARD.md` and
   the Codabench bundle. `OUT/record.json` should equal `validation/nova_cell/record.json` apart from the commit
   stamp.
3. To re-extract the data from the sources:
   `bash validation/nova_cell/fetch.sh SRC && pip install -e '.[extract]' && python validation/nova_cell/extract.py SRC`.

**Codabench.** `OUT/codabench/competition_bundle.zip` is a results-submission bundle (hep-simbench's builder)
with the reference solution as its solution. benchmark-builder's `validate_codabench_bundle.py` passes tiers 1-3 on
it with the solution zip. Tier 4 (Docker) was not run. The title, logo, terms and dates are placeholders, and no
upload is planned.

**Motivation.** Fast, differentiable detector simulations replace a full optical simulation by a few functions
with parameters. Whether those functions transfer between detectors, and are not just tuned to one, is the
question a fast model has to answer before it is used. Public NOvA data allow that question to be asked in the
open, with a train/test split across detectors.

**Background.**
- A NOvA cell is a PVC extrusion filled with liquid scintillator, read out by a looped wavelength-shifting fibre
  whose two ends go to one avalanche-photodiode pixel.
- Light from a point in the cell reaches the readout along a short path and a long path, so attenuation and cell
  length set the response along the cell.
- The front end shapes and samples the APD signal. The hit time comes from a fit to the samples, so its
  resolution depends on the shaping, the sampling and the noise.

**Known limitations.**
- The Far Detector curve is one cell, read from a raster figure as a band (the markers overlap), and labelled
  preliminary.
- The cosmic-ray dE/dx is a declared constant, not measured per hit.
- The empirical errors make the source's own fit the yardstick, with a floor of 1 by construction.
- Hit efficiency, charge distributions and noise hits are not in the dataset: no public figure gives them with
  enough information.
- The timing data are a collaboration fit's output: the resolution of their reconstruction, not of an ideal fit.

**Citation.** Cite scint-fibre-fastsim (`CITATION.cff`) and the three sources in `validation/nova_cell/sources.json`.

**Licence.** Code Apache-2.0. The data are numbers read from public papers, which keep their own licences.

## Rubric self-score

[`rubric.yaml`](rubric.yaml), scored with benchmark-builder's `score_benchmark.py` into
[`SCORE_REPORT.md`](SCORE_REPORT.md): 4.50 / 5. The remaining gaps are community-standard data metadata (Croissant or
HEPData), metrics that cover only part of the response (no fluctuations or efficiency), and a paper.
