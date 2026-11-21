# The NOvA cell model against public data

The NOvA cell pieces of this package are the extruded cell (`geometry.nova_cell_tdr`), the cell optics
(`response`: Birks quenching, looped-fibre two-exponential transport) and the APD and front end
(`backends.nova_apd`). This directory compares them with measurements published by the NOvA collaboration.
- Every input is public, and every point is extracted by a script from the source PDF.
- The criteria (`criteria.json`) were fixed before any model prediction was computed.
- The comparison runs on a CPU in a few seconds, in float32 and float64.

Two declared checks:
- The optics, fitted on the train references, **do not** predict the 15 m Far Detector cell.
- The front-end timing **does not** reproduce the measured resolution between 30 and 400 PE.

Both float precisions agree. The details, and what the failures point to, are below.

## References

| Ref | What | Source | Extraction | Split |
|---|---|---|---|---|
| R1 | mean cosmic-muon response (PE/cm) vs position, one 2.6 m Test Beam cell, with the paper's error bars and fits | NOvA Collaboration, *The NOvA Test Beam program*, arXiv:2609.10804v2, Fig. 22 | **vector**: the figure is drawn as PDF paths, so the 94 points, error bars and both fit curves are exact | train |
| R4 | fibre attenuation curve used by the NOvA simulation (the average over fibre quality-control spools), ln I at 1-26 m | Aurisano et al., *The NOvA simulation chain*, J. Phys. Conf. Ser. 664 (2015) 072002, Fig. 8 | raster: the purple ln I series, sampled at whole metres (half a pixel = 0.005 in ln I) | train |
| R2 | the same response along a 15 m Far Detector cell (plane 84, cell 12), with the thesis' double-exponential fit; labelled "NOvA Preliminary" | E. Niner, PhD thesis, Indiana University (2015), Fig. 3.16 | raster, **band**: the markers are 8 cm (2.5 px) apart and overlap, so the median marker row per pixel column is averaged in 50 cm bins | test |
| R5 | multi-point hit-time resolution vs hit charge, Near Detector (8 MHz) and Far Detector (2 MHz) data, with the thesis' fits | the same thesis, Figs. 4.9(b) and 4.8(b) | raster: marker centroids on the 25 PE bin grid (half a pixel = 0.04 ns ND, 0.14 ns FD) | test below 400 PE, train at 800 PE and above |

`sources.json` pins each file by URL and sha256; `fetch.sh` downloads and checks them; `extract.py` writes
`data/*.json` (each file records its figure, method, calibration and the source hash). Run with `--overlay DIR`,
`extract.py` draws the detected points on each raster figure, which is how the extraction was checked by eye.

Other public numbers set model constants and are not compared:
- the Birks constant: a JINR test stand measured (1.155 +- 0.065) x 10^-2 g MeV^-1 cm^-2 (Anfimov et al., JINST 15
  (2020) C06066), close to the default 0.125 mm/MeV (0.0107 g MeV^-1 cm^-2);
- the APD quantum efficiency (85 %), gain and noise scale (NOvA TDR, FERMILAB-DESIGN-2007-01);
- the ND and FD shaping times and sampling (the same thesis, Sec. 4.6).

## Method (`criteria.json`, declared before any prediction)

Five workflows in the [hep-simbench](https://github.com/RobMina/hep-simbench) reference layout (`reference.py`
writes them).
- **Stock arm:** the public data.
- **Floor arm:** the source's own fit curve, where the source has one.
- **Flag rule:** a histogram is flagged when chi2/ndf > max(2, 3 x floor) and p < 1e-3.
- **Yields:** a measured yield passes when |ratio - 1| - 3 se <= its tolerance.
- **Validity:** (1 - flagged fraction) x (fraction of yields passing).
- **Throughput:** not scored. There is no public rate for NOvA's own cell response code (`stock_rate_per_slot: null`).

| Workflow | Histograms (bins) | Errors | Floor | Yields (tolerance) |
|---|---|---|---|---|
| NC-TB | PE/cm, 3 cm bins, \|w\| <= 99 cm (66) | published | the paper's exponential fit (Eq. 6.2) | centre: mean of \|w\| < 15 cm (5 %) |
| NC-FIBRE | ln I, x = 1-26 m (26) | 0.01 (half the drawn line width) | none (threshold 2) | none |
| NC-FD | PE/cm, 50 cm bins, \|w\| <= 725 cm (30) | empirical: rms of data - thesis fit, 0.277 PE/cm | the thesis' fit | far_end (25 %), near_over_far (10 %) |
| NC-TND, NC-TFD | sigma_t, 25 PE bins: low (PE < 400, 16), high (PE >= 800, 32) | empirical per histogram: 0.36 / 1.18 ns (ND), 0.93 / 0.67 ns (FD) | the thesis' fit | none |

The model:
- **Response in PE/cm:** `yield_per_mev x qe x birks(dE/dx) x dE/dx x collection(d)`.
  - dE/dx = 1.8 MeV/cm in the scintillator for cosmic muons.
  - `collection(d)` is `response.fibre_transport` for a looped fibre, with d = L/2 - w measured to the readout end.
  - L = 2.6 m (Test Beam); 15 m for the Far Detector, which the caption gives as readout at +750 cm and far end at -750 cm.
- **Fibre curve:** ln of `response.attenuation`, with a free offset because the scanner's scale is arbitrary.
- **Timing:** either the sampler as merged in 0.1.6 (`time_res / sqrt(PE)`, 10 ns), or the new `nova_apd.timing_crb` with a constant
  floor added in quadrature.
  - `timing_crb` is the Cramér–Rao bound of a joint charge-and-time fit to the DCS samples of the front-end model:
    CR-RC shape, tick, DCS lag 3, 8 samples, white noise of 1.5 PE per raw sample, averaged over the clock phase.
  - It uses the published shaping and sampling per detector, and nothing in it is fitted.
  - `tests/test_nova_timing.py` checks that a least-squares template fit reaches it (within 6 % at 300 PE).

Configurations:
- **A:** the package defaults. A baseline with no pass line.
- **B:** fitted on the train split only, in this order:
  1. the three attenuation parameters, on the fibre curve;
  2. the yield, on the Test Beam cell;
  3. `time_res`, on the timing points at 800 PE and above.
- **C:** B, with the front-end timing bound plus a floor fitted on the points at 800 PE and above.

Pass lines:
- optics are validated if B passes NC-TB, NC-FIBRE and NC-FD;
- timing is validated if C's low-charge histograms pass;
- every arm agrees between float32 and float64 to 1e-4.

## Result (2026-11-21, scint-fibre-fastsim `04409cd`, hep-simbench `f966330`, CPU)

| Configuration | Mean validity | NC-TB | NC-FIBRE | NC-FD | NC-TND | NC-TFD |
|---|---|---|---|---|---|---|
| A defaults | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 |
| B train fit | 0.50 | 1.00 | 1.00 | 0.00 | 0.50 | 0.00 |
| C front-end timing | 0.60 | 1.00 | 1.00 | 0.00 | 0.50 | 0.50 |

Validity is identical in float32 and float64 for every arm. Predictions agree to 1.2e-6 relative.

| Pass line | Verdict |
|---|---|
| optics validated (B on NC-TB, NC-FIBRE, NC-FD) | **fails**: NC-FD |
| timing validated (C, low-charge histograms) | **fails**: both detectors |
| float32 / float64 agreement | passes |

The comparison per workflow (float64; float32 is the same):

| Workflow | A | B | C |
|---|---|---|---|
| NC-TB chi2/ndf (threshold 29.3) | 10832 | 4.98 | 4.98 |
| NC-TB centre ratio | 0.365 +- 0.002 | 0.984 +- 0.004 | same |
| NC-FIBRE chi2/ndf (threshold 2) | 112 | 0.10 | 0.10 |
| NC-FD chi2/ndf (threshold 3) | 804 | 99.4 | 99.4 |
| NC-FD far_end ratio | 0.232 +- 0.008 | 0.595 +- 0.020 | same |
| NC-FD near_over_far ratio | 1.46 +- 0.05 | 1.48 +- 0.05 | same |
| NC-TND low / high chi2/ndf (threshold 3) | 840 / 19.0 | 472 / 1.30 | 96.4 / 0.98 |
| NC-TFD low / high chi2/ndf (threshold 3) | 1384 / 175 | 69.7 / 3.43 | 169 / 0.94 |

Fitted values in B:
- fibre: short fraction 0.552, 2.65 m and 8.65 m;
- yield: 28.3 photons/MeV at the fibre (with QE 0.85);
- timing: `time_res` 178 ns (ND) and 308 ns (FD).

C's timing floors are 5.3 ns (ND) and 9.1 ns (FD).

## Reading the result

**Optics.**
- The two-exponential form describes the fibre curve to well within the line width.
- With that attenuation and one fitted yield, it describes the 2.6 m Test Beam cell. Its chi2/ndf of 5.0 is better
  than the paper's own exponential fit (9.8), which sets the threshold.
- The same parameters attenuate far too much over 15 m. The predicted readout-end to far-end ratio is 1.48 times
  the measured one, and the far end gets 0.60 of the measured light.
- Post hoc (`record.json` `post_hoc`, not a pass line):
  - The same looped two-exponential form fits the Far Detector curve well when its parameters are free: chi2 18
    for 30 bins with the empirical errors, short fraction 0.37, 2.1 m and 9.3 m.
  - One parameter set fitted to both cells' shapes, with a free normalisation per cell (short fraction 0.44, 1.9 m and
    10.0 m), gives chi2/ndf 4.8 on the Test Beam cell and 1.24 on the Far Detector cell. Both are under the declared
    thresholds.
  - The two normalisations differ by 18 % (27.9 and 32.8 photons/MeV), inside the 25 % far-end tolerance.
- So the model form holds. What fails is carrying the fibre quality-control attenuation, and the Test Beam
  brightness, over to another detector's cells. The spool scan excites the fibre differently from scintillation
  light in a cell, and the Test Beam and Far Detector differ in scintillator, fibre batch and gain.
- Default cell optics for the Far Detector need in-cell attenuation data, not the spool curve. Fitting the
  attenuation to cell curves is not allowed by this protocol, because the Far Detector cell is the test set. It
  needs a further public in-cell curve as its test.

**Timing.**
- The defaults (10 ns / sqrt(PE)) are an order of magnitude too optimistic at low charge: 2.6 ns against the
  measured 27 ns at 15 PE in the ND.
- The front-end bound has no free parameter at low charge. It reproduces the lowest bin of both detectors:
  27.3 ns against 27.4 ns (ND) and 96.7 ns against 99.1 ns (FD) at 12.5 PE.
- It falls as 1/PE, while the data fall more slowly. At 60-200 PE the measured resolution is about twice the
  bound plus floor.
- A missing 1/sqrt(PE) term cannot explain this: it would spoil the agreement at 12.5 PE.
- What the data need between 30 and 400 PE is open. Candidates: the collaboration fit not reaching the bound,
  correlated noise, and selection effects near threshold.
- Study T2 below adds a term for the missing part, and the sampler now uses that model.

## Study T2: a timing model across front ends (`timing_criteria.json`)

Declared after the comparison above and before any prediction of this model:
- **Model:** sigma_t^2 = timing_crb(PE)^2 + (time_kappa x tau_rise)^2 / PE + time_floor^2. One dimensionless
  `time_kappa` is shared by the two front ends, on the hypothesis that the extra spread scales with the rise time
  (140 ns ND, 460 ns FD).
- **Fit:** `time_kappa` and the ND floor on all ND points (both histograms); then the FD floor on the FD points at
  800 PE and above.
- **Test:** the FD points below 400 PE, under the same flag rule.
- Configuration D uses the ND points below 400 PE, which are a NOVA-CELL test histogram. So D is not a benchmark
  submission, and it is not on the leaderboard.

Result (`record.json`, configuration D and `t2_tick_variant`):

| | ND below 400 PE (fitted) | ND 800 PE and above (fitted) | FD below 400 PE (**test**) | FD 800 PE and above |
|---|---|---|---|---|
| chi2/ndf (threshold 3) | 30.0 (flagged) | 1.83 | **65.1 (flagged)** | 1.64 |

- Fitted values: `time_kappa` 0.449 (an extra 63 ns x sqrt(1/PE) in the ND); floors 6.09 ns (ND) and 6.79 ns (FD).
- The transfer test **fails**, and the form does not describe even the ND points it was fitted to. Between 12 and
  100 PE the data fall faster than a 1/sqrt(PE) term allows. The spread the data need beyond the bound and the floor
  falls from about 96 ns x sqrt(1/PE) at 62 PE to 48 at 362 PE: an excess going roughly as PE^-0.75.
- Scaling the extra spread by the tick instead of the rise time (descriptive) gives FD chi2/ndf 69.7.
- No mechanism derived from the public front-end description gives that exponent:
  - noise-limited timing goes as 1/PE (the bound);
  - photon-arrival statistics give 1/sqrt(PE). A rough estimate from the two fibre paths and the decay times
    puts the per-photon spread at about 15 ns in a 3.9 m cell, not 50-100 ns; this was not computed.
- The collaboration's own fit leaves the exponent free.

Adopted as declared: `nova_apd.sample` and `NovaAPDBackend.detect` now draw hit times with
`nova_apd.time_resolution`, which is this model.
- The defaults are the ND fit (`time_kappa` 0.449, `time_floor` 6.09 ns).
- It replaces `time_res / sqrt(PE)` (10 ns: 2.6 ns at 15 PE against the measured 27 ns).
- Against the ND data it is within about 25 % from 12 to 400 PE, at most 21 % high at 12.5 PE and 23 % low at
  62.5 PE.
- It is not validated for another front end.

## Running

```bash
bash validation/nova_cell/fetch.sh SRC_DIR                                  # 25 MB, sha256 checked
python validation/nova_cell/extract.py SRC_DIR [--overlay DIR]              # needs '.[extract]'
python validation/nova_cell/compare.py OUT_DIR                              # needs '.[validation]'; writes record.json
```

`compare.py` runs from the committed `data/*.json`, so the PDFs are needed only to re-extract. The Genesis-format
benchmark built on these references is [`benchmarks/nova_cell`](../../benchmarks/nova_cell/README.md).

Cite the sources above with any use of these points. The points are numbers read from the figures of public
papers, and the papers keep their own licences.
