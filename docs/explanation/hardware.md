# Hardware and JAX backends

Photon transport uses plain JAX/XLA. There is **no** NVIDIA-only (CUDA photon-loop
or tallymap) hard dependency in this repository.

| Backend | Status in this repo |
|---|---|
| CPU (JAX) | **Tested** — GitHub Actions CI; runs, but slow for production tables ([below](#jax-on-cpu)) |
| NVIDIA (CUDA) | **Measured** on A100, RTX A6000, A40 and RTX Pro 6000 ([below](#measured-dtype-and-gpu-type)); not in CI |
| AMD (ROCm) | Intended where JAX ROCm builds exist; not recorded in CI here |
| Intel (XPU / CPU) | Intended via JAX CPU or XPU plugins; not recorded in CI here |

Prefer dtype-explicit kernels (`float32` / `float64` arguments) over global JAX
defaults. One CI leg pins `jax`/`numpy` to the versions used by mu2e-fastsim so
transport goldens do not drift silently.

## Measured: dtype and GPU type

These results were measured on the production optical core (`scintfibrefastsim.core`,
then still inside mu2e-fastsim as `crvfastsim.lookupgen.core`, the same code) while it
made strip tables for a Mu2e CRV design study. Source:
[mu2e-fastsim `crv/docs/lookupgen.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/crv/docs/lookupgen.md#float32-float64-and-the-device).

**Test.** One table part: Cerenkov light in the fibre of a 24.71 x 25 mm strip,
56,375 bins x 1e4 photons, seed 1. Two runs are compared per observable by chi2/ndf,
with the last bits of each number ignored. The observables are:

- the per-bin arrival yield;
- the yield projected on z (and on the last 100 mm at each end), r, phi, theta and beta;
- the per-bin mean arrival time and mean emission count;
- the summed arrival-time and emission shapes.

**Floor.** The same float32 run with another seed gives chi2/ndf 0.46 to 1.60 over these
observables. Runs with the same seed are correlated, so they sit below that floor.

| run | time vs A100 float32 | compared with | worst chi2/ndf |
|---|---|---|---|
| RTX A6000, float32 | 1.45 | A100 float32, same seed | 0.10 |
| A40, float32 | 1.43 | A100 float32, same seed | 0.08 |
| RTX Pro 6000, float32 | 0.55 | A100 float32, same seed | 0.008 |
| A100, float64 | 2.0 x float32 on the card | A100 float32, same seed | 0.84 |
| RTX A6000 / A40, float64 | 5.4 x float32 on the card | float32 on the card, same seed | 0.67 / 0.82 |
| RTX Pro 6000, float64 | 6.8 x float32 on the card | float32 on the card, same seed | 0.58 |

- **float32 and float64 give the same tables at the physics level** on every card:
  every chi2/ndf is inside the seed-to-seed floor, and the float32 / float64 yield ratios
  are 0.991 to 0.997 (+- 0.004 to 0.008). float64 costs 2x on the A100, 5.4x on the
  A6000 and A40, and 6.8x on the RTX Pro 6000, which follows each card's FP64 rate.
  float32 is the production choice.
- **The four GPU types agree.** The other cards differ from the A100 by about a tenth of
  the seed-to-seed spread. The A6000 and A40 take about 1.45x the A100 time; the
  RTX Pro 6000 takes 0.55x (1.8x faster).
- **Peak device memory is under 1 GB** (0.6 to 0.7 GB in these runs), so any of these
  cards, or a slice of one, is large enough.

Since the device does not change the table, a table's provenance can record the device
per piece as information rather than as an input.

## JAX on CPU

The core runs on the JAX CPU backend (that is what CI tests), but it is slow for
production tables. In the same measurement, float32, one table part costs
**2,000 to 3,600 CPU core-hours per A100-hour** of the same work (fibre Cerenkov and
scintillation parts, timed on a fraction of each and scaled). The transport keeps a
wide wavefront of photon lanes, which suits a GPU; on a CPU the tail of slow lanes
(photons trapped in the fibre that bounce until they are absorbed) dominates.

Split CPU work into many single-thread processes: one 16-thread process ran slower than
16 one-thread processes on the same cores (it kept only about 4 cores busy). Random
numbers are keyed by bin and photon, so a table split over processes gives the same
counts as one run.

For comparison, the stock Geant4 table mode (CRVStandalone) is about 1,400 core-hours for
one stock-statistics table: cheaper than this core on CPU, but a different code.
