# Photons per bin and the time distributions

`scintfibrefastsim.core.run.generate(..., nphot, ...)` tracks `nphot` photons from every
bin and returns, per bin, the number of arrivals and their raw arrival-time (`ht`) and
fibre-emission (`he`) histograms. How many photons per bin a table needs depends on what
reads it, and the arrival probability is not the only number that matters.

## What a table stores

A stock Mu2e CRV table (and a writer following it) stores each bin's arrival-time and
emission distributions as uint8 weights, `uchar(255 k / n + 0.5)` over the n arrivals of
the bin. The encoding itself loses nothing that matters: the decoded distributions equal
the generator's counts (mixture mean and rms to 0.001 ns), and a weight of 255 means a
bin with one arrival, not saturation.

But a bin with few arrivals stores a **few-arrival sample** of its distribution. A
response simulation that samples the photons of one hit from the few bins its track
crosses (as Mu2e Offline does) then draws photon times from a handful of spikes:

- per hit, the photon-time distribution is **narrower and spikier** than the true one,
  so pulses are narrower, split more often and carry fewer photoelectrons each;
- averaged over all bins the distribution is **unbiased** (mixture mean within 0.1 ns),
  so a check of the summed time shape does not see the problem.

## Measured

On a 3200 mm stock-like strip, scintillation light:

| photons per bin | median arrivals per bin | per-bin rms of the arrival time |
|---|---|---|
| 1e4 | 9 | 0.89 x the 1e5 table's |
| 1e5 (stock) | 95 | reference |

Drawing the reference table's own distributions at the same counts and encoding them the
same way gives the same narrowing (0.84), so it is statistics, not the generator. In
the scintillator-Cerenkov part, 2e3 photons per bin leave a median of 1 arrival per bin
and a per-bin rms of 0.31 x the reference.

## Recommendation

- **Scintillation: at least 1e5 photons per bin** for stock-like strips (about 25 mm
  wide, metres long), the statistics of the stock Geant4 tables. A 24.71 x 25 mm,
  3200 mm strip then has a median of 181 arrivals per bin.
- For other parts and geometries, check the **median arrivals per bin**, not only the
  total light. The stock tables have medians of 95 (scintillation, 1e5 photons per bin)
  and 55 (scintillator Cerenkov, 1e5 photons per bin).

Source: [mu2e-fastsim `crv/docs/lookupgen.md`](https://github.com/RobMina/mu2e-fastsim/blob/main/crv/docs/lookupgen.md#photons-per-bin-and-the-time-distributions).
