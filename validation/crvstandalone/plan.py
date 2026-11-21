"""The R4 plan: which bins of CRVStandalone's default counter (3200 mm, reflector 0) are run, and the oracle commands.

    python validation/crvstandalone/plan.py OUT_DIR [--bins 20] [--seed 20261121]

Per sim type, `--bins` bins drawn without replacement (fixed seed: spread over every variable), plus bin 0 of sim type
0, whose piece carries the table header (the geometry check). Writes OUT_DIR/plan.json and OUT_DIR/commands.txt, one
`SIMTYPE BIN PHOTONS SEED` line per oracle run; each run is `wls -c -t SIMTYPE -l 3200 -R 0 -m BIN -M BIN -n PHOTONS
-r SEED` in its own directory holding CRVStandalone's photonMap.root and singlePEWaveform_v3.txt (wls-build/).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from scintfibrefastsim.validation import crvstandalone as C

PHOTONS = {0: 100_000, 1: 100_000, 2: 10_000}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20261121)
    a = ap.parse_args(argv)
    c = C.counter(3200.0, 0)
    b = C.binning(c)
    rng = np.random.default_rng(a.seed)
    plan = {"length": 3200.0, "reflector": 0, "seed": a.seed, "simtypes": {}}
    lines = []
    for st in (0, 1, 2):
        n = b.n_bins(st)
        bins = sorted(int(x) for x in rng.choice(n, a.bins, replace=False))
        if st == 0 and 0 not in bins:
            bins = [0] + bins
        plan["simtypes"][str(st)] = {"n_bins": n, "bins": bins, "photons": PHOTONS[st]}
        lines += [f"{st} {x} {PHOTONS[st]} {1000 + 7 * st + x % 997}" for x in bins]
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "plan.json").write_text(json.dumps(plan, indent=1))
    (a.out / "commands.txt").write_text("\n".join(lines) + "\n")
    print(f"{len(lines)} oracle runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
