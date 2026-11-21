"""The generator side of R4: scintfibrefastsim.core.run.generate on the plan's bins, model "current", one sim type.

    python validation/crvstandalone/gen.py PLAN_DIR --simtype 0 --dtype float64 --photons 30000 --out gen_s0_f64.npz
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("--simtype", type=int, required=True)
    ap.add_argument("--dtype", choices=("float64", "float32"), required=True)
    ap.add_argument("--photons", type=int, required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--lanes", type=int, default=1 << 15)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.dtype == "float64":
        import jax
        jax.config.update("jax_enable_x64", True)
    from scintfibrefastsim.core.run import generate
    from scintfibrefastsim.validation import crvstandalone as C
    plan = json.loads((a.plan / "plan.json").read_text())
    c = C.counter(plan["length"], plan["reflector"])
    bins = np.array(plan["simtypes"][str(a.simtype)]["bins"])
    out = generate(c, C.binning(c), a.simtype, bins, a.photons, dtype=a.dtype, seed=a.seed + 10 * a.simtype,
                   lanes=a.lanes, model="current", log=lambda s: print(s, flush=True))
    stats = out.pop("stats")
    np.savez_compressed(a.out, **out, stats=json.dumps(stats))
    print(json.dumps({k: stats.get(k) for k in ("wall_s", "dtype", "photons", "device")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
