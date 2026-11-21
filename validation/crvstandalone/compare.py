"""Score the generator outputs (gen.py) against the CRVStandalone pieces of the plan, with criteria.json.

    python validation/crvstandalone/compare.py PLAN_DIR PIECES_DIR GEN_DIR --output record.json

PIECES_DIR/s<simtype>/b<bin>/LookupTable_* are the oracle pieces; GEN_DIR/gen_s<simtype>_<dtype>.npz the generator.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
from pathlib import Path

import numpy as np

from scintfibrefastsim.validation import crvstandalone as C


def verdict(r: dict, crit: dict) -> dict:
    out = {}
    for key in ("prob_pull", "time_mean_pull", "emis_mean_pull"):
        s, lim = r[key], crit[key]
        if s.get("n", 0) == 0:
            out[key] = None
            continue
        ok = abs(s["mean"]) <= lim["abs_mean_max_sigma"] * max(s["mean_err"], 1e-12) and s["rms"] <= lim["rms_max"]
        if "frac_gt3_max" in lim:
            ok &= s["frac_gt3"] <= lim["frac_gt3_max"]
        out[key] = bool(ok)
    out["prob_ratio"] = bool(abs(r["prob_ratio_weighted"] - 1) <= 3 * r["prob_ratio_err"] + 0.02)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("plan", type=Path)
    ap.add_argument("pieces", type=Path)
    ap.add_argument("gen", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--criteria", type=Path, default=Path(__file__).with_name("criteria.json"))
    a = ap.parse_args(argv)
    crit_bytes = a.criteria.read_bytes()
    crit = json.loads(crit_bytes)
    plan = json.loads((a.plan / "plan.json").read_text())
    c = C.counter(plan["length"], plan["reflector"])
    rec = {"criteria_sha256": hashlib.sha256(crit_bytes).hexdigest(), "plan": plan, "simtypes": {}, "header": None}
    for st in (0, 1, 2):
        p = plan["simtypes"][str(st)]
        ref, missing = {}, []
        for b in p["bins"]:
            f = glob.glob(str(a.pieces / f"s{st}" / f"b{b}" / "LookupTable_*"))
            if not f:
                missing.append(b)
                continue
            head, recs = C.read_piece(f[0], st == 0 and b == 0)
            if head:
                rec["header"] = C.header_check(head, c)
            for i, prob, d, e in recs:
                ref[i] = (prob, d, e)
        s = {"missing_pieces": missing, "dtypes": {}}
        for dt in ("float64", "float32"):
            g = a.gen / f"gen_s{st}_{dt}.npz"
            if not g.is_file():
                continue
            z = np.load(g)
            out = {k: z[k] for k in ("bins", "gen", "ht", "he")}
            out["nphot"] = int(z["nphot"])
            r = C.compare(ref, out, p["photons"])
            r["verdict"] = verdict(r, crit["per_simtype_and_dtype"])
            s["dtypes"][dt] = r
        d = s["dtypes"]
        if "float64" in d and "float32" in d:
            x, y = d["float64"], d["float32"]
            s["dtype_agreement"] = bool(abs(x["prob_ratio_weighted"] - y["prob_ratio_weighted"])
                                        <= 3 * np.hypot(x["prob_ratio_err"], y["prob_ratio_err"]))
        rec["simtypes"][C.SIM_TYPES[st]] = s
        for dt, r in d.items():
            pp = r["prob_pull"]
            print(f"s{st} {dt}: {r['bins']} bins, ratio {r['prob_ratio_weighted']:.4f} +- {r['prob_ratio_err']:.4f}, "
                  f"prob pull {pp.get('mean', float('nan')):+.2f} +- {pp.get('mean_err', float('nan')):.2f} "
                  f"(rms {pp.get('rms', float('nan')):.2f}), verdict {r['verdict']}")
    h = rec["header"]
    rec["all_pass"] = bool(h and h["constants"] and h["edges"] and all(
        v is not False for s in rec["simtypes"].values() for r in s["dtypes"].values() for v in r["verdict"].values())
        and all(s.get("dtype_agreement", True) for s in rec["simtypes"].values()))
    a.output.write_text(json.dumps(rec, indent=1))
    print("header", {k: h[k] for k in ("reflector", "constants", "edges")} if h else None, "all pass", rec["all_pass"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
