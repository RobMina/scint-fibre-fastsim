"""The NOvA cell model against the public references: fit the configurations of criteria.json on the train split,
predict every workflow in float32 and float64, score with hep-simbench and apply the declared pass lines.

    python validation/nova_cell/compare.py OUT_DIR [--record validation/nova_cell/record.json]

writes OUT_DIR/reference/ (reference.py), OUT_DIR/submissions/<config>/<W>/<dtype>/hists.json, OUT_DIR/scores_<config>.json
and the record (fitted parameters, per-histogram statistics, verdicts, criteria and data sha256). Needs JAX and
hep-simbench (``pip install -e '.[validation]'``); CPU is enough (a few seconds).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import reference as REF  # noqa: E402

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

from scintfibrefastsim.backends import nova_apd  # noqa: E402
from scintfibrefastsim.backends.nova_apd import APDParams, Readout  # noqa: E402
from scintfibrefastsim.geometry import nova_cell_tdr  # noqa: E402
from scintfibrefastsim.response import CellOptics, attenuation, fibre_transport  # noqa: E402

DTYPES = ("float32", "float64")
READOUT = {"nd": (Readout(n_samples=8, tick=125.0, dcs_lag=3), dict(tau_rise=140.0, tau_fall=4500.0)),
           "fd": (Readout(n_samples=8, tick=500.0, dcs_lag=3), dict(tau_rise=460.0, tau_fall=7000.0))}


# ------------------------------------------------------------------------------------------- model predictions

def pe_per_cm(w_cm, length_mm, dedx_mev_cm, opt: CellOptics, apd: APDParams, dtype):
    """Mean PE per cm of a muon track at distance w (cm) from the cell centre, readout at +L/2."""
    w = jnp.asarray(w_cm, dtype)
    d = (0.5 * length_mm - 10.0 * w)                                              # mm to the readout end
    coll, _ = fibre_transport(d, nova_cell_tdr(length=length_mm), _cast(opt, dtype))
    dedx_mm = dedx_mev_cm / 10.0
    vis = dedx_mev_cm / (1.0 + opt.birks_kb * dedx_mm + opt.birks_kc * dedx_mm ** 2)
    return jnp.asarray(opt.yield_per_mev, dtype) * jnp.asarray(apd.qe, dtype) * jnp.asarray(vis, dtype) * coll


def ln_fibre(x_m, opt: CellOptics, offset, dtype):
    return jnp.log(attenuation(jnp.asarray(x_m, dtype) * 1000.0, _cast(opt, dtype))) + jnp.asarray(offset, dtype)


def sigma_sqrt(pe, time_res, dtype):
    return jnp.asarray(time_res, dtype) / jnp.sqrt(jnp.maximum(jnp.asarray(pe, dtype), 1.0))


def sigma_crb(pe, which, floor, apd: APDParams, dtype):
    ro, shaping = READOUT[which]
    p = _cast(apd._replace(**shaping), dtype)
    crb = nova_apd.timing_crb(jnp.asarray(pe, dtype), p, ro)
    return jnp.sqrt(crb ** 2 + jnp.asarray(floor, dtype) ** 2)


def _cast(nt, dtype):
    return type(nt)(*[jnp.asarray(v, dtype) for v in nt])


# ------------------------------------------------------------------------------------------- fits (float64)

def _lsq_scale(model, data):
    return float(np.sum(model * data) / np.sum(model * model))


def fit_fibre(x, y, err, start: CellOptics):
    """frac_short, atten_short, atten_long (+ offset, analytic) to ln I: Gauss-Newton on (logit f, log lengths)."""
    def resid(q):
        f, a_s, a_l = 1 / (1 + np.exp(-q[0])), np.exp(q[1]), np.exp(q[2])
        opt = start._replace(frac_short=f, atten_short=a_s, atten_long=a_l)
        m = np.asarray(ln_fibre(x, opt, 0.0, jnp.float64))
        c = np.mean(y - m)
        return (m + c - y) / err, c
    q = np.array([np.log(start.frac_short / (1 - start.frac_short)), np.log(start.atten_short),
                  np.log(start.atten_long)])
    lam = 1e-3
    for _ in range(200):
        r, _ = resid(q)
        J = np.stack([(resid(q + h)[0] - r) / 1e-6 for h in np.eye(3) * 1e-6], axis=1)
        step = np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12), -J.T @ r)
        q_new = q + step
        q_new[1:] = np.clip(q_new[1:], np.log(100.0), np.log(50000.0))
        if np.sum(resid(q_new)[0] ** 2) < np.sum(r ** 2):
            q, lam = q_new, lam / 3
        else:
            lam *= 5
        if np.max(np.abs(step)) < 1e-9:
            break
    r, c = resid(q)
    f, a_s, a_l = 1 / (1 + np.exp(-q[0])), np.exp(q[1]), np.exp(q[2])
    return {"frac_short": float(f), "atten_short": float(a_s), "atten_long": float(a_l), "offset": float(c),
            "chi2": float(np.sum(r ** 2)), "ndf": int(len(y) - 4)}


def post_hoc(tasks, floors) -> dict:
    """Diagnostics after the declared comparison (not validation): which attenuation the cell curves want, with a
    free normalisation per cell; and the timing bound at the lowest-charge bin."""
    cen = lambda h: 0.5 * (np.array(h["edges"][:-1]) + np.array(h["edges"][1:]))

    def parts(q, which):
        f, a_s, a_l = 1 / (1 + np.exp(-q[0])), np.exp(q[1]), np.exp(q[2])
        opt = CellOptics()._replace(frac_short=f, atten_short=a_s, atten_long=a_l, yield_per_mev=1.0)
        out = {}
        for w in which:
            h, cfg = tasks[w]["stock"]["hists"]["pe_per_cm"], tasks[w]["cfg"]
            x, y, e = cen(h), np.array(h["values"]), np.array(h["errors"])
            m = np.asarray(pe_per_cm(x, cfg["cell_length_mm"], cfg["dedx_mev_per_cm"], opt, APDParams(), jnp.float64))
            k = np.sum(m * y / e ** 2) / np.sum(m * m / e ** 2)
            out[w] = ((k * m - y) / e, float(k))
        return out

    def resid(q, which):
        return np.concatenate([r for r, _ in parts(q, which).values()])

    def fit(which):
        q, lam = np.array([0.0, np.log(2000.0), np.log(9000.0)]), 1e-3
        for _ in range(300):
            r = resid(q, which)
            J = np.stack([(resid(q + h, which) - r) / 1e-6 for h in np.eye(3) * 1e-6], axis=1)
            q_new = q + np.linalg.solve(J.T @ J + lam * np.diag(np.diag(J.T @ J) + 1e-12), -J.T @ r)
            q_new[1:] = np.clip(q_new[1:], np.log(100.0), np.log(200000.0))
            if np.sum(resid(q_new, which) ** 2) < np.sum(r ** 2):
                q, lam = q_new, lam / 3
            else:
                lam *= 5
        r = resid(q, which)
        per = {w: {"chi2ndf": float(np.sum(rr ** 2) / len(rr)), "yield_per_mev": k} for w, (rr, k) in parts(q, which).items()}
        return {"frac_short": float(1 / (1 + np.exp(-q[0]))), "atten_short": float(np.exp(q[1])),
                "atten_long": float(np.exp(q[2])), "chi2": float(np.sum(r ** 2)), "bins": int(len(r)),
                "free_normalisations": len(which), "per_cell": per}

    out = {"comment": "post hoc, after the declared comparison; not part of any pass line",
           "attenuation_fd_alone": fit(["NC-FD"]), "attenuation_tb_and_fd": fit(["NC-TB", "NC-FD"])}
    for which, w in (("nd", "NC-TND"), ("fd", "NC-TFD")):
        h = tasks[w]["stock"]["hists"]["sigma_low"]
        x = cen(h)
        crb = np.asarray(sigma_crb(x, which, 0.0, APDParams(), jnp.float64))
        out[f"timing_{which}_lowest_bin"] = {"pe": float(x[0]), "data_ns": float(h["values"][0]),
                                             "crb_ns": float(crb[0]), "crb_with_floor_ns": float(np.hypot(crb[0], floors[which]))}
    return out


def fit_floor(pe, y, which, apd):
    crb = np.asarray(sigma_crb(pe, which, 0.0, apd, jnp.float64))
    grid = np.linspace(0.0, 50.0, 50001)
    chi = [np.sum((np.sqrt(crb ** 2 + f ** 2) - y) ** 2) for f in grid[::50]]
    f0 = grid[::50][int(np.argmin(chi))]
    fine = np.linspace(max(f0 - 0.05, 0.0), f0 + 0.05, 1001)
    chi = [np.sum((np.sqrt(crb ** 2 + f ** 2) - y) ** 2) for f in fine]
    return float(fine[int(np.argmin(chi))])


# ------------------------------------------------------------------------------------------- arms

def configurations(tasks):
    opt0, apd0 = CellOptics(), APDParams()
    st = lambda w, h: tasks[w]["stock"]["hists"][h]
    cen = lambda h: 0.5 * (np.array(h["edges"][:-1]) + np.array(h["edges"][1:]))
    # B (1): fibre
    hf = st("NC-FIBRE", "ln_i")
    fib = fit_fibre(cen(hf), np.array(hf["values"]), np.array(hf["errors"]), opt0)
    optB = opt0._replace(frac_short=fib["frac_short"], atten_short=fib["atten_short"], atten_long=fib["atten_long"])
    # B (2): yield on the Test Beam histogram
    ht = st("NC-TB", "pe_per_cm")
    cfg = tasks["NC-TB"]["cfg"]
    m1 = np.asarray(pe_per_cm(cen(ht), cfg["cell_length_mm"], cfg["dedx_mev_per_cm"], optB._replace(yield_per_mev=1.0),
                              apd0, jnp.float64))
    w = 1 / np.array(ht["errors"]) ** 2
    optB = optB._replace(yield_per_mev=float(np.sum(w * m1 * ht["values"]) / np.sum(w * m1 * m1)))
    # B (3) and C: timing on the high histograms
    tres, floors = {}, {}
    for which, wname in (("nd", "NC-TND"), ("fd", "NC-TFD")):
        hh = st(wname, "sigma_high")
        x, y = cen(hh), np.array(hh["values"])
        tres[which] = _lsq_scale(np.asarray(sigma_sqrt(x, 1.0, jnp.float64)), y)
        floors[which] = fit_floor(x, y, which, apd0)
    return {
        "A_defaults": {"optics": opt0, "apd": apd0, "offset": float(np.mean(np.array(hf["values"]) - np.asarray(
            ln_fibre(cen(hf), opt0, 0.0, jnp.float64)))), "timing": ("sqrt", {k: apd0.time_res for k in tres})},
        "B_train_fit": {"optics": optB, "apd": apd0, "offset": fib["offset"], "timing": ("sqrt", tres), "fibre_fit": fib},
        "C_frontend_timing": {"optics": optB, "apd": apd0, "offset": fib["offset"], "timing": ("crb", floors)},
    }


def arm_hists(tasks, conf, dtype) -> dict:
    """{W: hists.json} of one configuration in one dtype."""
    dt = jnp.dtype(dtype)
    opt, apd = conf["optics"], conf["apd"]
    cen = lambda h: 0.5 * (np.array(h["edges"][:-1]) + np.array(h["edges"][1:]))
    def H(h, v):
        return {"edges": h["edges"], "values": [float(a) for a in np.asarray(v, np.float64)],
                "errors": [0.0] * len(h["values"])}
    out = {}
    for w in ("NC-TB", "NC-FD"):
        h = tasks[w]["stock"]["hists"]["pe_per_cm"]
        cfg = tasks[w]["cfg"]
        x = cen(h)
        v = np.asarray(pe_per_cm(x, cfg["cell_length_mm"], cfg["dedx_mev_per_cm"], opt, apd, dt), np.float64)
        y = {"hists": {"pe_per_cm": H(h, v)}, "yields": {}}
        if w == "NC-TB":
            y["yields"]["centre"] = {"value": float(v[np.abs(x) < 15].mean()), "error": 0.0}
        else:
            far, near = float(v[:2].mean()), float(v[-2:].mean())
            y["yields"] = {"far_end": {"value": far, "error": 0.0}, "near_over_far": {"value": near / far, "error": 0.0}}
        out[w] = y
    h = tasks["NC-FIBRE"]["stock"]["hists"]["ln_i"]
    out["NC-FIBRE"] = {"hists": {"ln_i": H(h, ln_fibre(cen(h), opt, conf["offset"], dt))}}
    kind, par = conf["timing"]
    for which, w in (("nd", "NC-TND"), ("fd", "NC-TFD")):
        hs = {}
        for name in ("sigma_low", "sigma_high"):
            h = tasks[w]["stock"]["hists"][name]
            x = cen(h)
            v = sigma_sqrt(x, par[which], dt) if kind == "sqrt" else sigma_crb(x, which, par[which], apd, dt)
            hs[name] = H(h, v)
        out[w] = {"hists": hs}
    return out


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(HERE), "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                              check=True).stdout.strip() + (" (dirty)" if subprocess.run(
            ["git", "-C", str(HERE), "status", "--porcelain", "--", "../../scintfibrefastsim"], capture_output=True,
            text=True).stdout.strip() else "")
    except Exception:
        return None


def main(argv=None) -> int:
    from hepsimbench.score import score
    import hepsimbench

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    ap.add_argument("--record", type=Path, default=HERE / "record.json")
    a = ap.parse_args(argv)
    ref = a.out / "reference"
    tasks = REF.write(ref)
    confs = configurations(tasks)
    record = {"criteria_sha256": _sha(HERE / "criteria.json"),
              "data_sha256": {p.name: _sha(p) for p in sorted((HERE / "data").glob("*.json"))},
              "scint_fibre_fastsim": _commit(), "hep_simbench": getattr(hepsimbench, "__version__", None),
              "empirical_errors": {w: t["cfg"].get("empirical_error", t["cfg"].get("empirical_errors"))
                                   for w, t in tasks.items()},
              "configurations": {}}
    for name, conf in confs.items():
        preds = {}
        for dt in DTYPES:
            preds[dt] = arm_hists(tasks, conf, dt)
            for w, hj in preds[dt].items():
                d = a.out / "submissions" / name / w / dt
                d.mkdir(parents=True, exist_ok=True)
                (d / "hists.json").write_text(json.dumps(hj, indent=1))
        res = score(a.out / "submissions" / name, ref)
        (a.out / f"scores_{name}.json").write_text(json.dumps(res, indent=1, default=float))
        # dtype agreement of the predictions
        worst = 0.0
        for w in preds["float64"]:
            for k, h in preds["float64"][w]["hists"].items():
                v64, v32 = np.array(h["values"]), np.array(preds["float32"][w]["hists"][k]["values"])
                worst = max(worst, float(np.max(np.abs(v32 - v64) / np.maximum(np.abs(v64), 1e-30))))
        per = {}
        for w, wr in res["workflows"].items():
            per[w] = {dt: {"validity": r["validity"],
                           "hists": {h["hist"]: {k: h[k] for k in ("chi2ndf", "ndf", "p", "floor", "threshold", "flag")}
                                     for h in r["hists"]},
                           "yields": {y["yield"]: {k: y[k] for k in ("ratio", "se", "pass")} for y in r["yields"]}}
                      for dt, r in wr.items()}
        par = {"optics": conf["optics"]._asdict(), "fibre_offset": conf["offset"], "timing": {
            "model": conf["timing"][0], "time_res" if conf["timing"][0] == "sqrt" else "time_floor": conf["timing"][1]}}
        if "fibre_fit" in conf:
            par["fibre_fit"] = conf["fibre_fit"]
        record["configurations"][name] = {"parameters": par, "validity_mean": res["validity_score"],
                                          "max_rel_dtype_diff": worst, "workflows": per}

    def unflagged(cn, w, hist=None):
        return all(not v["flag"] for dt in DTYPES for k, v in record["configurations"][cn]["workflows"][w][dt]["hists"].items()
                   if hist is None or k == hist)

    def yields_pass(cn, w):
        return all(v["pass"] for dt in DTYPES for v in record["configurations"][cn]["workflows"][w][dt]["yields"].values())

    B, C = "B_train_fit", "C_frontend_timing"
    record["post_hoc"] = post_hoc(tasks, confs[C]["timing"][1])
    same_validity = all(record["configurations"][cn]["workflows"][w]["float32"]["validity"]
                        == record["configurations"][cn]["workflows"][w]["float64"]["validity"]
                        for cn in confs for w in tasks)
    record["verdicts"] = {
        "optics_validated": all(unflagged(B, w) and yields_pass(B, w) for w in ("NC-TB", "NC-FIBRE", "NC-FD")),
        "timing_validated": all(unflagged(C, w, "sigma_low") for w in ("NC-TND", "NC-TFD")),
        "dtype_agreement": bool(same_validity and all(record["configurations"][cn]["max_rel_dtype_diff"] <= 1e-4
                                                      for cn in confs)),
    }
    a.record.write_text(json.dumps(record, indent=1, default=float) + "\n")
    for cn, c in record["configurations"].items():
        print(f"\n== {cn}   mean validity {c['validity_mean']}   max f32/f64 rel diff {c['max_rel_dtype_diff']:.1e}")
        for w, r in c["workflows"].items():
            r64 = r["float64"]
            hs = "  ".join(f"{k} {v['chi2ndf']:.2f}/{v['threshold']:.2f}{' FLAG' if v['flag'] else ''}"
                           for k, v in r64["hists"].items())
            ys = "  ".join(f"{k} {v['ratio']:.3f}+-{v['se']:.3f}{'' if v['pass'] else ' FAIL'}" for k, v in r64["yields"].items())
            print(f"  {w:9s} validity {r64['validity']:.3f} (f32 {r['float32']['validity']:.3f})  {hs}  {ys}")
    print("\nverdicts:", record["verdicts"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
