"""Build the hep-simbench reference tree of the NOvA cell references (data/*.json), as criteria.json defines it.

    python validation/nova_cell/reference.py OUT_DIR

writes OUT_DIR/<W>/{reference.json, stock/hists.json, floor/hists.json} for the workflows NC-TB, NC-FIBRE, NC-FD,
NC-TND and NC-TFD. NumPy only. The stock arm is the public data; the floor arm, where there is one, is the
source's own fit curve, so the flag threshold max(2, 3 x floor) is set by how well the source's own description
fits its data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SPLIT = {"NC-TB": "train", "NC-FIBRE": "train", "NC-FD": "test", "NC-TND": "test (low) / train (high)",
         "NC-TFD": "test (low) / train (high)"}
TIMING_LOW, TIMING_HIGH = 400.0, 800.0


def load(name: str) -> dict:
    return json.loads((DATA / f"{name}.json").read_text())


def _pick(x, centres, tol):
    x = np.asarray(x, float)
    idx = [int(np.argmin(np.abs(x - c))) for c in centres]
    assert all(abs(x[i] - c) < tol for i, c in zip(idx, centres)), "a bin centre has no data point"
    return np.array(idx)


def _h(edges, values, errors):
    return {"edges": [float(e) for e in edges], "values": [float(v) for v in values],
            "errors": [float(e) for e in errors]}


def tb() -> dict:
    d = load("r1_testbeam_cell")
    centres = np.arange(-97.5, 97.5 + 1e-9, 3.0)
    i = _pick(d["x_cm"], centres, 0.05)
    y, e = np.array(d["pe_per_cm"])[i], np.array(d["error"])[i]
    c = np.abs(centres) < 15
    fit = np.interp(centres, d["fit_exponential"]["x_cm"], d["fit_exponential"]["pe_per_cm"])
    edges = np.append(centres - 1.5, centres[-1] + 1.5)
    return {
        "centres": centres, "stock": {"hists": {"pe_per_cm": _h(edges, y, e)},
                                      "yields": {"centre": {"value": float(y[c].mean()),
                                                            "error": float(y[c].std(ddof=1) / np.sqrt(c.sum()))}}},
        "floor": {"hists": {"pe_per_cm": _h(edges, fit, np.zeros_like(fit))}},
        "cfg": {"boundary": "energy deposit of a cosmic-ray muon along a 2.6 m Test Beam cell -> mean PE per cm at "
                            "the readout", "mode": "values", "yield_tolerances": {"centre": 0.05},
                "cell_length_mm": 2600.0, "dedx_mev_per_cm": 1.8},
    }


def fibre() -> dict:
    d = load("r4_fibre")
    x, y = np.array(d["x_m"]), np.array(d["ln_i"])
    edges = np.append(x - 0.5, x[-1] + 0.5)
    return {"centres": x, "stock": {"hists": {"ln_i": _h(edges, y, np.full_like(y, 0.01))}}, "floor": None,
            "cfg": {"boundary": "fibre length -> transmitted fraction (ln, arbitrary offset)", "mode": "values"}}


def fd() -> dict:
    d = load("r2_fd_cell")
    centres = np.arange(-725.0, 725.0 + 1e-9, 50.0)
    i = _pick(d["x_cm"], centres, 0.5)
    y = np.array(d["pe_per_cm"])[i]
    fit = np.interp(centres, d["fit_double_exponential"]["x_cm"], d["fit_double_exponential"]["pe_per_cm"])
    sig = float(np.sqrt(np.mean((y - fit) ** 2)))
    far = float(y[:2].mean())
    near = float(y[-2:].mean())
    e_far = sig / np.sqrt(2.0)
    ratio = near / far
    e_ratio = ratio * np.sqrt((e_far / near) ** 2 + (e_far / far) ** 2)
    edges = np.append(centres - 25.0, centres[-1] + 25.0)
    return {
        "centres": centres,
        "stock": {"hists": {"pe_per_cm": _h(edges, y, np.full_like(y, sig))},
                  "yields": {"far_end": {"value": far, "error": float(e_far)},
                             "near_over_far": {"value": float(ratio), "error": float(e_ratio)}}},
        "floor": {"hists": {"pe_per_cm": _h(edges, fit, np.zeros_like(fit))}},
        "cfg": {"boundary": "energy deposit of a cosmic-ray muon along a 15 m Far Detector cell -> mean PE per cm "
                            "at the readout", "mode": "values",
                "yield_tolerances": {"far_end": 0.25, "near_over_far": 0.10},
                "cell_length_mm": 15000.0, "dedx_mev_per_cm": 1.8, "empirical_error": sig},
    }


def timing(which: str) -> dict:
    d = load(f"r5_timing_{which}")
    pe, y = np.array(d["pe"]), np.array(d["sigma_ns"])
    fit = np.interp(pe, d["fit"]["pe"], d["fit"]["sigma_ns"])
    out_s, out_f, sigs = {}, {}, {}
    for name, sel in (("sigma_low", pe < TIMING_LOW), ("sigma_high", pe >= TIMING_HIGH)):
        c = np.round((pe[sel] - 12.5) / 25.0) * 25.0 + 12.5
        assert np.all(np.abs(c - pe[sel]) < 3.0), "a timing point is off the 25 PE grid"
        edges = np.append(c - 12.5, c[-1] + 12.5)
        sig = float(np.sqrt(np.mean((y[sel] - fit[sel]) ** 2)))
        sigs[name] = sig
        out_s[name] = _h(edges, y[sel], np.full(sel.sum(), sig))
        out_f[name] = _h(edges, fit[sel], np.zeros(sel.sum()))
    det = {"nd": "Near", "fd": "Far"}[which]
    return {"centres": None, "stock": {"hists": out_s}, "floor": {"hists": out_f},
            "cfg": {"boundary": f"hit charge (PE) -> multi-point hit-time resolution, {det} Detector front end",
                    "mode": "values", "readout_tick_ns": d["readout_tick_ns"], "empirical_errors": sigs}}


def nd() -> dict:
    """R6, validation only (study O2): the Near Detector cell, errors from the scatter about the thesis' fit."""
    d = load("r6_nd_cell")
    x, y = np.array(d["x_cm"]), np.array(d["pe_per_cm"])
    fit = np.interp(x, d["fit_attenuation"]["x_cm"], d["fit_attenuation"]["pe_per_cm"])
    sig = float(np.sqrt(np.mean((y - fit) ** 2)))
    w = d["bin_cm"]
    edges = [float(x[0] - w / 2)] + [float(v) for v in 0.5 * (x[1:] + x[:-1])] + [float(x[-1] + w / 2)]
    return {"centres": x, "stock": {"hists": {"pe_per_cm": {"edges": edges, "values": y.tolist(),
                                                                                  "errors": [sig] * len(y)}}},
            "floor": {"hists": {"pe_per_cm": {"edges": edges, "values": fit.tolist(), "errors": [0.0] * len(y)}}},
            "cfg": {"boundary": "cosmic-ray muon along a 3.9 m Near Detector cell -> mean PE per cm", "mode": "values",
                    "cell_length_mm": 3900.0, "dedx_mev_per_cm": 1.8, "empirical_error": sig,
                    "note": "10 cm bins where markers are visible (bins hidden under the fit curves are missing); the edges are midpoints between the bin centres, so read the centres, not the widths"}}


def fdsim() -> dict:
    """R7, validation only (study O2): NOvA's simulated Far Detector attenuation, shape over |W| <= 700 cm; errors
    from the scatter of the calibrated series about its mean."""
    d = load("r7_fd_simulation")
    u, c = d["uncalibrated"], d["calibrated"]
    x, y = np.array(u["w_cm"]), np.array(u["ratio"])
    xc, yc = np.array(c["w_cm"]), np.array(c["ratio"])
    sel, selc = np.abs(x) <= 700.0, np.abs(xc) <= 700.0
    sig = float(np.std(yc[selc], ddof=1))
    x, y = x[sel], y[sel]
    edges = np.append(x - 8.0, x[-1] + 8.0)
    return {"centres": x, "stock": {"hists": {"ratio": _h(edges, y, np.full_like(y, sig))}}, "floor": None,
            "cfg": {"boundary": "position along a 15 m Far Detector cell -> relative response (NOvA simulation)",
                    "mode": "values", "cell_length_mm": 15000.0, "empirical_error": sig, "normalisation": "free"}}


def build() -> dict:
    return {"NC-TB": tb(), "NC-FIBRE": fibre(), "NC-FD": fd(), "NC-TND": timing("nd"), "NC-TFD": timing("fd")}


def build_extra() -> dict:
    """Validation-only workflows (study O2); not part of the NOVA-CELL benchmark."""
    return {"NC-ND": nd(), "NC-FDSIM": fdsim()}


def write(out: Path, tasks: dict | None = None) -> dict:
    tasks = tasks or build()
    for w, t in tasks.items():
        d = out / w
        (d / "stock").mkdir(parents=True, exist_ok=True)
        (d / "stock" / "hists.json").write_text(json.dumps(t["stock"], indent=1))
        if t["floor"] is not None:
            (d / "floor").mkdir(exist_ok=True)
            (d / "floor" / "hists.json").write_text(json.dumps(t["floor"], indent=1))
        cfg = {"stock_rate_per_slot": None, "split": SPLIT.get(w, "validation only"), "yield_tolerance": 0.05, **t["cfg"],
               "source": "scint-fibre-fastsim validation/nova_cell (public data; see its README)"}
        (d / "reference.json").write_text(json.dumps(cfg, indent=1))
    return tasks


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("out", type=Path)
    a = ap.parse_args(argv)
    for w, t in write(a.out).items():
        print(w, sorted(t["stock"]["hists"]), sorted(t["stock"].get("yields", {})))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
