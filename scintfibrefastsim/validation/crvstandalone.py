"""The public Geant4 oracle of the optical core: CRVStandalone's own counter, its table pieces and the bin-by-bin check.

CRVStandalone (github.com/Mu2e/CRVStandalone, Apache-2.0) builds one counter at compile time; its default build
(wls/src/WLSDetectorConstruction.cc, not TESTBEAM, not LDMX) is the counter here: a 19.78 x 51.34 mm extrusion with a
0.25 mm TiO2 coating and 2.0 mm corner radius, two elliptical holes (semi-axes 2.0 mm along the thickness, 1.0 mm along
the width) 26 mm apart, 1.4 mm fibres with two claddings of 3 % of the diameter each, 10 mm guide bars, 2 x 2 mm SiPMs
1 um from the fibre ends. Its table mode (`wls -c -t SIMTYPE -l LENGTH -R REFLECTOR -m BIN -M BIN -n PHOTONS -r SEED`)
writes one piece per run: the table header (only for sim type 0 starting at bin 0), then one record per bin
(probability, uint8 time-delay and fibre-emission distributions, as WLSEventAction writes them).

``counter`` and ``binning`` give the same counter and bins to ``scintfibrefastsim.core.run.generate``; ``read_piece``
and ``header_check`` read a piece; ``encode`` turns generator counts into the same uint8 distributions; ``compare``
gives the per-bin probability, mean-time and mean-emission pulls against the pieces.
"""
from __future__ import annotations

import math
import struct

import numpy as np

from scintfibrefastsim.core.binning import Binning
from scintfibrefastsim.core.counter import Counter

THICKNESS, WIDTH, SEPARATION, FIBER_DIAMETER = 19.78, 51.34, 26.0, 1.4
REST = dict(clad_fraction=0.03, hole_rx=2.0, hole_ry=1.0, coating=0.25, corner_radius=2.0, guide_bar=10.0,
            sipm_width=2.0, sipm_window=0.1, sipm_length=0.2, air_gap=0.001)
# bin edges of the default build [mm]: x >= 0 (mirror symmetric), y over the full width
X_EDGES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 6.0, 9.0, 9.5, 9.89)
Y_EDGES = (-25.67, -25.2, -24.7, -23.5, -20.5, -17.5, -15.5, -15.0, -14.5, -14.0, -13.5, -13.0, -12.5, -12.0, -11.5,
           -11.0, -10.5, -8.5, -6.5, -4.5, -1.5, 1.5, 4.5, 6.5, 8.5, 10.5, 11.0, 11.5, 12.0, 12.5, 13.0, 13.5, 14.0,
           14.5, 15.0, 15.5, 17.5, 20.5, 23.5, 24.7, 25.2, 25.67)
INV_BETA = (2.20, 1.80, 1.64, 1.58, 1.24, 1.0)
ORDER = {0: ("x", "y", "z"), 1: ("x", "y", "z", "beta"), 2: ("beta", "theta", "phi", "r", "z")}
SIM_TYPES = {0: "scintillation", 1: "Cerenkov in the scintillator", 2: "Cerenkov in the fibre"}


def counter(length: float = 3200.0, reflector: int = 0) -> Counter:
    return Counter(length=float(length), reflector=int(reflector), thickness=THICKNESS, width=WIDTH,
                   fiber_separation=SEPARATION, fiber_diameter=FIBER_DIAMETER, **REST)


def z_edges(c: Counter) -> np.ndarray:
    """10 mm and 25 mm bins over 300 mm at each end, about 100 mm bins in between (counters longer than 600 mm)."""
    if c.length <= 600.0:
        raise ValueError("the z binning needs a counter longer than 600 mm")
    hl = c.length / 2.0
    n = int(np.rint((c.length - 600.0) / 100.0))
    bw = (c.length - 600.0) / n if n > 0 else 0.0
    z = [-hl + 10.0 * i for i in range(6)] + [-hl + 50.0 + 25.0 * i for i in range(1, 11)]
    z += [-hl + 300.0 + bw * i for i in range(1, n + 1)]
    z += [hl - 300.0 + 25.0 * i for i in range(1, 11)] + [hl - 50.0 + 10.0 * i for i in range(1, 6)]
    return np.array(z, np.float64)


def edges(c: Counter) -> dict:
    return {"x": np.array(X_EDGES), "y": np.array(Y_EDGES), "z": z_edges(c),
            "beta": np.array([1.0 / b for b in INV_BETA]),
            "theta": np.array([0.0] + [math.pi / 20.0 + math.pi * i / 10.0 for i in range(10)] + [math.pi]),
            "phi": np.array([0.0] + [math.pi / 8.0 + math.pi * i / 4.0 for i in range(4)] + [math.pi]),
            "r": np.array([c.r_core * i / 5.0 for i in range(6)])}


def binning(c: Counter) -> Binning:
    return Binning(edges(c), ORDER)


def read_piece(path, with_header: bool):
    """(header bytes, [(bin, prob, delays uint8, emissions uint8), ...]) of one piece."""
    b = open(path, "rb").read()
    o = 0
    if with_header:
        o = 96
        for _ in range(2):
            n = struct.unpack_from("<Q", b, o)[0]
            o += 8 + 16 * n
        for _ in range(7):
            n = struct.unpack_from("<Q", b, o)[0]
            o += 8 + 8 * n
    head, recs = b[:o], []
    while o < len(b):
        i, p = struct.unpack_from("<If", b, o)
        o += 8
        nd = struct.unpack_from("<Q", b, o)[0]
        d = np.frombuffer(b, np.uint8, nd, o + 8)
        o += 8 + nd
        ne = struct.unpack_from("<Q", b, o)[0]
        e = np.frombuffer(b, np.uint8, ne, o + 8)
        o += 8 + ne
        recs.append((int(i), float(p), d, e))
    return head, recs


def header_check(head: bytes, c: Counter) -> dict:
    """The piece's geometry constants and bin edges against this module's counter and edges."""
    const = struct.unpack_from("<10d", head, 16)
    ours = (c.hx, c.hy, c.half_length, c.fiber_separation, c.hole_rx, c.hole_ry, c.r_clad2, c.rc)
    o = 96
    for _ in range(2):
        n = struct.unpack_from("<Q", head, o)[0]
        o += 8 + 16 * n
    e, ok = edges(c), True
    for k in ("x", "y", "z", "beta", "theta", "phi", "r"):
        n = struct.unpack_from("<Q", head, o)[0]
        got = np.frombuffer(head, "<f8", n, o + 8)
        o += 8 + 8 * n
        ok &= got.shape == e[k].shape and bool(np.array_equal(got, e[k]))
    return {"reflector": struct.unpack_from("<3i", head, 0)[2] == c.reflector,
            "constants": bool(np.allclose(const[:8], ours, rtol=1e-12, atol=0)), "edges": bool(ok),
            "constants_piece": list(const[:8]), "constants_ours": [float(v) for v in ours]}


def _encode(counts, total) -> np.ndarray:
    """uint8(255 * count / total + 0.5) in float32, up to the last non-empty entry (WLSEventAction)."""
    counts = np.asarray(counts)
    nz = np.nonzero(counts)[0]
    m = nz[-1] + 1 if len(nz) else 0
    if total == 0 or m == 0:
        return np.zeros(0, np.uint8)
    p = counts[:m].astype(np.float32) / np.float32(total)
    return np.floor((np.float32(255) * p).astype(np.float64) + 0.5).astype(np.uint8)


def encode(out: dict, j: int):
    """(prob, delays, emissions) of bin j of generator output, as CRVStandalone writes them: only arrivals inside both
    histograms count, for the probability too."""
    n = int(np.sum(out["ht"][j]))
    prob = float(np.float32(n) / np.float32(out["gen"][j])) if n > 0 else 0.0
    return prob, _encode(out["ht"][j], n), _encode(out["he"][j], n)


def _moments(w):
    w = np.asarray(w, np.float64)
    s = w.sum(axis=-1)
    i = np.arange(w.shape[-1])
    with np.errstate(invalid="ignore", divide="ignore"):
        m = (w * i).sum(-1) / s
        v = (w * i * i).sum(-1) / s - m * m
    return m, np.maximum(v, 0)


def compare(ref: dict, out: dict, nphot_ref: int) -> dict:
    """Pulls per bin of generator output `out` (bins, gen, ht, he, nphot) against reference records
    ref = {bin: (prob, delays, emissions)}; the reference's generated count is taken as the generator's scaled to its
    photons per bin (both count the photons of the bin box inside the scintillator)."""
    keep = np.array([b in ref for b in out["bins"]])
    bins = np.asarray(out["bins"])[keep]
    gen = np.asarray(out["gen"], np.float64)[keep]
    sub = {k: np.asarray(out[k])[keep] for k in ("ht", "he", "gen")}
    enc = [encode(sub, j) for j in range(len(bins))]
    p_new = np.array([e[0] for e in enc])
    p_ref = np.array([ref[b][0] for b in bins])
    gen_ref = gen * float(nphot_ref) / float(out["nphot"])
    with np.errstate(invalid="ignore", divide="ignore"):
        var = p_new * (1 - p_new) / gen + p_ref * (1 - p_ref) / gen_ref
        pull = (p_new - p_ref) / np.sqrt(var)
    use = (gen > 0) & (var > 0) & ((p_new > 0) | (p_ref > 0))
    L = 160
    dn, en, dr, er = (np.zeros((len(bins), L)) for _ in range(4))
    for j, b in enumerate(bins):
        for arr, src in ((dn, enc[j][1]), (en, enc[j][2]), (dr, ref[b][1]), (er, ref[b][2])):
            arr[j, :min(len(src), L)] = src[:L]
    arr_new = sub["ht"].sum(axis=1).astype(np.float64)
    arr_ref = p_ref * gen_ref
    (mt_n, vt_n), (mt_r, vt_r), (me_n, ve_n), (me_r, ve_r) = (_moments(x) for x in (dn, dr, en, er))
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = (mt_n - mt_r) / np.sqrt(vt_n / arr_new + vt_r / arr_ref)
        pe = (me_n - me_r) / np.sqrt(ve_n / arr_new + ve_r / arr_ref)
    ut = use & (arr_new >= 20) & (arr_ref >= 20) & np.isfinite(pt)
    ue = ut & np.isfinite(pe) & ((ve_n > 0) | (ve_r > 0))

    def summ(x, m):
        x = x[m]
        if len(x) == 0:
            return {"n": 0}
        return {"n": int(len(x)), "mean": float(x.mean()), "mean_err": float(x.std() / np.sqrt(len(x))),
                "rms": float(x.std()), "frac_gt3": float((np.abs(x) > 3).mean())}

    w = arr_new[use]
    ratio = float((p_new[use] * gen[use]).sum() / (p_ref[use] * gen[use]).sum()) if use.any() else float("nan")
    return {"bins": int(len(bins)), "nphot": int(out["nphot"]), "nphot_ref": int(nphot_ref),
            "prob_pull": summ(pull, use), "prob_ratio_weighted": ratio,
            "prob_ratio_err": float(np.sqrt(2.0 / max(w.sum(), 1))),
            "time_mean_pull": summ(pt, ut), "emis_mean_pull": summ(pe, ue),
            "per_bin": {"bin": bins.tolist(), "prob_new": p_new.tolist(), "prob_ref": p_ref.tolist(),
                        "prob_pull": np.where(use, pull, np.nan).tolist()}}
