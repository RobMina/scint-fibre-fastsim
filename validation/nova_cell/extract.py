"""Extract the NOvA cell reference points from the public sources (sources.json) into data/*.json.

    python validation/nova_cell/extract.py SRC_DIR [--out validation/nova_cell/data] [--overlay DIR]

SRC_DIR holds the files fetch.sh downloaded. Needs pdfminer.six, pypdf, Pillow and SciPy (``pip install -e
'.[extract]'``). The committed data/*.json are this script's output; rerun it to reproduce them.

Two methods:
- vector: the figure is drawn as PDF paths (ROOT output), so every point, error bar and fit curve is read exactly
  from the path coordinates (pdfminer.six); the axis is calibrated on the drawn tick marks;
- raster: the figure is an embedded image (pypdf + Pillow). The plot frame and the major tick marks are found in
  the pixels and mapped to the printed tick labels; markers are the centroids of dark blobs (a fit line drawn over
  them is opened away first); a curve is the mean row of its colour per pixel column. The quoted digitisation
  error is half a pixel in each axis.

``--overlay DIR`` writes each raster figure with the detected points drawn on it, for checking by eye.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


# --------------------------------------------------------------------------------------------- vector (pdfminer)

def _walk(o):
    from pdfminer.layout import LTFigure
    for c in o:
        if isinstance(c, LTFigure):
            yield from _walk(c)
        else:
            yield c


def _color(o):
    c = o.stroking_color
    return tuple(round(float(v), 3) for v in c) if isinstance(c, (tuple, list)) else c


def testbeam_fig22(pdf: Path) -> dict:
    """Test Beam paper Fig. 22 (page 31): mean cosmic-muon response (PE/cm) against distance from the cell centre,
    plane 17 cell 36, readout at +x; with the paper's exponential fit (Eq. 6.2) and full (LOWESS) fit."""
    from pdfminer.high_level import extract_pages
    from pdfminer.layout import LTCurve, LTLine, LTRect

    page = next(extract_pages(str(pdf), page_numbers=[30]))
    objs = list(_walk(page))
    frame = [o for o in objs if isinstance(o, LTRect) and _color(o) == (0.0, 0.0, 0.0) and o.width > 200]
    x0, y0, x1, y1 = frame[0].bbox
    # y ticks: thin black lines from the left frame edge; the long ones are the labelled majors 0, 5, ..., 25
    ticks = sorted({round(o.pts[0][1], 4) for o in objs if isinstance(o, LTLine) and _color(o) == (0.0, 0.0, 0.0)
                    and o.pts[0][1] == o.pts[1][1] and min(o.pts[0][0], o.pts[1][0]) == x0
                    and 5 < abs(o.pts[0][0] - o.pts[1][0]) < 10})
    labels = 5.0 * np.arange(len(ticks))
    ky, cy = np.polyfit(ticks, labels, 1)
    assert np.max(np.abs(np.polyval([ky, cy], ticks) - labels)) < 1e-3, "y ticks not equally spaced"
    fy = lambda y: ky * y + cy
    fx = lambda x: -150.0 + 300.0 * (x - x0) / (x1 - x0)       # ROOT draws the frame at the axis range
    # points: thick black vertical segments (error bars) grouped by x
    bars: dict[float, list] = {}
    for o in objs:
        if isinstance(o, LTLine) and _color(o) == (0.0, 0.0, 0.0) and round(o.linewidth, 2) == 0.79:
            (xa, ya), (xb, yb) = o.pts
            if xa == xb and ya != yb and x0 < xa < x1:
                bars.setdefault(round(xa, 3), []).extend([ya, yb])
    pts = []
    for xp, ys in sorted(bars.items()):
        x = fx(xp)
        if abs((x + 148.5) / 3.0 - round((x + 148.5) / 3.0)) > 0.02:   # not on the 3 cm bin grid: legend sample
            continue
        lo, hi = fy(min(ys)), fy(max(ys))
        pts.append((round(x, 3), (lo + hi) / 2, (hi - lo) / 2))
    curves = {}
    for name, col in (("fit_exponential", (1.0, 0.0, 0.0)), ("fit_full", (0.0, 0.0, 1.0))):
        xy = sorted({(round(fx(px), 3), fy(py)) for o in objs if isinstance(o, LTCurve) and not isinstance(o, LTLine)
                     and _color(o) == col and len(o.pts) > 10 for px, py in o.pts})
        curves[name] = {"x_cm": [p[0] for p in xy], "pe_per_cm": [round(p[1], 4) for p in xy]}
    x, y, e = map(list, zip(*pts))
    return {
        "reference": "R1", "source": "testbeam", "figure": "Fig. 22 (page 31)", "method": "vector",
        "what": "mean response of cosmic-ray muon hits (PE per cm of path) against distance from the cell centre; "
                "Test Beam detector plane 17 (horizontal), cell 36; cell length 260 cm; readout at +x",
        "cell_length_cm": 260.0, "readout_at": "+x",
        "x_cm": x, "pe_per_cm": [round(v, 4) for v in y], "error": [round(v, 4) for v in e], **curves,
        "calibration": f"y from the {len(ticks)} major tick marks (0-25, linear to 1e-3); x from the frame edges "
                       "(-150, 150 cm), bins 3 cm",
    }


# --------------------------------------------------------------------------------------------- raster

def _page_image(pdf: Path, page: int, index: int) -> np.ndarray:
    from pypdf import PdfReader
    im = PdfReader(str(pdf)).pages[page - 1].images[index].image.convert("RGB")
    return np.asarray(im).astype(int)


def _frame(dark: np.ndarray) -> tuple[int, int, int, int]:
    """Rows/columns of the plot frame: the outermost long dark lines."""
    H, W = dark.shape
    rows = np.where(dark.sum(1) > 0.6 * W)[0]
    cols = np.where(dark.sum(0) > 0.6 * H)[0]
    return int(cols.min()), int(cols.max()), int(rows.min()), int(rows.max())


def _ticks(dark, L, R, T, B, axis: str) -> list[float]:
    """Positions of the major tick marks (the long ones) inside the frame along the bottom or left axis."""
    if axis == "x":
        pos = np.arange(L + 3, R - 2)
        n = np.array([next(k for k in range(40) if not dark[B - 1 - k, c]) for c in pos])
    else:
        pos = np.arange(T + 3, B - 2)
        n = np.array([next(k for k in range(40) if not dark[r, L + 1 + k]) for r in pos])
    minor = np.median(n[n >= 3])
    major = pos[(n >= 1.6 * minor) & (n < 3.5 * minor)]
    groups, cur = [], [major[0]]
    for p in major[1:]:
        if p - cur[-1] <= 1:
            cur.append(p)
        else:
            groups.append(cur)
            cur = [p]
    groups.append(cur)
    return [float(np.mean(g)) for g in groups]


def _axis(pix, labels):
    k, c = np.polyfit(pix, labels, 1)
    resid = np.max(np.abs(np.polyval([k, c], pix) - np.asarray(labels)))
    assert resid < 0.6 * abs(k), f"ticks not equally spaced ({resid})"
    return k, c


def _markers(im, L, R, T, B, open_px: int):
    from scipy import ndimage
    dark = im.sum(-1) < 300
    red = (im[..., 0] > 140) & (im[..., 1] < 120) & (im[..., 2] < 120)
    m = np.zeros_like(dark)
    sl = (slice(T + 3, B - 2), slice(L + 3, R - 2))
    m[sl] = (dark | red)[sl]
    m = ndimage.binary_opening(m, np.ones((open_px, open_px)))
    lab, n = ndimage.label(m)
    idx = range(1, n + 1)
    cy, cx = np.array(ndimage.center_of_mass(m, lab, idx)).reshape(-1, 2).T
    blk = np.array(ndimage.sum(dark, lab, idx))
    keep = blk >= 0.25 * open_px ** 2
    return cx[keep], cy[keep], m


def _red_curve(im, L, R, T, B, marker_mask):
    red = (im[..., 0] > 140) & (im[..., 1] < 120) & (im[..., 2] < 120)
    xs, ys = [], []
    for c in range(L + 3, R - 2):
        r = np.where(red[T + 3:B - 2, c] & ~marker_mask[T + 3:B - 2, c])[0]
        if len(r):
            xs.append(c)
            ys.append(T + 3 + r.mean())
    return np.array(xs, float), np.array(ys)


def root_raster(pdf, page, index, x_major, y_major, open_px, overlay=None, dark_level=600, grid=None):
    """A ROOT plot embedded as an image: markers and the red fit curve in axis units. With ``grid = (start,
    step)`` (the histogram's bin centres in axis units) each marker is the centroid of the marker pixels within half
    a step of a bin centre, which separates markers that touch."""
    im = _page_image(pdf, page, index)
    dark = im.sum(-1) < dark_level
    L, R, T, B = _frame(im.sum(-1) < 300)
    tx, ty = _ticks(dark, L, R, T, B, "x"), _ticks(dark, L, R, T, B, "y")
    assert len(tx) == len(x_major) and len(ty) == len(y_major), (tx, ty)
    kx, cx0 = _axis(tx, x_major)
    ky, cy0 = _axis(ty, sorted(y_major, reverse=True))          # rows grow downward
    mx, my, mask = _markers(im, L, R, T, B, open_px)
    if grid is not None:
        cols = np.arange(im.shape[1])
        xa = kx * cols + cx0
        gx, gy = [], []
        for g in np.arange(grid[0], kx * (R - 3) + cx0, grid[1]):
            sel = np.abs(xa - g) < 0.5 * grid[1]
            rr, cc = np.nonzero(mask[:, sel])
            if len(rr) >= 0.5 * open_px ** 2:
                gx.append(cols[sel][cc].mean())
                gy.append(rr.mean())
        mx, my = np.array(gx), np.array(gy)
    o = np.argsort(mx)
    fx, fyc = _red_curve(im, L, R, T, B, mask)
    if overlay is not None:
        from PIL import Image, ImageDraw
        img = Image.fromarray(im.astype(np.uint8))
        d = ImageDraw.Draw(img)
        for a, b in zip(mx, my):
            d.ellipse([a - 4, b - 4, a + 4, b + 4], outline=(0, 200, 0))
        for a, b in zip(fx[::4], fyc[::4]):
            d.point((a, b), fill=(0, 160, 255))
        img.resize((img.width * 2, img.height * 2)).save(overlay)
    return {
        "x": (kx * mx[o] + cx0).round(3).tolist(), "y": (ky * my[o] + cy0).round(4).tolist(),
        "digitisation_error": {"x": round(abs(kx) / 2, 4), "y": round(abs(ky) / 2, 4)},
        "fit_curve": {"x": (kx * fx + cx0).round(3).tolist(), "y": (ky * fyc + cy0).round(4).tolist()},
        "calibration": f"frame cols {L}-{R}, rows {T}-{B}; major ticks x {np.round(tx, 1).tolist()} -> {x_major}, "
                       f"y {np.round(ty, 1).tolist()} -> {sorted(y_major, reverse=True)}",
        "image": f"page {page}, image {index}, {im.shape[1]} x {im.shape[0]} px",
    }


def niner_fd_attenuation(pdf, overlay=None, bin_cm=50.0) -> dict:
    """The markers are about 8 cm (2.5 px) apart and overlap, so single points cannot be separated. The data are
    read as a band instead: per pixel column the median row of the dark (marker) pixels, red fit pixels excluded;
    then averaged in ``bin_cm`` bins, with the spread of the column values in each bin."""
    im = _page_image(pdf, 89, 0)
    dark = im.sum(-1) < 600
    L, R, T, B = _frame(im.sum(-1) < 300)
    tx, ty = _ticks(dark, L, R, T, B, "x"), _ticks(dark, L, R, T, B, "y")
    x_major, y_major = [-500, 0, 500], [25, 20, 15, 10, 5]
    assert len(tx) == 3 and len(ty) == 5, (tx, ty)
    kx, cx0 = _axis(tx, x_major)
    ky, cy0 = _axis(ty, y_major)
    marker = im.sum(-1) < 300
    red = (im[..., 0] > 140) & (im[..., 1] < 120) & (im[..., 2] < 120)
    cols, rows = [], []
    for c in range(L + 12, R - 2):                         # skip the left tick marks
        r = np.where(marker[T + 3:B - 12, c] & ~red[T + 3:B - 12, c])[0]
        if len(r):
            cols.append(c)
            rows.append(T + 3 + np.median(r))
    xc = kx * np.array(cols, float) + cx0
    yc = ky * np.array(rows) + cy0
    edges = np.arange(-800.0, 800.0 + 1e-9, bin_cm)
    xs, ys, sd, n = [], [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (xc >= lo) & (xc < hi)
        if s.sum() >= 5:
            xs.append((lo + hi) / 2)
            ys.append(float(np.mean(yc[s])))
            sd.append(float(np.std(yc[s], ddof=1)))
            n.append(int(s.sum()))
    fx, fyc = _red_curve(im, L, R, T, B, np.zeros_like(red))
    fit_x, fit_y = kx * fx + cx0, ky * fyc + cy0
    if overlay is not None:
        from PIL import Image, ImageDraw
        img = Image.fromarray(im.astype(np.uint8))
        d = ImageDraw.Draw(img)
        for a, b in zip(cols, rows):
            d.point((a, b), fill=(0, 200, 0))
        for a, b in zip(xs, ys):
            px, py = (a - cx0) / kx, (b - cy0) / ky
            d.rectangle([px - 3, py - 3, px + 3, py + 3], outline=(0, 90, 255))
        img.resize((img.width * 2, img.height * 2)).save(overlay)
    return {
        "reference": "R2", "source": "niner", "figure": "Fig. 3.16 (page 89 of the PDF, thesis page 55)",
        "method": "raster (band)",
        "what": "mean response (PE/cm) of cosmic-ray muon hits against distance from the cell centre; Far Detector "
                "plane 84 (horizontal), cell 12; readout end at +750 cm, far end at -750 cm (caption); labelled "
                "NOvA Preliminary; red: the thesis' double-exponential fit",
        "cell_length_cm": 1500.0, "readout_at": "+x", "bin_cm": bin_cm,
        "x_cm": xs, "pe_per_cm": [round(v, 4) for v in ys], "column_sd": [round(v, 4) for v in sd],
        "columns": n, "digitisation_error": {"x": round(abs(kx) / 2, 4), "y": round(abs(ky) / 2, 4)},
        "fit_double_exponential": {"x_cm": fit_x.round(3).tolist(), "pe_per_cm": fit_y.round(4).tolist()},
        "calibration": f"frame cols {L}-{R}, rows {T}-{B}; major ticks x {np.round(tx, 1).tolist()} -> {x_major}, "
                       f"y {np.round(ty, 1).tolist()} -> {y_major}; per column the median row of the marker "
                       f"pixels, averaged in {bin_cm:g} cm bins",
        "image": f"page 89, image 0, {im.shape[1]} x {im.shape[0]} px",
    }


def niner_timing(pdf, which: str, overlay=None) -> dict:
    index, y_major, det, tick = {"fd": (1, [20, 40, 60, 80, 100], "Far", 500.0),
                                 "nd": (3, [5, 10, 15, 20, 25], "Near", 125.0)}[which]
    fig = {"fd": "Fig. 4.8(b)", "nd": "Fig. 4.9(b)"}[which]
    r = root_raster(pdf, 104, index, [500, 1000, 1500], y_major, open_px=4, overlay=overlay, grid=(12.5, 25.0))
    return {
        "reference": "R5" + which, "source": "niner", "figure": f"{fig} (page 104 of the PDF, thesis page 70)",
        "method": "raster",
        "what": f"multi-point hit timing resolution (ns) against hit photoelectrons, {det} Detector data; red: the "
                "thesis' fit sigma_t = p0 / (p1 + npe^p2) + p3",
        "readout_tick_ns": tick,
        "pe": r["x"], "sigma_ns": r["y"], "digitisation_error": r["digitisation_error"],
        "fit": {"pe": r["fit_curve"]["x"], "sigma_ns": r["fit_curve"]["y"]},
        "calibration": r["calibration"], "image": r["image"],
    }


def aurisano_fibre(pdf, overlay=None) -> dict:
    """Aurisano et al. Fig. 8: fibre attenuation curve (average over fibre QC spools), ln I against x (m). The chart
    has grey axes and gridless ticks; the ln I series is purple with markers every metre."""
    im = _page_image(pdf, 6, 0)
    grey = ((np.abs(im[..., 0] - im[..., 1]) < 12) & (np.abs(im[..., 1] - im[..., 2]) < 12)
            & (im[..., 0] > 100) & (im[..., 0] < 175))
    ax = int(np.argmax(grey.sum(0)))                                     # vertical axis (x = 0)
    yt = [r for r in range(im.shape[0]) if grey[r, ax - 5:ax - 1].sum() >= 3]
    yt = [float(np.mean(g)) for g in np.split(np.array(yt), np.where(np.diff(yt) > 1)[0] + 1)]
    y_labels = [2, 1, 0, -1, -2, -3, -4]
    assert len(yt) == len(y_labels), yt
    ky, cy0 = _axis(yt, y_labels)
    row0 = int(round(np.polyval([1 / ky, -cy0 / ky], 0.0)))              # the x axis is drawn at y = 0
    xt = [c for c in range(ax, im.shape[1]) if grey[row0 + 1:row0 + 5, c].sum() >= 3]
    xt = [float(np.mean(g)) for g in np.split(np.array(xt), np.where(np.diff(xt) > 1)[0] + 1)]
    x_labels = [0, 5, 10, 15, 20, 25, 30]
    assert len(xt) == len(x_labels), xt
    kx, cx0 = _axis(xt, x_labels)
    pur = ((im[..., 0] > 120) & (im[..., 0] < 175) & (im[..., 1] > 90) & (im[..., 1] < 135)
           & (im[..., 2] > 150) & (im[..., 2] < 205))
    pur[:row0 + 2] = False                                               # the ln I series lies below the x axis
    cols = np.where(pur.any(0))[0]
    cx = np.array([c for c in cols], float)
    cyv = np.array([np.where(pur[:, c])[0].mean() for c in cols])
    xm = kx * cx + cx0
    lni = ky * cyv + cy0
    xs = np.arange(1.0, np.floor(xm.max()) + 1e-9)
    ys = np.interp(xs, xm, lni)
    if overlay is not None:
        from PIL import Image, ImageDraw
        img = Image.fromarray(im.astype(np.uint8))
        d = ImageDraw.Draw(img)
        for a, b in zip(xs, ys):
            px, py = (a - cx0) / kx, (b - cy0) / ky
            d.ellipse([px - 4, py - 4, px + 4, py + 4], outline=(0, 200, 0))
        img.resize((img.width * 2, img.height * 2)).save(overlay)
    return {
        "reference": "R4", "source": "simchain", "figure": "Fig. 8 (page 6)", "method": "raster",
        "what": "fibre attenuation curve used by the NOvA simulation, the average over fibre quality-control spool "
                "scans; ln I (I in volts of the scanner, 1.0 at the first point) against distance along the fibre",
        "x_m": xs.round(3).tolist(), "ln_i": ys.round(4).tolist(),
        "digitisation_error": {"x": round(abs(kx) / 2, 4), "y": round(abs(ky) / 2, 4)},
        "calibration": f"vertical axis col {ax}; y ticks rows {np.round(yt, 1).tolist()} -> {y_labels}; x ticks cols "
                       f"{np.round(xt, 1).tolist()} -> {x_labels}; curve = mean row of the purple series per column, "
                       "sampled at whole metres",
        "image": f"page 6, image 0, {im.shape[1]} x {im.shape[0]} px",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", type=Path)
    ap.add_argument("--out", type=Path, default=HERE / "data")
    ap.add_argument("--overlay", type=Path)
    a = ap.parse_args(argv)
    src = json.loads((HERE / "sources.json").read_text())["sources"]
    path = {}
    for key, s in src.items():
        f = a.src / s["file"]
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        if h != s["sha256"]:
            raise SystemExit(f"{f}: sha256 {h} != {s['sha256']} (run fetch.sh)")
        path[key] = f
    ov = (lambda n: a.overlay / n) if a.overlay else (lambda n: None)
    if a.overlay:
        a.overlay.mkdir(parents=True, exist_ok=True)
    a.out.mkdir(parents=True, exist_ok=True)
    out = {
        "r1_testbeam_cell": testbeam_fig22(path["testbeam"]),
        "r2_fd_cell": niner_fd_attenuation(path["niner"], ov("r2.png")),
        "r4_fibre": aurisano_fibre(path["simchain"], ov("r4.png")),
        "r5_timing_nd": niner_timing(path["niner"], "nd", ov("r5nd.png")),
        "r5_timing_fd": niner_timing(path["niner"], "fd", ov("r5fd.png")),
    }
    for name, d in out.items():
        d["cite"] = src[d["source"]]["cite"]
        d["source_sha256"] = src[d["source"]]["sha256"]
        (a.out / f"{name}.json").write_text(json.dumps(d, indent=1) + "\n")
        n = len(d.get("x_cm") or d.get("x_m") or d.get("pe"))
        print(f"{name:18s} {d['figure']:42s} {d['method']:6s} {n:4d} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
