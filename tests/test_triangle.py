"""Polygon cross-sections: the triangular strip (one fiber at the centroid) and the sharp-cornered box.

The box runs the polygon code on the rectangle: with corner_radius = coating it must give the "rect" strip's counts.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from scintfibrefastsim.core.counter import Counter
from scintfibrefastsim.geometry.families import toy_strip


def toy_triangle(length=300.0, thickness=20.0, width=40.0):
    r = 0.7
    return Counter(length=length, thickness=thickness, width=width, fiber_separation=0.0, fiber_diameter=2 * r,
                   clad_fraction=0.03, hole_rx=r + 1.3, hole_ry=r + 0.3, coating=0.25, corner_radius=1.0,
                   guide_bar=5.0, sipm_width=2.0, sipm_window=0.1, sipm_length=0.3, air_gap=0.001,
                   profile="triangle", n_fibers=1)


def _clip_area(poly, x0, x1, y0, y1):
    """Area of a convex polygon [(x, y)] inside the box [x0, x1] x [y0, y1] (Sutherland-Hodgman)."""
    def clip(pts, inside, cut):
        out = []
        for i, p in enumerate(pts):
            q = pts[i - 1]
            if inside(p):
                if not inside(q):
                    out.append(cut(q, p))
                out.append(p)
            elif inside(q):
                out.append(cut(q, p))
        return out

    def cx(v):
        return lambda a, b: (v, a[1] + (b[1] - a[1]) * (v - a[0]) / (b[0] - a[0]))

    def cy(v):
        return lambda a, b: (a[0] + (b[0] - a[0]) * (v - a[1]) / (b[1] - a[1]), v)
    pts = list(poly)
    for ins, cut in ((lambda p: p[0] >= x0, cx(x0)), (lambda p: p[0] <= x1, cx(x1)),
                     (lambda p: p[1] >= y0, cy(y0)), (lambda p: p[1] <= y1, cy(y1))):
        pts = clip(pts, ins, cut)
        if not pts:
            return 0.0
    return 0.5 * abs(sum(pts[i - 1][0] * pts[i][1] - pts[i][0] * pts[i - 1][1] for i in range(len(pts))))


def test_triangle_faces_and_vertices():
    c = replace(toy_triangle(thickness=30.0, width=42.38), coating=0.0)
    v = c.vertices()
    assert sorted((round(x, 9), round(y, 9)) for x, y in v) == [(-10.0, -21.19), (-10.0, 21.19), (20.0, 0.0)]
    assert abs(sum(x for x, _ in v) / 3) < 1e-12 and abs(sum(y for _, y in v) / 3) < 1e-12   # centroid at the origin
    c2 = toy_triangle(thickness=30.0, width=42.38)
    for (a, b, h), (a0, b0, h0) in zip(c2.faces, c.faces):
        assert (a, b) == (a0, b0) and h == pytest.approx(h0 - 0.25)
    assert c2.rc == 0.0 and toy_strip().rc == pytest.approx(0.75)


def test_counter_validation():
    with pytest.raises(ValueError, match="one fiber"):
        replace(toy_triangle(), n_fibers=2, fiber_separation=10.0)
    with pytest.raises(ValueError, match="fiber_separation must be 0"):
        replace(toy_triangle(), fiber_separation=1.0)
    with pytest.raises(ValueError, match="profile"):
        replace(toy_triangle(), profile="hexagon")
    with pytest.raises(ValueError, match="inside the scintillator"):
        toy_triangle(thickness=4.0, width=8.0)


def _kernel(c, simtype, binning, dtype="float64"):
    from scintfibrefastsim.core import transport as T
    return T.Kernel(T.Config(counter=c, binning=binning, simtype=simtype, dtype=dtype, lanes=256))


def test_triangle_source_fills_the_triangle():
    """Scintillation starts: the fraction inside the scintillator per (x, y) bin equals the exact area of the
    triangle minus the fiber hole in that bin."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from scintfibrefastsim.core.binning import Binning
    c = toy_triangle()
    xs = [p[0] for p in c.vertices()]
    ys = [p[1] for p in c.vertices()]
    ex = np.array([min(xs) - 1.0, 0.0, max(xs) + 1.0])
    ey = np.array([min(ys) - 1.0, 0.0, max(ys) + 1.0])
    b = Binning({"x": ex, "y": ey, "z": np.array([-c.half_length, c.half_length])}, {0: ("x", "y", "z")})
    k = _kernel(c, 0, b)
    n = 50_000
    bins = np.arange(4)
    box = {kk: (np.asarray(v[0], np.float64), np.asarray(v[1], np.float64)) for kk, v in b.boxes(0, bins).items()}
    lb = jnp.asarray(np.repeat(bins, n))
    pid = jnp.asarray(np.arange(4 * n, dtype=np.int64))
    _, inside, _ = k.source(lb, box, pid, k.key_src)
    frac = np.asarray(inside).reshape(4, n).mean(1)
    hole = math.pi * c.hole_rx * c.hole_ry / 4                      # a quarter of the hole in each bin
    for i, (ix, iy) in enumerate(np.ndindex(2, 2)):
        area = (ex[ix + 1] - ex[ix]) * (ey[iy + 1] - ey[iy])
        want = (_clip_area(c.vertices(), ex[ix], ex[ix + 1], ey[iy], ey[iy + 1]) - hole) / area
        assert abs(frac[i] - want) < 4 * math.sqrt(want * (1 - want) / n) + 1e-9, (i, frac[i], want)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_polygon_exit_distance(dtype):
    """polygon_exit: distance and normal of the first face along random rays from inside the triangle."""
    pytest.importorskip("jax")
    import jax.numpy as jnp
    from scintfibrefastsim.core.binning import Binning
    c = toy_triangle()
    b = Binning({"x": [0.0, 1.0], "y": [0.0, 1.0], "z": [0.0, 1.0]}, {0: ("x", "y", "z")})
    k = _kernel(c, 0, b, dtype)
    rng = np.random.default_rng(3)
    v = np.array(c.vertices())
    w = rng.dirichlet(np.ones(3), 2000)
    p = w @ v
    ang = rng.uniform(0, 2 * np.pi, len(p))
    d = np.stack([np.cos(ang), np.sin(ang)], 1)
    f = jnp.float64 if dtype == "float64" else jnp.float32
    t, nrm = k.polygon_exit(*(jnp.asarray(a, f) for a in (p[:, 0], p[:, 1], d[:, 0], d[:, 1])), jnp.inf)
    t, nrm = np.asarray(t, np.float64), np.asarray(nrm, np.float64)
    q = p + t[:, None] * d
    faces = np.array(c.faces)
    resid = q @ faces[:, :2].T - faces[:, 2]                      # on one face, inside the others
    tol = 1e-9 if dtype == "float64" else 2e-4
    assert np.all(resid.max(1) < tol) and np.all(np.abs(resid).min(1) < tol)
    hit = np.abs(resid).argmin(1)
    assert np.allclose(nrm[:, :2], faces[hit, :2]) and np.all(nrm[:, 2] == 0)


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_box_profile_matches_rect(dtype):
    """The polygon transport on the rectangle (profile "box") gives the counts of the "rect" strip with sharp
    corners (corner_radius = coating): same photons, same random numbers."""
    pytest.importorskip("jax")
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.run import generate
    rect = replace(toy_strip(length=300.0), corner_radius=0.25)
    box = replace(rect, profile="box")
    edges = {"x": np.linspace(0.0, rect.hx, 3), "y": np.linspace(-rect.hy, rect.hy, 3),
             "z": np.linspace(-rect.half_length, rect.half_length, 3)}
    b = Binning(edges=edges, order={0: ["x", "y", "z"]})
    out = [generate(c, b, 0, np.arange(b.n_bins(0)), 200, dtype=dtype, seed=7, lanes=256, chunk_iters=500)
           for c in (rect, box)]
    for key in ("gen", "nall", "ht", "he"):
        assert np.array_equal(out[0][key], out[1][key]), key
    assert out[0]["stats"]["steps"] == out[1]["stats"]["steps"]


@pytest.mark.parametrize("dtype", ["float32", "float64"])
def test_triangle_generate_completes(dtype):
    """Scintillation and fiber-Cerenkov photons of the triangle: every bin completes; fiber-Cerenkov photons start in
    the (single) fiber and are scored on it."""
    pytest.importorskip("jax")
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.run import generate
    c = toy_triangle(length=200.0)
    xs = [p[0] for p in c.vertices()]
    edges = {"x": np.array([min(xs), 0.0, max(xs)]), "y": np.array([0.0, 2.0, 10.0]),
             "z": np.linspace(-c.half_length, c.half_length, 3), "beta": np.array([0.7, 1.0]),
             "theta": np.array([0.0, math.pi]), "phi": np.array([0.0, math.pi]),
             "r": np.array([0.0, c.r_core])}
    b = Binning(edges=edges, order={0: ("x", "y", "z"), 2: ("beta", "theta", "phi", "r", "z")})
    r0 = generate(c, b, 0, np.arange(b.n_bins(0)), 60, dtype=dtype, seed=2, lanes=256, chunk_iters=400)
    assert r0["complete"].all() and int(r0["gen"].sum()) > 0
    assert np.all(r0["ht"].sum(1) == r0["he"].sum(1))
    r2 = generate(c, b, 2, np.arange(b.n_bins(2)), 300, dtype=dtype, seed=2, lanes=256, chunk_iters=400)
    assert r2["complete"].all() and int(r2["nall"].sum()) > 0        # light trapped in the fiber reaches SiPM 0
