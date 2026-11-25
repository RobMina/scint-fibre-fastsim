"""Geometry of one scintillator strip with wavelength-shifting fibers (dimensions in mm).

The strip is an extrusion along z (``length``, origin at its centre) with a reflective coating of thickness
``coating`` on its outer faces. Its cross-section (``profile``):

- ``"rect"`` (default): a rectangle, ``thickness`` along x and ``width`` along y, with rounded outer corners;
- ``"triangle"``: an isosceles triangle with its base (``width``) along y and its height (``thickness``) along x,
  base at x = -thickness/3 and apex at x = +2 thickness/3, so that its centroid is at the origin;
- ``"box"``: the rectangle with sharp corners, traced by the polygon code of the triangle (tests: with
  corner_radius = coating it is the ``"rect"`` strip).

Polygon profiles have sharp corners (``corner_radius`` is not used) and a coating of the same thickness on every
face. Fibers: ``n_fibers`` = 2 puts them at y = +-fiber_separation/2 (x = 0); 1 puts one fiber at the origin (the
centroid of the triangle), with fiber_separation 0. Each fiber is double-clad (each cladding ``clad_fraction`` of the
diameter) in an elliptical hole (semi-axes ``hole_rx``, ``hole_ry``), runs the full length plus a fiber guide bar of
length ``guide_bar`` at each end, and has at each end a photodetector (or, at +z, a reflector or black tape) behind an
air gap. Every dimension is a parameter; an experiment adapter supplies its own counters.
"""
import math
from dataclasses import dataclass, replace

PROFILES = ("rect", "box", "triangle")


@dataclass(frozen=True)
class Counter:
    """One strip. ``reflector``: 0 photodetectors at both ends, 1 reflector at +z, 2 black tape at +z."""
    length: float
    thickness: float
    width: float
    fiber_separation: float
    fiber_diameter: float
    clad_fraction: float                 # each cladding layer, as a fraction of the fiber diameter
    hole_rx: float                       # fiber hole semi-axis along x (thickness)
    hole_ry: float                       # fiber hole semi-axis along y (width)
    coating: float
    corner_radius: float                 # outer (extrusion) corner radius of the "rect" profile
    guide_bar: float
    # end readout of the "gap" optical model; the "contact" model puts a photodetector / reflector disc of the fiber
    # radius directly on the fiber end
    sipm_width: float
    sipm_window: float                   # G4Box half-length of the epoxy window
    sipm_length: float
    air_gap: float
    reflector: int = 0
    profile: str = "rect"                # cross-section: PROFILES (module docstring)
    n_fibers: int = 2                    # 2: fibers at y = +-fiber_separation/2; 1: one fiber at the origin

    def __post_init__(self):
        if self.reflector not in (0, 1, 2):
            raise ValueError("reflector must be 0 (none), 1 (reflector at +z) or 2 (black tape at +z)")
        if self.profile not in PROFILES:
            raise ValueError(f"profile must be one of {PROFILES}")
        if self.n_fibers not in (1, 2):
            raise ValueError("n_fibers must be 1 or 2")
        if self.n_fibers == 1 and self.fiber_separation != 0:
            raise ValueError("a single fiber sits at the origin: fiber_separation must be 0")
        if self.profile == "triangle" and self.n_fibers != 1:
            raise ValueError("the triangle has one fiber, at its centroid")
        if self.r_clad2 >= self.hole_ry or self.r_clad2 >= self.hole_rx:
            raise ValueError("the fiber must fit inside its hole")
        if self.profile == "rect":
            if self.fiber_separation / 2 + self.hole_ry >= self.hy or self.hole_rx >= self.hx:
                raise ValueError("the fiber holes must lie inside the scintillator")
        else:
            s2 = self.fiber_separation / 2
            for nx, ny, h in self.faces:
                for yc in ((-s2, s2) if self.n_fibers == 2 else (0.0,)):
                    if ny * yc + math.hypot(nx * self.hole_rx, ny * self.hole_ry) >= h:
                        raise ValueError("the fiber holes must lie inside the scintillator")

    # derived dimensions (the expressions of the Geant4 geometry, same floating-point order)
    @property
    def hx(self):
        return self.thickness / 2.0 - self.coating

    @property
    def hy(self):
        return self.width / 2.0 - self.coating

    @property
    def half_length(self):
        return self.length / 2.0

    @property
    def rc(self):
        """Corner radius of the scintillator ("rect"); 0 for the sharp polygon profiles."""
        return self.corner_radius - self.coating if self.profile == "rect" else 0.0

    @property
    def faces(self):
        """Polygon profiles: the scintillator faces as (nx, ny, h), outward unit normal and distance from the origin
        (inside: nx x + ny y <= h for every face); None for "rect"."""
        if self.profile == "box":
            return ((1.0, 0.0, self.hx), (-1.0, 0.0, self.hx), (0.0, 1.0, self.hy), (0.0, -1.0, self.hy))
        if self.profile == "triangle":
            t, w = self.thickness, self.width
            n = math.hypot(w / 2.0, t)
            h = w * t / 3.0 / n - self.coating
            return ((-1.0, 0.0, t / 3.0 - self.coating), (w / 2.0 / n, t / n, h), (w / 2.0 / n, -t / n, h))
        return None

    def vertices(self):
        """Polygon profiles: the scintillator corners (x, y), counter-clockwise; None for "rect"."""
        f = self.faces
        if f is None:
            return None
        pts = []
        for i in range(len(f)):
            for j in range(i + 1, len(f)):
                (a1, b1, h1), (a2, b2, h2) = f[i], f[j]
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-12:
                    continue
                x, y = (h1 * b2 - h2 * b1) / det, (a1 * h2 - a2 * h1) / det
                if all(a * x + b * y <= h + 1e-9 for a, b, h in f):
                    pts.append((x, y))
        cx, cy = sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts)
        return sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))

    @property
    def r_clad2(self):
        return self.fiber_diameter / 2.0

    @property
    def r_clad1(self):
        return self.fiber_diameter / 2.0 - self.clad_fraction * self.fiber_diameter

    @property
    def r_core(self):
        c = self.clad_fraction * self.fiber_diameter
        return self.fiber_diameter / 2.0 - c - c


def with_length(c, length, reflector):
    return replace(c, length=float(length), reflector=int(reflector))
