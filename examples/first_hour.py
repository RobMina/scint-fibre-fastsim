"""Tutorial script: toy strip, geometry families, backend stubs, G4 plan.

Run from the repo root after ``pip install -e '.[test]'``:

    python examples/first_hour.py
"""

from __future__ import annotations

from scintfibrefastsim.backends import Mu2eSiPMBackend, NovaAPDBackend
from scintfibrefastsim.geometry.families import (
    cell_tdr_caricature,
    strip_family,
    toy_strip,
)
from scintfibrefastsim.validation import compare_plan_crvstandalone


def main() -> None:
    c = toy_strip(length=300.0)
    print(f"toy strip: L={c.length} mm, fiber={c.fiber_diameter} mm, r_core={c.r_core:.3f} mm")
    print(f"family: {strip_family().name} ({strip_family().kind})")
    print(f"cell:   {cell_tdr_caricature().description}")
    print(f"backends: {Mu2eSiPMBackend().sensor_kind}, {NovaAPDBackend().sensor_kind}")
    plan = compare_plan_crvstandalone()
    print(f"G4 plan: {plan.name} vs {plan.oracle}")


if __name__ == "__main__":
    main()
