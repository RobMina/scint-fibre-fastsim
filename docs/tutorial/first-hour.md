# First hour

Build a public toy strip and inspect the shared core + backend contracts.
Run the same steps as `examples/first_hour.py`:

```bash
python examples/first_hour.py
```

```python
from scintfibrefastsim.geometry.families import (
    cell_tdr_caricature,
    strip_family,
    toy_strip,
)
from scintfibrefastsim.backends import Mu2eSiPMBackend, NovaAPDBackend
from scintfibrefastsim.validation import compare_plan_crvstandalone

c = toy_strip(length=300.0)
print(f"toy strip: L={c.length} mm, fiber={c.fiber_diameter} mm, r_core={c.r_core:.3f} mm")
print(f"family: {strip_family().name} ({strip_family().kind})")
print(f"cell:   {cell_tdr_caricature().description}")
print(f"backends: {Mu2eSiPMBackend().sensor_kind}, {NovaAPDBackend().sensor_kind}")
plan = compare_plan_crvstandalone()
print(f"G4 plan: {plan.name} vs {plan.oracle}")
```

Next: [install](../howto/install.md).
