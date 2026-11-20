# First hour

Build a public toy strip and inspect the shared core + backend contracts.
Run the same steps as `examples/first_hour.py`:

```bash
python examples/first_hour.py
```

```python
from scintfibrefastsim.geometry.families import toy_strip, strip_family, cell_tdr_caricature
from scintfibrefastsim.backends import Mu2eSiPMBackend, NovaAPDBackend
from scintfibrefastsim.validation import compare_plan_crvstandalone

c = toy_strip(length=300.0)
print(strip_family().name, c.length, c.fiber_diameter)
print(cell_tdr_caricature().description)
print(Mu2eSiPMBackend().sensor_kind, NovaAPDBackend().sensor_kind)
print(compare_plan_crvstandalone().oracle)
```

Next: [install](../howto/install.md), then the Phase-1 extract steps in the
repository README.
