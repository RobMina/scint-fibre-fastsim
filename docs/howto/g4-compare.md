# Geant4 / CRVStandalone compare hooks

```python
from scintfibrefastsim.validation import compare_plan_crvstandalone, compare_plan_toy
from scintfibrefastsim.validation.g4_hooks import run_compare

plan = compare_plan_crvstandalone()
# run_compare(plan)  # Phase 1 — raises NotImplementedError in 0.1.0
```

Rules:

- Public toy geometries only in this repository.
- Oracle for the strip family: [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone).
- No Mu2e-internal released tables or conditions.
