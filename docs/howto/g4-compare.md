# Geant4 / CRVStandalone compare hooks

```python
from scintfibrefastsim.validation import (
    compare_plan_crvstandalone, compare_plan_toy, run_compare, run_toy_selfcheck,
)

# CI / laptop: JAX core self-check (no Geant4)
receipt = run_toy_selfcheck(dtype="float64", seed=1)
print(receipt.status, receipt.metrics)

# Declared CRVStandalone plan: needs $SCINTFIBRE_CRVSTANDALONE (wls or wrapper)
plan = compare_plan_crvstandalone()
receipt = run_compare(plan, mode="auto")  # oracle_missing if unset
```

Oracle wrapper contract: argv = ``geometry nphot seed dtype``; last stdout line is JSON
with at least ``gen`` and ``hits``.

Rules:

- Public toy geometries only in this repository.
- Oracle for the strip family: [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone).
- No Mu2e-internal released tables or conditions.
