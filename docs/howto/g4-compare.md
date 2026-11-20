# Geant4 / CRVStandalone compare hooks

```python
from scintfibrefastsim.validation import (
    compare_plan_crvstandalone, compare_plan_toy, run_compare, run_toy_selfcheck,
)

# CI / laptop: JAX core self-check (no Geant4)
receipt = run_toy_selfcheck(dtype="float64", seed=1)
print(receipt.status, receipt.metrics)

# Declared CRVStandalone plan: needs a WRAPPER on $SCINTFIBRE_CRVSTANDALONE
plan = compare_plan_crvstandalone()
receipt = run_compare(plan, mode="auto")  # oracle_missing if unset
```

## Wrapper required (not raw `wls`)

`$SCINTFIBRE_CRVSTANDALONE` must point at a **wrapper script**, not the raw
CRVStandalone `wls` binary. The contract:

- argv: `geometry nphot seed dtype`
- last stdout line: JSON with at least `gen` and `hits`

`geometry=toy_strip` is this package's public toy name; the wrapper maps it to a
CRVStandalone counter card. See `examples/crvstandalone_oracle_wrapper.sh`.

Modes: `selfcheck`, `oracle`, `auto` (unknown modes raise `ValueError`). Oracle
timeouts become `CompareReceipt(status="failed")`.

Rules:

- Public toy geometries only in this repository.
- Oracle for the strip family: [Mu2e/CRVStandalone](https://github.com/Mu2e/CRVStandalone).
- No Mu2e-internal released tables or conditions.
