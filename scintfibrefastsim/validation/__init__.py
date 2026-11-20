"""Geant4 and public-oracle validation hooks.

The public optical oracle for the strip family is Mu2e/CRVStandalone. Hooks here
are drivers and contracts; they do not vendor Geant4 or CRVStandalone.
"""

from .g4_hooks import (
    CompareReceipt,
    G4ComparePlan,
    compare_plan_crvstandalone,
    compare_plan_toy,
    crvstandalone_path,
    run_compare,
    run_toy_selfcheck,
)

__all__ = [
    "CompareReceipt",
    "G4ComparePlan",
    "compare_plan_crvstandalone",
    "compare_plan_toy",
    "crvstandalone_path",
    "run_compare",
    "run_toy_selfcheck",
]
