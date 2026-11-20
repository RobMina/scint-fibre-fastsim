"""Geant4 and public-oracle validation hooks.

The public optical oracle for the strip family is Mu2e/CRVStandalone. Hooks here
are drivers and contracts; they do not vendor Geant4 or CRVStandalone.
"""

from .g4_hooks import G4ComparePlan, compare_plan_crvstandalone, compare_plan_toy

__all__ = ["G4ComparePlan", "compare_plan_crvstandalone", "compare_plan_toy"]
