"""Stubs for Geant4 / CRVStandalone comparison drivers.

Phase 1 fills these with a public toy strip+fibre bench against CRVStandalone
table mode. No Mu2e-internal tables or conditions belong here.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class G4ComparePlan:
    """What a Geant4 comparison job must declare before it runs."""

    name: str
    oracle: str  # e.g. "Mu2e/CRVStandalone", "toy_g4_optical"
    geometry: str
    nphot_per_bin: int
    simtypes: tuple[int, ...]
    notes: str = ""


def compare_plan_crvstandalone() -> G4ComparePlan:
    """Declared plan for the Phase-1 CRVStandalone bench (not executed here)."""
    return G4ComparePlan(
        name="strip_vs_crvstandalone",
        oracle="Mu2e/CRVStandalone",
        geometry="toy_strip",
        nphot_per_bin=10_000,
        simtypes=(0, 1, 2),
        notes="Public toy strip; no Mu2e-internal released tables",
    )


def compare_plan_toy() -> G4ComparePlan:
    """Minimal self-check plan for CI-sized runs once the driver exists."""
    return G4ComparePlan(
        name="toy_optical_smoke",
        oracle="toy_g4_optical",
        geometry="toy_strip",
        nphot_per_bin=100,
        simtypes=(0,),
        notes="Smoke only; fills in when a public G4 optical driver is wired",
    )


def run_compare(plan: G4ComparePlan) -> dict:
    """Execute a compare plan. Stub until Phase 1 wires the driver."""
    raise NotImplementedError(
        f"Geant4 compare driver for {plan.name!r} is Phase 1; "
        f"oracle={plan.oracle!r}, geometry={plan.geometry!r}"
    )
