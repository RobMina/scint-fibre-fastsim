"""Geant4 / CRVStandalone comparison drivers and receipts.

Phase 1: a toy self-check that runs the JAX core on a public strip, and a declared
CRVStandalone compare plan. Full oracle runs need Mu2e/CRVStandalone on
``$SCINTFIBRE_CRVSTANDALONE`` (the ``wls`` binary or a wrapper script).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class G4ComparePlan:
    """What a Geant4 comparison job must declare before it runs."""

    name: str
    oracle: str  # e.g. "Mu2e/CRVStandalone", "toy_selfcheck"
    geometry: str
    nphot_per_bin: int
    simtypes: tuple[int, ...]
    notes: str = ""


@dataclass
class CompareReceipt:
    """Machine-readable outcome of a compare or self-check run."""

    plan: dict[str, Any]
    status: str  # "ok" | "oracle_missing" | "failed"
    dtype: str
    seed: int
    wall_s: float
    metrics: dict[str, Any] = field(default_factory=dict)
    notes: str = ""

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    def write(self, path: str | Path) -> Path:
        p = Path(path)
        p.write_text(self.to_json() + "\n")
        return p


def compare_plan_crvstandalone() -> G4ComparePlan:
    """Declared plan for the Phase-1 CRVStandalone bench."""
    return G4ComparePlan(
        name="strip_vs_crvstandalone",
        oracle="Mu2e/CRVStandalone",
        geometry="toy_strip",
        nphot_per_bin=10_000,
        simtypes=(0, 1, 2),
        notes="Public toy strip; no Mu2e-internal released tables",
    )


def compare_plan_toy() -> G4ComparePlan:
    """CI-sized self-check plan (JAX core only; no external Geant4)."""
    return G4ComparePlan(
        name="toy_optical_smoke",
        oracle="toy_selfcheck",
        geometry="toy_strip",
        nphot_per_bin=50,
        simtypes=(0,),
        notes="JAX core smoke on a public toy strip; not a Geant4 oracle",
    )


def crvstandalone_path() -> str | None:
    """Path to a CRVStandalone ``wls`` (or wrapper) if configured."""
    return os.environ.get("SCINTFIBRE_CRVSTANDALONE") or None


def _toy_generate(dtype: str, seed: int, nphot: int) -> dict[str, Any]:
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.run import generate
    from scintfibrefastsim.geometry.families import toy_strip

    c = toy_strip(length=200.0)
    edges = {
        "x": np.linspace(0.0, float(c.hx), 3),
        "y": np.linspace(-float(c.hy), float(c.hy), 3),
        "z": np.linspace(-float(c.half_length), float(c.half_length), 3),
    }
    b = Binning(edges=edges, order={0: ["x", "y", "z"]})
    bins = np.arange(b.n_bins(0), dtype=np.int64)
    return generate(
        c, b, 0, bins, int(nphot), dtype=dtype, seed=int(seed),
        lanes=256, chunk_iters=400, helix=True,
    )


def run_toy_selfcheck(*, dtype: str = "float64", seed: int = 1,
                      nphot: int | None = None) -> CompareReceipt:
    """Run the JAX core on the public toy strip; return a receipt."""
    plan = compare_plan_toy()
    nphot = int(nphot if nphot is not None else plan.nphot_per_bin)
    t0 = time.perf_counter()
    r = _toy_generate(dtype, seed, nphot)
    wall = time.perf_counter() - t0
    gen = int(np.asarray(r["gen"]).sum())
    hits = int(np.asarray(r["ht"]).sum())
    return CompareReceipt(
        plan=asdict(plan),
        status="ok",
        dtype=dtype,
        seed=int(seed),
        wall_s=wall,
        metrics={
            "gen": gen,
            "hits": hits,
            "n_bins": int(len(r["bins"])),
            "stats": {k: (int(v) if hasattr(v, "item") else v)
                      for k, v in dict(r.get("stats") or {}).items()
                      if k in ("capped", "forced", "badsrc", "device", "platform", "model")},
        },
        notes="toy_selfcheck: JAX core only",
    )


def run_compare(plan: G4ComparePlan | None = None, *, dtype: str = "float64",
                seed: int = 1, mode: str = "auto") -> CompareReceipt:
    """Execute a compare plan.

    ``mode``:
      - ``selfcheck``: always run ``run_toy_selfcheck`` (CI default path)
      - ``oracle``: require ``$SCINTFIBRE_CRVSTANDALONE`` and invoke it
      - ``auto``: oracle if the env is set, else selfcheck for toy plans,
        ``oracle_missing`` for CRVStandalone plans without the binary
    """
    plan = plan or compare_plan_crvstandalone()
    if mode == "selfcheck" or (mode == "auto" and plan.oracle == "toy_selfcheck"):
        return run_toy_selfcheck(dtype=dtype, seed=seed, nphot=plan.nphot_per_bin)

    path = crvstandalone_path()
    if mode == "auto" and not path:
        return CompareReceipt(
            plan=asdict(plan),
            status="oracle_missing",
            dtype=dtype,
            seed=int(seed),
            wall_s=0.0,
            notes="Set SCINTFIBRE_CRVSTANDALONE to the CRVStandalone wls binary (or wrapper)",
        )
    if not path:
        raise FileNotFoundError(
            "SCINTFIBRE_CRVSTANDALONE is unset; cannot run oracle mode"
        )

    t0 = time.perf_counter()
    # Wrapper contract: argv[1]=geometry name, argv[2]=nphot, argv[3]=seed, argv[4]=dtype
    # stdout: one JSON object with at least gen, hits (same keys as a receipt metrics block).
    try:
        proc = subprocess.run(
            [path, plan.geometry, str(plan.nphot_per_bin), str(seed), dtype],
            check=True, capture_output=True, text=True, timeout=3600,
        )
        metrics = json.loads(proc.stdout.strip().splitlines()[-1])
        status = "ok"
        notes = f"oracle={path}"
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, IndexError) as exc:
        return CompareReceipt(
            plan=asdict(plan),
            status="failed",
            dtype=dtype,
            seed=int(seed),
            wall_s=time.perf_counter() - t0,
            notes=f"oracle invoke failed: {exc}",
        )
    return CompareReceipt(
        plan=asdict(plan),
        status=status,
        dtype=dtype,
        seed=int(seed),
        wall_s=time.perf_counter() - t0,
        metrics=metrics,
        notes=notes,
    )
