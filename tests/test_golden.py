"""Golden regression: fixed toy strip / seed / bins; hashes of gen, ht, he per dtype."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

GOLDEN = Path(__file__).resolve().parent / "data" / "golden_toy.json"


def _digest(arr) -> str:
    return hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_golden_toy_transport(dtype: str) -> None:
    pytest.importorskip("jax")
    from scintfibrefastsim.core.binning import Binning
    from scintfibrefastsim.core.run import generate
    from scintfibrefastsim.geometry.families import toy_strip

    gold = json.loads(GOLDEN.read_text())
    exp = gold["dtypes"][dtype]
    c = toy_strip(length=float(gold["length"]))
    edges = {
        "x": np.linspace(0.0, float(c.hx), 3),
        "y": np.linspace(-float(c.hy), float(c.hy), 3),
        "z": np.linspace(-float(c.half_length), float(c.half_length), 3),
    }
    b = Binning(edges=edges, order={0: ["x", "y", "z"]})
    bins = np.arange(b.n_bins(0), dtype=np.int64)
    r = generate(
        c, b, int(gold["simtype"]), bins, int(gold["nphot"]),
        dtype=dtype, seed=int(gold["seed"]),
        lanes=int(gold["lanes"]), chunk_iters=int(gold["chunk_iters"]), helix=True,
    )
    assert int(np.asarray(r["gen"]).sum()) == exp["gen_sum"]
    assert int(np.asarray(r["ht"]).sum()) == exp["ht_sum"]
    assert _digest(r["gen"]) == exp["gen_sha256"]
    assert _digest(r["ht"]) == exp["ht_sha256"]
    assert _digest(r["he"]) == exp["he_sha256"]
    assert r["stats"].get("package_version")
