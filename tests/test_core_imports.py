"""Core must stay experiment-neutral (stdlib + numpy + jax + own modules)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

CORE = Path(__file__).resolve().parents[1] / "scintfibrefastsim" / "core"
FORBIDDEN = ("mu2e", "crvfastsim", "offline", "novasoft", "hepsimlib", "art")


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module.split(".")[0])
    return names


@pytest.mark.parametrize("path", sorted(CORE.glob("*.py")))
def test_core_imports_nothing_experiment_specific(path: Path) -> None:
    for name in _imports(path):
        low = name.lower()
        assert not any(f in low for f in FORBIDDEN), f"{path.name} imports {name!r}"
        if name not in {""} and not name.startswith("."):
            # relative imports appear as empty module with level>0; top-level only
            pass
    allowed_top = {"numpy", "jax", "math", "bisect", "dataclasses", "time", "typing", "collections", "functools", "itertools", "warnings", "abc", "enum", "numbers", "operator", "sys", "os", "re", "json", "pathlib", "copy", "struct", "hashlib", "array"}
    for name in _imports(path):
        if not name or name.startswith("scintfibrefastsim"):
            continue
        # relative: ast.ImportFrom with module like "optics" when level==0 inside package
        # In core files, imports are `from . import optics` (module None) or `from .optics`
        root = name.split(".")[0]
        if root in {"optics", "materials", "binning", "counter", "limits", "transport", "run"}:
            continue
        assert root in allowed_top or root == "numpy" or root == "jax", (
            f"{path.name}: unexpected import {name!r}"
        )


def test_core_modules_importable() -> None:
    from scintfibrefastsim.core import binning, counter, limits, materials, optics, run, transport

    assert counter.Counter
    assert binning.Binning
    assert optics.DEFAULT_MODEL == "current"
    assert materials.CURRENT.name == "current"
    assert limits.STEP_BITS == 19
    assert transport is not None
    assert run is not None
