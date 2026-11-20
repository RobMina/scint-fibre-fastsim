"""Examples run as written."""

from __future__ import annotations

import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"


@pytest.mark.parametrize("name", ["first_hour.py", "toy_strip_core.py"])
def test_example_runs(name: str) -> None:
    runpy.run_path(str(EXAMPLES / name), run_name="__main__")
