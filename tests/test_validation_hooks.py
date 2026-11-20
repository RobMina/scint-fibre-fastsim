"""Geant4 validation hook stubs and toy self-check."""

from __future__ import annotations

import json

import pytest

from scintfibrefastsim.validation import (
    compare_plan_crvstandalone,
    compare_plan_toy,
    run_compare,
    run_toy_selfcheck,
)


def test_crvstandalone_plan() -> None:
    p = compare_plan_crvstandalone()
    assert p.oracle == "Mu2e/CRVStandalone"
    assert 0 in p.simtypes


def test_toy_plan() -> None:
    p = compare_plan_toy()
    assert p.nphot_per_bin == 50
    assert p.oracle == "toy_selfcheck"


def test_run_compare_oracle_missing_without_env(monkeypatch) -> None:
    monkeypatch.delenv("SCINTFIBRE_CRVSTANDALONE", raising=False)
    receipt = run_compare(compare_plan_crvstandalone(), mode="auto")
    assert receipt.status == "oracle_missing"


def test_run_toy_selfcheck_float64() -> None:
    pytest.importorskip("jax")
    receipt = run_toy_selfcheck(dtype="float64", seed=1, nphot=20)
    assert receipt.status == "ok"
    assert receipt.metrics["gen"] > 0
    assert receipt.metrics["hits"] >= 0
    payload = json.loads(receipt.to_json())
    assert payload["dtype"] == "float64"


def test_run_toy_selfcheck_float32() -> None:
    pytest.importorskip("jax")
    receipt = run_toy_selfcheck(dtype="float32", seed=1, nphot=20)
    assert receipt.status == "ok"
    assert receipt.metrics["gen"] > 0


def test_run_compare_selfcheck_mode() -> None:
    pytest.importorskip("jax")
    receipt = run_compare(compare_plan_toy(), mode="selfcheck", dtype="float64")
    assert receipt.status == "ok"
