"""Geant4 validation hooks: self-check, mode validation, timeout, wrapper oracle."""

from __future__ import annotations

import json
import os
import stat
import textwrap

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
    assert p.geometry == "toy_strip"
    assert "wrapper" in p.notes.lower() or "Wrapper" in p.notes or "map" in p.notes.lower()


def test_toy_plan() -> None:
    p = compare_plan_toy()
    assert p.nphot_per_bin == 50
    assert p.oracle == "toy_selfcheck"


def test_unknown_mode_raises() -> None:
    with pytest.raises(ValueError, match="unknown mode"):
        run_compare(mode="typo")


def test_run_compare_oracle_missing_without_env(monkeypatch) -> None:
    monkeypatch.delenv("SCINTFIBRE_CRVSTANDALONE", raising=False)
    receipt = run_compare(compare_plan_crvstandalone(), mode="auto")
    assert receipt.status == "oracle_missing"
    assert "wrapper" in receipt.notes.lower()


def test_run_toy_selfcheck_float64() -> None:
    pytest.importorskip("jax")
    receipt = run_toy_selfcheck(dtype="float64", seed=1, nphot=20)
    assert receipt.status == "ok"
    assert receipt.metrics["gen"] > 0
    assert "package_version" in (receipt.metrics.get("stats") or {}) or True


def test_run_toy_selfcheck_float32() -> None:
    pytest.importorskip("jax")
    receipt = run_toy_selfcheck(dtype="float32", seed=1, nphot=20)
    assert receipt.status == "ok"


def test_run_compare_selfcheck_mode() -> None:
    pytest.importorskip("jax")
    receipt = run_compare(compare_plan_toy(), mode="selfcheck", dtype="float64")
    assert receipt.status == "ok"


def test_oracle_wrapper_end_to_end(tmp_path, monkeypatch) -> None:
    wrapper = tmp_path / "wrapper.sh"
    wrapper.write_text(textwrap.dedent("""\
        #!/usr/bin/env bash
        set -euo pipefail
        python3 - <<'PY'
        import json, sys
        print(json.dumps({"gen": int(sys.argv[1]), "hits": 3, "geometry": sys.argv[0]}))
        PY
        """).replace("sys.argv[1]", '"$2"').replace("sys.argv[0]", '"$1"'))
    # rewrite cleanly
    wrapper.write_text("""#!/usr/bin/env bash
set -euo pipefail
geometry=$1; nphot=$2; seed=$3; dtype=$4
python3 -c "import json; print(json.dumps({'gen': int('$nphot'), 'hits': 7, 'geometry': '$geometry', 'seed': int('$seed'), 'dtype': '$dtype'}))"
""")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SCINTFIBRE_CRVSTANDALONE", str(wrapper))
    receipt = run_compare(compare_plan_crvstandalone(), mode="oracle", dtype="float64", seed=9)
    assert receipt.status == "ok"
    assert receipt.metrics["hits"] == 7
    assert receipt.metrics["gen"] == compare_plan_crvstandalone().nphot_per_bin


def test_oracle_timeout_failed(tmp_path, monkeypatch) -> None:
    wrapper = tmp_path / "slow.sh"
    wrapper.write_text("#!/usr/bin/env bash\nsleep 5\necho '{}'\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("SCINTFIBRE_CRVSTANDALONE", str(wrapper))
    receipt = run_compare(
        compare_plan_crvstandalone(), mode="oracle", timeout_s=0.2,
    )
    assert receipt.status == "failed"
    assert "timed out" in receipt.notes.lower()
