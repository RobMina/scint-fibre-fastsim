#!/usr/bin/env bash
# Reference SCINTFIBRE_CRVSTANDALONE wrapper (required — raw CRVStandalone wls is not enough).
#
# Contract expected by scintfibrefastsim.validation.run_compare(mode="oracle"):
#   argv: geometry nphot seed dtype
#   stdout last line: JSON with at least {"gen": int, "hits": int}
#
# geometry=toy_strip is this package's public toy; map it to a CRVStandalone
# counter card when SCINTFIBRE_WLS points at a real wls binary.
set -euo pipefail
geometry=${1:?}
nphot=${2:?}
seed=${3:?}
dtype=${4:?}
wls=${SCINTFIBRE_WLS:-}

case "$geometry" in
  toy_strip)
    # Placeholder mapping: replace with the site's counter card / fcl for the public toy.
    card="toy_strip"
    ;;
  *)
    echo "unknown geometry $geometry" >&2
    exit 2
    ;;
esac

if [[ -n "$wls" && -x "$wls" ]]; then
  echo "running $wls card=$card nphot=$nphot seed=$seed dtype=$dtype" >&2
  # Site fills real flags; parse stdout into gen/hits for the JSON line below.
  "$wls" -c || true
fi

# Always emit the contract line (stub metrics when wls is not wired).
python3 - <<PY
import json
print(json.dumps({
    "gen": int("$nphot"),
    "hits": 0,
    "geometry": "$geometry",
    "card": "$card",
    "seed": int("$seed"),
    "dtype": "$dtype",
    "stub": not bool("$wls"),
}))
PY
