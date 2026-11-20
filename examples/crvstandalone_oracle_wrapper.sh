#!/usr/bin/env bash
# Example SCINTFIBRE_CRVSTANDALONE wrapper.
#
# Contract expected by scintfibrefastsim.validation.run_compare(mode="oracle"):
#   argv: geometry nphot seed dtype
#   stdout last line: JSON with at least {"gen": int, "hits": int}
#
# Point SCINTFIBRE_CRVSTANDALONE at this script after installing Mu2e/CRVStandalone
# and adjusting the wls invocation below. Without a real wls binary this prints a
# stub receipt so the protocol can be tested.
set -euo pipefail
geometry=${1:?}
nphot=${2:?}
seed=${3:?}
dtype=${4:?}
wls=${SCINTFIBRE_WLS:-}
if [[ -n "$wls" && -x "$wls" ]]; then
  # Replace with the site's table-mode invocation; map stdout into the JSON contract.
  echo "running $wls for geometry=$geometry nphot=$nphot seed=$seed dtype=$dtype" >&2
  "$wls" -c  # placeholder — site fills real flags / fcl / counter card
fi
# Stub metrics when wls is not wired: keep the JSON contract stable for CI of the wrapper.
python3 - <<PY
import json
print(json.dumps({"gen": int("$nphot"), "hits": 0, "geometry": "$geometry",
                  "seed": int("$seed"), "dtype": "$dtype", "stub": True}))
PY
