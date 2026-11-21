#!/usr/bin/env bash
# Download the public sources of the NOvA cell references into DIR and check every sha256 (sources.json).
#   bash validation/nova_cell/fetch.sh DIR
set -euo pipefail
dir=${1:?usage: fetch.sh DIR}
here=$(cd "$(dirname "$0")" && pwd)
mkdir -p "$dir"
python3 - "$here/sources.json" "$dir" <<'EOF'
import hashlib, json, sys, urllib.request
from pathlib import Path
src, out = json.load(open(sys.argv[1]))["sources"], Path(sys.argv[2])
bad = 0
for key, s in src.items():
    f = out / s["file"]
    if not f.is_file():
        req = urllib.request.Request(s["url"], headers={"User-Agent": "scint-fibre-fastsim-validation"})
        with urllib.request.urlopen(req, timeout=120) as r:
            f.write_bytes(r.read())
    h = hashlib.sha256(f.read_bytes()).hexdigest()
    ok = h == s["sha256"]
    bad += not ok
    print(f"{'ok ' if ok else 'BAD'} {key:9s} {s['file']}  {h}")
sys.exit(1 if bad else 0)
EOF
