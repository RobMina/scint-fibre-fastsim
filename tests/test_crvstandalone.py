"""CRVStandalone default counter, its bins and the piece format (no Geant4 needed)."""
import struct

import numpy as np

from scintfibrefastsim.validation import crvstandalone as C


def test_default_counter_bins():
    c = C.counter(3200.0, 0)
    b = C.binning(c)
    assert len(C.z_edges(c)) == 57                     # 56 z bins for 3200 mm
    assert b.n_bins(0) == 10 * 41 * 56
    assert b.n_bins(1) == 10 * 41 * 56 * 5
    assert b.n_bins(2) == 5 * 11 * 5 * 5 * 56


def test_piece_round_trip(tmp_path):
    out = {"bins": np.array([3, 7]), "gen": np.array([1000, 1000]), "nphot": 1000,
           "ht": np.array([[0, 5, 10, 5] + [0] * 146, [0] * 150]),
           "he": np.array([[20] + [0] * 9, [0] * 10])}
    blob = b""
    for j, i in enumerate(out["bins"]):
        prob, d, e = C.encode(out, j)
        blob += struct.pack("<If", int(i), prob) + struct.pack("<Q", len(d)) + d.tobytes() + struct.pack("<Q", len(e)) + e.tobytes()
    p = tmp_path / "LookupTable_piece"
    p.write_bytes(blob)
    head, recs = C.read_piece(p, False)
    assert head == b"" and [r[0] for r in recs] == [3, 7]
    assert abs(recs[0][1] - 0.02) < 1e-7 and recs[1][1] == 0.0
    ref = {i: (pr, d, e) for i, pr, d, e in recs}
    r = C.compare(ref, out, 1000)
    assert r["prob_pull"]["n"] == 1 and abs(r["prob_pull"]["mean"]) < 1e-9
