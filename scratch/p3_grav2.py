"""HARDENING-P3 R6 wrapper: GRAV-2 contractivity-tax paired births
on the D2-EXCISED diet (frozen driver grav2_spacetime.py untouched;
import-and-override, the p3_umoe_soft pattern). OUT names collide
with the originals at SEED=1 — this wrapper REFUSES SEED values
whose checkpoint already exists (cited-evidence guard) unless
OTAG-style suffixing is added upstream; run seeds 2/3/4.

Usage: ARM=ctl|contract SEED=<n> \
       .venv/bin/python scratch/p3_grav2.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import grav2_spacetime as U  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

if Path(U.OUT).exists():
    raise SystemExit(f"REFUSING: {U.OUT} exists (cited evidence); "
                     "use an unspent SEED")

_orig = U.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r6] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    return kept


U.load_rows = excised_load_rows

if __name__ == "__main__":
    U.main()
