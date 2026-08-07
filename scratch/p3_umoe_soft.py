"""HARDENING-P3 R5 wrapper: UMOE soft-routing seeds on the
D2-EXCISED diet (frozen driver umoe_conserve.py untouched —
import-and-override; the loader filter is the Phase-1 law applied
at load time, receipt printed).

Usage: ARM=soft SEED=<n> OTAG=_x3 \
       .venv/bin/python scratch/p3_umoe_soft.py
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import umoe_conserve as U  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig = U.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r5] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} band-touching excised)",
          flush=True)
    return kept


U.load_rows = excised_load_rows

if __name__ == "__main__":
    U.main()
