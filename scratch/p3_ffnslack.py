"""HARDENING-P3 R3 wrapper: ffn-slack ENDPOINTS (d56 f224 v f128)
on the D2-EXCISED diet — frozen driver scratch/sym_birth.py
untouched (import-and-override, the p3_umoe_soft/p3_grav2 pattern).
train_mathnative.load_rows is patched BEFORE sym_birth's from-import
binds it, so the excision covers the driver's diet verbatim.

Hard gates carried: (1) D2 excision with printed receipt; (2)
REFUSE-IF-EXISTS on the constructed OUT name (sym_birth writes
checkpoints/sym_birth_{ARM}{TAG}.pt — seed-free unless TAG carries
the seed; this wrapper REQUIRES the seed in TAG); (3) device-of-
origin = cuda (the NIGHT-29 battery-2 line; run on the 3080 only).

Usage: TAG=_cu56_f224_s2 SEED=2 D=56 FFN=224 HEADS=4 EMA=0.999 \
       ARM=dense .venv/bin/python scratch/p3_ffnslack.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

TAG = os.environ.get("TAG", "")
SEED = os.environ.get("SEED", "")
ARM = os.environ.get("ARM", "dense")
if not SEED or f"s{SEED}" not in TAG:
    raise SystemExit("REFUSING: TAG must carry the seed "
                     "(seed-free OUT names are the collision risk)")
out = Path(f"checkpoints/sym_birth_{ARM}{TAG}.pt")
if out.exists():
    raise SystemExit(f"REFUSING: {out} exists (cited-evidence "
                     "guard); use an unspent TAG/seed")

import train_mathnative  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig = train_mathnative.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r3] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    return kept


train_mathnative.load_rows = excised_load_rows

import sym_birth  # noqa: E402,F401  (top-level script: import runs the birth)
