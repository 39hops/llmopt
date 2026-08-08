"""HARDENING-P4 row 2 wrapper: the 45M ZX scale-lever null gets
its seed ladder. The 45M union verdict (2026-07-27: ZX 36 = +0.8
sigma inside the 19M seed fence mean 32.7 sd ~4.2) was n=1 on a
gate class with a measured 8-point seed swing — the highest-value
seed-starved null in the file after pincer. This wrapper reruns
the FROZEN recipe (night_45m_union.sh: union diet math gen-4 +
zx_farm1, vocab-47, fp32, d512/L12/ffn2048/h8, 3ep) at fresh
seeds; the pooled 3-seed read (booked s1=36 + fresh) goes against
the 19M fence per the original pre-reg's own framing.

Hard gates: D2 excision on load_rows (the ~130 excised rows are
math-gate-band rows — outside the ZX column entirely; seed-1
comparator predates excision, fenced in the pre-reg),
refuse-if-exists, BIRTH_SEED env (set by caller, asserted here).

Usage (envs as night_zx45_x2.sh):
    SEED=<n> .venv/bin/python scratch/rev4_zx45.py
Device: 3080 cuda (device-of-origin: the 45M union line is 3080).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ["SEED"]
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/union_45m_s{SEED}.pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (use unspent SEED)")

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[rev4-zx45] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

TM.main(v2=False, d=512, layers=12, ffn=2048, heads=8,
        out=str(OUT), v21=False, fast=False,
        v22=False, gen4=False, epochs=3,
        diet="data/union_math_zx.jsonl")
