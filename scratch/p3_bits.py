"""HARDENING-P3 R9 wrapper: the bits-dimension 19M row pooled at
n=3 — frozen drivers untouched (tournament_birth.main for the
ternary arm; train_mathnative.main for the fp32 comparator), with
the standard hard gates: D2 excision on load_rows (patched BEFORE
either driver binds it), refuse-if-exists on every OUT, seed via
BIRTH_SEED env.

ARMS: t = ternary d384 at 6 epochs (the discrete-learning dose);
fp32 = fp32 d384 at 3 epochs (its own proven dose). Same corpus,
same width/params — the pooled read is the ALPHABET TAX at matched
width with each alphabet at its own dose.

Usage: ARM=t|fp32 SEED=<n> .venv/bin/python scratch/p3_bits.py
Device: 3080 cuda (the tournament line's device-of-origin).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

ARM = os.environ["ARM"]
SEED = os.environ["SEED"]
assert ARM in ("t", "fp32"), ARM
os.environ["BIRTH_SEED"] = SEED

if ARM == "t":
    targets = [Path(f"checkpoints/tourn_T_p3r9_s{SEED}.pt"),
               Path(f"checkpoints/tourn_T_p3r9_s{SEED}_latent.pt")]
else:
    targets = [Path(f"checkpoints/p3r9_fp32_s{SEED}.pt")]
for t in targets:
    if t.exists():
        raise SystemExit(f"REFUSING: {t} exists (use unspent SEED)")

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[p3-r9] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

if ARM == "t":
    sys.argv = ["tournament_birth.py", "--alpha", "T",
                "--epochs", "6", "--tag", f"_p3r9_s{SEED}"]
    from tournament_birth import main as tourn_main  # noqa: E402
    tourn_main()
else:
    TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
            out=str(targets[0]), v21=False, fast=False,
            v22=True, gen4=True, epochs=3)
