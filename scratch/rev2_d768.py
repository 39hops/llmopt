"""REVIVE track item 3 wrapper: the d768 ternary-v-fp32 crossover
gets n=3 SAME-DEVICE paired seeds (the p3_bits pattern at d768;
p3_bits itself is R9-cited and stays frozen).

Why this revive matters doubly: the original crossover ("ternary
65 beats fp32 58 at d768") compared a MAC-born fp32 cell
(night2_mac.sh) against a 3080-born ternary cell (tournament) —
cross-device by modern doctrine. This re-run pairs both arms on
ONE device (Mac, Artin's call 2026-08-08), fixing the fence while
testing the claim. Both arms at 3 epochs (the original grid's own
dose — this replicates the GRID's comparison).

Hard gates: D2 excision on load_rows (patched before either driver
binds it); refuse-if-exists; BIRTH_SEED env; GRAD_CKPT for d768.

Usage: ARM=t|fp32 SEED=<n> .venv/bin/python scratch/rev2_d768.py
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
os.environ.setdefault("GRAD_CKPT", "1")

if ARM == "t":
    targets = [Path(f"checkpoints/tourn_T_rev2d768_s{SEED}.pt"),
               Path(f"checkpoints/tourn_T_rev2d768_s{SEED}_latent.pt")]
else:
    targets = [Path(f"checkpoints/rev2_fp32_768_s{SEED}.pt")]
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
    print(f"[rev2-d768] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

if ARM == "t":
    sys.argv = ["tournament_birth.py", "--alpha", "T",
                "--epochs", "3", "--d", "768", "--layers", "8",
                "--ffn", "3072", "--heads", "12",
                "--tag", f"_rev2d768_s{SEED}"]
    from tournament_birth import main as tourn_main  # noqa: E402
    tourn_main()
else:
    TM.main(v2=False, d=768, layers=8, ffn=3072, heads=12,
            out=str(targets[0]), v21=False, fast=False,
            v22=True, gen4=True, epochs=3)
