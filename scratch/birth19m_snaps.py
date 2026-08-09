"""Gallery instrument run: a fresh 19M-class birth with PER-EPOCH
snapshots, so the 113M-style growth render (plot_neurons
--displace, the recovered whisper-zoom instrument) exists for the
19M line as [R]-reproducible (Artin's ask 2026-08-08; the frozen
crystal-era files have no surviving pair). Also feeds the
calibrated internet-vs-native displacement comparison
(qwen_displace_extract.py made the internet pair).

Mechanism: torch.save tee — every save of OUT also lands a copy
at OUT stem + _ep{N}.pt (N increments per save; the trainer saves
once per epoch end). Standard hard gates: D2 excision,
refuse-if-exists, BIRTH_SEED. NOT a gate experiment: no
capability claims, portrait only (no pre-reg needed; any future
capability read of these weights needs its own pre-reg).

Usage: SEED=1 .venv/bin/python scratch/birth19m_snaps.py
Device: whichever is free (portrait-only, no gate comparison).
"""
import os
import shutil
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ.get("SEED", "1")
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/gallery19m_s{SEED}.pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (use unspent SEED)")

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[gallery19m] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    return kept


TM.load_rows = excised_load_rows

_orig_save = torch.save
_n = [0]


def tee_save(obj, f, *a, **kw):
    _orig_save(obj, f, *a, **kw)
    if str(f) == str(OUT):
        snap = OUT.with_name(f"{OUT.stem}_ep{_n[0]}.pt")
        shutil.copyfile(OUT, snap)
        print(f"[gallery19m] snap -> {snap}", flush=True)
        _n[0] += 1


torch.save = tee_save

TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
        out=str(OUT), v21=False, fast=True, nopack=True,
        v22=True, gen4=True, epochs=3)
