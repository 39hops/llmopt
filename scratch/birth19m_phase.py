"""PHASE-PORTRAIT-1 instrument run: a fresh 19M-class birth with
STEP-LEVEL milestone saves INCLUDING optimizer state, so a true
per-neuron (angle, angular-velocity) phase portrait exists — position
from the weights, momentum from Adam's exp_avg, velocity from
adjacent milestones. The pendulum riff's residue (RIFF-LEDGER
2026-08-13), instrument-grade.

Precedent: scratch/birth19m_snaps.py (2026-08-08) — portrait-only
instrument births carry no capability claims and need no pre-reg;
any future capability read of these weights needs its own pre-reg.
Same recipe as that run (D2 excision, refuse-if-exists, BIRTH_SEED,
gen4 diet, d384/8L/ffn1536/h6, 3 epochs), two changes: the tee fires
every MILESTONE_EVERY optimizer steps (not per epoch), and each
milestone carries {"model", "opt", "step"} — Adam state included.

Milestones land in checkpoints/phase19m/ (untracked, ~227MB each;
budget ~17 x 227MB = 3.9GB (~14,840 steps projected)). Set MILESTONE_EVERY to tune density;
default aims at ~15 milestones across the run.

Usage: SEED=2 .venv/bin/python scratch/birth19m_phase.py
Device: whichever is free (portrait-only, no gate comparison).
Smoke:  SMOKE=1 SEED=99 ... (2 tiny epochs, milestone every 5 steps,
        output under checkpoints/phase19m_smoke/, deleted freely).
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ.get("SEED", "2")
SMOKE = os.environ.get("SMOKE") == "1"
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/gallery19m_phase_s{SEED}.pt")
MDIR = Path("checkpoints/phase19m_smoke" if SMOKE
            else "checkpoints/phase19m")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (use unspent SEED)")
if MDIR.exists() and any(MDIR.iterdir()):
    raise SystemExit(f"REFUSING: {MDIR} is non-empty")
MDIR.mkdir(parents=True, exist_ok=True)

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
    print(f"[phase19m] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    if SMOKE:
        kept = kept[:400]
        print(f"[phase19m] SMOKE: rows cut to {len(kept)}", flush=True)
    return kept


TM.load_rows = excised_load_rows

# Capture the live model: build_model returns it before the opt exists.
_model = [None]
_orig_build = TM.build_model


def capture_build(*a, **kw):
    _model[0] = _orig_build(*a, **kw)
    return _model[0]


TM.build_model = capture_build

# Milestone tee on the optimizer step. MILESTONE_EVERY defaults to a
# value the launcher prints and can be tuned; the run also tees step 1
# so the portrait has its birth frame.
EVERY = int(os.environ.get("MILESTONE_EVERY", "5" if SMOKE else "900"))
_step = [0]
_orig_opt_step = torch.optim.AdamW.step


def tee_step(self, *a, **kw):
    out = _orig_opt_step(self, *a, **kw)
    _step[0] += 1
    if _step[0] == 1 or _step[0] % EVERY == 0:
        snap = MDIR / f"m{_step[0]:06d}.pt"
        torch.save({"model": _model[0].state_dict(),
                    "opt": self.state_dict(), "step": _step[0]}, snap)
        print(f"[phase19m] milestone step {_step[0]} -> {snap}",
              flush=True)
    return out


torch.optim.AdamW.step = tee_step

TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
        out=str(OUT), v21=False, fast=True, nopack=True,
        v22=True, gen4=True, epochs=(1 if SMOKE else 3))
print(f"[phase19m] complete: {_step[0]} steps, "
      f"{len(list(MDIR.glob('m*.pt')))} milestones in {MDIR}",
      flush=True)
