"""BACKWARD-SCHEDULE-1 instrument run (pre-reg RESULTS 2026-08-13):
the phase19m recipe with ONE variable changed — the OneCycle lr
sequence is served REVERSED in time (anneal-first, warm-last). Same
D2 excision, BIRTH_SEED, arch, diet, epochs, device as the booked
phase19m birth; milestone tee kept so the backwards phase portrait
comes free. Paired control is the BOOKED m015300 gate 64/120.

No-op precondition (asserted at import): with REVERSE=0 the patched
scheduler reproduces stock OneCycleLR's lr sequence element-wise on
a dummy 100-step optimizer.

Usage:  SEED=2 .venv/bin/python scratch/birth19m_backsched.py
Smoke:  SMOKE=1 SEED=99 ... (2 tiny epochs, milestone every 5 steps)
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ.get("SEED", "2")
SMOKE = os.environ.get("SMOKE") == "1"
REVERSE = os.environ.get("REVERSE", "1") == "1"
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/gallery19m_backsched_s{SEED}.pt")
MDIR = Path("checkpoints/backsched19m_smoke" if SMOKE
            else "checkpoints/backsched19m")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (use unspent SEED)")
if MDIR.exists() and any(MDIR.iterdir()):
    raise SystemExit(f"REFUSING: {MDIR} is non-empty")
MDIR.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_onecycle = torch.optim.lr_scheduler.OneCycleLR


def _stock_lr_sequence(opt, max_lr, total_steps, pct_start):
    """The stock OneCycle per-step lr values, computed on a dummy
    optimizer so the real one's state is untouched."""
    dummy = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    sch = _orig_onecycle(dummy, max_lr=max_lr, total_steps=total_steps,
                         pct_start=pct_start)
    lrs = []
    for _ in range(total_steps):
        lrs.append(sch.get_last_lr()[0])
        dummy.step()
        sch.step()
    return lrs


class SequenceLR(torch.optim.lr_scheduler.LambdaLR):
    """Serve a precomputed lr sequence by step index (clamped at
    the end, matching OneCycle's headroom-overrun tolerance)."""

    def __init__(self, opt, seq):
        self._seq = seq
        base = [g["lr"] for g in opt.param_groups]
        super().__init__(opt, lr_lambda=[
            (lambda i, b=b: self._seq[min(i, len(self._seq) - 1)] / b)
            for b in base])


def patched_onecycle(opt, max_lr, total_steps, pct_start=0.3, **kw):
    seq = _stock_lr_sequence(opt, max_lr, total_steps, pct_start)
    if REVERSE:
        seq = seq[::-1]
        print(f"[backsched] REVERSED lr sequence: starts {seq[0]:.3e}, "
              f"peaks {max(seq):.3e} at step {seq.index(max(seq))}, "
              f"ends {seq[-1]:.3e} ({total_steps} steps)", flush=True)
    return SequenceLR(opt, seq)


# No-op precondition: REVERSE=0 path == stock OneCycle, element-wise.
def _assert_noop():
    dummy_a = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    dummy_b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    stock = _orig_onecycle(dummy_a, max_lr=3e-3, total_steps=100,
                           pct_start=0.03)
    seq = SequenceLR(dummy_b, _stock_lr_sequence(
        dummy_b, 3e-3, 100, 0.03))
    for i in range(100):
        a, b = stock.get_last_lr()[0], seq.get_last_lr()[0]
        assert a == b, f"no-op precondition FAILED at step {i}: {a} != {b}"
        dummy_a.step(); stock.step()
        dummy_b.step(); seq.step()
    print("[backsched] no-op precondition PASSED: 100/100 lr values "
          "element-wise equal to stock OneCycle", flush=True)


_assert_noop()
torch.optim.lr_scheduler.OneCycleLR = patched_onecycle

_orig_rows = TM.load_rows


def excised_load_rows(*a, **kw):
    rows = _orig_rows(*a, **kw)
    band = set(gate_band_exprs())
    kept = [r for r in rows
            if norm(str(r.get("cur", ""))) not in band
            and norm(str(r.get("nxt", ""))) not in band]
    print(f"[backsched] D2 excision: {len(rows)} -> {len(kept)} "
          f"rows ({len(rows) - len(kept)} excised)", flush=True)
    if SMOKE:
        kept = kept[:400]
        print(f"[backsched] SMOKE: rows cut to {len(kept)}", flush=True)
    return kept


TM.load_rows = excised_load_rows

_model = [None]
_orig_build = TM.build_model


def capture_build(*a, **kw):
    _model[0] = _orig_build(*a, **kw)
    return _model[0]


TM.build_model = capture_build

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
        print(f"[backsched] milestone step {_step[0]} -> {snap}",
              flush=True)
    return out


torch.optim.AdamW.step = tee_step

TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
        out=str(OUT), v21=False, fast=True, nopack=True,
        v22=True, gen4=True, epochs=(1 if SMOKE else 3))
print(f"[backsched] complete: {_step[0]} steps, "
      f"{len(list(MDIR.glob('m*.pt')))} milestones in {MDIR}",
      flush=True)

# Final gate, streamed into the same log (stock harness, unwrapped).
from llmopt.lab.gate import gate_checkpoint  # noqa: E402
import torch as _t  # noqa: E402
sd = _t.load(OUT, map_location="cpu")
sd = sd["model"] if isinstance(sd, dict) and "model" in sd else sd
tmp = MDIR / "final_model_only.pt"
_t.save(sd, tmp)
gate_checkpoint(str(tmp), 384, 8, 1536, 6,
                f"backsched_s{SEED}", device="mps")
