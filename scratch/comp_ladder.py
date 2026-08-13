"""COMP-LADDER-1 instrument run (pre-reg RESULTS 2026-08-13): the
phase19m recipe with the OneCycle schedule COMPRESSED — the full
stock shape (warmup, peak, anneal) squeezed into N = RATIO x 15,420
steps, training stopped at step N by this driver (save, gate, exit).
The arm trains on fewer tokens too; that is the product being
priced, per the pre-reg fence.

No-op precondition (asserted at import): the sequence-serving
scheduler at ratio 1.0 reproduces stock OneCycleLR element-wise on
a dummy 100-step optimizer.

Usage:  RATIO=0.5 SEED=2 .venv/bin/python scratch/comp_ladder.py
Smoke:  SMOKE=1 RATIO=0.5 SEED=99 ...
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SEED = os.environ.get("SEED", "2")
SMOKE = os.environ.get("SMOKE") == "1"
RATIO = float(os.environ.get("RATIO", "0.5"))
TAG = f"comp{int(RATIO * 100)}"
os.environ["BIRTH_SEED"] = SEED
OUT = Path(f"checkpoints/gallery19m_{TAG}_s{SEED}.pt")
if OUT.exists():
    raise SystemExit(f"REFUSING: {OUT} exists (use unspent SEED)")
RECEIPTS = Path("logs/comp_ladder/arms.jsonl")
RECEIPTS.parent.mkdir(parents=True, exist_ok=True)

import torch  # noqa: E402

import train_mathnative as TM  # noqa: E402
from tenet_d2_revdiet import gate_band_exprs, norm  # noqa: E402

_orig_onecycle = torch.optim.lr_scheduler.OneCycleLR


def _stock_lr_sequence(max_lr, total_steps, pct_start):
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
    def __init__(self, opt, seq):
        self._seq = seq
        base = [g["lr"] for g in opt.param_groups]
        super().__init__(opt, lr_lambda=[
            (lambda i, b=b: self._seq[min(i, len(self._seq) - 1)] / b)
            for b in base])


N_STEPS = [None]  # set when the patched scheduler is constructed


def patched_onecycle(opt, max_lr, total_steps, pct_start=0.3, **kw):
    n = max(int(total_steps * RATIO), 10)
    N_STEPS[0] = n
    seq = _stock_lr_sequence(max_lr, n, pct_start)
    print(f"[{TAG}] compressed schedule: stock shape over {n} steps "
          f"(ratio {RATIO} of {total_steps}); peak {max(seq):.3e} at "
          f"step {seq.index(max(seq))}, ends {seq[-1]:.3e}", flush=True)
    return SequenceLR(opt, seq)


def _assert_noop():
    dummy_a = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    dummy_b = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
    stock = _orig_onecycle(dummy_a, max_lr=3e-3, total_steps=100,
                           pct_start=0.03)
    seq = SequenceLR(dummy_b, _stock_lr_sequence(3e-3, 100, 0.03))
    for i in range(100):
        a, b = stock.get_last_lr()[0], seq.get_last_lr()[0]
        assert a == b, f"no-op precondition FAILED at step {i}: {a} != {b}"
        dummy_a.step(); stock.step()
        dummy_b.step(); seq.step()
    print(f"[{TAG}] no-op precondition PASSED: 100/100 lr values "
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
    print(f"[{TAG}] D2 excision: {len(rows)} -> {len(kept)} rows "
          f"({len(rows) - len(kept)} excised)", flush=True)
    if SMOKE:
        kept = kept[:400]
        print(f"[{TAG}] SMOKE: rows cut to {len(kept)}", flush=True)
    return kept


TM.load_rows = excised_load_rows

_model = [None]
_orig_build = TM.build_model


def capture_build(*a, **kw):
    _model[0] = _orig_build(*a, **kw)
    return _model[0]


TM.build_model = capture_build


class _Done(Exception):
    pass


_step = [0]
_t0 = time.time()
_orig_opt_step = torch.optim.AdamW.step


def stop_step(self, *a, **kw):
    out = _orig_opt_step(self, *a, **kw)
    _step[0] += 1
    if N_STEPS[0] is not None and _step[0] >= N_STEPS[0]:
        raise _Done()
    return out


torch.optim.AdamW.step = stop_step

try:
    TM.main(v2=False, d=384, layers=8, ffn=1536, heads=6,
            out=str(OUT), v21=False, fast=True, nopack=True,
            v22=True, gen4=True, epochs=(1 if SMOKE else 3))
except _Done:
    pass
wall = time.time() - _t0
torch.save(_model[0].state_dict(), OUT)
print(f"[{TAG}] stopped at step {_step[0]} ({wall:.0f}s), saved {OUT}",
      flush=True)

from llmopt.lab.gate import gate_checkpoint  # noqa: E402
solves, valid, tot = gate_checkpoint(
    str(OUT), 384, 8, 1536, 6, f"{TAG}_s{SEED}", device="mps")
with RECEIPTS.open("a") as f:
    f.write(json.dumps({
        "arm": TAG, "seed": SEED, "ratio": RATIO, "steps": _step[0],
        "solves": solves, "total": tot, "valid_pct": round(valid, 2),
        "train_wall_s": round(wall, 1), "device": "mps"}) + "\n")
print(f"[{TAG}] receipt appended to {RECEIPTS}", flush=True)
