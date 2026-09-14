"""ONECYCLE-SCHEDULER-COMPONENT-AUDIT-0 (Phase 1 of the 2026-09-14 GO):
zero-training, mechanical audit of WHAT the two booked writers' schedulers
actually changed in the AdamW param groups, under the installed torch.

The trainer law (scripts/train_mathnative.py, asserted by source segment
below): AdamW(model.parameters(), lr=LR, weight_decay=0.01) with the
torch defaults betas (0.9, 0.999), eps 1e-8; OneCycleLR(opt, max_lr=LR,
total_steps=steps_total, pct_start=0.03) with every other OneCycle
argument at its torch default; per step: backward, clip_grad_norm_ 1.0,
opt.step() (the phase19m tee saves {"model", "opt", "step"} AFTER the
AdamW update, inside step), sched.step() unless past total_steps - 1,
zero_grad. The booked births: LR 3e-4, steps_total 15,420 (3 epochs x
5,140 batches of the 164,490 encodable D2-excised rows).

The backward writer (scratch/birth19m_backsched.py) replaces OneCycleLR
by SequenceLR, a LambdaLR serving the stock per-step LR sequence
(computed on a dummy SGD optimizer), reversed when REVERSE=1; its no-op
precondition compared LR only. This audit reproduces both scheduler
laws on a dummy AdamW built with the trainer's constructor call, records
before every optimizer step (i.e. the values USED for step s, s = 1..N):
lr, betas[0], betas[1], weight_decay, and the scheduler's last_epoch,
for STOCK, BACKWARD (REVERSE=1) and PATCHED-FORWARD (REVERSE=0), and
answers mechanically:
  1. does stock OneCycle cycle beta1 on this AdamW?
  2. does SequenceLR leave beta1 fixed?
  3. is REVERSE=0 optimizer-law equivalent to stock, or LR-equivalent only?
  4. at the milestone steps of the geometry work, do the serialized
     optimizer param groups (lr, betas) equal the reconstructed values
     used for the just-completed step s?
It also writes the reconstruction law consumed by OPTIMIZER-GEOMETRY-DESK-0:
values used for step s are sequence[s - 1]; values for step s + 1 are
sequence[s] (the milestone at step s was serialized BEFORE sched.step()).
The SequenceLR / patched_onecycle bodies are copied verbatim from the
frozen writer (import has side effects) and asserted byte-equal by AST
segment. Outputs logs/schedaudit0/audit.json (refuse-if-exists) with the
full per-step tables for the three laws (15,420 x 3 rows) and the
milestone parity. Usage: .venv/bin/python scratch/onecycle_component_audit.py
"""
import ast
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
os.environ.setdefault("ARM", "off")
os.environ.setdefault("BIRTH_SEED", "0")

import torch  # noqa: E402

LR = 3e-4
TOTAL = 15_420
PCT = 0.03
WD = 0.01
GRID = [900, 3600, 7200, 10800, 13500]
A_PATHS = {s: f"checkpoints/phase19m/m{s:06d}.pt" for s in GRID}
B_PATHS = {s: f"checkpoints/backsched19m/m{s:06d}.pt" for s in GRID}
OUT = Path("logs/schedaudit0/audit.json")
TOL = 0.0                         # exact float equality is the registered bar for the serialized v reconstructed values

_orig_onecycle = torch.optim.lr_scheduler.OneCycleLR


# ---- verbatim copies of the frozen backward writer's scheduler bodies (asserted below)
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


def assert_verbatim():
    src = Path("scratch/birth19m_backsched.py").read_text()
    mine = Path(__file__).read_text()
    for name in ("_stock_lr_sequence", "SequenceLR"):
        def seg(text):
            tree = ast.parse(text)
            for n in tree.body:
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name:
                    return ast.get_source_segment(text, n)
            raise KeyError(name)
        assert seg(src) == seg(mine), f"{name} drifted from the frozen writer"
    tm = Path("scripts/train_mathnative.py").read_text()
    assert "opt = torch.optim.AdamW(model.parameters(), lr=lr,\n                            weight_decay=0.01)" in tm
    assert "sched = torch.optim.lr_scheduler.OneCycleLR(\n        opt, max_lr=lr, total_steps=steps_total, pct_start=0.03)" in tm
    assert "            loss.backward()\n            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)\n            opt.step()\n            if sched.last_epoch < steps_total - 1:\n                sched.step()" in tm
    assert "LR = 3e-4" in tm and "EPOCHS = 3" in tm
    ph = Path("scratch/birth19m_phase.py").read_text()
    assert "    out = _orig_opt_step(self, *a, **kw)\n    _step[0] += 1\n    if _step[0] == 1 or _step[0] % EVERY == 0:" in ph and '"opt": self.state_dict(), "step": _step[0]}' in ph
    return {"backsched_sha256": hashlib.sha256(src.encode()).hexdigest(), "train_mathnative_sha256": hashlib.sha256(tm.encode()).hexdigest(), "phase_sha256": hashlib.sha256(ph.encode()).hexdigest()}


def dummy_adamw():
    p = torch.nn.Parameter(torch.zeros(3))
    return torch.optim.AdamW([p], lr=LR, weight_decay=WD)     # the trainer's constructor call (defaults for betas / eps)


def table(kind):
    """Per-step optimizer fields USED for step s (read before opt.step), s = 1..TOTAL, plus the post-run state."""
    opt = dummy_adamw()
    if kind == "stock":
        sch = _orig_onecycle(opt, max_lr=LR, total_steps=TOTAL, pct_start=PCT)
    else:
        seq = _stock_lr_sequence(opt, LR, TOTAL, PCT)
        if kind == "backward":
            seq = seq[::-1]
        sch = SequenceLR(opt, seq)
    rows = []
    for s in range(1, TOTAL + 1):
        g = opt.param_groups[0]
        rows.append([s, float(g["lr"]), float(g["betas"][0]), float(g["betas"][1]), float(g["weight_decay"]), int(sch.last_epoch)])
        opt.step()
        if sch.last_epoch < TOTAL - 1:
            sch.step()
    g = opt.param_groups[0]
    post = {"lr": float(g["lr"]), "beta1": float(g["betas"][0]), "beta2": float(g["betas"][1]), "weight_decay": float(g["weight_decay"]), "last_epoch": int(sch.last_epoch), "keys": sorted(k for k in g if k != "params")}
    return rows, post


def summarize(rows):
    lr = [r[1] for r in rows]; b1 = [r[2] for r in rows]; b2 = [r[3] for r in rows]; wd = [r[4] for r in rows]
    return {"lr_min": min(lr), "lr_max": max(lr), "lr_first": lr[0], "lr_last": lr[-1], "lr_argmax_step": rows[lr.index(max(lr))][0],
            "beta1_min": min(b1), "beta1_max": max(b1), "beta1_first": b1[0], "beta1_last": b1[-1], "beta1_distinct": len(set(b1)), "beta1_argmin_step": rows[b1.index(min(b1))][0],
            "beta2_distinct": sorted(set(b2)), "wd_distinct": sorted(set(wd))}


def milestone_parity(paths, rows):
    out = {}
    for s, p in paths.items():
        d = torch.load(p, map_location="cpu")
        assert d["step"] == s, (p, d["step"])
        g = d["opt"]["param_groups"][0]
        used = rows[s - 1]; nxt = rows[s] if s < TOTAL else None
        st = d["opt"]["state"]
        steps = sorted({int(v["step"]) if not torch.is_tensor(v["step"]) else int(v["step"].item()) for v in st.values()})
        out[s] = {"path": p, "saved": {"lr": float(g["lr"]), "beta1": float(g["betas"][0]), "beta2": float(g["betas"][1]), "weight_decay": float(g["weight_decay"]), "eps": float(g["eps"]),
                                       "amsgrad": bool(g.get("amsgrad", False)), "maximize": bool(g.get("maximize", False)), "group_keys": sorted(k for k in g if k != "params")},
                  "reconstructed_used_for_step_s": {"lr": used[1], "beta1": used[2], "beta2": used[3], "weight_decay": used[4], "sched_last_epoch_before_step": used[5]},
                  "reconstructed_for_step_s_plus_1": ({"lr": nxt[1], "beta1": nxt[2], "beta2": nxt[3], "weight_decay": nxt[4]} if nxt else None),
                  "adam_step_counters": steps, "n_state_tensors": len(st)}
        o = out[s]
        o["parity_exact"] = (abs(o["saved"]["lr"] - used[1]) <= TOL and abs(o["saved"]["beta1"] - used[2]) <= TOL and abs(o["saved"]["beta2"] - used[3]) <= TOL and abs(o["saved"]["weight_decay"] - used[4]) <= TOL and steps == [s])
    return out


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())
    src = assert_verbatim()
    tables = {k: table(k) for k in ("stock", "backward", "patched_forward")}
    summ = {k: summarize(v[0]) for k, v in tables.items()}
    rows = {k: v[0] for k, v in tables.items()}
    lr_eq = all(a[1] == b[1] for a, b in zip(rows["stock"], rows["patched_forward"]))
    full_eq = all(a[1:5] == b[1:5] for a, b in zip(rows["stock"], rows["patched_forward"]))
    b1_diff_steps = sum(a[2] != b[2] for a, b in zip(rows["stock"], rows["backward"]))
    answers = {
        "q1_stock_cycles_beta1": summ["stock"]["beta1_distinct"] > 1,
        "q1_stock_beta1_range": [summ["stock"]["beta1_min"], summ["stock"]["beta1_max"]],
        "q2_sequencelr_beta1_fixed": summ["backward"]["beta1_distinct"] == 1 and summ["patched_forward"]["beta1_distinct"] == 1,
        "q2_sequencelr_beta1_value": summ["backward"]["beta1_first"],
        "q3_reverse0_lr_equivalent_to_stock": lr_eq,
        "q3_reverse0_optimizer_law_equivalent_to_stock": full_eq,
        "steps_where_stock_and_backward_beta1_differ": b1_diff_steps,
        "onecycle_defaults_used": {"cycle_momentum": True, "base_momentum": 0.85, "max_momentum": 0.95, "div_factor": 25.0, "final_div_factor": 1e4, "anneal_strategy": "cos", "three_phase": False},
    }
    parity = {"A": milestone_parity(A_PATHS, rows["stock"]), "B": milestone_parity(B_PATHS, rows["backward"])}
    parity_all = all(v["parity_exact"] for fam in parity.values() for v in fam.values())
    rec = {"prereg": "ONECYCLE-SCHEDULER-COMPONENT-AUDIT-0", "kind": "onecycle_component_audit", "commit": commit, "tree_dirty": dirty, "torch_version": torch.__version__,
           "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "asserted_sources": src, "law": {"LR": LR, "TOTAL": TOTAL, "PCT": PCT, "WD": WD, "adamw_defaults": {k: v for k, v in dummy_adamw().defaults.items() if k != "params"}},
           "reconstruction_law": "values USED for optimizer step s are row s (1-based) of the table, i.e. sequence[s - 1]; values for step s + 1 are sequence[s]; a milestone at step s is serialized after the AdamW update of step s and before sched.step()",
           "summary": summ, "post_run_param_group": {k: v[1] for k, v in tables.items()}, "answers": answers, "milestone_parity": parity, "milestone_parity_all_exact": parity_all,
           "tables": {k: v for k, v in rows.items()}, "written_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")}
    OUT.write_text(json.dumps(rec, indent=1))
    print(json.dumps({"answers": answers, "parity_all_exact": parity_all, "summary": summ}, indent=1))
    print(f"[schedaudit] written {OUT}")


if __name__ == "__main__":
    main()
