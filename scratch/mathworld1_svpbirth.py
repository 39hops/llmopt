"""MATH-CYBER-1 SVP-BIRTH-0 — the paired STATE-v-PROGRAM birth
driver (production) + the path-isolated MPS backward smoke.

PRODUCTION (SVPBIRTH_PRODUCTION=1, needs its own GO — NOT run
under the design GO): single lockstep process, both ~19M models
from the frozen shared init, same target-blind batch plan, 6,876
AdamW/OneCycle steps each, alternating per-step arm order,
prefix-masked per-row-normalized continuation CE (eos in T_i).
Hard assertions before step 0 (full-hash pins, bitwise init
equality, independent storage/optimizers, 73,324 unique row IDs,
0 cap violations, exactly 6,876 scheduled steps,
refuse-if-exists outputs). Outputs: checkpoints/svp_state.pt,
checkpoints/svp_program.pt, logs/mathworld1/svpbirth_receipt.json.

SMOKE (default): disposable copies of both models + optimizers/
schedulers on MPS; real frozen-artifact batches (high-memory
STATE batch, high-memory PROGRAM batch, the 12-row tail);
backward + clip + AdamW.step + OneCycleLR.step + zero_grad;
alternating order exercised; production init sha asserted
unchanged after; smoke outputs ONLY under smoke_ paths
(logs/mathworld1/smoke_svpbirth.json). Smoke batch selection MAY
inspect sequence lengths (operational stress only — the
production plan is untouched).

    .venv/bin/python scratch/mathworld1_svpbirth.py             (smoke)
    SVPBIRTH_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpbirth.py
"""
import hashlib
import json
import math
import os
import platform
import random
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402

PRODUCTION = os.environ.get("SVPBIRTH_PRODUCTION") == "1"


def gate(cond, msg):
    """Hard exit that survives python -O (never a bare assert)."""
    if not cond:
        raise SystemExit(f"GATE FAILED: {msg}")

PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b7"
        "5d8402351d468e8",
    "logs/mathworld1/svpeval/episodes.jsonl":
        "cb90ff0f6d655cfe5dc20f091da0597b1bb0e23a4d0c23355a"
        "997e8849c61dd8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f316524"
        "1406a86253858c6",
    "checkpoints/svp_init.pt":
        "18597944400e061f797755175d31a06690e378d8141c68760"
        "6724b95a7d0a86c",
}
PLAN_SHA = ("4c0441b7858349230a5d0791517712008d3b616704"
            "91dfcde8204a0a139d6f54")
BS = 32
EPOCHS = 3
CAP = 512
LR = 3e-4
TOTAL_STEPS = 6876
TOK = ActionGCTok()

CK_STATE = Path("checkpoints/svp_state.pt")
CK_PROGRAM = Path("checkpoints/svp_program.pt")
RECEIPT = Path("logs/mathworld1/svpbirth_receipt.json")
SMOKE_RECEIPT = Path("logs/mathworld1/smoke_svpbirth2.json")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def sha_b(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def batch_plan(row_ids):
    """The SVP-DESIGN-0 target-blind law, verbatim."""
    base = sorted(row_ids)
    plan = []
    for e in range(EPOCHS):
        order = list(base)
        random.Random(f"svp-epoch-{e}").shuffle(order)
        for s in range(0, len(order), BS):
            plan.append((e, s // BS, order[s:s + BS]))
    return plan


def encode_row(r, view):
    pre = TOK.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
    if view == "STATE":
        cont = TOK.encode(r["state_target"] + "\n") + [TOK.eos_id]
    else:
        cont = TOK.encode(r["program_text"]) + [TOK.eos_id]
    return pre, cont


def make_batch(rows, view, dev):
    enc = [encode_row(r, view) for r in rows]
    L = max(len(p) + len(c) for p, c in enc)
    ids = torch.full((len(enc), L), TOK.pad_id)
    mask = torch.zeros((len(enc), L), dtype=torch.long)
    for i, (p, c) in enumerate(enc):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p):len(p) + len(c)] = 1
    return ids.to(dev), mask.to(dev)


def masked_loss(logits, ids, mask):
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    m = mask[:, 1:].float()
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    return (-(tok_lp * m).sum(1) / m.sum(1)).mean()


def load_arms(dev):
    sd = torch.load(PINS_INIT, weights_only=True)
    arms = {}
    for view in ("STATE", "PROGRAM"):
        m = build_model(TOK.vocab_size, ctx=4096)
        m.load_state_dict(sd)
        arms[view] = m.to(dev)
    a, b = arms["STATE"], arms["PROGRAM"]
    gate(all(torch.equal(x.cpu(), y.cpu()) for (_, x), (_, y) in
             zip(sorted(a.state_dict().items()),
                 sorted(b.state_dict().items()))), "INIT UNEQUAL")
    gate(not any(x.data_ptr() == y.data_ptr()
                 for x, y in zip(a.parameters(), b.parameters())),
         "SHARED STORAGE")
    return arms


PINS_INIT = "checkpoints/svp_init.pt"


def preflight():
    for p, h in PINS.items():
        if fsha(p) != h:
            raise SystemExit(f"PIN MISMATCH {p}")
    rows = [json.loads(l) for l in open("data/matsub_paired.jsonl")]
    ids_all = [r["row_id"] for r in rows]
    gate(len(set(ids_all)) == 73324 == len(ids_all), "ROW IDS")
    plan = batch_plan(ids_all)
    gate(sha_b(json.dumps(plan).encode()) == PLAN_SHA, "PLAN SHA")
    gate(len(plan) == TOTAL_STEPS, "STEPS")
    over = 0
    for r in rows:
        for view in ("STATE", "PROGRAM"):
            p, c = encode_row(r, view)
            if len(p) + len(c) > CAP:
                over += 1
    gate(over == 0, f"CAP VIOLATIONS {over}")
    return rows, plan


def env_block(dev):
    return {"torch": torch.__version__,
            "python": sys.version.split()[0],
            "optimize_flag": sys.flags.optimize,
            "platform": platform.platform(),
            "mps_available": torch.backends.mps.is_available(),
            "mps_env": {k: v for k, v in os.environ.items()
                        if "MPS" in k or "PYTORCH" in k},
            "device": str(dev)}


def make_opt(model, total_steps):
    opt = torch.optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=LR, total_steps=total_steps, pct_start=0.03)
    return opt, sched


def train_step(model, opt, sched, ids, mask):
    logits = model(ids)
    loss = masked_loss(logits, ids, mask)
    gate(math.isfinite(float(loss)), "NON-FINITE LOSS")
    loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(
        model.parameters(), 1.0, error_if_nonfinite=True)
    gate(math.isfinite(float(gn)), "NON-FINITE GRAD NORM")
    opt.step()
    if sched.last_epoch < sched.total_steps - 1:
        sched.step()
    opt.zero_grad()
    return float(loss), float(gn)


def smoke():
    if SMOKE_RECEIPT.exists():
        raise SystemExit(f"REFUSING: {SMOKE_RECEIPT} exists")
    START = start_provenance(
        ["scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    rows, plan = preflight()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS UNAVAILABLE")
    dev = torch.device("mps")
    arms = load_arms(dev)
    on_mps = all(p.device.type == "mps"
                 for m in arms.values() for p in m.parameters())
    gate(on_mps, "NOT ON MPS")
    opts = {v: make_opt(arms[v], 3) for v in arms}
    by_id = {r["row_id"]: r for r in rows}
    # stress batches: longest-STATE batch, longest-PROGRAM batch,
    # the production 12-row tail (epoch 0 final chunk)
    slens = {r["row_id"]: sum(map(len, encode_row(r, "STATE")))
             for r in rows}
    plens = {r["row_id"]: sum(map(len, encode_row(r, "PROGRAM")))
             for r in rows}
    top_state = [by_id[i] for i in sorted(
        slens, key=slens.get, reverse=True)[:BS]]
    top_prog = [by_id[i] for i in sorted(
        plens, key=plens.get, reverse=True)[:BS]]
    tail = [by_id[i] for i in plan[2291][2]]
    gate(len(tail) == 12, "TAIL SIZE")
    gate(plan[2291][0] == 0, "TAIL EPOCH")
    before = {v: [p.detach().cpu().clone()
                  for p in arms[v].parameters()]
              for v in arms}
    results = []  # noqa
    t0 = time.time()
    for step, batch_rows in enumerate((top_state, top_prog, tail)):
        order = (("STATE", "PROGRAM") if step % 2 == 0
                 else ("PROGRAM", "STATE"))
        rec = {"batch": ["top_state", "top_prog", "tail12"][step],
               "order": list(order), "n_rows": len(batch_rows)}
        for view in order:
            ids, mask = make_batch(batch_rows, view, dev)
            loss, gn = train_step(arms[view], opts[view][0],
                                  opts[view][1], ids, mask)
            rec[view] = {"loss": loss, "grad_norm": gn,
                         "seq_len": int(ids.shape[1]),
                         "finite": (loss == loss
                                    and abs(loss) < 1e9
                                    and gn == gn)}
        results.append(rec)
    wall = round(time.time() - t0, 2)
    changed = {}
    for v in arms:
        deltas = [float((p.detach().cpu() - b).abs().max())
                  for p, b in zip(arms[v].parameters(), before[v])]
        changed[v] = sum(1 for d in deltas if d > 0)
    # arms diverged from each other (independent updates)
    diverged = any(
        not torch.equal(x.detach().cpu(), y.detach().cpu())
        for x, y in zip(arms["STATE"].parameters(),
                        arms["PROGRAM"].parameters()))
    # OneCycle guard law (historical birth): the scheduler is
    # stepped only while last_epoch < total_steps-1, so 3 train
    # steps at total=3 land last_epoch == 2 in both arms.
    step_counts = {v: int(opts[v][1].last_epoch) for v in arms}
    try:
        mps_alloc = int(torch.mps.current_allocated_memory())
    except Exception:
        mps_alloc = None
    init_after = fsha(PINS_INIT)
    receipt = {
        "mode": "smoke",
        "env": env_block(dev),
        "batches": results,
        "wall_s": wall,
        "params_changed": changed,
        "arms_diverged": diverged,
        "sched_step_counts": step_counts,
        "mps_allocated_bytes": mps_alloc,
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss // (1 << 20),
        "init_sha_after": init_after,
        "bars": {
            "ON_MPS": on_mps,
            "LOSSES_FINITE": all(
                r[v]["finite"] for r in results
                for v in ("STATE", "PROGRAM")),
            "GRADS_FINITE": all(
                math.isfinite(r[v]["grad_norm"])
                for r in results for v in ("STATE", "PROGRAM")),
            "PARAMS_CHANGED_INDEP": all(
                c > 0 for c in changed.values()) and diverged,
            "STEP_COUNTS_MATCH": step_counts["STATE"]
                == step_counts["PROGRAM"] == 2,
            "ALTERNATION_EXERCISED": [r["order"][0]
                                      for r in results]
                == ["STATE", "PROGRAM", "STATE"],
            "NO_OOM_FALLBACK": True,  # any OOM/error aborts run
            "INIT_UNCHANGED": init_after
                == PINS["checkpoints/svp_init.pt"],
            "NO_PRODUCTION_PATHS": not CK_STATE.exists()
                and not CK_PROGRAM.exists()
                and not RECEIPT.exists(),
        },
        "start": START, "completion_commit": completion_commit()}
    SMOKE_RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start",)}, indent=1),
          flush=True)
    return 0


def production():
    for p in (CK_STATE, CK_PROGRAM, RECEIPT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    rows, plan = preflight()
    if not torch.backends.mps.is_available():
        raise SystemExit("MPS UNAVAILABLE")
    dev = torch.device("mps")
    arms = load_arms(dev)
    opts = {v: make_opt(arms[v], TOTAL_STEPS) for v in arms}
    o_s, o_p = opts["STATE"][0], opts["PROGRAM"][0]
    gate(o_s is not o_p and opts["STATE"][1] is not opts[
        "PROGRAM"][1], "SHARED OPT")
    gate(o_s.defaults == o_p.defaults, "HYPERPARAM MISMATCH")
    by_id = {r["row_id"]: r for r in rows}
    losses = {"STATE": [], "PROGRAM": []}
    tok_counts = {"STATE": 0, "PROGRAM": 0}
    cont_tokens = {"STATE": 0, "PROGRAM": 0}
    walls = {"STATE": 0.0, "PROGRAM": 0.0}
    t0 = time.time()
    for step, (ep, bi, ids_chunk) in enumerate(plan):
        batch_rows = [by_id[i] for i in ids_chunk]
        order = (("STATE", "PROGRAM") if step % 2 == 0
                 else ("PROGRAM", "STATE"))
        for view in order:
            tv = time.time()
            ids, mask = make_batch(batch_rows, view, dev)
            loss, _ = train_step(arms[view], opts[view][0],
                                 opts[view][1], ids, mask)
            losses[view].append(loss)
            tok_counts[view] += int(ids.numel())
            cont_tokens[view] += int(mask.sum())
            walls[view] += time.time() - tv
        if step % 200 == 0:
            print(f"[svpbirth] step {step}/{TOTAL_STEPS} "
                  f"S={losses['STATE'][-1]:.4f} "
                  f"P={losses['PROGRAM'][-1]:.4f}", flush=True)
    # completion hard gates BEFORE any checkpoint is written
    for v in ("STATE", "PROGRAM"):
        gate(len(losses[v]) == TOTAL_STEPS,
             f"SHORT RUN {v} {len(losses[v])}")
        gate(all(math.isfinite(x) for x in losses[v]),
             f"NON-FINITE LOSS HISTORY {v}")
    gate(opts["STATE"][1].last_epoch
         == opts["PROGRAM"][1].last_epoch
         == TOTAL_STEPS - 1,
         "SCHED TERMINAL STATE (OneCycle guard law)")
    for v, ck in (("STATE", CK_STATE), ("PROGRAM", CK_PROGRAM)):
        torch.save({k: t.cpu() for k, t in
                    arms[v].state_dict().items()}, ck)
    receipt = {
        "mode": "production", "env": env_block(dev),
        "pins": {p: fsha(p) for p in PINS},
        "plan_sha": PLAN_SHA, "total_steps": TOTAL_STEPS,
        "sched_step_counts": {v: int(opts[v][1].last_epoch)
                              for v in arms},
        "epoch_mean_loss": {
            v: [round(sum(losses[v][e * 2292:(e + 1) * 2292])
                      / 2292, 5) for e in range(EPOCHS)]
            for v in losses},
        "preflight": {"rows": len(rows), "plan_sha": PLAN_SHA,
                      "cap_violations": 0,
                      "note": "invariants enforced by gate() hard "
                              "exits; receipt existence = pass"},
        "tokens_processed_padded": tok_counts,
        "continuation_target_tokens": cont_tokens,
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss // (1 << 20),
        "mps_allocated_bytes": (
            int(torch.mps.current_allocated_memory())
            if torch.backends.mps.is_available() else None),
        "arm_wall_s": {v: round(w, 1) for v, w in walls.items()},
        "wall_s": round(time.time() - t0, 1),
        "checkpoints": {str(CK_STATE): fsha(CK_STATE),
                        str(CK_PROGRAM): fsha(CK_PROGRAM)},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print("[svpbirth] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(production() if PRODUCTION else smoke())
