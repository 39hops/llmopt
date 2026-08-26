"""MATH-CYBER-1 SVP-FACTOR-HASH-BIRTH-0 — the frozen three-arm
mechanism birth: CANONICAL-340 / FACTOR-OPAQUE / HASH-OPAQUE
from ONE bit-identical shared vocab-340 init (seed 12001), one
target-blind batch plan, one frozen optimization law. ZERO
scoring; no dependency on the third band anywhere in this file.

Law by IMPORT from the frozen svpbirth driver: batch_plan,
masked_loss, make_opt, train_step, gate (the -HARDEN
finite/clip/scheduler laws travel inside train_step). FACTOR/
HASH encodings from the qualified svpcode module (ad7df05c).
Code atoms occupy ids 332..339 of the vocab-340 model; the
CANONICAL arm uses the same model shape and never emits them.

Targets per SAME row: C = canonical program_text + EOS;
F = factor_symbols(tuple) + EOS; H = hash_symbols(tuple) + EOS
(T=9 exactly, gated per row with exact semantic inverse).
Execution rotation per step: step%3==0 C,F,H; ==1 F,H,C;
==2 H,C,F — the same batch is consumed by all three arms before
the plan advances. Each arm owns its optimizer/scheduler.

Paths (refuse-if-exists):
  checkpoints/svp_fh_init_s12001.pt        (shared init)
  checkpoints/svp_fh_canonical_s12001.pt
  checkpoints/svp_fh_factor_s12001.pt
  checkpoints/svp_fh_hash_s12001.pt
  logs/mathworld1/svpfhbirth_s12001_receipt.json
Smoke (isolated): logs/mathworld1/smoke_svpfhbirth.json — no
production path is written in smoke mode.

    .venv/bin/python scratch/mathworld1_svpfhbirth.py          (smoke)
    SVPFH_PRODUCTION=1 .venv/bin/python scratch/mathworld1_svpfhbirth.py
"""
import hashlib
import io
import json
import math
import os
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
from scratch.mathworld1_svpbirth import (batch_plan,  # noqa: E402
                                         gate, make_opt, sha_b,
                                         train_step)
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        factor_symbols,
                                        hash_decode, hash_symbols,
                                        in_domain)

SEED = 12001
VOCAB = 340
CODE_BASE = 332  # <c:s> -> id 332+s
PAIRED = "data/matsub_paired.jsonl"
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
PLAN_SHA = ("4c0441b7858349230a5d0791517712008d3b616704"
            "91dfcde8204a0a139d6f54")
EPOCHS = 3
BS = 32
TOTAL_STEPS = 6876
ARMS = ["CANONICAL", "FACTOR", "HASH"]
ROT = {0: ["CANONICAL", "FACTOR", "HASH"],
       1: ["FACTOR", "HASH", "CANONICAL"],
       2: ["HASH", "CANONICAL", "FACTOR"]}
INIT_CK = Path("checkpoints/svp_fh_init_s12001.pt")
CKS = {"CANONICAL": Path("checkpoints/svp_fh_canonical_s12001.pt"),
       "FACTOR": Path("checkpoints/svp_fh_factor_s12001.pt"),
       "HASH": Path("checkpoints/svp_fh_hash_s12001.pt")}
RECEIPT = Path("logs/mathworld1/svpfhbirth_s12001_receipt.json")
SMOKE_RECEIPT = Path("logs/mathworld1/smoke_svpfhbirth.json")
TOK = ActionGCTok()
PRODUCTION = os.environ.get("SVPFH_PRODUCTION") == "1"


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def state_bytes(seed):
    torch.manual_seed(seed)
    m = build_model(VOCAB, ctx=4096)
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(m.state_dict().items())}, buf)
    return buf.getvalue()


def ensure_init():
    if not INIT_CK.exists():
        b1 = state_bytes(SEED)
        b2 = state_bytes(SEED)
        gate(b1 == b2, "INIT NONDETERMINISTIC")
        INIT_CK.write_bytes(b1)
    return fsha(INIT_CK)


def load_rows():
    gate(fsha(PAIRED) == PAIRED_SHA, "PAIRED PIN")
    rows = [json.loads(l) for l in open(PAIRED)]
    gate(len(rows) == 73324, "ROWS")
    for r in rows:
        tup = (r["rule"], r["site_kind"], r["site_ordinal"],
               r["param_kind"], r["param_index"])
        gate(in_domain(*tup), "ROW OUT OF DOMAIN")
        fs = factor_symbols(*tup)
        hs = hash_symbols(*tup)
        gate(factor_decode(fs) == tup, "F INVERSE")
        gate(hash_decode(hs) == tup, "H INVERSE")
        r["_F"] = [CODE_BASE + s for s in fs] + [TOK.eos_id]
        r["_H"] = [CODE_BASE + s for s in hs] + [TOK.eos_id]
        gate(len(r["_F"]) == 9 and len(r["_H"]) == 9, "T!=9")
    return rows


def encode_row(r, view):
    pre = TOK.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
    if view == "CANONICAL":
        cont = TOK.encode(r["program_text"]) + [TOK.eos_id]
    elif view == "FACTOR":
        cont = r["_F"]
    else:
        cont = r["_H"]
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


def load_arm_models(dev, init_sha):
    gate(fsha(INIT_CK) == init_sha, "INIT MUTATED")
    sd = torch.load(INIT_CK, weights_only=True)
    arms = {}
    n_params = None
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(sd)
        for k, v in m.state_dict().items():
            gate(torch.equal(v.cpu(), sd[k]), f"INIT DRIFT {a}")
        np_ = sum(p.numel() for p in m.parameters())
        if n_params is None:
            n_params = np_
        gate(np_ == n_params, "PARAM COUNT DRIFT")
        arms[a] = m.to(dev)
    return arms, n_params


def run(plan, rows_by_id, dev, init_sha, n_steps, tag):
    arms, n_params = load_arm_models(dev, init_sha)
    opts = {a: make_opt(arms[a], n_steps) for a in ARMS}
    losses = {a: [] for a in ARMS}
    tok_padded = {a: 0 for a in ARMS}
    tok_cont = {a: 0 for a in ARMS}
    order_counts = {i: 0 for i in range(3)}
    wall = {a: 0.0 for a in ARMS}
    for step, (e, bi, ids_) in enumerate(plan[:n_steps]):
        rows = [rows_by_id[i] for i in ids_]
        order = ROT[step % 3]
        order_counts[step % 3] += 1
        for a in order:
            t0 = time.monotonic()
            ids, mask = make_batch(rows, a, dev)
            tok_padded[a] += int(ids.numel())
            tok_cont[a] += int(mask.sum())
            opt, sched = opts[a]
            l, _ = train_step(arms[a], opt, sched, ids, mask)
            losses[a].append(l)
            wall[a] += time.monotonic() - t0
        counts = {a: opts[a][1].last_epoch for a in ARMS}
        gate(len(set(counts.values())) == 1,
             f"STEP-COUNT DRIFT {counts}")
        if step % 200 == 0 or step == n_steps - 1:
            print(f"[svpfh:{tag}] step {step}/{n_steps} " +
                  " ".join(f"{a[0]}={losses[a][-1]:.4f}"
                           for a in ARMS), flush=True)
    for a in ARMS:
        gate(len(losses[a]) == n_steps, f"LOSS COUNT {a}")
        gate(all(math.isfinite(x) for x in losses[a]),
             f"NON-FINITE {a}")
        gate(opts[a][1].last_epoch == n_steps - 1,
             f"SCHED TERMINAL {a}")
    return arms, losses, tok_padded, tok_cont, order_counts, \
        wall, n_params


def main():
    if PRODUCTION:
        for p in list(CKS.values()) + [RECEIPT]:
            if p.exists():
                raise SystemExit(f"REFUSING: {p} exists")
    else:
        if SMOKE_RECEIPT.exists():
            raise SystemExit(f"REFUSING: {SMOKE_RECEIPT} exists")
    START = start_provenance(
        ["scratch/mathworld1_svpfhbirth.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    gate(torch.backends.mps.is_available(), "MPS UNAVAILABLE")
    dev = torch.device("mps")
    init_sha = ensure_init()
    rows = load_rows()
    rows_by_id = {r["row_id"]: r for r in rows}
    plan = batch_plan(list(rows_by_id))
    gate(len(plan) == TOTAL_STEPS, "PLAN LEN")
    plan_sha = sha_b(json.dumps(plan).encode())
    gate(plan_sha == PLAN_SHA, "PLAN SHA")

    t0 = time.time()
    if not PRODUCTION:
        n = 2
        arms, losses, tp, tc, oc, wall, n_params = run(
            plan, rows_by_id, dev, init_sha, n, "smoke")
        receipt = {
            "mode": "smoke", "seed": SEED, "vocab": VOCAB,
            "n_params": n_params, "init_sha": init_sha,
            "plan_sha": plan_sha, "steps": n,
            "losses": losses, "tokens_padded": tp,
            "tokens_continuation": tc,
            "order_counts": {str(k): v for k, v in oc.items()},
            "bars": {
                "INIT_BITWISE_ALL_ARMS": True,
                "VOCAB_340": VOCAB == 340,
                "PARAM_EQUAL": True,
                "PLAN_SHA_MATCH": plan_sha == PLAN_SHA,
                "FH_T9_INVERSE": True,
                "ROTATION_EXERCISED": oc[0] >= 1 and oc[1] >= 1,
                "FINITE": True,
                "STEP_LOCKSTEP": True,
                "NO_PRODUCTION_PATHS": not any(
                    p.exists() for p in CKS.values())
                and not RECEIPT.exists(),
            },
            "wall_s": round(time.time() - t0, 1),
            "start": START,
            "completion_commit": completion_commit()}
        SMOKE_RECEIPT.write_text(json.dumps(receipt, indent=1))
        print(json.dumps(receipt["bars"], indent=1), flush=True)
        return 0

    arms, losses, tp, tc, oc, wall, n_params = run(
        plan, rows_by_id, dev, init_sha, TOTAL_STEPS, "prod")
    # completion gates BEFORE checkpoint writes
    for a in ARMS:
        gate(len(losses[a]) == TOTAL_STEPS, f"INCOMPLETE {a}")
    for a in ARMS:
        torch.save({k: v.cpu() for k, v in
                    sorted(arms[a].state_dict().items())}, CKS[a])
    epoch_mean = {a: [round(sum(losses[a][e*2292:(e+1)*2292])
                            / 2292, 5) for e in range(3)]
                  for a in ARMS}
    receipt = {
        "mode": "production", "seed": SEED, "vocab": VOCAB,
        "n_params": n_params,
        "init_sha": init_sha, "plan_sha": plan_sha,
        "total_steps": TOTAL_STEPS,
        "updates_per_arm": {a: TOTAL_STEPS for a in ARMS},
        "sched_terminal": {a: TOTAL_STEPS - 1 for a in ARMS},
        "epoch_mean_loss": epoch_mean,
        "tokens_padded": tp, "tokens_continuation": tc,
        "order_counts": {str(k): v for k, v in oc.items()},
        "arm_wall_s": {a: round(wall[a], 1) for a in ARMS},
        "wall_s": round(time.time() - t0, 1),
        "peak_rss_mb": resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss // (1024 * 1024),
        "mps_allocated_bytes":
            torch.mps.current_allocated_memory(),
        "env": {"torch": torch.__version__,
                "device": str(dev)},
        "checkpoints": {str(CKS[a]): fsha(CKS[a]) for a in ARMS},
        "pins": {PAIRED: fsha(PAIRED),
                 str(INIT_CK): fsha(INIT_CK)},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print("[svpfh] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
