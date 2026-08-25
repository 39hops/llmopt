"""MATH-CYBER-1 STATE-v-PROGRAM DESIGN-0 — design-qualification
smoke harness. Verifies the five frozen design mechanisms with
ZERO training: (1) frozen-init determinism + pin, (2) target-blind
batch-plan determinism, (3) sequence-cap census, (4) prefix-masked
per-row-normalized CE arithmetic v an independent hand loop,
(5) one forward pass per arm (no backward, no optimizer).

Receipt logs/mathworld1/svpdesign_receipt.json (refuse-if-exists).
Init artifact checkpoints/svp_init.pt (refuse-if-exists;
untrained by construction — saved before any forward).

    .venv/bin/python scratch/mathworld1_svpdesign.py          (Mac)
"""
import hashlib
import io
import json
import math
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

PAIRED = Path("data/matsub_paired.jsonl")
PAIRED_SHA = ("a943ba7fc581db743b07192e5d951fadd"
              "dd2ba19bca3225b75d8402351d468e8")
INIT = Path("checkpoints/svp_init.pt")
RECEIPT = Path("logs/mathworld1/svpdesign_receipt.json")
BS = 32
EPOCHS = 3
CAP = 512
SEED = 9001


def sha_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def fsha(p) -> str:
    return sha_bytes(Path(p).read_bytes())


def state_bytes(model) -> bytes:
    buf = io.BytesIO()
    torch.save({k: v.cpu() for k, v in
                sorted(model.state_dict().items())}, buf)
    return buf.getvalue()


def batch_plan(row_ids):
    """Target-blind plan: row_id-sorted, per-epoch seeded shuffle,
    consecutive BS chunks, tail retained."""
    base = sorted(row_ids)
    plan = []
    for e in range(EPOCHS):
        order = list(base)
        random.Random(f"svp-epoch-{e}").shuffle(order)
        for s in range(0, len(order), BS):
            plan.append((e, s // BS, order[s:s + BS]))
    return plan


def masked_loss(logits, ids, mask):
    """Prefix-masked per-row-normalized continuation CE.
    ids [B,L]; mask [B,L] 1 on continuation targets; logits
    predict ids[t] from position t-1."""
    logp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    tgt = ids[:, 1:]
    m = mask[:, 1:].float()
    tok_lp = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
    per_row = -(tok_lp * m).sum(1) / m.sum(1)
    return per_row.mean(), per_row


def encode_row(tok, r, view):
    pre = tok.encode(f"Current: {r['cur']}\nHints: none\nStep: ")
    if view == "STATE":
        cont = tok.encode(r["state_target"] + "\n") + [tok.eos_id]
    else:
        cont = tok.encode(r["program_text"]) + [tok.eos_id]
    return pre, cont


def main():
    for p in (RECEIPT, INIT):
        if p.exists():
            raise SystemExit(f"REFUSING: {p} exists")
    if fsha(PAIRED) != PAIRED_SHA:
        raise SystemExit("PAIRED ARTIFACT SHA MISMATCH")
    START = start_provenance(
        ["scratch/mathworld1_svpdesign.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    tok = ActionGCTok()
    rows = [json.loads(l) for l in open(PAIRED)]

    # (1) init determinism + pin
    torch.manual_seed(SEED)
    m1 = build_model(tok.vocab_size, ctx=4096)
    torch.manual_seed(SEED)
    m2 = build_model(tok.vocab_size, ctx=4096)
    b1, b2 = state_bytes(m1), state_bytes(m2)
    init_ok = b1 == b2
    INIT.parent.mkdir(exist_ok=True)
    INIT.write_bytes(b1)
    n_params = sum(p.numel() for p in m1.parameters())
    # both-arms load + tensor equality (the birth's step-0 assert)
    sd = torch.load(INIT, weights_only=True)
    mA = build_model(tok.vocab_size, ctx=4096)
    mB = build_model(tok.vocab_size, ctx=4096)
    mA.load_state_dict(sd)
    mB.load_state_dict(sd)
    load_eq = all(torch.equal(a, b) for (_, a), (_, b) in zip(
        sorted(mA.state_dict().items()),
        sorted(mB.state_dict().items())))

    # (2) batch plan determinism (target-blind by construction)
    ids_all = [r["row_id"] for r in rows]
    if len(set(ids_all)) != len(ids_all):
        raise SystemExit("ROW_ID NOT UNIQUE")
    p1 = batch_plan(ids_all)
    p2 = batch_plan(ids_all)
    plan_blob = json.dumps(p1).encode()
    plan_ok = plan_blob == json.dumps(p2).encode()
    steps_per_epoch = sum(1 for (e, _, _) in p1 if e == 0)
    tail = len(p1[-1][2])

    # (3) full cap census, both views
    mx_pre = mx_st = mx_pg = over = 0
    st_tok_total = pg_tok_total = 0
    for r in rows:
        pre, st = encode_row(tok, r, "STATE")
        _, pg = encode_row(tok, r, "PROGRAM")
        mx_pre = max(mx_pre, len(pre))
        mx_st = max(mx_st, len(pre) + len(st))
        mx_pg = max(mx_pg, len(pre) + len(pg))
        st_tok_total += len(st)
        pg_tok_total += len(pg)
        if len(pre) + len(st) > CAP or len(pre) + len(pg) > CAP:
            over += 1

    # (4) loss arithmetic v independent hand loop (fp64, 3 rows)
    torch.manual_seed(0)
    dev = "cpu"
    sample = rows[:3]
    batch = []
    for r in sample:
        pre, cont = encode_row(tok, r, "PROGRAM")
        batch.append((pre, cont))
    L = max(len(p) + len(c) for p, c in batch)
    ids = torch.full((len(batch), L), tok.pad_id)
    mask = torch.zeros((len(batch), L), dtype=torch.long)
    for i, (p, c) in enumerate(batch):
        ids[i, :len(p) + len(c)] = torch.tensor(p + c)
        mask[i, len(p):len(p) + len(c)] = 1
    small = build_model(tok.vocab_size, d=64, layers=2, heads=2,
                        ffn=128, ctx=CAP).to(dev)
    with torch.no_grad():
        logits = small(ids)
        loss, per_row = masked_loss(logits, ids, mask)
        # independent hand loop
        hand = []
        logp = torch.log_softmax(logits[:, :-1].double(), -1)
        for i, (p, c) in enumerate(batch):
            acc = 0.0
            for t in range(len(p), len(p) + len(c)):
                acc += logp[i, t - 1, ids[i, t]].item()
            hand.append(-acc / len(c))
    loss_ok = all(abs(a - b.item()) < 1e-4
                  for a, b in zip(hand, per_row))
    prefix_masked_ok = int(mask[:, :1].sum()) == 0 and all(
        mask[i, :len(p)].sum().item() == 0
        for i, (p, _) in enumerate(batch))

    # (5) one forward per arm on one REAL batch, no backward
    plan_rows = {r["row_id"]: r for r in rows}
    first = [plan_rows[i] for i in p1[0][2]]
    fwd = {}
    for view, model in (("STATE", mA), ("PROGRAM", mB)):
        bt = [encode_row(tok, r, view) for r in first]
        L = max(len(p) + len(c) for p, c in bt)
        ids = torch.full((len(bt), L), tok.pad_id)
        mask = torch.zeros((len(bt), L), dtype=torch.long)
        for i, (p, c) in enumerate(bt):
            ids[i, :len(p) + len(c)] = torch.tensor(p + c)
            mask[i, len(p):len(p) + len(c)] = 1
        t0 = time.time()
        with torch.no_grad():
            loss, _ = masked_loss(model(ids), ids, mask)
        fwd[view] = {"loss": float(loss), "seq_len": L,
                     "wall_s": round(time.time() - t0, 2),
                     "finite": math.isfinite(float(loss))}

    peak_mb = resource.getrusage(
        resource.RUSAGE_SELF).ru_maxrss // (1 << 20)  # bytes on mac
    receipt = {
        "init": {"bitwise_repeat_ok": init_ok,
                 "both_arms_load_equal": load_eq,
                 "sha256": sha_bytes(b1), "n_params": n_params,
                 "path": str(INIT)},
        "batch_plan": {"deterministic_ok": plan_ok,
                       "sha256": sha_bytes(plan_blob),
                       "steps_per_epoch": steps_per_epoch,
                       "total_steps": len(p1),
                       "tail_rows": tail, "bs": BS,
                       "epochs": EPOCHS},
        "cap_census": {"rows": len(rows), "cap": CAP,
                       "max_prefix": mx_pre,
                       "max_state_seq": mx_st,
                       "max_program_seq": mx_pg,
                       "over_cap_either": over,
                       "state_cont_tokens_total": st_tok_total,
                       "program_cont_tokens_total": pg_tok_total},
        "loss_check": {"hand_loop_agree": loss_ok,
                       "prefix_fully_masked": prefix_masked_ok,
                       "per_row": [round(h, 6) for h in hand]},
        "forward": fwd,
        "peak_rss_mb": peak_mb,
        "bars": {
            "INIT": init_ok and load_eq,
            "BATCH_PLAN": plan_ok and tail == 12
                and len(p1) == 6876,
            "CAP": over == 0 and mx_st == 441 and mx_pg == 223,
            "LOSS": loss_ok and prefix_masked_ok,
            "FORWARD": all(v["finite"] for v in fwd.values()),
        },
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k != "start"}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
