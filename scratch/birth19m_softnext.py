"""SOFT-NEXT-1 instrument (pre-reg RESULTS 2026-08-15): branch
distributions attached to conflicted rows. Two arms (ARM env:
control | soft), BIRTH_SEED 2, 15,420 steps, cuda bf16 autocast
(falls back mps/cpu for smoke tests). Same rows, batches, and step
count in both arms — the ONE variable is the training target at
branch tokens:

- control: one-hot CE on every position (the stock recipe).
- soft: for each conflicted cur (>= 2 distinct nxt in the diet), a
  token-level prefix trie over its distinct answers; on positions
  where this row's answer shares its prefix with siblings, the
  target is the empirical next-token distribution (weights = row
  counts); past the divergence point targets are one-hot again.
  Unconflicted rows (84.3% of the diet) are untouched.

Data-identity guards (NOT-RUN conditions from the pre-reg): the
pre-excision row count must equal 165,028 and the in-driver
conflict census must land within 1% of the booked 4,356 curs.

Post-train calibration probe (both arms, identical): valid-set
mass over a string-seeded 500-cur conflicted sample = mean total
teacher-forced probability assigned to the set of that cur's
distinct valid answers.

Usage: ARM=soft .venv/bin/python scratch/birth19m_softnext.py
"""
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

SN_ARM = os.environ.get("ARM", "")
assert SN_ARM in ("control", "soft"), SN_ARM
SEED = 2
SMOKE = os.environ.get("SMOKE", "") == "1"

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)
OUT = Path(f"checkpoints/gallery19m_softnext_{SN_ARM}_s{SEED}.pt")
if OUT.exists() and not SMOKE:
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
RECEIPTS = Path("logs/softnext1/arms.jsonl")
STEPS_TOTAL_PIN = 15_420
BOOKED_PRE_EXCISION = 165_028
BOOKED_CONFLICTED_CURS = 4_356
CAL_SAMPLE = 500


def build_conflict_tries(rows, tok):
    """Group diet rows by whitespace-stripped cur; for groups with
    >= 2 distinct nxt, build a token-prefix trie over the ENCODED
    answer tails with per-node next-token counts.

    Returns (census_curs, soft_map) where soft_map maps a row index
    -> list of (position_in_answer, [(token, prob), ...]) covering
    only shared-prefix positions."""
    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        groups["".join(str(r["cur"]).split())].append(idx)

    census = 0
    soft_map = {}
    for curkey, idxs in groups.items():
        distinct = defaultdict(list)   # nxt-text -> row idxs
        for i in idxs:
            distinct["".join(str(rows[i]["nxt"]).split())].append(i)
        if len(distinct) < 2:
            continue
        census += 1
        # encode each distinct answer ONCE (answer region tokens
        # only — the prompt region is shared by construction)
        seqs = []
        for ridxs in distinct.values():
            r = rows[ridxs[0]]
            try:
                ids = tok.encode(f"Step: {r['nxt']}\n") + [tok.eos_id]
            except ValueError:
                continue
            seqs.append((tuple(ids), len(ridxs), ridxs))
        if len(seqs) < 2:
            continue
        # per shared position, count mass per next token
        for s, _, ridxs in seqs:
            softs = []
            for pos in range(len(s)):
                # siblings sharing this row's prefix s[:pos]
                sharing = [(s2, w2) for s2, w2, _ in seqs
                           if s2[:pos] == s[:pos]]
                if len(sharing) < 2:
                    break   # diverged: one-hot from here on
                dist = defaultdict(float)
                for s2, w2 in sharing:
                    if pos < len(s2):
                        dist[s2[pos]] += w2
                tot = sum(dist.values())
                if tot <= 0:
                    break
                probs = [(t, c / tot) for t, c in dist.items()]
                if len(probs) > 1:
                    softs.append((pos, probs))
            if softs:
                for ri in ridxs:
                    soft_map[ri] = softs
    return census, soft_map


def main():
    tok = TM.MathTokenizer()
    rows_pre = C._STOCK_LOAD(True, True, True, True, False, False, None)
    n_pre = len(rows_pre)
    stock_rows = C.load_excised_rows()
    if not SMOKE:
        assert n_pre == BOOKED_PRE_EXCISION, \
            f"NOT-RUN: pre-excision {n_pre} != {BOOKED_PRE_EXCISION}"

    census, soft_map = build_conflict_tries(stock_rows, tok)
    print(f"[softnext] census: {census} conflicted curs "
          f"(booked {BOOKED_CONFLICTED_CURS}), soft rows "
          f"{len(soft_map)}", flush=True)
    if not SMOKE:
        assert abs(census - BOOKED_CONFLICTED_CURS) \
            <= 0.01 * BOOKED_CONFLICTED_CURS, "NOT-RUN: census drift"

    # trainer's encode path, with row index carried so soft targets
    # can be attached after the length sort
    triples = []
    for ri, r in enumerate(stock_rows):
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            # answer-region offset: position of the first token of
            # "Step: ..." within the full sequence
            head = f"Current: {r['cur']}\nHints: none\n"
            off = len(tok.encode(head))
            triples.append((ids, ri, off))
    triples.sort(key=lambda p: len(p[0]))
    enc = [p[0] for p in triples]
    row_of = [p[1] for p in triples]
    off_of = [p[2] for p in triples]
    enc_stock, _ = C.encode_with_levels(stock_rows, tok)
    assert len(enc) == len(enc_stock), (len(enc), len(enc_stock))

    C.assert_noop(enc_stock)    # precondition, fresh, in-process

    if SMOKE:
        enc = enc[:2 * BS]
        row_of = row_of[:2 * BS]
        off_of = off_of[:2 * BS]

    steps_per_epoch = len(C.stock_epoch_stream(len(enc_stock), 0))
    steps_total = EPOCHS * (len(enc_stock) // BS)
    if not SMOKE:
        assert steps_total == STEPS_TOTAL_PIN, steps_total

    dev = ("cuda" if torch.cuda.is_available() else
           "mps" if torch.backends.mps.is_available() else "cpu")
    use_bf16 = dev == "cuda"
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4,
        total_steps=(steps_total if not SMOKE else 2), pct_start=0.03)
    print(f"[softnext] arm={SN_ARM} dev={dev} bf16={use_bf16} "
          f"steps_total {steps_total}", flush=True)

    step = 0
    t0 = time.time()
    n_soft_positions = 0
    stream_all = []
    for ep in range(EPOCHS):
        s = C.stock_epoch_stream(len(enc), ep)
        stream_all.append(s[:steps_per_epoch] if not SMOKE else s[:2])
    for ep, stream in enumerate(stream_all):
        for a, b in stream:
            batch = enc[a:b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            with torch.autocast("cuda", torch.bfloat16,
                                enabled=use_bf16):
                logits = model(ids[:, :-1], mask[:, :-1])
            logits = logits.float()
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            logp = torch.log_softmax(logits, dim=-1)
            # one-hot CE, then subtract/add the soft corrections
            nll = torch.nn.functional.nll_loss(
                logp.reshape(-1, logp.shape[-1]),
                labels.reshape(-1), ignore_index=-100,
                reduction="sum")
            n_tok = (labels != -100).sum()
            if SN_ARM == "soft":
                corr = logits.new_zeros(())
                for bi in range(len(batch)):
                    softs = soft_map.get(row_of[a + bi])
                    if not softs:
                        continue
                    off = off_of[a + bi]
                    for pos, probs in softs:
                        # answer token at sequence index off+pos is
                        # PREDICTED at logits position off+pos-1
                        li = off + pos - 1
                        if li < 0 or li >= logp.shape[1]:
                            continue
                        onehot_tok = int(ids[bi, off + pos])
                        # replace -log p(onehot) with
                        # -sum_v q(v) log p(v)
                        corr = corr + logp[bi, li, onehot_tok]
                        for v, q in probs:
                            corr = corr - q * logp[bi, li, v]
                        n_soft_positions += 1
                loss = (nll + corr) / n_tok
            else:
                loss = nll / n_tok
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < sched.total_steps - 1:
                sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps_total} loss "
                      f"{float(loss.detach()):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)",
                      flush=True)

    if not SMOKE:
        torch.save(model.state_dict(), OUT)
    print(f"[softnext] saved {OUT} after {step} steps "
          f"({time.time()-t0:.0f}s), soft positions touched "
          f"{n_soft_positions}", flush=True)

    # calibration probe: valid-set mass on a fixed conflicted sample
    model.eval()
    groups = defaultdict(list)
    for r in stock_rows:
        groups["".join(str(r["cur"]).split())].append(r)
    conflicted = []
    for curkey, rs in groups.items():
        answers = {"".join(str(r["nxt"]).split()): r for r in rs}
        if len(answers) >= 2:
            conflicted.append((rs[0]["cur"], [r["nxt"] for r in
                               answers.values()]))
    conflicted.sort(key=lambda p: str(p[0]))
    random.Random("softnext-cal-1").shuffle(conflicted)
    sample = conflicted[:CAL_SAMPLE if not SMOKE else 5]
    masses = []
    with torch.no_grad():
        for cur, answers in sample:
            head = f"Current: {cur}\nHints: none\n"
            try:
                hid = tok.encode(head)
            except ValueError:
                continue
            mass = 0.0
            for nxt in answers:
                try:
                    aid = tok.encode(f"Step: {nxt}\n") + [tok.eos_id]
                except ValueError:
                    continue
                full = hid + aid
                ids = torch.tensor([full], device=dev)
                am = torch.ones_like(ids)
                with torch.autocast("cuda", torch.bfloat16,
                                    enabled=use_bf16):
                    lg = model(ids[:, :-1], am[:, :-1])
                lp = torch.log_softmax(lg.float(), -1)
                s = 0.0
                for t_i in range(len(hid) - 1, len(full) - 1):
                    s += float(lp[0, t_i, full[t_i + 1]])
                mass += math.exp(s)
            masses.append(mass)
    valid_mass = sum(masses) / max(len(masses), 1)

    from llmopt.lab.gate import gate_eval
    from llmopt.lab.hash import git_sha
    solves, valid = gate_eval(model, tok, dev)
    tot = sum(solves.values())
    RECEIPTS.parent.mkdir(parents=True, exist_ok=True)
    row = {"arm": SN_ARM, "seed": SEED, "steps": step,
           "solves": solves, "total": tot,
           "valid_pct": round(valid, 2), "device": dev,
           "bf16": use_bf16, "census_curs": census,
           "n_soft_rows": len(soft_map),
           "valid_set_mass": round(valid_mass, 4),
           "cal_sample": len(masses),
           "code_commit": git_sha(short=True)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[softnext] GATE arm={SN_ARM} {solves} = {tot}/120 "
          f"@ {valid:.2f}% | valid_set_mass {valid_mass:.4f}",
          flush=True)


if __name__ == "__main__":
    main()
