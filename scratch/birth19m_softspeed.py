"""SOFT-SPEED-1 instrument (pre-reg RESULTS 2026-08-15, commit
be3b8e4): collapse each conflicted cur group (>= 2 distinct nxt in
the excised diet) into ONE soft-target row and train MATCHED EPOCHS,
so the ~13% diet cut becomes a ~13% step cut. Two arms (ARM env):

- control: the frozen ladder stock recipe verbatim (plain CE,
  mean reduction, full excised diet, 15,420 steps) — must
  reproduce the booked stock s3 cell exactly, 64/120
  {3:24, 4:6, 5:15, 6:8, 7:11} sha bf2dc94b1d9712cb (L29465).
- soft: collapsed diet. Per conflicted group the representative is
  the highest-weight answer (weight = rows carrying that
  whitespace-stripped nxt; ties lexically smallest); soft targets
  over shared-prefix positions are the group's empirical
  next-token distribution (the frozen SOFT-NEXT trie
  construction), one-hot past divergence; the row's answer-region
  loss is weighted by the group's TOTAL row count so per-epoch
  expected gradient mass at branch positions matches the
  uncollapsed diet. Tail supervision past divergence is
  representative-only (disclosed in the pre-reg fences).
  OneCycleLR total_steps rescales to the soft arm's own count.

NOT-RUN guards (from the pre-reg): pre-excision rows == 165,028;
census within 1% of 4,347 groups / 25,852 conflicted rows;
collapsed diet within 1% of 143,391. train_seconds lands in the
receipt for the wall-clock half of BAR 1.

Usage: ARM=soft SEED=3 .venv/bin/python scratch/birth19m_softspeed.py
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

SS_ARM = os.environ.get("ARM", "")
SEED = int(os.environ.get("SEED", "3"))
assert SS_ARM in ("control", "soft"), SS_ARM
SMOKE = os.environ.get("SMOKE", "") == "1"

os.environ["ARM"] = "off"       # frozen module import side-effects only
os.environ["BIRTH_SEED"] = str(SEED)
OUT = Path(f"checkpoints/gallery19m_softspeed_{SS_ARM}_s{SEED}.pt")
if OUT.exists() and not SMOKE:
    raise SystemExit(f"REFUSING: {OUT} exists")

import torch  # noqa: E402

import birth19m_curric as C  # noqa: E402  (frozen, import-only)
import train_mathnative as TM  # noqa: E402

EPOCHS, BS = C.EPOCHS, C.BS
RECEIPTS = Path("logs/softspeed1/arms.jsonl")
STEPS_TOTAL_PIN = 15_420
BOOKED_PRE_EXCISION = 165_028
BOOKED_GROUPS = 4_347
BOOKED_CONF_ROWS = 25_852
BOOKED_COLLAPSED = 143_391
CAL_SAMPLE = 500


def conflict_groups(rows):
    """cur-key -> list of row indices, conflicted keys only."""
    groups = defaultdict(list)
    for idx, r in enumerate(rows):
        groups["".join(str(r["cur"]).split())].append(idx)
    out = {}
    for k, idxs in groups.items():
        distinct = {"".join(str(rows[i]["nxt"]).split()) for i in idxs}
        if len(distinct) >= 2:
            out[k] = idxs
    return out


def collapse(rows, tok):
    """-> (soft_rows, weight_of, soft_targets_of, n_conf_rows)

    soft_rows: the collapsed diet (unconflicted rows verbatim, one
    representative row per conflicted group). weight_of /
    soft_targets_of are keyed by index into soft_rows; targets are
    [(pos_in_answer, [(token, prob), ...])] over shared-prefix
    positions, built exactly as the frozen SOFT-NEXT trie."""
    conf = conflict_groups(rows)
    conf_idx = {i for idxs in conf.values() for i in idxs}
    soft_rows, weight_of, targets_of = [], {}, {}
    for i, r in enumerate(rows):
        if i not in conf_idx:
            soft_rows.append(r)
    for k in sorted(conf):            # deterministic order
        idxs = conf[k]
        distinct = defaultdict(list)  # stripped nxt -> row idxs
        for i in idxs:
            distinct["".join(str(rows[i]["nxt"]).split())].append(i)
        # encode each distinct answer once (answer region only)
        seqs = []
        for key in sorted(distinct):
            ridxs = distinct[key]
            r = rows[ridxs[0]]
            try:
                ids = tok.encode(f"Step: {r['nxt']}\n") + [tok.eos_id]
            except ValueError:
                continue
            seqs.append((tuple(ids), len(ridxs), r))
        if len(seqs) < 2:
            # group degenerates after encode: keep all rows as-is
            for i in idxs:
                soft_rows.append(rows[i])
            continue
        # representative: max weight, ties lexically smallest nxt
        best_w = max(s[1] for s in seqs)
        rep = min((s for s in seqs if s[1] == best_w),
                  key=lambda s: "".join(str(s[2]["nxt"]).split()))
        s_rep = rep[0]
        softs = []
        for pos in range(len(s_rep)):
            sharing = [(s2, w2) for s2, w2, _ in seqs
                       if s2[:pos] == s_rep[:pos]]
            if len(sharing) < 2:
                break
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
        ri = len(soft_rows)
        soft_rows.append(rep[2])
        weight_of[ri] = float(len(idxs))
        if softs:
            targets_of[ri] = softs
    return soft_rows, weight_of, targets_of, len(conf_idx), len(conf)


def encode_indexed(rows, tok):
    """encode_with_levels' text/filter path with the source row
    index and answer-region offset carried through the length sort."""
    triples = []
    for ri, r in enumerate(rows):
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = tok.encode(t) + [tok.eos_id]
        except ValueError:
            continue
        if len(ids) <= int(os.environ.get("SEQ_CAP", "512")):
            head = f"Current: {r['cur']}\nHints: none\n"
            triples.append((ids, ri, len(tok.encode(head))))
    triples.sort(key=lambda p: len(p[0]))
    return ([p[0] for p in triples], [p[1] for p in triples],
            [p[2] for p in triples])


def main():
    tok = TM.MathTokenizer()
    rows_pre = C._STOCK_LOAD(True, True, True, True, False, False, None)
    if not SMOKE:
        assert len(rows_pre) == BOOKED_PRE_EXCISION, \
            f"NOT-RUN: pre-excision {len(rows_pre)}"
    stock_rows = C.load_excised_rows()
    enc_stock, _ = C.encode_with_levels(stock_rows, tok)

    C.assert_noop(enc_stock)    # precondition, fresh, in-process

    if SS_ARM == "control":
        enc = enc_stock
        row_of = off_of = None
        weight_of, targets_of = {}, {}
        steps_total = EPOCHS * (len(enc_stock) // BS)
        if not SMOKE:
            assert steps_total == STEPS_TOTAL_PIN, steps_total
    else:
        soft_rows, weight_of, targets_of, n_conf, n_groups = \
            collapse(stock_rows, tok)
        print(f"[softspeed] census: {n_groups} groups, {n_conf} "
              f"conflicted rows, collapsed diet {len(soft_rows)}",
              flush=True)
        if not SMOKE:
            assert abs(n_groups - BOOKED_GROUPS) <= 0.01 * BOOKED_GROUPS
            assert abs(n_conf - BOOKED_CONF_ROWS) \
                <= 0.01 * BOOKED_CONF_ROWS
            assert abs(len(soft_rows) - BOOKED_COLLAPSED) \
                <= 0.01 * BOOKED_COLLAPSED
        enc, row_of, off_of = encode_indexed(soft_rows, tok)
        steps_total = EPOCHS * (len(enc) // BS)

    if SMOKE:
        enc = enc[:2 * BS]
        if row_of:
            row_of = row_of[:2 * BS]
            off_of = off_of[:2 * BS]
        steps_total = 2

    dev = ("mps" if torch.backends.mps.is_available() else
           "cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    model = TM.build_model(len(tok.vocab), d=384, layers=8,
                           heads=6, ffn=1536).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4,
                            weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=3e-4, total_steps=steps_total, pct_start=0.03)
    print(f"[softspeed] arm={SS_ARM} seed={SEED} dev={dev} train "
          f"{len(enc)} seq, steps_total {steps_total}", flush=True)

    step = 0
    t0 = time.time()
    n_soft_positions = 0
    steps_per_epoch = steps_total // EPOCHS if not SMOKE else 2
    for ep in range(EPOCHS):
        stream = C.stock_epoch_stream(len(enc), ep)[:steps_per_epoch]
        for a, b in stream:
            batch = enc[a:b]
            L = max(len(s) for s in batch)
            ids = torch.tensor([s + [tok.pad_id] * (L - len(s))
                                for s in batch], device=dev)
            mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                                 for s in batch], device=dev)
            logits = model(ids[:, :-1], mask[:, :-1])
            labels = ids[:, 1:].clone()
            labels[mask[:, 1:] == 0] = -100
            if SS_ARM == "control":
                # the frozen ladder loss path, verbatim: bit-identity
                # with the booked stock s3 cell is the precondition
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    labels.reshape(-1), ignore_index=-100)
            else:
                logp = torch.log_softmax(logits, dim=-1)
                tokloss = torch.nn.functional.nll_loss(
                    logp.reshape(-1, logp.shape[-1]),
                    labels.reshape(-1), ignore_index=-100,
                    reduction="none").reshape(labels.shape)
                w = torch.ones_like(tokloss)
                corr = logits.new_zeros(())
                for bi in range(len(batch)):
                    ri = row_of[a + bi]
                    k = weight_of.get(ri)
                    if k is None:
                        continue
                    off = off_of[a + bi]
                    # answer region: label positions off-1 .. end
                    w[bi, max(off - 1, 0):] = k
                    for pos, probs in targets_of.get(ri, []):
                        li = off + pos - 1
                        if li < 0 or li >= logp.shape[1]:
                            continue
                        onehot_tok = int(ids[bi, off + pos])
                        corr = corr + k * logp[bi, li, onehot_tok]
                        for v, q in probs:
                            corr = corr - k * q * logp[bi, li, v]
                        n_soft_positions += 1
                live = (labels != -100).float()
                loss = ((tokloss * w * live).sum() + corr) \
                    / (w * live).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched.last_epoch < steps_total - 1:
                sched.step()
            opt.zero_grad()
            step += 1
            if step % 200 == 0:
                print(f"  step {step}/{steps_total} loss "
                      f"{float(loss.detach()):.3f} "
                      f"({step/(time.time()-t0):.1f} it/s)",
                      flush=True)
    train_seconds = time.time() - t0

    if not SMOKE:
        torch.save(model.state_dict(), OUT)
    print(f"[softspeed] {'saved' if not SMOKE else 'skipped save'} "
          f"{OUT} after {step} steps "
          f"({train_seconds:.0f}s), soft positions "
          f"{n_soft_positions}", flush=True)

    # unregistered observable: valid-set mass on the fixed
    # conflicted sample (Mac-fp32 family; not comparable to 3080)
    model.eval()
    groups = defaultdict(list)
    for r in stock_rows:
        groups["".join(str(r["cur"]).split())].append(r)
    conflicted = []
    for curkey, rs in groups.items():
        answers = {"".join(str(r["nxt"]).split()): r for r in rs}
        if len(answers) >= 2:
            conflicted.append((rs[0]["cur"],
                               [r["nxt"] for r in answers.values()]))
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
                tin = torch.tensor([full], device=dev)
                am = torch.ones_like(tin)
                lp = torch.log_softmax(
                    model(tin[:, :-1], am[:, :-1]).float(), -1)
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
    row = {"arm": SS_ARM, "seed": SEED, "steps": step,
           "steps_total": steps_total, "train_seconds":
           round(train_seconds, 1), "solves": solves, "total": tot,
           "valid_pct": round(valid, 2), "device": dev,
           "n_soft_rows": len(weight_of),
           "n_soft_positions": n_soft_positions,
           "valid_set_mass": round(valid_mass, 4),
           "cal_sample": len(masses),
           "code_commit": git_sha(short=True)}
    with RECEIPTS.open("a") as f:
        f.write(json.dumps(row) + "\n")
    print(f"[softspeed] GATE arm={SS_ARM} s{SEED} {solves} = "
          f"{tot}/120 @ {valid:.2f}% | mass {valid_mass:.4f} | "
          f"{train_seconds:.0f}s train", flush=True)


if __name__ == "__main__":
    main()
