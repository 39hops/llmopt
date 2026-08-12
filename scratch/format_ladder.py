"""The format ladder (spec 2026-07-26-format-ladder, pre-reg in
RESULTS). One birth per invocation:

    FORMAT=traces|skip|dechain|oneshot|delta|randpack \
    SCHED=1p|3e BIRTH_SEED=1 python scratch/format_ladder.py

1p = the v4 streaming recipe (mixed shuffled batches, final-10%
cooldown, surprise rider); 3e = standard 3ep OneCycle (control
construction: length-sorted BS=32, shuffled batch order).
Checkpoint: checkpoints/fmt_{FORMAT}_{SCHED}.pt
"""
from llmopt.common.device import pick_device
import json
import os
import random
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch
import torch.nn.functional as F

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

FORMAT = os.environ["FORMAT"]
SCHED = os.environ.get("SCHED", "1p")
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BS = 32
BASE_LR = 3e-4
WARMUP = 200
EMA_DECAY = 0.99
CLAMP = (0.25, 4.0)
CAP = 1024 if FORMAT in ("traces", "delta", "randpack") else 512
OUT = f"checkpoints/fmt_{FORMAT}_{SCHED}.pt"

tok = MathTokenizer()
rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
norm = lambda s: s.replace(" ", "")  # noqa: E731


def pair_text(r):
    return f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"


def build_chains():
    """State-linked greedy chains from roots; consumes every row
    exactly once. Returns list of lists of row indices."""
    by_cur = {}
    for i, r in enumerate(rows):
        by_cur.setdefault(norm(r["cur"]), []).append(i)
    nxt_set = {norm(r["nxt"]) for r in rows}
    consumed = [False] * len(rows)
    chains = []
    order = sorted(range(len(rows)),
                   key=lambda i: norm(rows[i]["cur"]) in nxt_set)
    for i in order:  # roots first
        if consumed[i]:
            continue
        ch = [i]
        consumed[i] = True
        while True:
            cands = [j for j in by_cur.get(norm(rows[ch[-1]]["nxt"]), [])
                     if not consumed[j]]
            if not cands:
                break
            j = cands[0]
            consumed[j] = True
            ch.append(j)
        chains.append(ch)
    return chains


texts = []
if FORMAT == "traces":
    for ch in build_chains():
        t = (f"Current: {rows[ch[0]]['cur']}\nHints: none\n"
             + "".join(f"Step: {rows[j]['nxt']}\n" for j in ch))
        texts.append(t)
elif FORMAT == "skip":
    rnd = random.Random(3)
    chains = build_chains()
    skips = []
    for ch in chains:
        for a in range(0, len(ch) - 1, 2):
            skips.append(f"Current: {rows[ch[a]]['cur']}\n"
                         f"Hints: none\nStep: {rows[ch[a+1]]['nxt']}\n")
    plain = [pair_text(r) for r in rows]
    rnd.shuffle(plain)
    texts = skips + plain[:max(0, len(rows) - len(skips))]
elif FORMAT == "dechain":
    for ch in build_chains():
        for k in range(0, len(ch), 2):
            texts.append(pair_text(rows[ch[k]]))
elif FORMAT == "oneshot":
    for ch in build_chains():
        texts.append(f"Current: {rows[ch[0]]['cur']}\nHints: none\n"
                     f"Step: {rows[ch[-1]]['nxt']}\n")
elif FORMAT == "altpairs":
    # distribution-rows forward edition (pre-reg 2026-07-26 night):
    # full pairs diet + farmed verified-alternative successors
    # (make_altpairs.py) — teaches that the legal set has more than
    # one valid branch. Share is whatever the farm minted (~14%).
    texts = [pair_text(r) for r in rows]
    alts = [json.loads(l) for l in open("data/altpairs_rows.jsonl")]
    alts = [r for r in alts if norm(r["cur"]) != norm(r["nxt"])]
    texts += [pair_text(r) for r in alts]
    print(f"[altpairs] {len(rows)} base + {len(alts)} alt rows",
          flush=True)
elif FORMAT == "backpairs":
    # pincer R1a (spec 2026-07-26-reverse-llmue-pincer): the
    # backward crystal's diet — EVERY row reversed, zero forward
    # mixing (its whole world is backward; separate-crystals
    # doctrine, no direction marker needed). Same pair frame,
    # roles flipped: later state as Current, predecessor as Step.
    texts = [f"Current: {r['nxt']}\nHints: none\nStep: {r['cur']}\n"
             for r in rows]
elif FORMAT == "revpairs10":
    # low-dose backward ration (pre-reg 2026-07-28 night): 90/10
    # forward/reversed at matched total dose — the 50/50 cell's
    # -18 direction tax at its untested low dose ("stupid corner").
    rnd = random.Random(9)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    cut = len(idx) // 10
    texts = [pair_text(rows[i]) for i in idx[cut:]]
    texts += [f"Current: {rows[i]['nxt']}\nHints: none\n"
              f"Step: {rows[i]['cur']}\n" for i in idx[:cut]]
elif FORMAT == "revpairs":
    # dual-direction crystal (reverse-LLMUE bank): 50/50 forward +
    # reversed pairs at matched total dose; reverse rows spell the
    # LATER state as Current and the predecessor as Step (vocab-40
    # native, no new atoms). Fence: reverse steps are equivalence-
    # valid, so gate pollution shows as WANDERING, not unsoundness.
    rnd = random.Random(9)
    idx = list(range(len(rows)))
    rnd.shuffle(idx)
    half = len(idx) // 2
    texts = [pair_text(rows[i]) for i in idx[:half]]
    texts += [f"Current: {rows[i]['nxt']}\nHints: none\n"
              f"Step: {rows[i]['cur']}\n" for i in idx[half:]]
elif FORMAT in ("delta", "randpack"):
    K = 4
    if FORMAT == "randpack":
        order = list(range(len(rows)))
        random.Random(5).shuffle(order)
        groups = [order[i:i + K] for i in range(0, len(order), K)]
    else:
        emb = torch.load("scratch/fmt_row_emb.pt")  # built by
        # format_delta_prep.py: (N, d) unit vectors, row-aligned
        visited = torch.zeros(len(rows), dtype=torch.bool)
        groups = []
        rnd = random.Random(7)
        sims = []
        while int(visited.sum()) < len(rows):
            free = (~visited).nonzero().flatten()
            cur = int(free[rnd.randrange(len(free))])
            g = [cur]
            visited[cur] = True
            while len(g) < K:
                s = emb @ emb[g[-1]]
                s[visited] = -2
                j = int(s.argmax())
                if s[j] < 0.5:  # the delta floor
                    break
                sims.append(float(s[j]))
                g.append(j)
                visited[j] = True
            groups.append(g)
        print(f"[delta] mean intra-group sim "
              f"{sum(sims)/max(len(sims),1):.3f} over {len(sims)} hops",
              flush=True)
    texts = ["".join(pair_text(rows[j]) for j in g) for g in groups]
else:
    raise SystemExit(f"unknown FORMAT {FORMAT}")

enc, dropped = [], 0
for t in texts:
    try:
        ids = tok.encode(t) + [tok.eos_id]
    except ValueError:
        dropped += 1
        continue
    if len(ids) <= CAP:
        enc.append(ids)
    else:
        dropped += 1
tokmass = sum(len(s) for s in enc)
print(f"[{FORMAT}/{SCHED}] {len(texts)} sequences -> {len(enc)} "
      f"encoded (dropped {dropped}), token mass {tokmass/1e6:.1f}M",
      flush=True)

dev = pick_device()
torch.manual_seed(int(os.environ.get("BIRTH_SEED", "1")))
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                        weight_decay=0.01)


def run_batches(batches, lr_fn, epochs_label=""):
    ema = None
    t0 = time.time()
    for step, b in enumerate(batches):
        batch = [enc[j] for j in b]
        L = max(len(s) for s in batch)
        ids = torch.tensor(
            [s + [tok.pad_id] * (L - len(s)) for s in batch], device=dev)
        mask = torch.tensor(
            [[1] * len(s) + [0] * (L - len(s)) for s in batch],
            device=dev)
        logits = model(ids[:, :-1], mask[:, :-1])
        labels = ids[:, 1:].clone()
        labels[mask[:, 1:] == 0] = -100
        loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                               labels.reshape(-1), ignore_index=-100)
        lv = float(loss.detach())
        ema = lv if ema is None else EMA_DECAY * ema + (1 - EMA_DECAY) * lv
        surprise = max(CLAMP[0], min(CLAMP[1], lv / max(ema, 1e-8)))
        for g in opt.param_groups:
            g["lr"] = lr_fn(step, len(batches)) * (
                surprise if SCHED == "1p" else 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        if (step + 1) % 200 == 0:
            r = (step + 1) / (time.time() - t0)
            print(f"  {epochs_label}{step+1}/{len(batches)} loss "
                  f"{lv:.3f} ({r:.1f} it/s)", flush=True)
    return time.time() - t0


t_all = time.time()
if SCHED == "1p":
    perm = list(range(len(enc)))
    random.Random(1).shuffle(perm)
    batches = [perm[i:i + BS] for i in range(0, len(perm), BS)]

    def lr_fn(step, total):
        warm = min(1.0, (step + 1) / WARMUP)
        tail = total // 10
        left = total - step
        cool = left / tail if left <= tail else 1.0
        return BASE_LR * warm * cool
    run_batches(batches, lr_fn)
else:
    order = sorted(range(len(enc)), key=lambda j: len(enc[j]))
    base = [order[i:i + BS] for i in range(0, len(order), BS)]
    total = 3 * len(base)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=BASE_LR, total_steps=total, pct_start=0.03)
    done = 0

    def lr_fn(step, _):
        return sched.get_last_lr()[0]
    for ep in range(3):
        bt = list(base)
        random.Random(ep).shuffle(bt)
        # step the schedule alongside (surprise off at 3e)
        orig_step = opt.step

        def step_and_sched(*a, **k):
            orig_step(*a, **k)
            if sched.last_epoch < total - 1:
                sched.step()
        opt.step = step_and_sched
        run_batches(bt, lr_fn, epochs_label=f"ep{ep} ")
        opt.step = orig_step

torch.save(model.state_dict(), OUT)
print(f"saved {OUT}  wall {time.time()-t_all:.0f}s", flush=True)
