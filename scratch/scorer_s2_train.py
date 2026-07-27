"""S2 trainer (calibrated-scorer spec): the listwise objective
race. State + enumerated legal set -> distribution in ONE forward
pass (teacher-forced seq log-probs, softmax over the set, zero
generation). Two arms, one variable (the target distribution):

  imit — smoothed one-hot on the replayed true move (the free
         corpus labels; needs true_child_idx)
  dist — softmax(-plies) over value-labeled children (the
         Dijkstra head; unsolved children get PMAX)

Matched everything else: same warm-start crystal (pairs_3e), same
state set (rows where BOTH label kinds exist), same steps/lr/seed.
Battery states are exclude=-guarded out of training. Eval on
data/scorer_battery_v1.jsonl mixed states: Spearman vs value,
top1-solves, mass-on-solving, calls-to-first-hit under
mass-descending order (R0b economics) — vs length-only and
raw-pairs_3e controls.

    python scratch/scorer_s2_train.py [epochs=3]
Out: checkpoints/scorer_s2_{imit,dist}.pt + logs/pp_s2_train.jsonl
"""
import json
import math
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch

from llmopt.train.mathnative import MathTokenizer, build_model

EPOCHS = int(sys.argv[1]) if len(sys.argv) > 1 else 3
WARM = "checkpoints/mathnative_wfloor_d256.pt"
PMAX, LR, SMOOTH = 12, 1e-4, 0.1
norm = lambda s: s.replace(" ", "")  # noqa: E731
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(0)

# ---- data: rows where BOTH label kinds exist; battery excluded --
battery = [json.loads(x) for x in open("data/scorer_battery_v1.jsonl")]
excl = {norm(b["state"]) for b in battery}
rows = []
for r in map(json.loads, open("data/scorer_train_v1.jsonl")):
    if norm(r["state"]) in excl:
        continue
    ch = [c for c in r["children"] if c["n_tok"] is not None]
    if len(ch) < 2 or r["true_child_idx"] is None:
        continue
    true_child = r["children"][r["true_child_idx"]]
    if true_child["n_tok"] is None or true_child not in ch:
        continue
    if not any(c["solved"] is not None for c in ch):
        continue
    rows.append({"state": r["state"], "level": r["level"],
                 "children": ch, "true_idx": ch.index(true_child)})
rnd = random.Random("s2-train-1")
rnd.shuffle(rows)
n_hold = max(1, len(rows) // 10)
hold, train = rows[:n_hold], rows[n_hold:]
print(f"{len(train)} train / {len(hold)} holdout states "
      f"(battery-excluded: {len(excl)} fenced)", flush=True)


def targets(ch, true_idx, arm):
    if arm == "imit":
        n = len(ch)
        t = [SMOOTH / (n - 1)] * n
        t[true_idx] = 1.0 - SMOOTH
        return t
    w = [math.exp(-(c["plies"] if c["solved"] else PMAX))
         for c in ch]
    z = sum(w)
    return [x / z for x in w]


def batch_logps(model, state, ch, grad=False):
    """Summed seq logp of each child continuation (padded batch)."""
    pre = tok.encode(f"Current: {state}\nHints: none\nStep: ")
    encs = [(pre + tok.encode(c["child"]) + [tok.id["\n"]],
             c["n_tok"] + 1) for c in ch]
    L = max(len(e) for e, _ in encs)
    ids = torch.tensor([e + [tok.pad_id] * (L - len(e))
                        for e, _ in encs], device=dev)
    mask = torch.tensor([[1] * len(e) + [0] * (L - len(e))
                         for e, _ in encs], device=dev)
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        logits = model(ids[:, :-1], mask[:, :-1])
        lsm = torch.log_softmax(logits.float(), -1)
        lps = []
        for b, (e, nt) in enumerate(encs):
            idx = torch.tensor(e[len(e) - nt:], device=dev)
            pos = torch.arange(len(e) - nt - 1, len(e) - 1, device=dev)
            lps.append(lsm[b, pos, idx].sum())
        return torch.stack(lps)


def run_arm(arm):
    m = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
    m.load_state_dict(torch.load(WARM, map_location="cpu"))
    opt = torch.optim.Adam(m.parameters(), lr=LR)
    log = open(f"logs/pp_s2_train.jsonl", "a")
    order = list(range(len(train)))
    r_ep = random.Random(f"s2-{arm}-ep")
    for ep in range(EPOCHS):
        r_ep.shuffle(order)
        m.train()
        tot = 0.0
        for i in order:
            r = train[i]
            lps = batch_logps(m, r["state"], r["children"], grad=True)
            t = torch.tensor(targets(r["children"], r["true_idx"], arm),
                             device=dev)
            loss = -(t * torch.log_softmax(lps, 0)).sum()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        m.eval()
        hl = ht = 0.0
        for r in hold:
            lps = batch_logps(m, r["state"], r["children"])
            t = torch.tensor(targets(r["children"], r["true_idx"], arm),
                             device=dev)
            hl += float(-(t * torch.log_softmax(lps, 0)).sum())
            ht += int(int(lps.argmax()) == r["true_idx"])
        rec = {"arm": arm, "ep": ep, "train_loss": tot / len(train),
               "hold_loss": hl / len(hold),
               "hold_top1_true": ht / len(hold)}
        print(rec, flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()
    torch.save(m.state_dict(), f"checkpoints/scorer_s2_{arm}.pt")
    log.close()
    return m


# ---- battery eval (mixed states): model vs controls ------------
def spearman(xs, ys):
    def rank(v):
        o = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for j, i in enumerate(o):
            rk[i] = j
        return rk
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return (sum((a - mx) * (b - my) for a, b in zip(rx, ry))
            / (dx * dy)) if dx and dy else None


def eval_battery(name, score_fn):
    sps, top1, hits = [], [0, 0], []
    for b in battery:
        ch = [c for c in b["children"]
              if c["n_tok"] is not None and c["solved"] is not None]
        solved = [bool(c["solved"]) for c in ch]
        if len(ch) < 3 or not (0 < sum(solved) < len(ch)):
            continue  # mixed states only
        xs = score_fn(b["state"], ch)
        sp_ = spearman(xs, [1.0 if s else 0.0 for s in solved])
        if sp_ is not None:
            sps.append(sp_)
        best = max(range(len(ch)), key=lambda i: xs[i])
        top1[0] += solved[best]
        top1[1] += 1
        order = sorted(range(len(ch)), key=lambda i: -xs[i])
        hits.append(next(k + 1 for k, i in enumerate(order)
                         if solved[i]))
    return {"model": name, "n": top1[1],
            "spearman": round(sum(sps) / len(sps), 3),
            "top1_solves": f"{top1[0]}/{top1[1]}",
            "mean_calls_to_hit": round(sum(hits) / len(hits), 2)}


if __name__ == "__main__":
    results = []
    for arm in ("imit", "dist"):
        print(f"=== arm {arm} ===", flush=True)
        m = run_arm(arm)
        results.append(eval_battery(
            f"s2_{arm}",
            lambda s, ch, m=m: [float(x) for x in batch_logps(m, s, ch)]))
    base = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                       ffn=1024).to(dev)
    base.load_state_dict(torch.load(WARM, map_location="cpu"))
    base.eval()
    results.append(eval_battery(
        "pairs_3e_raw",
        lambda s, ch: [float(x) for x in batch_logps(base, s, ch)]))
    results.append(eval_battery(
        "length_only", lambda s, ch: [-c["n_tok"] for c in ch]))
    for r in results:
        print(r, flush=True)
    with open("logs/pp_s2_train.jsonl", "a") as f:
        f.write(json.dumps({"battery_eval": results}) + "\n")
