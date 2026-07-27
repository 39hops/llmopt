"""Pincer distribution readout (Artin's quantum-frame ask,
2026-07-26 night; pre-reg in RESULTS): the engine enumerates the
COMPLETE legal move set for a state (the classical superposition,
exact by construction); each crystal supplies a distribution over
it (teacher-forced sequence log-probs, no generation). Question:
are the amplitudes CALIBRATED — does model mass track which moves
actually lead to the root (fork-isolated engine solves as value
labels)?

Scored models (one probe pass logs all four):
  pairs_3e / pairs_1p — forward chain crystals (the incumbents)
  oneshot_1p          — the conjecturer (predicted miscalibrated
                        on 1-ply moves: trained to skip)
  backpairs_1p        — REVERSE scorer: logp_b(s | child) — the
                        backward crystal as Bayes-style reranker
                        (per-edge score, own normalization noted)
Per-child log row (streamed; the sidecar is the deliverable):
rule@site, child, token len, solved/nodes (value), raw logp per
model. Per-state: n_legal, n_solving, per-model entropy, mass on
solving set, top-1-solves, best-move rank, Spearman(logp, value).
Aggregates: per-model, per-level, per-rule-family, length-bias
corr (the confound named), calibration deciles.

    python scratch/pincer_dist_probe.py [wall_per_solve=20]
Sidecar: logs/pp_dist_probe.jsonl
"""
import glob
import json
import math
import multiprocessing as mp
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp
import torch

from llmopt.train.mathnative import MathTokenizer, build_model
from llmopt.search.derivation import State, successors

WALL = int(sys.argv[1]) if len(sys.argv) > 1 else 20
BUDGET = 150
norm = lambda s: s.replace(" ", "")  # noqa: E731

# ---- probe states: the R1 pool (fresh gate-band mid-chain) ----
seen, states = set(), []
for f in sorted(glob.glob("logs/pp_*.jsonl")):
    if "r0" in f or "backpairs" in f or "dist" in f:
        continue
    for r in map(json.loads, open(f)):
        for c in r.get("chain", []):
            if c != "SOLVED" and norm(c) not in seen:
                seen.add(norm(c))
                states.append((r["level"], c))
print(f"{len(states)} probe states", flush=True)

# ---- value oracle: fork-isolated engine solve (SIGALRM doctrine) --
def _solve_worker(expr_s, q):
    try:
        from llmopt.search.engine import solve
        r = solve(sp.sympify(expr_s), budget=BUDGET)
        q.put({"solved": bool(r.solved), "nodes": r.nodes,
               "plies": r.state.plies})
    except Exception:
        q.put(None)


def solve_isolated(expr_s):
    ctx = mp.get_context("fork")
    q = ctx.Queue()
    p = ctx.Process(target=_solve_worker, args=(expr_s, q))
    p.start()
    p.join(WALL)
    if p.is_alive():
        p.kill()
        p.join()
        return None
    try:
        return q.get(timeout=10)
    except Exception:
        return None


# ---- models ----
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")


def load(ckpt):
    m = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


MODELS = {name: load(p) for name, p in [
    ("pairs_3e", "checkpoints/mathnative_wfloor_d256.pt"),
    ("pairs_1p", "checkpoints/mathnative_wfloor_d256_stream4.pt"),
    ("oneshot_1p", "checkpoints/fmt_oneshot_1p.pt"),
    ("backpairs_1p", "checkpoints/fmt_backpairs_1p.pt"),
]}


def seq_logp(model, prefix, targets):
    """Raw (T=1) summed log-prob of each target continuation after
    prefix; returns list of (logp, n_target_tokens)."""
    pre = tok.encode(prefix)
    encs = []
    for t in targets:
        tt = tok.encode(t) + [tok.id["\n"]]
        encs.append((pre + tt, len(tt)))
    L = max(len(e) for e, _ in encs)
    ids = torch.tensor([e + [tok.pad_id] * (L - len(e))
                        for e, _ in encs], device=dev)
    mask = torch.tensor([[1] * len(e) + [0] * (L - len(e))
                         for e, _ in encs], device=dev)
    with torch.no_grad():
        logits = model(ids[:, :-1], mask[:, :-1])
        lsm = torch.log_softmax(logits.float(), -1)
    out = []
    for b, (e, nt) in enumerate(encs):
        lp = sum(float(lsm[b, j - 1, e[j]])
                 for j in range(len(e) - nt, len(e)))
        out.append((lp, nt))
    return out


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        for r, i in enumerate(order):
            rk[i] = r
        return rk
    if len(xs) < 3:
        return None
    rx, ry = rank(xs), rank(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx and dy else None


out = open("logs/pp_dist_probe.jsonl", "w")
skipped_enum = 0
for si, (lv, s) in enumerate(states):
    try:
        legal = [(name, sp.sstr(ch.expr))
                 for name, ch in successors(
                     State(sp.sympify(s)), use_macros=True)]
    except Exception:
        skipped_enum += 1
        continue
    if len(legal) < 2:
        skipped_enum += 1
        continue
    vals = [solve_isolated(c) for _, c in legal]
    children = [c for _, c in legal]
    try:
        scores = {m: seq_logp(mod, f"Current: {s}\nHints: none\nStep: ",
                              children) for m, mod in MODELS.items()
                  if m != "backpairs_1p"}
        # reverse frame: logp_b(s | child) per edge
        scores["backpairs_1p"] = [
            seq_logp(MODELS["backpairs_1p"],
                     f"Current: {c}\nHints: none\nStep: ", [s])[0]
            for c in children]
    except ValueError:  # unencodable state (charset)
        skipped_enum += 1
        continue
    solved = [bool(v and v["solved"]) for v in vals]
    row = {
        "level": lv, "state": s, "n_legal": len(legal),
        "n_solving": sum(solved), "n_val_skip": sum(v is None for v in vals),
        "children": [{
            "rule": legal[i][0], "child": children[i],
            "solved": solved[i],
            "nodes": vals[i]["nodes"] if vals[i] else None,
            "plies_to_root": vals[i]["plies"] if vals[i] and solved[i] else None,
            "n_tok": scores["pairs_3e"][i][1],
            "logp": {m: scores[m][i][0] for m in scores},
        } for i in range(len(legal))],
    }
    # per-state per-model derived reads
    for m in scores:
        lps = [scores[m][i][0] for i in range(len(legal))]
        mx = max(lps)
        ps = [math.exp(l - mx) for l in lps]
        z = sum(ps)
        ps = [p / z for p in ps]
        ent = -sum(p * math.log(p + 1e-12) for p in ps)
        row[f"d_{m}"] = {
            "entropy": round(ent, 4),
            "entropy_norm": round(ent / math.log(len(ps)), 4),
            "mass_solving": round(sum(p for p, sv in zip(ps, solved)
                                      if sv), 4),
            "top1_solves": solved[max(range(len(ps)),
                                      key=lambda i: ps[i])],
            "spearman_v_solved": spearman(lps, [1.0 if x else 0.0
                                                for x in solved]),
            "len_corr": spearman(lps, [-scores["pairs_3e"][i][1]
                                       for i in range(len(legal))]),
        }
    out.write(json.dumps(row) + "\n")
    out.flush()
    if (si + 1) % 10 == 0:
        print(f"  {si+1}/{len(states)}", flush=True)
out.close()
print(f"done; enum-skipped {skipped_enum}. Aggregate with "
      f"scratch/pincer_dist_report.py", flush=True)
