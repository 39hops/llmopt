"""Pincer R8: meet v1 — full protocol (conjecture + peel + meet)
vs let-it-finish forward re-roll at equal sampled-token budget
(spec amendment 3; pre-reg in RESULTS 2026-07-26 night).

Battery = the problems pairs_3e failed at its greedy gate (the
sidecar's misses — where the variance lives). Soundness: every
edge is oracle-verified at mint; a spliced chain is equivalent by
transitivity, solved iff its endpoint is integral-free.

    python scratch/pincer_r8.py
Sidecar: logs/pp_r8_meet.jsonl (both arms, per problem)
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp
import torch

from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

K = 8
PEEL_J = 2
MAX_B = 24  # goal-set cap per problem
norm = lambda s: s.replace(" ", "")  # noqa: E731

tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")


def load(p):
    m = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
    m.load_state_dict(torch.load(p, map_location="cpu"))
    m.eval()
    return m


fwd = load("checkpoints/mathnative_wfloor_d256.pt")      # pairs_3e
one = load("checkpoints/fmt_oneshot_1p.pt")
bwd = load("checkpoints/fmt_backpairs_1p.pt")

battery = [(r["level"], r["i"]) for r in
           map(json.loads, open("logs/pp_pairs_3e.jsonl"))
           if not r["solved"]]
print(f"battery: {len(battery)} pairs_3e misses", flush=True)
tokens = {"A": 0, "B": 0}


def wave(model, cur, seeds, arm):
    texts, _, _ = G.sample_wave_lp(
        model, tok, tok.encode(f"Current: {cur}\nHints: none\nStep: "),
        seeds, dev)
    tokens[arm] += sum(len(tok.encode(t)) for t in texts if t)
    return [t for t in dict.fromkeys(texts) if t]


def chain_search(model, root, seed0, arm, goal=None, plies=12):
    """Greedy verified chain (gate discipline); if goal set given,
    contact => spliced solve. Returns (solved, how, chain)."""
    cur = root
    visited = {norm(root)}
    chain = []
    for ply in range(plies):
        cands = wave(model, cur, [seed0 + ply * 7 + b for b in range(G.B)],
                     arm)
        cands = [t for t in cands if norm(t) not in visited]
        wv = verify_wave(cur, cands) if cands else {}
        nxt = None
        for t in cands:
            ok, so = wv.get(t, (False, False))
            if ok:
                if so:
                    return True, "chain", chain + [t]
                if nxt is None:
                    nxt = t
        if nxt is None:
            return False, "stall", chain
        cur = nxt
        chain.append(cur)
        visited.add(norm(cur))
        if goal and norm(cur) in goal:
            return True, "meet", chain
    return False, "budget", chain


out = open("logs/pp_r8_meet.jsonl", "w")
score = {"A": 0, "B": 0}
b_how = {}
for lv, i in battery:
    p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
    if p is None:
        continue
    root = f"Integral({sp.sstr(p._expr)}, x)"
    row = {"level": lv, "i": i, "root": root}

    # ---- ARM A: forward re-roll, fresh seed band ----
    sa, how_a, _ = chain_search(fwd, root, 77_000_000 + i * 311 + lv,
                                "A")
    row["A_solved"], row["A_how"] = sa, how_a
    score["A"] += sa

    # ---- ARM B: conjecture -> peel -> meet ----
    solved_b, how_b = False, "none"
    cands = wave(one, root, [88_000_000 + i * 31 + b for b in range(K)],
                 "B")
    wv = verify_wave(root, cands) if cands else {}
    n_conj_valid = 0
    for t in cands:
        ok, so = wv.get(t, (False, False))
        n_conj_valid += ok
        if ok and so:
            solved_b, how_b = True, "conjecture"
            break
    goal, peel_kept = {}, 0
    if not solved_b:
        frontier = cands[:4]  # peel the top distinct candidates
        for d in range(PEEL_J):
            nxt_frontier = []
            for c in frontier:
                if len(goal) >= MAX_B:
                    break
                preds = wave(bwd, c, [99_000_000 + i * 17 + d * 5 + b
                                      for b in range(K)], "B")
                preds = [q for q in preds if norm(q) != norm(c)]
                for q in preds:
                    pv = verify_wave(q, [c])
                    if pv.get(c, (False, False))[0]:
                        goal[norm(q)] = c
                        peel_kept += 1
                        nxt_frontier.append(q)
            frontier = nxt_frontier[:4]
        sb, how, _ = chain_search(fwd, root, 66_000_000 + i * 211 + lv,
                                  "B", goal=set(goal))
        solved_b, how_b = sb, how
    row.update({"B_solved": solved_b, "B_how": how_b,
                "n_conj_valid": n_conj_valid, "peel_kept": peel_kept,
                "goal_size": len(goal)})
    score["B"] += solved_b
    out.write(json.dumps(row) + "\n")
    out.flush()
    b_how[how_b] = b_how.get(how_b, 0) + 1
    print(f"  L{lv} i{i}: A={int(sa)}({how_a}) B={int(solved_b)}"
          f"({how_b}) goal={len(goal)}", flush=True)
out.close()
print(f"R8: arm A {score['A']}/{len(battery)} v arm B "
      f"{score['B']}/{len(battery)}; B mechanisms {b_how}; "
      f"tokens A {tokens['A']} B {tokens['B']} "
      f"[sidecar logs/pp_r8_meet.jsonl]", flush=True)
