"""Judge-collapsed decoding (spec 2026-07-28 rung 4, pre-reg
2026-07-28). Three arms on 30 fresh L5-L7 states, 12 plies:
(a) wave-8 (production semantics), (b) greedy-1, (c) greedy with
top-2 branching at near-tie steps (margin < 0.02), oracle judge,
both branches' tokens charged. Tokens + per-state sidecar logged.
"""
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

CKPT = "checkpoints/calib_d256_ctl.pt"
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
SEED = 99_300_000
MARGIN = 0.02
PLIES, MAX_NEW = 12, 120

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                 weights_only=True))
model.eval()
NL = tok.encode("\n")[-1]

states = []
for lv in (5, 6, 7):
    for i in range(10):
        p = _gen_isolated(lv, SEED + 1000 * lv + i)
        if p is not None:
            states.append((lv, f"Integral({sp.sstr(p._expr)}, x)"))
print(f"{len(states)} states", flush=True)


@torch.no_grad()
def greedy_step(prefix, spend, branch=False):
    """Greedy decode one Step line. branch=True: at the FIRST
    near-tie, fork both tokens, greedy-finish both, return both."""
    out = list(prefix)
    for _ in range(MAX_NEW):
        lg = model(torch.tensor([out], device=dev))[0, -1]
        top2 = lg.topk(2)
        spend[0] += 1
        gap = float(top2.values[0] - top2.values[1])
        if branch and gap < MARGIN:
            a, _ = greedy_step(out + [int(top2.indices[0])], spend)
            b, _ = greedy_step(out + [int(top2.indices[1])], spend)
            return a, b
        t = int(top2.indices[0])
        out.append(t)
        if t in (tok.eos_id, NL):
            break
    return out, None


def run_chain(cur0, arm, seed0):
    spend = [0]
    cur = cur0
    visited = {cur.replace(" ", "")}
    for ply in range(PLIES):
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        if arm == "wave":
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [seed0 + ply * 7 + b for b in range(8)], dev)
            spend[0] += sum(len(tok.encode(t)) + 1 for t in texts)
            cands = [t for t in dict.fromkeys(texts)
                     if t and t.replace(" ", "") not in visited]
        else:
            r = greedy_step(prompt, spend, branch=(arm == "judge"))
            outs = [r[0]] if r[1] is None else [r[0], r[1]]
            cands = []
            for o in outs:
                t = tok.decode(o[len(prompt):]).strip("\n")
                if t and t.replace(" ", "") not in visited:
                    cands.append(t)
            cands = list(dict.fromkeys(cands))
        wv = verify_wave(cur, cands) if cands else {}
        nxt = None
        for t in cands:
            ok, so = wv.get(t, (False, False))
            if ok:
                nxt = "SOLVED" if so else t
                if so:
                    break
        if nxt == "SOLVED":
            return True, spend[0]
        if nxt is None:
            return False, spend[0]
        cur = nxt
        visited.add(cur.replace(" ", ""))
    return False, spend[0]


side = open("logs/pp_judge_decode.jsonl", "w")
tot = {a: [0, 0] for a in ("wave", "greedy", "judge")}
for si, (lv, cur) in enumerate(states):
    row = {"lv": lv, "cur": cur}
    for arm in ("wave", "greedy", "judge"):
        solved, spent = run_chain(cur, arm, SEED + si * 977)
        tot[arm][0] += solved
        tot[arm][1] += spent
        row[arm] = {"solved": solved, "tokens": spent}
    side.write(json.dumps(row) + "\n")
    side.flush()
    print(f"[{si+1}/{len(states)}] L{lv} " +
          " ".join(f"{a}:{'S' if row[a]['solved'] else '.'}"
                   f"({row[a]['tokens']})" for a in tot), flush=True)
for a, (s, t) in tot.items():
    print(f"{a:7s} solves {s}/{len(states)} tokens {t}", flush=True)
