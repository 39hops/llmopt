"""Chain gate for PREFIX-substrate models (rung 1, spec
2026-07-25-native-transformer). Mirrors gate_eval exactly — same
seeds, same _gen_isolated problems, same verify_wave oracle — with
prefix<->infix conversion at the two boundaries: prompts serialize
cur to prefix; model emissions parse prefix->sympy and re-render
infix (sp.sstr) before the oracle. Emissions that fail the prefix
parser are invalid candidates (counted tried, never valid).

    .venv/bin/python scratch/gate_prefix.py <ckpt> <d> <layers> <ffn> <heads> <label>
"""
import sys

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")

import sympy as sp

import step_grpo_micro as G
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave
from llmopt.mathgen.prefix import from_prefix, to_prefix
from llmopt.train.mathnative import MathTokenizer, build_model

ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
tok = MathTokenizer()
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

solves = {}
valid = tried = parse_fail = 0
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        s = 0
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            cur = sp.Integral(p._expr, sp.Symbol("x"))
            visited = {to_prefix(cur)}
            done = False
            for ply in range(12):
                prompt = tok.encode(
                    f"Current: {to_prefix(cur)}\nHints: none\nStep: ")
                texts, _, _ = G.sample_wave_lp(
                    model, tok, prompt,
                    [G.GATE_BAND + i * 31 + ply * 7 + b
                     for b in range(G.B)], dev)
                tried += len(texts)
                cand = {}   # raw emission -> (canonical prefix, infix)
                for t_ in dict.fromkeys(texts):
                    if not t_:
                        continue
                    try:
                        e = from_prefix(t_.strip())
                    except Exception:
                        parse_fail += 1
                        continue
                    canon = to_prefix(e)
                    if canon not in visited:
                        cand[t_] = (canon, sp.sstr(e))
                wv = (verify_wave(sp.sstr(cur),
                                  list({i for _, i in cand.values()}))
                      if cand else {})
                nxt = None
                for t_ in texts:
                    if t_ not in cand:
                        continue
                    canon, infix = cand[t_]
                    ok, so = wv.get(infix, (False, False))
                    if ok:
                        valid += 1
                        if nxt is None:
                            nxt = "SOLVED" if so else canon
                if nxt == "SOLVED":
                    done = True
                    break
                if nxt is None:
                    break
                visited.add(nxt)
                cur = from_prefix(nxt)
            s += done
        solves[lv] = s
tot = sum(solves.values())
print(f"{label} gate: {solves} = {tot}/120 @ "
      f"{100 * valid / max(tried, 1):.2f}% "
      f"(parse-fail {parse_fail}/{tried})", flush=True)
