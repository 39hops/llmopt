"""Pincer R0: conjecture-leg readout (spec
2026-07-26-reverse-llmue-pincer.md, cell R0; pre-reg in RESULTS).

fmt_oneshot_1p.pt proposes k=8 T-sampled answers per gate problem
(same 120 problems/seeds as gate_ckpt); a conjecture HIT is a
candidate that forward-verifies from the ROOT as equivalence-valid
AND integral-free (verify_wave's solved flag) — i.e. the whole
problem falls at ply 0. Books yield per level + pp sidecar.

    python scratch/pincer_r0.py [ckpt=checkpoints/fmt_oneshot_1p.pt]
Sidecar: logs/pp_r0_conjecture.jsonl
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
ckpt = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/fmt_oneshot_1p.pt"
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

out = open("logs/pp_r0_conjecture.jsonl", "w")
hits = {}
valid_nonsolve = 0
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        h = 0
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            root = f"Integral({sp.sstr(p._expr)}, x)"
            prompt = tok.encode(f"Current: {root}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [G.GATE_BAND + i * 31 + b for b in range(K)], dev)
            distinct = [t for t in dict.fromkeys(texts) if t]
            wv = verify_wave(root, distinct) if distinct else {}
            solved_cands = [t for t in distinct
                            if wv.get(t, (False, False))[1]]
            valid_cands = [t for t in distinct
                           if wv.get(t, (False, False))[0]]
            valid_nonsolve += len(set(valid_cands) - set(solved_cands))
            hit = bool(solved_cands)
            h += hit
            out.write(json.dumps({
                "level": lv, "i": i, "root": root, "hit": hit,
                "n_distinct": len(distinct),
                "n_valid": len(valid_cands),
                "n_solved": len(solved_cands),
                "answers": solved_cands[:3],
            }) + "\n")
        hits[lv] = h
        print(f"  L{lv}: {h}/{G.GATE_N}", flush=True)
out.close()
tot = sum(hits.values())
print(f"R0 conjecture yield: {hits} = {tot}/120 (k={K}); "
      f"valid-but-unsolved candidates {valid_nonsolve} "
      f"[sidecar logs/pp_r0_conjecture.jsonl]", flush=True)
