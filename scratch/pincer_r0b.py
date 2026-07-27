"""R0b: collapse-ordered readout (pre-reg in RESULTS 2026-07-26
late). The honest Grover residue: does checking candidates in
descending model-mass order reach the first verified solution in
fewer ORACLE CALLS than random/sampling order? All candidates are
oracle-checked once (instrument cost, not protocol cost); orders
are then evaluated on the recorded truth.

    python scratch/pincer_r0b.py
Sidecar: logs/pp_r0b_readout.jsonl
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

K = 16
norm = lambda s: s.replace(" ", "")  # noqa: E731
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
model.load_state_dict(torch.load("checkpoints/fmt_oneshot_1p.pt",
                                 map_location="cpu"))
model.eval()


def seq_logp(prefix, targets):
    pre = tok.encode(prefix)
    encs = [(pre + tok.encode(t) + [tok.id["\n"]],
             len(tok.encode(t)) + 1) for t in targets]
    L = max(len(e) for e, _ in encs)
    ids = torch.tensor([e + [tok.pad_id] * (L - len(e))
                        for e, _ in encs], device=dev)
    mask = torch.tensor([[1] * len(e) + [0] * (L - len(e))
                         for e, _ in encs], device=dev)
    with torch.no_grad():
        lsm = torch.log_softmax(model(ids[:, :-1], mask[:, :-1]).float(),
                                -1)
    return [sum(float(lsm[b, j - 1, e[j]])
                for j in range(len(e) - nt, len(e)))
            for b, (e, nt) in enumerate(encs)]


out = open("logs/pp_r0b_readout.jsonl", "w")
agg = {"mass": [], "sample": [], "random": []}
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            root = f"Integral({sp.sstr(p._expr)}, x)"
            prompt = tok.encode(f"Current: {root}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [G.GATE_BAND + i * 31 + b for b in range(K)], dev)
            cands = [t for t in dict.fromkeys(texts) if t]
            if not cands:
                continue
            wv = verify_wave(root, cands)
            hit = [wv.get(t, (False, False))[1] for t in cands]
            h, n = sum(hit), len(cands)
            row = {"level": lv, "i": i, "n": n, "h": h}
            if h:
                lps = seq_logp(f"Current: {root}\nHints: none\nStep: ",
                               cands)
                order = sorted(range(n), key=lambda j: -lps[j])
                row["calls_mass"] = next(
                    k + 1 for k, j in enumerate(order) if hit[j])
                row["calls_sample"] = next(
                    k + 1 for k, x in enumerate(hit) if x)
                row["calls_random"] = (n + 1) / (h + 1)
                for m in ("mass", "sample", "random"):
                    agg[m].append(row[f"calls_{m}"])
            out.write(json.dumps(row) + "\n")
out.close()
n = len(agg["mass"])
print(f"R0b over {n} problems with >=1 hit: mean oracle calls "
      f"mass {sum(agg['mass'])/n:.2f} | sample "
      f"{sum(agg['sample'])/n:.2f} | random "
      f"{sum(agg['random'])/n:.2f}  [sidecar logs/pp_r0b_readout.jsonl]",
      flush=True)
