"""Per-problem gate (step-3 item (d), first cut): the standard
chain gate with a jsonl sidecar — per-problem outcome + the full
greedy chain + wandering/identity signatures. Same seeds/oracle
as gate_eval (results comparable to gate_ckpt numbers).

    gate_pp.py <ckpt> <d> <layers> <ffn> <heads> <label>
Sidecar: logs/pp_<label>.jsonl
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

ckpt, d, layers, ffn, heads, label = (
    sys.argv[1], int(sys.argv[2]), int(sys.argv[3]),
    int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(tok.vocab), d=d, layers=layers,
                    heads=heads, ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

out = open(f"logs/pp_{label}.jsonl", "w")
solves = {}
valid = tried = 0
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        s = 0
        for i in range(G.GATE_N):
            p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
            if p is None:
                continue
            root = cur = f"Integral({sp.sstr(p._expr)}, x)"
            visited = {cur.replace(" ", "")}
            chain, done = [], False
            invalid_waves = 0
            for ply in range(12):
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                texts, _, _ = G.sample_wave_lp(
                    model, tok, prompt,
                    [G.GATE_BAND + i * 31 + ply * 7 + b
                     for b in range(G.B)], dev)
                tried += len(texts)
                distinct = [t for t in dict.fromkeys(texts)
                            if t and t.replace(" ", "") not in visited]
                wv = verify_wave(cur, distinct) if distinct else {}
                nxt = None
                for t in texts:
                    ok, so = wv.get(t, (False, False))
                    if ok and t.replace(" ", "") not in visited:
                        valid += 1
                        if nxt is None:
                            nxt = "SOLVED" if so else t
                if nxt == "SOLVED":
                    done = True
                    chain.append("SOLVED")
                    break
                if nxt is None:
                    invalid_waves += 1
                    break
                cur = nxt
                chain.append(cur)
                visited.add(cur.replace(" ", ""))
            s += done
            # wandering signature: did Integral-count ever rise?
            ic = [c.count("Integral(") for c in [root] + chain
                  if c != "SOLVED"]
            out.write(json.dumps({
                "level": lv, "i": i, "root": root, "solved": done,
                "plies": len(chain), "chain": chain,
                "wander": bool(any(b > a for a, b in zip(ic, ic[1:]))),
            }) + "\n")
        solves[lv] = s
out.close()
tot = sum(solves.values())
print(f"{label} gate: {solves} = {tot}/120 @ "
      f"{100*valid/max(tried,1):.2f}%  [pp sidecar logs/pp_{label}.jsonl]",
      flush=True)
