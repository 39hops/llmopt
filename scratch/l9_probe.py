"""L9 probe: 24 fresh L9a problems (band 90M — disjoint from the
farm's 72/73M and roots_c1), gate_eval-style rollout, 12 plies.
Usage: l9_probe.py <ckpt> <d> <layers> <ffn> <heads> <label>"""
import sys
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
BAND = 90_000_000
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

solved = tried = 0
with torch.no_grad():
    seed = 0
    while tried < 24 and seed < 400:
        seed += 1
        p = _gen_isolated(9, BAND + seed)
        if p is None:
            continue
        tried += 1
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        visited = {cur.replace(" ", "")}
        done = False
        for ply in range(12):
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [BAND + tried * 31 + ply * 7 + b for b in range(G.B)],
                dev)
            distinct = [t for t in dict.fromkeys(texts)
                        if t and t.replace(" ", "") not in visited]
            wv = verify_wave(cur, distinct) if distinct else {}
            nxt = None
            for t in texts:
                ok, so = wv.get(t, (False, False))
                if ok and t.replace(" ", "") not in visited:
                    nxt = "SOLVED" if so else t
                    break
            if nxt == "SOLVED":
                done = True
                break
            if nxt is None:
                break
            cur = nxt
            visited.add(cur.replace(" ", ""))
        solved += done
print(f"{label} L9 probe: {solved}/{tried} (band {BAND})", flush=True)
