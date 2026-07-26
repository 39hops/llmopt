import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave
tok = MathTokenizer()
dev = "cuda"
torch.backends.cuda.matmul.allow_tf32 = True
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2304).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_grown.pt", map_location="cpu"))
model.eval()
BAND = 90_000_000
solved = tried = 0
with torch.no_grad():
    seed = 0
    while tried < 24 and seed < 400:
        seed += 1
        p = _gen_isolated(9, BAND + seed)
        if p is None: continue
        tried += 1
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        visited = {cur.replace(" ", "")}
        done = False
        for ply in range(12):
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [BAND + tried*31 + ply*7 + b for b in range(G.B)], dev)
            distinct = [t for t in dict.fromkeys(texts)
                        if t and t.replace(" ", "") not in visited]
            wv = verify_wave(cur, distinct) if distinct else {}
            nxt = None
            for t in texts:
                ok, so = wv.get(t, (False, False))
                if ok and t.replace(" ", "") not in visited:
                    nxt = "SOLVED" if so else t
                    break
            if nxt == "SOLVED": done = True; break
            if nxt is None: break
            cur = nxt; visited.add(cur.replace(" ", ""))
        solved += done
print(f"CHAMPION-ON-CUDA L9 probe: {solved}/{tried}", flush=True)
