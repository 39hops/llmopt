"""Deploy-ternarize the NNUE-metabolized latents, honest gate +
L9 probe on cuda. Doctrine: gate the DEPLOYED 1.58-bit snapshot."""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5*s, torch.zeros_like(w),
                       torch.sign(w)*s)

sd = torch.load("checkpoints/mathnative_gen6_ternary_latent.pt",
                map_location="cpu")
dep = {}
for k, W in sd.items():
    if W.dim() == 2 and "emb" not in k and W.shape[0] != 40:
        dep[k] = ternary(W.float())
    else:
        dep[k] = W
torch.save(dep, "checkpoints/ternary_control_deployed.pt")
tok = MathTokenizer()
dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32 = True
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2048).to(dev)
model.load_state_dict(dep)
model.eval()
solves, valid = G.gate_eval(model, tok, dev)
print(f"TERNARY-CONTROL deployed gate: {solves} = "
      f"{sum(solves.values())}/120 @ {valid:.2f}%", flush=True)

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
print(f"TERNARY-CONTROL L9 probe: {solved}/{tried}", flush=True)
