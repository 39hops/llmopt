"""Ceiling probe: which L7/L8 integrals can a checkpoint actually
solve? Same machinery as gate_eval but per-problem printout."""
from llmopt.common.device import pick_device
import sys, torch, sympy as sp
sys.path.insert(0, 'scripts'); sys.path.insert(0, '.')
ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

tok = MathTokenizer()
dev = pick_device()
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

N = 24
for lv in (7, 8):
    solved = 0
    for i in range(N):
        p = _gen_isolated(lv, G.GATE_BAND + 1000 * lv + i)
        if p is None:
            continue
        expr = sp.sstr(p._expr)
        cur = f"Integral({expr}, x)"
        visited = {cur.replace(" ", "")}
        done = False
        with torch.no_grad():
            for ply in range(12):
                prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
                texts, _, _ = G.sample_wave_lp(
                    model, tok, prompt,
                    [G.GATE_BAND + i * 31 + ply * 7 + b
                     for b in range(G.B)], dev)
                distinct = [t_ for t_ in dict.fromkeys(texts)
                            if t_ and t_.replace(" ", "") not in visited]
                wv = verify_wave(cur, distinct) if distinct else {}
                nxt = None
                for t_ in texts:
                    ok, so = wv.get(t_, (False, False))
                    if ok and t_.replace(" ", "") not in visited:
                        if nxt is None:
                            nxt = "SOLVED" if so else t_
                if nxt == "SOLVED":
                    done = True
                    break
                if nxt is None:
                    break
                cur = nxt
                visited.add(cur.replace(" ", ""))
        solved += done
        if done:
            print(f"  L{lv} SOLVED: {expr}", flush=True)
    print(f"{label} L{lv}: {solved}/{N}", flush=True)
