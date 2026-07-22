"""Holdout v2: exclude-guarded (the doctrine I broke in v1 — 281
collisions caught by the audit). Probes drawn from band 88M but
each slot advances its seed until the expr is NOT in the corpus
cur-set. Usage: holdout_v2.py <ckpt> <d> <layers> <ffn> <heads> <label>"""
import sys, glob, json
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated

ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
BAND = 88_000_000
corpus = set()
for f in glob.glob("data/*.jsonl"):
    for line in open(f):
        try:
            c = json.loads(line).get("cur", "")
        except Exception:
            continue
        if c.startswith("Integral("):
            corpus.add(c[9:-4].replace(" ", ""))
print(f"[exclude set] {len(corpus)} corpus exprs", flush=True)
probes = []
skipped = 0
for lv in G.GATE_LEVELS:
    made = 0
    off = 0
    while made < G.GATE_N and off < 4000:
        p = _gen_isolated(lv, BAND + 1000 * lv + off)
        off += 1
        if p is None:
            continue
        e = sp.sstr(p._expr)
        if e.replace(" ", "") in corpus:
            skipped += 1
            continue
        probes.append((lv, made, e))
        made += 1
print(f"[holdout-v2] {len(probes)} probes, {skipped} collisions "
      f"skipped", flush=True)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
from bench_verify_fast import verify_wave
solves = {lv: 0 for lv in G.GATE_LEVELS}
valid = tried = 0
with torch.no_grad():
    for lv, i, e in probes:
        cur = f"Integral({e}, x)"
        visited = {cur.replace(" ", "")}
        done = False
        for ply in range(12):
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [BAND + i * 31 + ply * 7 + b for b in range(G.B)], dev)
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
                done = True; break
            if nxt is None:
                break
            cur = nxt; visited.add(cur.replace(" ", ""))
        solves[lv] += done
print(f"{label} HOLDOUT-v2 gate: {solves} = "
      f"{sum(solves.values())}/120 @ {100*valid/max(tried,1):.2f}%",
      flush=True)
