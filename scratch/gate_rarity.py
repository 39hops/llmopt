"""Rarity-stratified gate (schedule-law queue item 2): capability as
a curve over expression rarity, not a scalar. Rarity = skeleton
frequency — integer constants normalized to '#', skeleton counted in
the corpus cur-set. Probes drawn WITHOUT exclude-filtering (the full
spectrum is the point); bins: common / mid / rare / unseen-skeleton.
Usage: gate_rarity.py <ckpt> <d> <layers> <ffn> <heads> <label>"""
import sys, glob, json, re
from collections import Counter
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated

ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
BAND = 88_000_000
N_PER_LEVEL = 40   # wider than GATE_N so every bin gets mass

def skeleton(e: str) -> str:
    return re.sub(r"\d+", "#", e.replace(" ", ""))

skel_count = Counter()
for f in glob.glob("data/*.jsonl"):
    if "diet" in f:  # diet files recompose existing rows: no double count
        continue
    for line in open(f):
        try:
            c = json.loads(line).get("cur", "")
        except Exception:
            continue
        if c.startswith("Integral("):
            skel_count[skeleton(c[9:-4])] += 1
print(f"[skeletons] {len(skel_count)} distinct in corpus cur-set",
      flush=True)

def binof(n: int) -> str:
    if n == 0: return "unseen"
    if n < 5: return "rare"
    if n < 50: return "mid"
    return "common"

probes = []
for lv in G.GATE_LEVELS:
    made, off = 0, 0
    while made < N_PER_LEVEL and off < 6000:
        p = _gen_isolated(lv, BAND + 1000 * lv + off)
        off += 1
        if p is None:
            continue
        e = sp.sstr(p._expr)
        probes.append((lv, made, e, binof(skel_count[skeleton(e)])))
        made += 1
dist = Counter(b for _, _, _, b in probes)
print(f"[probes] {len(probes)} total; bins {dict(dist)}", flush=True)

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
from bench_verify_fast import verify_wave
solved = Counter(); total = Counter()
by_level = {lv: Counter() for lv in G.GATE_LEVELS}
valid = tried = 0
with torch.no_grad():
    for lv, i, e, bn in probes:
        total[bn] += 1
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
        if done:
            solved[bn] += 1
        by_level[lv][bn, done] += 1
curve = {b: f"{solved[b]}/{total[b]}"
         for b in ("common", "mid", "rare", "unseen")}
print(f"{label} RARITY curve: {curve} | overall "
      f"{sum(solved.values())}/{len(probes)} @ "
      f"{100*valid/max(tried,1):.2f}%", flush=True)
for lv in G.GATE_LEVELS:
    row = {b: f"{by_level[lv][b, True]}/"
              f"{by_level[lv][b, True] + by_level[lv][b, False]}"
           for b in ("common", "mid", "rare", "unseen")
           if by_level[lv][b, True] + by_level[lv][b, False]}
    print(f"  L{lv}: {row}", flush=True)
