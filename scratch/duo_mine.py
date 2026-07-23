"""Duo miner (overnight flywheel): duo wave over a fresh band (spec 2026-07-22-duo-substrate, exp 1):
per ply, B/2 samples from TERNARY + B/2 from CHAMPION (budget-matched
vs a single model's B), merged and oracle-verified. Same 200-probe
rarity battery as gate_rarity.py (same seeds, same census)."""
import sys, glob, json, re
from collections import Counter
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated

import time
BAND = 91_000_000  # fresh mining band
N_PER_LEVEL = 400  # mine wide
HALF = G.B // 2

def skeleton(e: str) -> str:
    return re.sub(r"\d+", "#", e.replace(" ", ""))

skel_count = Counter()
for f in glob.glob("data/*.jsonl"):
    if "diet" in f:
        continue
    for line in open(f):
        try:
            c = json.loads(line).get("cur", "")
        except Exception:
            continue
        if c.startswith("Integral("):
            skel_count[skeleton(c[9:-4])] += 1

def binof(n):
    return ("unseen" if n == 0 else "rare" if n < 5
            else "mid" if n < 50 else "common")

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
print(f"[probes] {len(probes)}; bins "
      f"{dict(Counter(b for *_, b in probes))}", flush=True)

tok = MathTokenizer()
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
champ = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2304).to(dev)
champ.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_grown.pt", map_location="cpu"))
champ.eval()
tern = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                   ffn=2048).to(dev)
tern.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_ternary.pt", map_location="cpu"))
tern.eval()

from bench_verify_fast import verify_wave
solved = Counter(); total = Counter()
valid = tried = 0
dump = open("data/duo_mined_shard1.jsonl", "a")  # STREAM rows
with torch.no_grad():
    for lv, i, e, bn in probes:
        total[bn] += 1
        cur = f"Integral({e}, x)"
        visited = {cur.replace(" ", "")}
        done = False
        chain = []
        for ply in range(12):
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            seeds = [BAND + i * 31 + ply * 7 + b for b in range(G.B)]
            t_tern, _, _ = G.sample_wave_lp(tern, tok, prompt,
                                            seeds[:HALF], dev)
            t_champ, _, _ = G.sample_wave_lp(champ, tok, prompt,
                                             seeds[HALF:], dev)
            texts = t_tern + t_champ
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
                for t in texts:
                    ok, so = wv.get(t, (False, False))
                    if ok and so:
                        chain.append(t); break
                done = True; break
            if nxt is None:
                break
            chain.append(nxt)
            cur = nxt; visited.add(cur.replace(" ", ""))
        solved[bn] += done
        if done and bn in ("rare", "unseen"):
            c2 = f"Integral({e}, x)"
            for stp in chain:
                dump.write(json.dumps({"cur": c2, "nxt": stp,
                                       "level": lv}) + "\n")
                c2 = stp
            dump.flush()
print(f"DUO (8T+8fp32) RARITY curve: "
      f"{ {b: f'{solved[b]}/{total[b]}' for b in ('common','mid','rare','unseen')} }"
      f" | overall {sum(solved.values())}/{len(probes)} @ "
      f"{100*valid/max(tried,1):.2f}%", flush=True)
