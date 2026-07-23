"""PRACTICE MODE, model-side (the mirror of axiom's arg-10):
duo-wave rollouts that (1) BANK verified steps from ALL attempts —
solved or not (the solved-only leak fix, Artin) — tagging rows by
outcome so the gen-8 A/B can split them; (2) LOG stuck states —
the exact cur where every unsolved attempt died — to a worklist in
axiom's format ({id, level, root, from, why, plies}), ready for the
stuck-state exchange AND as maximum-surprise metabolic v4 food.

Usage: practice_mine.py <n_per_level> <band> <out_tag>
Outputs (streamed): data/practice_rows_<tag>.jsonl
                    data/stuck_states_<tag>.jsonl
"""
import sys, glob, json, re
from collections import Counter
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

N_PER_LEVEL, BAND, TAG = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
HALF = G.B // 2

def skeleton(e):
    return re.sub(r"\d+", "#", e.replace(" ", ""))

skel_count = Counter()
for f in glob.glob("data/*.jsonl"):
    if "diet" in f or "practice" in f or "stuck" in f:
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

rows_f = open(f"data/practice_rows_{TAG}.jsonl", "a")
stuck_f = open(f"data/stuck_states_{TAG}.jsonl", "a")
stats = Counter()
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        made, off = 0, 0
        while made < N_PER_LEVEL and off < 8000:
            p = _gen_isolated(lv, BAND + 1000 * lv + off)
            off += 1
            if p is None:
                continue
            made += 1
            e = sp.sstr(p._expr)
            bn = binof(skel_count[skeleton(e)])
            root = f"Integral({e}, x)"
            cur = root
            visited = {cur.replace(" ", "")}
            chain = []
            solved = False
            for ply in range(12):
                prompt = tok.encode(
                    f"Current: {cur}\nHints: none\nStep: ")
                seeds = [BAND + made * 31 + ply * 7 + b
                         for b in range(G.B)]
                t1, _, _ = G.sample_wave_lp(tern, tok, prompt,
                                            seeds[:HALF], dev)
                t2, _, _ = G.sample_wave_lp(champ, tok, prompt,
                                            seeds[HALF:], dev)
                texts = t1 + t2
                distinct = [t for t in dict.fromkeys(texts)
                            if t and t.replace(" ", "") not in visited]
                wv = verify_wave(cur, distinct) if distinct else {}
                nxt = None
                for t in texts:
                    ok, so = wv.get(t, (False, False))
                    if ok and t.replace(" ", "") not in visited:
                        if nxt is None:
                            nxt = "SOLVED" if so else t
                if nxt == "SOLVED":
                    solved = True
                    for t in texts:
                        ok, so = wv.get(t, (False, False))
                        if ok and so:
                            chain.append((cur, t)); break
                    break
                if nxt is None:
                    # STUCK: every candidate failed here — log it
                    if ply > 0:  # zero-progress walls excluded (axiom's rule)
                        stuck_f.write(json.dumps({
                            "id": f"m-l{lv}-{made}#s{ply}",
                            "level": lv, "root": cur,
                            "from": root, "why": "dead_wave",
                            "plies": ply, "bin": bn}) + "\n")
                        stuck_f.flush()
                        stats[f"stuck_{bn}"] += 1
                    break
                chain.append((cur, nxt))
                cur = nxt
                visited.add(cur.replace(" ", ""))
            # BANK verified steps from ALL attempts (the leak fix)
            for c_, n_ in chain:
                rows_f.write(json.dumps({
                    "cur": c_, "nxt": n_, "level": lv,
                    "outcome": "solved" if solved else "unsolved",
                    "bin": bn}) + "\n")
            rows_f.flush()
            stats[f"{'solved' if solved else 'unsolved'}_{bn}"] += 1
        print(f"[L{lv}] {dict(stats)}", flush=True)
print(f"DONE {dict(stats)}", flush=True)
