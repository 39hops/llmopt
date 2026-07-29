"""FARMER PROBE (pre-reg 2026-07-29: escalation-engine cell 6,
Artin's reverse-self-learner riff). A full-reverse d64 birth
(sym_birth REV=2, SKIP_GATE) plays farmer: sample predecessor
candidates for NOVEL band expressions (gate-band + 50k offset,
disjoint from the gate), verify each by FORWARD rule application
(fork-boxed verify_wave: cand -> seed must be a valid step), and
score verified-distinct-NOVEL yield per 1000 samples + wall time.
Novel = candidate absent from the entire gen-4 corpus (cur+nxt).
Usage: CKPT=checkpoints/sym_birth_dense_revfarm_ema.pt \
       .venv/bin/python scratch/farmer_probe.py
"""
import os
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402
from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = os.environ.get("CKPT",
                      "checkpoints/sym_birth_dense_revfarm_ema.pt")
D, LAYERS, FFN, HEADS = 64, 8, 256, 4
SEED_OFF = 50_000  # novel band, disjoint from the gate's i<24
N_PER_LV = 25  # x5 levels x B=8 samples = 1000

tok = MathTokenizer()
dev = ("cuda" if torch.cuda.is_available() else
       "mps" if torch.backends.mps.is_available() else "cpu")
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu",
                                 weights_only=True))
model.eval()

SEEDMODE = os.environ.get("SEEDMODE", "band")  # band | corpus
known = set()
corpus = load_rows(gen4=True)
for r in corpus:
    known.add(r["cur"].replace(" ", ""))
    known.add(r["nxt"].replace(" ", ""))
print(f"corpus expressions (novelty fence): {len(known)}", flush=True)

if SEEDMODE == "corpus":  # control: in-distribution prompts —
    import random  # seeds are corpus LATER-states (the flipped
    by_lv = {lv: [] for lv in G.GATE_LEVELS}  # training prompt
    for r in corpus:  # distribution); novelty fence unchanged
        if r.get("level") in by_lv:
            by_lv[r["level"]].append(r["nxt"])
    for lv in by_lv:
        random.Random(13).shuffle(by_lv[lv])

stats = {lv: dict(samp=0, ver=0, vdn=0) for lv in G.GATE_LEVELS}
seen_preds = set()  # distinct fence, global
t0 = time.time()
with torch.no_grad():
    for lv in G.GATE_LEVELS:
        for i in range(N_PER_LV):
            if SEEDMODE == "corpus":
                if i >= len(by_lv[lv]):
                    continue
                s = by_lv[lv][i]
            else:
                p = _gen_isolated(lv, G.GATE_BAND + SEED_OFF
                                  + 1000 * lv + i)
                if p is None:
                    continue
                s = f"Integral({sp.sstr(p._expr)}, x)"
            prompt = tok.encode(f"Current: {s}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [G.GATE_BAND + SEED_OFF + i * 31 + b
                 for b in range(G.B)], dev)
            stats[lv]["samp"] += len(texts)
            s_norm = s.replace(" ", "")
            for cand in dict.fromkeys(texts):
                c_norm = cand.replace(" ", "") if cand else ""
                if not c_norm or c_norm == s_norm:
                    continue  # verified AND distinct doctrine
                ok, _solved = verify_wave(cand, [s]).get(
                    s, (False, False))
                if not ok:
                    continue
                stats[lv]["ver"] += 1
                if c_norm not in known and c_norm not in seen_preds:
                    seen_preds.add(c_norm)
                    stats[lv]["vdn"] += 1
        print(f"L{lv}: {stats[lv]}", flush=True)

dt = time.time() - t0
tot = {k: sum(s[k] for s in stats.values())
       for k in ("samp", "ver", "vdn")}
print(f"FARMER: {tot['vdn']} verified-distinct-novel / "
      f"{tot['samp']} samples "
      f"({1000 * tot['vdn'] / max(tot['samp'], 1):.1f}/1000) | "
      f"verified {tot['ver']} | {dt:.0f}s "
      f"({tot['samp'] / dt:.2f} samp/s, "
      f"{tot['vdn'] / dt:.3f} novel/s)", flush=True)
