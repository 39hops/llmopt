"""Pincer R1 backward-validity probe (spec
2026-07-26-reverse-llmue-pincer.md, cell R1; pre-reg in RESULTS).

For fresh gate-band states t (pooled distinct mid-chain states
from the nine pp sidecars — exclude-guarded by band; skew honest:
L4/L6 thin), the backward crystal proposes k=8 T-sampled
predecessors p. A candidate is VALID iff p != t (skeleton) and
the FORWARD step p -> t verifies (existing verify machinery; the
bidirectional-cheat fence: forward-verify, never corpus match).

    python scratch/pincer_r1_probe.py <ckpt> <label>
Sidecar: logs/pp_<label>.jsonl
"""
import glob
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch

from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G
from bench_verify_fast import verify_wave

K = 8
ckpt, label = sys.argv[1], sys.argv[2]
norm = lambda s: s.replace(" ", "")  # noqa: E731

seen, states = set(), []
for f in sorted(glob.glob("logs/pp_*.jsonl")):
    if "r0" in f or label in f:
        continue
    for r in map(json.loads, open(f)):
        for c in r.get("chain", []):
            if c != "SOLVED" and norm(c) not in seen:
                seen.add(norm(c))
                states.append((r["level"], c))
print(f"{len(states)} probe states", flush=True)

tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

out = open(f"logs/pp_{label}.jsonl", "w")
per_lv = {}
cand_valid = cand_tried = 0
with torch.no_grad():
    for si, (lv, t) in enumerate(states):
        prompt = tok.encode(f"Current: {t}\nHints: none\nStep: ")
        texts, _, _ = G.sample_wave_lp(
            model, tok, prompt,
            [G.GATE_BAND + si * 31 + b for b in range(K)], dev)
        distinct = [p for p in dict.fromkeys(texts)
                    if p and norm(p) != norm(t)]
        cand_tried += len(texts)
        good = []
        for p in distinct:
            wv = verify_wave(p, [t])
            if wv.get(t, (False, False))[0]:
                good.append(p)
        cand_valid += len(good)
        cov, n = per_lv.get(lv, (0, 0))
        per_lv[lv] = (cov + bool(good), n + 1)
        out.write(json.dumps({
            "level": lv, "state": t, "n_distinct": len(distinct),
            "n_valid": len(good), "preds": good[:3],
        }) + "\n")
out.close()
cov = {lv: f"{c}/{n}" for lv, (c, n) in sorted(per_lv.items())}
tot_c = sum(c for c, _ in per_lv.values())
print(f"{label}: per-cand validity {100*cand_valid/max(cand_tried,1):.1f}% "
      f"({cand_valid}/{cand_tried}); state coverage {tot_c}/{len(states)} "
      f"{cov}  [sidecar logs/pp_{label}.jsonl]", flush=True)
