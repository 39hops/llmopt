"""Sensitivity-wall anatomy (Artin 2026-07-27: "find WHERE the wall
lives"): single-tensor Q=16 snap ablation on the 19M crystal. For
each 2-D tensor alone-snapped (rest fp32), measure teacher-forced
divergence vs control on gen-4 rows: mean KL + argmax-flip rate.
Localization instrument (CPU, no gate contention with the births);
top culprits earn real gates later. House pre-reg guess: head/attn
out-projections carry the wall, ffn interiors tolerant.
"""
import json
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
from llmopt.train.mathnative import MathTokenizer, build_model

Q = 16
tok = MathTokenizer()
model = build_model(len(tok.vocab), d=384, layers=8, heads=6, ffn=1536)
sd0 = torch.load("checkpoints/mathnative_19m.pt", map_location="cpu")
model.load_state_dict(sd0)
model.eval()

rows = []
for line in open("data/micromodel_gen4_sidecar.jsonl"):
    r = json.loads(line)
    t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
    try:
        ids = tok.encode(t)
    except Exception:
        continue
    if 32 <= len(ids) <= 256:
        rows.append(ids)
    if len(rows) >= 48:
        break
batch = torch.full((len(rows), max(len(r) for r in rows)), 0,
                   dtype=torch.long)
mask = torch.zeros_like(batch, dtype=torch.bool)
for i, r in enumerate(rows):
    batch[i, :len(r)] = torch.tensor(r)
    mask[i, :len(r)] = True

def snap(w):
    s = w.abs().mean().clamp(min=1e-8)
    v = w / s
    best = torch.round(v)
    err = (v - best).abs()
    for q in range(2, Q + 1):
        c = torch.round(v * q) / q
        e = (v - c).abs()
        m = e < err
        best = torch.where(m, c, best)
        err = torch.where(m, e, err)
    return s * best

with torch.no_grad():
    ref = model(batch)
    ref = ref[0] if isinstance(ref, tuple) else ref
    refp = F.log_softmax(ref[mask], -1)
    refa = refp.argmax(-1)

out = []
msd = model.state_dict()
for name, w in msd.items():
    if w.ndim != 2 or not w.is_floating_point():
        continue
    orig = w.clone()
    w.copy_(snap(w.float()).to(w.dtype))
    with torch.no_grad():
        lg = model(batch)
        lg = lg[0] if isinstance(lg, tuple) else lg
        lp = F.log_softmax(lg[mask], -1)
        kl = F.kl_div(lp, refp, log_target=True,
                      reduction="batchmean").item()
        flip = (lp.argmax(-1) != refa).float().mean().item()
    w.copy_(orig)
    out.append((kl, flip, name, w.numel()))
    print(f"{kl:10.6f} kl  {flip:7.4f} flip  {name}", flush=True)

out.sort(reverse=True)
print("\n=== TOP 10 by KL ===")
for kl, flip, name, n in out[:10]:
    print(f"{kl:10.6f} kl  {flip:7.4f} flip  {n/1e3:8.0f}k  {name}")
tot = sum(kl for kl, *_ in out)
top5 = sum(kl for kl, *_ in out[:5])
print(f"\ntop-5 tensors carry {100*top5/tot:.1f}% of total single-tensor KL")
