"""ZX gate (pre-reg 2026-07-26, the factorial's ZX column).

120 held-out cur states (zx_farm1_held.jsonl, spiders<=12 so the
pyzx tensor oracle is affordable), greedy decode, a solve =
in-grammar parse AND invariant-distinct AND compare_tensors-equal
(fork-walled 10s; walls excluded from denominator, counted).

    VOCAB_EXTRA="in(,out(,Z(,X(,P(,H(,:" .venv/bin/python \
        scratch/gate_zx.py <ckpt> <real|cplx> <label>
"""
import json
import os
import random
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")

import torch

from adjudicate_zx import parse, semantic
from llmopt.train.mathnative import MathTokenizer, build_model

ckpt, arch, label = sys.argv[1], sys.argv[2], sys.argv[3]
extra = os.environ["VOCAB_EXTRA"]
tok = MathTokenizer(extra=extra.split(","))
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
if arch == "cplx":
    import complex_model as C
    # latent checkpoints need the training-time STE forward
    # (raw latents without quantize = a never-used function)
    C.set_alpha(os.environ.get("CPLX_ALPHA", "none"))
    model = C.build_complex_model(len(tok.vocab), d=384, layers=8,
                                  heads=6, ffn=1536).to(dev)
else:
    model = build_model(len(tok.vocab), d=384, layers=8, heads=6,
                        ffn=1536).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()

rows = [json.loads(l) for l in open("data/zx_farm1_held.jsonl")]


def n_spiders(s):
    return len(parse(s)[2])


pool = [r for r in rows if n_spiders(r["cur"]) <= 12]
random.Random("zx-gate-0").shuffle(pool)
# stratify by size class as available
by_size = {}
for r in pool:
    by_size.setdefault(r["size"], []).append(r)
probes = []
sizes = sorted(by_size)
i = 0
while len(probes) < 120 and any(by_size.values()):
    s = sizes[i % len(sizes)]
    if by_size[s]:
        probes.append(by_size[s].pop())
    i += 1
print(f"{label}: {len(probes)} probes "
      f"(sizes {sorted({p['size'] for p in probes})}, "
      f"kinds {sorted({p['kind'] for p in probes})})", flush=True)


@torch.no_grad()
def greedy(cur, max_new=700):
    prompt = f"Current: {cur}\nHints: none\nStep:"
    ids = torch.tensor([tok.encode(prompt)], device=dev)
    logits, past = model(ids, use_cache=True)
    out = []
    nl = tok.id["\n"]
    for _ in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if nxt in (nl, tok.eos_id, tok.pad_id):
            break
        out.append(nxt)
        logits, past = model(
            torch.tensor([[nxt]], device=dev), past=past)
    return tok.decode(out).strip()


def invariants(s):
    ins, outs, sp, ed = parse(s)
    return (sorted(sp.values()),
            sorted(ed.values()), len(sp), len(ed))


solved = walls = parsefail = identity = unsound = 0
per_kind = {}
for i, r in enumerate(probes):
    cand = greedy(r["cur"])
    try:
        pv = invariants(cand)
    except Exception:
        parsefail += 1
        continue
    if pv == invariants(r["cur"]):
        identity += 1  # conservative: all farmed kinds move an
        continue       # invariant, so equal-invariants = identity
    v = semantic(r["cur"], cand, wall=10)
    if v == "WALL":
        walls += 1
    elif v is True:
        solved += 1
        per_kind[r["kind"]] = per_kind.get(r["kind"], 0) + 1
    else:
        unsound += 1
    if (i + 1) % 20 == 0:
        print(f"  {i+1}/{len(probes)} solved {solved} "
              f"parsefail {parsefail} identity {identity} "
              f"unsound {unsound} walls {walls}", flush=True)

den = len(probes) - walls
print(f"{label} ZX gate: {solved}/{den} (walls {walls} excluded) "
      f"| parsefail {parsefail} identity {identity} "
      f"unsound {unsound} | by kind {per_kind}", flush=True)
