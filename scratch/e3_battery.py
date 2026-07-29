"""E3 battery (axiom GO 2026-07-29, 50 rows): exact-mode paired
GREEDY gate. House side: 50 fresh gate-style prompts (seed band
disjoint from battery20 and the GATE band), fp32 eager greedy
continuations (<=64 tokens, stop at eos) from the S2 scorer.
Axiom decodes the same prompts in exact mode and diffs token-
identically. Emits data/e3_battery50{,_meta.jsonl,_greedy.txt}
+ sha256 pins.
"""
import hashlib
import json
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

from bench_step_tokens import _gen_isolated  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

E3_BAND = 777000  # disjoint from GATE_BAND and battery20 rows
tok = MathTokenizer()
model = build_model(len(tok.vocab), d=256, layers=8, heads=4, ffn=1024)
model.load_state_dict(torch.load("checkpoints/scorer_s2_dist.pt",
                                 map_location="cpu", weights_only=True))
model = model.float().eval()

rows = []
i = 0
while len(rows) < 120:  # candidates; margin filter trims to 50
    p = _gen_isolated(3 + len(rows) % 5, E3_BAND + i)
    i += 1
    if p is None:
        continue
    text = f"Current: Integral({sp.sstr(p._expr)}, x)\nHints: none\nStep: "
    try:
        ids = tok.encode(text)
    except ValueError:
        continue
    if len(ids) <= 400:
        rows.append((text, ids))

MARGIN_BAR = 0.05  # rows below this are near-tie land; excluded so
# any axiom-side token diff is a REAL decode divergence by construction
nl = tok.encode("\n")[-1]
with torch.no_grad():
    conts, margins, kept = [], [], []
    for text, ids in rows:
        cur = list(ids)
        out, mmin = [], float("inf")
        for _ in range(64):
            lg = model(torch.tensor([cur]))[0, -1]
            top2 = torch.topk(lg, 2).values
            m = float(top2[0] - top2[1])
            if m < MARGIN_BAR:  # truncate BEFORE the first near-tie:
                break  # every retained token is certified high-margin
            mmin = min(mmin, m)
            nxt = int(lg.argmax())
            out.append(nxt)
            if nxt in (tok.eos_id, nl):  # Step lines end at newline
                break
            cur.append(nxt)
        if len(out) >= 8:  # enough tokens to exercise the decode path
            conts.append(out)
            margins.append(mmin)
            kept.append((text, ids))
assert len(kept) >= 50, f"only {len(kept)} margin-clean rows"
rows, conts, margins = kept[:50], conts[:50], margins[:50]
print(f"margin filter: kept {len(kept)} of 120 candidates; "
      f"battery min margin {min(margins):.3f}")

with open("data/e3_battery50.txt", "w") as f:
    for _, ids in rows:
        f.write(" ".join(map(str, ids)) + "\n")
with open("data/e3_battery50_meta.jsonl", "w") as f:
    for k, (text, _) in enumerate(rows):
        f.write(json.dumps({"idx": k, "text": text}) + "\n")
with open("data/e3_expected_greedy.txt", "w") as f:
    for out in conts:
        f.write(" ".join(map(str, out)) + "\n")

for fn in ("data/e3_battery50.txt", "data/e3_battery50_meta.jsonl",
           "data/e3_expected_greedy.txt"):
    sha = hashlib.sha256(open(fn, "rb").read()).hexdigest()[:16]
    print(f"{fn} sha256[:16]={sha}")
nl = tok.encode("\n")[-1]
print(f"greedy lens: min {min(map(len, conts))} "
      f"max {max(map(len, conts))} "
      f"terminated {sum(o[-1] in (tok.eos_id, nl) for o in conts)}/50")
