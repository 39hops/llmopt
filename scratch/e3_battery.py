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
while len(rows) < 50:
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

with torch.no_grad():
    conts = []
    for text, ids in rows:
        cur = list(ids)
        out = []
        for _ in range(64):
            nxt = int(model(torch.tensor([cur]))[0, -1].argmax())
            out.append(nxt)
            if nxt == tok.eos_id:
                break
            cur.append(nxt)
        conts.append(out)

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
print(f"greedy lens: min {min(map(len, conts))} "
      f"max {max(map(len, conts))} "
      f"eos-terminated {sum(o[-1] == tok.eos_id for o in conts)}/50")
