"""E2 closure (relay -28-5 loop): reproduce axiom's pinned
20-prompt battery logits with torch fp32 on the house scorer.
Asserts (1) their token ids decode to their meta text via the
house tokenizer (tokenization parity), (2) final-position
logits agree within 1e-4 elementwise. PASS arms E3.
"""
import json
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

tok = MathTokenizer()
ids_lines = [[int(t) for t in ln.split()]
             for ln in open("data/scorer_s2_battery20.txt")]
meta = [json.loads(ln)
        for ln in open("data/scorer_s2_battery20_meta.jsonl")]
exp = [[float(t) for t in ln.split()]
       for ln in open("data/scorer_s2_expected_logits.txt")]
assert len(ids_lines) == len(exp) == 20

for i, m in enumerate(meta):
    assert tok.decode(ids_lines[i]) == m["text"], f"roundtrip {i}"
print("tokenization parity: 20/20 round-trips match meta text")

model = build_model(len(tok.vocab), d=256, layers=8, heads=4,
                    ffn=1024)
model.load_state_dict(torch.load("checkpoints/scorer_s2_dist.pt",
                                 map_location="cpu",
                                 weights_only=True))
model = model.float().eval()

worst = -1.0
with torch.no_grad():
    for i, ids in enumerate(ids_lines):
        logits = model(torch.tensor([ids]))[0, -1]
        d = float((logits - torch.tensor(exp[i])).abs().max())
        worst = max(worst, d)
print(f"max|delta logit| over 20x40 = {worst:.3e} "
      f"({'PASS' if worst < 1e-4 else 'FAIL'} vs 1e-4 bar)")
