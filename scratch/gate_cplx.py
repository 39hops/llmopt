"""Gate a complex-FFN checkpoint (mirror of gate_ckpt.py).

    python scratch/gate_cplx.py <ckpt> <alpha> <label> [d layers ffn heads]
"""
import sys

import torch

sys.path.insert(0, "scripts")
sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import complex_model as C  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer  # noqa: E402

ckpt, alpha, label = sys.argv[1], sys.argv[2], sys.argv[3]
d, layers, ffn, heads = (int(x) for x in (sys.argv[4:8] or
                                          [384, 8, 1536, 6]))
C.set_alpha(alpha)
tok = MathTokenizer()
dev = ("mps" if torch.backends.mps.is_available() else
       "cuda" if torch.cuda.is_available() else "cpu")
model = C.build_complex_model(len(tok.vocab), d=d, layers=layers,
                              heads=heads, ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
solves, valid = G.gate_eval(model, tok, dev)
print(f"{label} gate: {solves} = {sum(solves.values())}/120 "
      f"@ {valid:.2f}%", flush=True)
