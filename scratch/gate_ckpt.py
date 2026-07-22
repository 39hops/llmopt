import sys, torch
sys.path.insert(0, 'scripts')
sys.path.insert(0, '.')
ckpt, d, layers, ffn, heads, label = (sys.argv[1], int(sys.argv[2]),
    int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6])
from llmopt.train.mathnative import MathTokenizer, build_model
import step_grpo_micro as G
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
model = build_model(len(tok.vocab), d=d, layers=layers, heads=heads,
                    ffn=ffn).to(dev)
model.load_state_dict(torch.load(ckpt, map_location="cpu"))
model.eval()
solves, valid = G.gate_eval(model, tok, dev)
tot = sum(solves.values())
print(f"{label} gate: {solves} = {tot}/120 @ {valid:.2f}%", flush=True)
