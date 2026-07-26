"""Absorption decider: LR 1e-5 (the pilot's regime), 25-min STE
burst on cuda, late layers, band 98M. Counts fp32 updates where
w+delta == w (learning lost to rounding). Paired proxy pre/post."""
import sys, json, time
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5*s, torch.zeros_like(w),
                       torch.sign(w)*s)

class TLin(nn.Linear):
    def forward(self, x):
        if self.out_features == 40:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        wq = w + (ternary(w) - w).detach()
        return F.linear(x, wq, self.bias)

nn.Linear = TLin
torch.backends.cuda.matmul.allow_tf32 = True
tok = MathTokenizer()
dev = "cuda"
model = build_model(len(tok.vocab), d=512, layers=12, heads=8,
                    ffn=2048).to(dev)
model.load_state_dict(torch.load(
    "checkpoints/mathnative_gen6_ternary_latent.pt",
    map_location="cpu"))
for li, blk in enumerate(model.blocks):
    if li < 8:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]
opt = torch.optim.AdamW(params, lr=float(sys.argv[1]) if len(sys.argv) > 1 else 1e-5, weight_decay=0.0)
model.eval()
s0, v0 = G.gate_eval(model, tok, dev, n=8)
print(f"pre: proxy {sum(s0.values())} @ {v0:.1f}%", flush=True)
absorbed = total = 0
buf = []
SEED0 = 98_000_000
t0 = time.time()
cycle = 0
while time.time() - t0 < 25 * 60:
    cycle += 1
    for k in range(4):
        p = _gen_isolated(9, SEED0 + cycle * 17 + k)
        if p is None: continue
        cur = f"Integral({sp.sstr(p._expr)}, x)"
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [SEED0 + cycle * 31 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts) if t]
        wv = verify_wave(cur, distinct) if distinct else {}
        for t_ in distinct:
            ok, _s = wv.get(t_, (False, False))
            if ok:
                ids = tok.encode(f"Current: {cur}\nHints: none\n"
                                 f"Step: {t_}\n") + [tok.eos_id]
                if len(ids) <= 512:
                    buf.append(ids)
    if len(buf) >= 16:
        model.train()
        L = max(len(q) for q in buf)
        x = torch.tensor([q + [tok.pad_id]*(L-len(q)) for q in buf],
                         device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id)
        pre = [p.detach().clone() for p in params]
        opt.zero_grad(); loss.backward(); opt.step()
        for p_new, p_old in zip(params, pre):
            absorbed += int((p_new == p_old).sum())
            total += p_new.numel()
        buf = []
        model.eval()
    if cycle % 20 == 0:
        print(f"cycle {cycle}: absorbed {absorbed}/{total} "
              f"({100*absorbed/max(total,1):.4f}%) "
              f"{(time.time()-t0)/60:.0f} min", flush=True)
s1, v1 = G.gate_eval(model, tok, dev, n=8)
print(f"post: proxy {sum(s1.values())} @ {v1:.1f}% | FINAL "
      f"absorption {100*absorbed/max(total,1):.4f}%", flush=True)
