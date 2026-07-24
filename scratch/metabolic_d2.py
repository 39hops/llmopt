"""DISAGREEMENT #2 test — exact vs fp64 accumulation at the validity
level (v5-mini, 2 of the 4 race arms). ONE variable: arm fp64
accumulates AdamW steps into fp64 masters (rounds 2^-53/step); arm
dd accumulates via two-sum double-double (EXACT — absorption
structurally impossible). Identical manual AdamW, food stream,
seeds. Streaming: every row eaten once, no epochs.
Usage: metabolic_d2.py <ckpt> <worklist> <minutes> <fp64|dd>
"""
import os, sys, time, json, random
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import torch
import torch.nn as nn
import torch.nn.functional as F
import sympy as sp
import step_grpo_micro as G
from llmopt.train.mathnative import MathTokenizer, build_model
from bench_step_tokens import _gen_isolated
from bench_verify_fast import verify_wave

CKPT, WORKLIST, MINUTES, ARM = (sys.argv[1], sys.argv[2],
                                int(sys.argv[3]), sys.argv[4])
D, LAYERS, FFN, HEADS = 768, 8, 3840, 12
LR = 1e-5
B1, B2, EPS = 0.9, 0.999, 1e-8
FRESH_BAND = 103_000_000
FRESH_LEVELS = [6, 7, 8, 9]
CENSUS_SEC = 600

def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s, torch.zeros_like(w),
                       torch.sign(w) * s)

class TLin(nn.Linear):
    def forward(self, x):
        if self.out_features == 40:
            return F.linear(x, self.weight, self.bias)
        w = self.weight
        return F.linear(x, w + (ternary(w) - w).detach(), self.bias)

nn.Linear = TLin
tok = MathTokenizer()
dev = "cuda"
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
model.load_state_dict(torch.load(CKPT, map_location="cpu"))
for li, blk in enumerate(model.blocks):
    if li < LAYERS // 2:
        for p in blk.parameters():
            p.requires_grad_(False)
model.emb.weight.requires_grad_(False)
params = [p for p in model.parameters() if p.requires_grad]

hi = [p.detach().double().clone() for p in params]
lo = [torch.zeros_like(h) for h in hi]          # dd tail (dd arm only)
mom = [torch.zeros_like(h) for h in hi]
vel = [torch.zeros_like(h) for h in hi]
step_t = 0

def opt_step():
    """manual AdamW, identical both arms except accumulation"""
    global step_t
    step_t += 1
    bc1 = 1 - B1 ** step_t
    bc2 = 1 - B2 ** step_t
    with torch.no_grad():
        for k, p in enumerate(params):
            g = p.grad.double()
            mom[k].mul_(B1).add_(g, alpha=1 - B1)
            vel[k].mul_(B2).addcmul_(g, g, value=1 - B2)
            s = -(LR / bc1) * mom[k] / ((vel[k] / bc2).sqrt() + EPS)
            if ARM == "dd":                     # exact two-sum accumulate
                snew = hi[k] + s
                bb = snew - hi[k]
                lo[k] += (hi[k] - (snew - bb)) + (s - bb)
                hi[k] = snew
            else:                               # fp64 masters (rounds)
                hi[k] += s
            p.copy_((hi[k] + lo[k]).float() if ARM == "dd"
                    else hi[k].float())

def sign_state():
    return {k: torch.sign(ternary(p.detach().float().cpu()))
            for k, p in enumerate(params) if p.dim() == 2}

start_sign = sign_state()
stuck = [json.loads(l) for l in open(WORKLIST)]
print(f"[d2:{ARM}] worklist {len(stuck)} | LR {LR} | streaming",
      flush=True)

def try_state(cur0, seed0, plies=8):
    cur = cur0
    visited = {cur.replace(" ", "")}
    for ply in range(plies):
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [seed0 + ply * 7 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts)
                    if t and t.replace(" ", "") not in visited]
        wv = verify_wave(cur, distinct) if distinct else {}
        nxt = None
        for t in texts:
            ok, so = wv.get(t, (False, False))
            if ok and t.replace(" ", "") not in visited:
                if nxt is None:
                    nxt = "SOLVED" if so else t
        if nxt == "SOLVED":
            return True
        if nxt is None:
            return False
        cur = nxt
        visited.add(cur.replace(" ", ""))
    return False

def probe(tag):
    model.eval()
    res = {}
    for i, s in enumerate(stuck):
        res[s["id"]] = try_state(s["root"], 55_000_000 + i * 101)
    print(f"[d2:{ARM}] {tag} resolution: {sum(res.values())}/"
          f"{len(stuck)} {[k for k, v in res.items() if v]}",
          flush=True)
    return res

pre = probe("PRE")
s0, v0 = G.gate_eval(model, tok, dev, n=8)
print(f"[d2:{ARM}] anchor proxy {sum(s0.values())} @ {v0:.1f}%",
      flush=True)

buf, neg = [], []
last_census = time.time()
t0 = time.time()
cycle = 0
rng = random.Random("d2-food")                  # identical both arms
while time.time() - t0 < MINUTES * 60:
    cycle += 1
    menu = [rng.choice(stuck)["root"] for _ in range(2)]
    lv = FRESH_LEVELS[cycle % len(FRESH_LEVELS)]
    p_ = _gen_isolated(lv, FRESH_BAND + cycle)
    if p_ is not None:
        menu.append(f"Integral({sp.sstr(p_._expr)}, x)")
    model.eval()
    for mi, cur0 in enumerate(menu):
        seed0 = FRESH_BAND + cycle * 131 + mi * 17
        prompt = tok.encode(f"Current: {cur0}\nHints: none\nStep: ")
        with torch.no_grad():
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [seed0 + b for b in range(8)], dev)
        distinct = [t for t in dict.fromkeys(texts) if t]
        wv = verify_wave(cur0, distinct) if distinct else {}
        ver = [t for t in distinct if wv.get(t, (False, False))[0]]
        rej = [t for t in distinct if not wv.get(t, (False, False))[0]]
        vfrac = sum(wv.get(t, (False, False))[0]
                    for t in texts) / max(len(texts), 1)
        w_row = 1.0 - vfrac
        for t_ in ver:
            ids = tok.encode(f"Current: {cur0}\nHints: none\n"
                             f"Step: {t_}\n") + [tok.eos_id]
            if len(ids) <= 512 and w_row > 0.01:
                buf.append((ids, w_row))
        if ver and rej:
            ids = tok.encode(f"Current: {cur0}\nHints: none\n"
                             f"Step: {rej[0]}\n") + [tok.eos_id]
            if len(ids) <= 512:
                neg.append((ids, 0.1 * w_row))
    if len(buf) >= 16:
        model.train()
        rows = buf[:24] + [(i, -w) for i, w in neg[:8]]
        buf, neg = buf[24:], neg[8:]
        L = max(len(q) for q, _ in rows)
        x = torch.tensor([q + [tok.pad_id] * (L - len(q))
                          for q, _ in rows], device=dev)
        wts = torch.tensor([w for _, w in rows], device=dev)
        logits = model(x)[:, :-1]
        y = x[:, 1:]
        per_tok = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1),
            ignore_index=tok.pad_id, reduction="none").view(y.shape)
        per_row = per_tok.sum(1) / (y != tok.pad_id).sum(1).clamp(min=1)
        loss = (per_row * wts).sum() / wts.abs().sum().clamp(min=1e-8)
        for p in params:
            p.grad = None
        loss.backward()
        opt_step()
        model.eval()
    if time.time() - last_census > CENSUS_SEC:
        last_census = time.time()
        now = sign_state()
        net = sum(int((now[k] != start_sign[k]).sum()) for k in now)
        print(f"[d2:{ARM}] cyc {cycle} steps {step_t} NET flips {net} "
              f"| {(time.time()-t0)/60:.0f}m", flush=True)

now = sign_state()
net = sum(int((now[k] != start_sign[k]).sum()) for k in now)
post = probe("POST")
s1, v1 = G.gate_eval(model, tok, dev, n=8)
tail = (sum(l.abs().max().item() for l in lo) if ARM == "dd" else 0)
print(f"[d2:{ARM}] FINAL: flips {net} | steps {step_t} | resolution "
      f"{sum(pre.values())} -> {sum(post.values())} | proxy "
      f"{sum(s0.values())} -> {sum(s1.values())} @ {v1:.1f}% | "
      f"dd-tail max {tail:.2e}", flush=True)
torch.save(model.state_dict(), f"checkpoints/metab_d2_{ARM}.pt")
