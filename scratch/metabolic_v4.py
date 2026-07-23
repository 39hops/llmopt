"""METABOLIC V4 — practice food + persistence census (spec
2026-07-23-metabolic-v4, v4.1 revisions). Single arm, fp64 masters
ON, LR 1e-5 (hot-but-guarded), food = stuck-state worklist cycled +
fresh unseen-biased problems; rollouts START at the stuck cur; new
stuck states eaten in-session; skip-pair banking on resolutions;
pre/post resolution probes (paired); flip census every 20 min.

Usage: metabolic_v4.py <ckpt_latent> <stuck_worklist.jsonl> <minutes>
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

CKPT, WORKLIST, MINUTES = sys.argv[1], sys.argv[2], int(sys.argv[3])
D, LAYERS, FFN, HEADS = 768, 8, 3840, 12
LR = 1e-5
FRESH_BAND = 97_000_000
FRESH_LEVELS = [6, 7, 8, 9]
CENSUS_SEC = 1200

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

def sign_state():
    return {id(p): torch.sign(ternary(p.detach().float().cpu()))
            for p in params if p.dim() == 2}

start_sign = sign_state()
masters = [p.detach().double().clone() for p in params]
for m in masters:
    m.requires_grad_(True)
opt = torch.optim.AdamW(masters, lr=LR, weight_decay=0.0)

stuck = [json.loads(l) for l in open(WORKLIST)]
print(f"[v4] worklist {len(stuck)} states | LR {LR} | fp64 ON",
      flush=True)

def try_state(cur0, seed0, plies=10):
    """One duo... single-model rollout from a state. Returns
    (resolved, chain[(cur,nxt)...])."""
    cur = cur0
    visited = {cur.replace(" ", "")}
    chain = []
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
            for t in texts:
                ok, so = wv.get(t, (False, False))
                if ok and so:
                    chain.append((cur, t)); break
            return True, chain
        if nxt is None:
            return False, chain
        chain.append((cur, nxt))
        cur = nxt
        visited.add(cur.replace(" ", ""))
    return False, chain

def probe_worklist(tag, seed_base):
    model.eval()
    res = {}
    for i, s in enumerate(stuck):
        ok, _ = try_state(s["root"], seed_base + i * 101, plies=8)
        res[s["id"]] = ok
    n = sum(res.values())
    print(f"[v4] {tag} resolution: {n}/{len(stuck)} "
          f"{[k for k, v in res.items() if v]}", flush=True)
    return res

pre = probe_worklist("PRE", 55_000_000)
s0, v0 = G.gate_eval(model, tok, dev, n=8)
ANCHOR = sum(s0.values())
print(f"[v4] anchor proxy {ANCHOR} @ {v0:.1f}%", flush=True)
snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

rows_f = open("data/practice_rows_v4.jsonl", "a")
buf, neg = [], []
prev_sign = sign_state()
flip_hist = []
last_census = time.time()
t0 = time.time()
cycle = 0
resolved_live = 0
rng = random.Random("v4-food")
while time.time() - t0 < MINUTES * 60:
    cycle += 1
    # food: 3 stuck states (cycled) + 1 fresh hard problem
    menu = [rng.choice(stuck)["root"] for _ in range(3)]
    lv = FRESH_LEVELS[cycle % len(FRESH_LEVELS)]
    p = _gen_isolated(lv, FRESH_BAND + cycle)
    if p is not None:
        menu.append(f"Integral({sp.sstr(p._expr)}, x)")
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
        # full resolution attempt on one stuck item per cycle
        if mi == 0 and cycle % 3 == 0:
            ok, chain = try_state(cur0, seed0 + 999, plies=8)
            if ok and chain:
                resolved_live += 1
                for c_, n_ in chain:
                    rows_f.write(json.dumps(
                        {"cur": c_, "nxt": n_, "kind": "step"}) + "\n")
                    ids = tok.encode(f"Current: {c_}\nHints: none\n"
                                     f"Step: {n_}\n") + [tok.eos_id]
                    if len(ids) <= 512:
                        buf.append((ids, 1.0))
                if len(chain) > 1:  # skip pair, transitivity-verified
                    rows_f.write(json.dumps(
                        {"cur": chain[0][0], "nxt": chain[-1][1],
                         "kind": "skip"}) + "\n")
                rows_f.flush()
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
        opt.zero_grad(); loss.backward()
        for m, p in zip(masters, params):
            m.grad = p.grad.double()
        opt.step()
        with torch.no_grad():
            for m, p in zip(masters, params):
                p.copy_(m.float())
        model.eval()
    if time.time() - last_census > CENSUS_SEC:
        last_census = time.time()
        now = sign_state()
        new = back = 0
        for pid in now:
            d_prev = (now[pid] != prev_sign[pid])
            d_start = (now[pid] != start_sign[pid])
            new += int(d_prev.sum())
            # flip-back: differed from start at prev census, now matches
            was = (prev_sign[pid] != start_sign[pid])
            back += int((was & ~d_start).sum())
        net = sum(int((now[pid] != start_sign[pid]).sum())
                  for pid in now)
        flip_hist.append((cycle, new, back, net))
        prev_sign = now
        s1, v1 = G.gate_eval(model, tok, dev, n=8)
        cur_p = sum(s1.values())
        print(f"[v4] cyc {cycle} proxy {cur_p} live-resolved "
              f"{resolved_live} | census new {new} back {back} "
              f"NET {net} | {(time.time()-t0)/60:.0f}m", flush=True)
        if cur_p < ANCHOR - 3:
            model.load_state_dict(snap); model.to(dev)
            with torch.no_grad():
                for m, p in zip(masters, params):
                    m.copy_(p.detach().double())
            print("[v4] ROLLBACK", flush=True)
        else:
            snap = {k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}

post = probe_worklist("POST", 55_000_000)  # same seeds = paired
delta = sum(post.values()) - sum(pre.values())
s1, v1 = G.gate_eval(model, tok, dev, n=8)
print(f"[v4] FINAL: resolution {sum(pre.values())} -> "
      f"{sum(post.values())} (delta {delta:+d}/{len(stuck)}) | "
      f"proxy {sum(s1.values())} | live-resolved {resolved_live} | "
      f"census {flip_hist}", flush=True)
torch.save(model.state_dict(), "checkpoints/metab_v4.pt")
