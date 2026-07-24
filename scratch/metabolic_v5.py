"""METABOLIC V5 session 1 (spec 2026-07-23-metabolic-v5; dd arm
retired per disagreement-2 verdict). fp64 masters, streaming, long
horizon. Three jobs in one session:
  (1) practice: worklist = p1 residue + p2 deep states (14), stuck
      food + fresh L6-9, paired PRE/POST fixed-seed probes;
  (2) MINER V2: bank ALL verified steps outcome-tagged (solved /
      unsolved) -> data/practice_rows_v5.jsonl — the failed-step
      shard the gen-9 solved-only-leak A/B needs;
  (3) fresh-wall logging: zero-verified fresh roots -> axiom
      exchange format, data/stuck_states_v5.jsonl (morning relay).
Usage: metabolic_v5.py <ckpt> <worklist> <minutes>
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
B1, B2, EPS = 0.9, 0.999, 1e-8
FRESH_BAND = 111_000_000
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
masters = [p.detach().double().clone() for p in params]
mom = [torch.zeros_like(m) for m in masters]
vel = [torch.zeros_like(m) for m in masters]
step_t = 0

def opt_step():
    global step_t
    step_t += 1
    bc1 = 1 - B1 ** step_t
    bc2 = 1 - B2 ** step_t
    with torch.no_grad():
        for k, p in enumerate(params):
            g = p.grad.double()
            mom[k].mul_(B1).add_(g, alpha=1 - B1)
            vel[k].mul_(B2).addcmul_(g, g, value=1 - B2)
            masters[k] += -(LR / bc1) * mom[k] / ((vel[k] / bc2).sqrt() + EPS)
            p.copy_(masters[k].float())

def sign_state():
    return {k: torch.sign(ternary(p.detach().float().cpu()))
            for k, p in enumerate(params) if p.dim() == 2}

start_sign = sign_state()
stuck = [json.loads(l) for l in open(WORKLIST)]
print(f"[v5] worklist {len(stuck)} | LR {LR} | {MINUTES}m streaming",
      flush=True)

def try_state(cur0, seed0, plies=8):
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

def probe(tag):
    model.eval()
    res = {}
    for i, s in enumerate(stuck):
        ok, _ = try_state(s["root"], 55_000_000 + i * 101)
        res[s["id"]] = ok
    print(f"[v5] {tag} resolution: {sum(res.values())}/{len(stuck)} "
          f"{[k for k, v in res.items() if v]}", flush=True)
    return res

pre = probe("PRE")
s0, v0 = G.gate_eval(model, tok, dev, n=8)
ANCHOR = sum(s0.values())
print(f"[v5] anchor proxy {ANCHOR} @ {v0:.1f}%", flush=True)
snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

rows_f = open("data/practice_rows_v5.jsonl", "a")
walls_f = open("data/stuck_states_v5.jsonl", "a")
walls_seen = set()
buf, neg = [], []
prev_net = 0
last_census = time.time()
t0 = time.time()
cycle = 0
resolved_live = 0
rng = random.Random("v5-s1")
while time.time() - t0 < MINUTES * 60:
    cycle += 1
    menu = [(rng.choice(stuck)["root"], None, None) for _ in range(2)]
    lv = FRESH_LEVELS[cycle % len(FRESH_LEVELS)]
    p_ = _gen_isolated(lv, FRESH_BAND + cycle)
    if p_ is not None:
        menu.append((f"Integral({sp.sstr(p_._expr)}, x)", lv, cycle))
    model.eval()
    for mi, (cur0, flv, fcyc) in enumerate(menu):
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
        # miner v2: bank every verified step, outcome-tagged later
        for t_ in ver:
            solved = wv.get(t_, (False, False))[1]
            rows_f.write(json.dumps(
                {"cur": cur0, "nxt": t_, "level": flv or 5,
                 "outcome": "solved" if solved else "unsolved"}) + "\n")
            ids = tok.encode(f"Current: {cur0}\nHints: none\n"
                             f"Step: {t_}\n") + [tok.eos_id]
            if len(ids) <= 512 and w_row > 0.01:
                buf.append((ids, w_row))
        # fresh wall: zero verified on an L6+ fresh root -> log
        if flv and not ver and cur0.replace(" ", "") not in walls_seen:
            walls_seen.add(cur0.replace(" ", ""))
            walls_f.write(json.dumps(
                {"id": f"m-l{flv}-v5s1-{fcyc}", "level": flv,
                 "root": cur0, "from": "v5-s1", "why": "zero-verified",
                 "plies": 0, "bin": "fresh"}) + "\n")
            walls_f.flush()
        if ver and rej:
            ids = tok.encode(f"Current: {cur0}\nHints: none\n"
                             f"Step: {rej[0]}\n") + [tok.eos_id]
            if len(ids) <= 512:
                neg.append((ids, 0.1 * w_row))
        if mi == 0 and cycle % 3 == 0:
            ok, chain = try_state(cur0, seed0 + 999, plies=8)
            if ok and chain:
                resolved_live += 1
                for c_, n_ in chain:
                    rows_f.write(json.dumps(
                        {"cur": c_, "nxt": n_, "level": 5,
                         "outcome": "solved"}) + "\n")
                if len(chain) > 1:
                    rows_f.write(json.dumps(
                        {"cur": chain[0][0], "nxt": chain[-1][1],
                         "level": 5, "outcome": "skip"}) + "\n")
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
        for p in params:
            p.grad = None
        loss.backward()
        opt_step()
        model.eval()
    if time.time() - last_census > CENSUS_SEC:
        last_census = time.time()
        now = sign_state()
        net = sum(int((now[k] != start_sign[k]).sum()) for k in now)
        s1, v1 = G.gate_eval(model, tok, dev, n=8)
        cur_p = sum(s1.values())
        print(f"[v5] cyc {cycle} steps {step_t} NET {net} (+{net-prev_net}) "
              f"proxy {cur_p} live {resolved_live} walls {len(walls_seen)} "
              f"| {(time.time()-t0)/60:.0f}m", flush=True)
        prev_net = net
        if cur_p < ANCHOR - 3:
            model.load_state_dict(snap); model.to(dev)
            with torch.no_grad():
                for m, p in zip(masters, params):
                    m.copy_(p.detach().double())
            print("[v5] ROLLBACK", flush=True)
        else:
            snap = {k: v.detach().cpu().clone()
                    for k, v in model.state_dict().items()}

post = probe("POST")
s1, v1 = G.gate_eval(model, tok, dev, n=8)
now = sign_state()
net = sum(int((now[k] != start_sign[k]).sum()) for k in now)
print(f"[v5] FINAL: resolution {sum(pre.values())} -> "
      f"{sum(post.values())}/{len(stuck)} | proxy {ANCHOR} -> "
      f"{sum(s1.values())} @ {v1:.1f}% | NET {net} | steps {step_t} | "
      f"live {resolved_live} | new walls {len(walls_seen)}", flush=True)
torch.save(model.state_dict(), "checkpoints/metab_v5_s1.pt")
