"""Clade-gated streaming pilot, arm G (pre-reg 2026-07-26).

One variable vs v4 (57/120): stream order + gating. Bands
(L1,L2)->(L3)->(L4,L5)->(L6,L7); advancement by a verified wave
probe on the current band (>= 0.55 valid fraction, 8 states x 8
samples, 1 ply) or band exhaustion; 30% ration from previous bands
after advance. Mixed shuffled batches, final-10% cooldown,
surprise multiplier, BS=32, seed 1 — all matched to v4.
"""
import os
import random
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp
import torch
import torch.nn.functional as F

from train_mathnative import load_rows  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402
import step_grpo_micro as G  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from bench_verify_fast import verify_wave  # noqa: E402

D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
BS = 32
BASE_LR = 3e-4
WARMUP = 200
EMA_DECAY = 0.99
CLAMP = (0.25, 4.0)
MAX_STEPS = 4140
BANDS = [(1, 2), (3,), (4, 5), (6, 7)]
PROBE_EVERY = 200
PROBE_STATES = 8
PROBE_WAVE = 8
ADVANCE_AT = 0.55
RATION = 0.30
PROBE_BAND = 77_000_000  # fresh seed space, never a training band
OUT = ("checkpoints/mathnative_wfloor_d256_clade2.pt"
       if os.environ.get("CLADE_V2") == "1"
       else "checkpoints/mathnative_wfloor_d256_clade.pt")

tok = MathTokenizer()
rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
by_band: list[list[int]] = [[] for _ in BANDS]
skipped = 0
enc = []
for r in rows:
    t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
    try:
        ids = tok.encode(t) + [tok.eos_id]
    except ValueError:
        skipped += 1
        continue
    if len(ids) > 512:
        continue
    lv = r.get("level") or 1
    bi = next((i for i, b in enumerate(BANDS) if lv in b), len(BANDS) - 1)
    enc.append(ids)
    by_band[bi].append(len(enc) - 1)
print(f"{len(enc)} sequences (skipped {skipped}); band sizes "
      f"{[len(b) for b in by_band]}", flush=True)

rng = random.Random(1)
for b in by_band:
    rng.shuffle(b)

dev = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(int(os.environ.get("BIRTH_SEED", "1")))
model = build_model(len(tok.vocab), d=D, layers=LAYERS, heads=HEADS,
                    ffn=FFN).to(dev)
print(f"model: {sum(p.numel() for p in model.parameters())/1e6:.1f}M "
      f"on {dev} [clade-gated stream]", flush=True)
opt = torch.optim.AdamW(model.parameters(), lr=BASE_LR,
                        weight_decay=0.01)


def probe_band(band: tuple[int, ...]) -> float:
    """Verified 1-ply valid fraction on fresh band states."""
    model.eval()
    valid = tried = 0
    with torch.no_grad():
        for i in range(PROBE_STATES):
            lv = band[i % len(band)]
            p = _gen_isolated(lv, PROBE_BAND + 1000 * lv + i)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
            texts, _, _ = G.sample_wave_lp(
                model, tok, prompt,
                [PROBE_BAND + i * 17 + b for b in range(PROBE_WAVE)], dev)
            tried += len(texts)
            distinct = [t for t in dict.fromkeys(texts)
                        if t and t.replace(" ", "") != cur.replace(" ", "")]
            wv = verify_wave(cur, distinct) if distinct else {}
            valid += sum(1 for t in texts if wv.get(t, (False, False))[0])
    model.train()
    return valid / max(tried, 1)


cursors = [0] * len(BANDS)
band_i = 0
advances = []
ema = None
t0 = time.time()
step = 0
while step < MAX_STEPS:
    # advancement check
    if step % PROBE_EVERY == 0 and step > 0 and band_i < len(BANDS) - 1:
        vf = probe_band(BANDS[band_i])
        print(f"  [probe] step {step} band {BANDS[band_i]} "
              f"valid-frac {vf:.3f}", flush=True)
        if vf >= ADVANCE_AT or cursors[band_i] >= len(by_band[band_i]):
            advances.append((step, band_i, vf))
            band_i += 1
            print(f"  [advance] -> band {BANDS[band_i]} at step {step}",
                  flush=True)
    if cursors[band_i] >= len(by_band[band_i]):
        if band_i < len(BANDS) - 1:
            advances.append((step, band_i, -1.0))
            band_i += 1
            print(f"  [exhausted] -> band {BANDS[band_i]} at step {step}",
                  flush=True)
        elif os.environ.get("CLADE_V2") == "1":
            # v2 (pre-reg 2026-07-26): recycle leftover budget into the
            # weakest band by probe; revisits allowed (aimed rations).
            probes = [(probe_band(b), i) for i, b in enumerate(BANDS)]
            vf, weak = min(probes)
            print(f"  [recycle] step {step} probes "
                  f"{[round(p, 3) for p, _ in probes]} -> band "
                  f"{BANDS[weak]} (vf {vf:.3f})", flush=True)
            band_i = weak
            rng.shuffle(by_band[weak])
            cursors[weak] = 0
        else:
            break
    # build batch: current band + rations from mastered bands
    idxs = []
    n_ration = int(BS * RATION) if band_i > 0 else 0
    for _ in range(n_ration):
        past = rng.randrange(band_i)
        pool = by_band[past]
        idxs.append(pool[rng.randrange(len(pool))])  # revisit (rations)
    while len(idxs) < BS and cursors[band_i] < len(by_band[band_i]):
        idxs.append(by_band[band_i][cursors[band_i]])
        cursors[band_i] += 1
    if not idxs:
        break
    batch = [enc[j] for j in idxs]
    L = max(len(s) for s in batch)
    ids = torch.tensor([s + [tok.pad_id] * (L - len(s)) for s in batch],
                       device=dev)
    mask = torch.tensor([[1] * len(s) + [0] * (L - len(s))
                         for s in batch], device=dev)
    logits = model(ids[:, :-1], mask[:, :-1])
    labels = ids[:, 1:].clone()
    labels[mask[:, 1:] == 0] = -100
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                           labels.reshape(-1), ignore_index=-100)
    lv_ = float(loss.detach())
    ema = lv_ if ema is None else EMA_DECAY * ema + (1 - EMA_DECAY) * lv_
    surprise = max(CLAMP[0], min(CLAMP[1], lv_ / max(ema, 1e-8)))
    warm = min(1.0, (step + 1) / WARMUP)
    tail = MAX_STEPS // 10
    cool = min(1.0, (MAX_STEPS - step) / tail)
    for g in opt.param_groups:
        g["lr"] = BASE_LR * warm * cool * surprise
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
    step += 1
    if step % 200 == 0:
        print(f"  {step}/{MAX_STEPS} band {BANDS[band_i]} loss {lv_:.3f} "
              f"({step/(time.time()-t0):.1f} it/s)", flush=True)

torch.save(model.state_dict(), OUT)
print(f"advances: {advances}", flush=True)
print(f"saved {OUT}  wall {time.time()-t0:.0f}s  steps {step}", flush=True)
