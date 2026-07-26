"""The CE-gate study (pre-reg 2026-07-26, RESULTS.md).

Leg A: teacher-forced mean CE on 400 fixed diet rows (seed 7).
Leg B: 40 fresh states (L3-L7, 8/level), 8-sample waves at T=0.7,
oracle-verified; valid fraction / distinct-valid coverage /
identity fraction. Specimens: muon 34, stream3 45, stream4 57,
wfloor 65 — does the gate track coverage while CE anti-tracks?
"""
import random
import sys

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

MODELS = {
    "muon":    ("checkpoints/mathnative_wfloor_d256_muon.pt", 34),
    "stream3": ("checkpoints/mathnative_wfloor_d256_stream3.pt", 45),
    "stream4": ("checkpoints/mathnative_wfloor_d256_stream4.pt", 57),
    "control": ("checkpoints/mathnative_wfloor_d256.pt", 65),
}
D, LAYERS, FFN, HEADS = 256, 8, 1024, 4
STATE_SEED = 88_000_000  # fresh space (clade probe used 77M)

tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"

rows = load_rows(gen4=True)
rows = [r for r in rows
        if r["cur"].replace(" ", "") != r["nxt"].replace(" ", "")]
random.Random(7).shuffle(rows)
ce_rows = rows[:400]

states = []
for lv in (3, 4, 5, 6, 7):
    for i in range(8):
        p = _gen_isolated(lv, STATE_SEED + 1000 * lv + i)
        if p is not None:
            states.append((lv, f"Integral({sp.sstr(p._expr)}, x)"))
print(f"{len(ce_rows)} CE rows, {len(states)} probe states", flush=True)


@torch.no_grad()
def mean_ce(model):
    tot = n = 0
    for r in ce_rows:
        t = f"Current: {r['cur']}\nHints: none\nStep: {r['nxt']}\n"
        try:
            ids = torch.tensor([tok.encode(t) + [tok.eos_id]],
                               device=dev)
        except ValueError:
            continue
        logits = model(ids[:, :-1])
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids[0, 1:])
        tot += float(loss)
        n += 1
    return tot / n


@torch.no_grad()
def coverage(model):
    valid = tried = dviil = ident = 0
    dv_total = 0
    for si, (lv, cur) in enumerate(states):
        prompt = tok.encode(f"Current: {cur}\nHints: none\nStep: ")
        texts, _, _ = G.sample_wave_lp(
            model, tok, prompt,
            [STATE_SEED + si * 17 + b for b in range(8)], dev)
        tried += len(texts)
        ident += sum(1 for t in texts
                     if t.replace(" ", "") == cur.replace(" ", ""))
        distinct = [t for t in dict.fromkeys(texts)
                    if t and t.replace(" ", "") != cur.replace(" ", "")]
        wv = verify_wave(cur, distinct) if distinct else {}
        valid += sum(1 for t in texts if wv.get(t, (False, False))[0])
        dv_total += sum(1 for t in distinct
                        if wv.get(t, (False, False))[0])
    return (valid / max(tried, 1), dv_total / len(states),
            ident / max(tried, 1))


print(f"{'model':8s} {'gate':>4s} {'trainCE':>8s} {'valid%':>7s} "
      f"{'dv/state':>8s} {'ident%':>7s}", flush=True)
for name, (path, gate) in MODELS.items():
    model = build_model(len(tok.vocab), d=D, layers=LAYERS,
                        heads=HEADS, ffn=FFN).to(dev)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    ce = mean_ce(model)
    vf, dv, idf = coverage(model)
    print(f"{name:8s} {gate:4d} {ce:8.4f} {100*vf:7.2f} {dv:8.2f} "
          f"{100*idf:7.2f}", flush=True)
    del model
