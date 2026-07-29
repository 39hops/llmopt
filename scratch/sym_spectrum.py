"""Symmetry spectrum (pre-reg 2026-07-29: Artin's superposition
riff): isotypic decomposition of wfloor d256 gate weights under
C8 conjugation into 5 real frequency bands; report band masses;
gate CUMULATIVE reconstructions in descending-mass order.
comp_k(W) = (1/8) sum_s w^{-ks} R^s W R^{-s}; real bands pair
k with 8-k. Desk only (no training), MPS.
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

NB, D, LAYERS, FFN, HEADS = 8, 256, 8, 1024, 4
BANDS = [(0,), (1, 7), (2, 6), (3, 5), (4,)]


def shift_perm(n, sh):
    return torch.tensor([NB * (r // NB) + (r % NB - sh) % NB
                         for r in range(n)])


def conj_s(W, s):
    return W[shift_perm(W.shape[0], s)][:, shift_perm(W.shape[1], s)]


def band(W, ks):
    """Real isotypic component for the frequency set ks."""
    acc = torch.zeros_like(W)
    for s in range(NB):
        coef = sum(math.cos(2 * math.pi * k * s / NB) for k in ks)
        acc += coef * conj_s(W, s)
    return acc / NB


base = torch.load("checkpoints/mathnative_wfloor_d256.pt",
                  map_location="cpu", weights_only=True)
keys = [f"blocks.{li}.gate.weight" for li in range(LAYERS)]
comps = {ks: {k: band(base[k].float(), ks) for k in keys}
         for ks in map(tuple, BANDS)}

# sanity: bands sum back to W exactly
for k in keys:
    recon = sum(comps[tuple(b)][k] for b in BANDS)
    assert float((recon - base[k].float()).abs().max()) < 1e-4
masses = {}
for b in BANDS:
    m = sum(float(comps[tuple(b)][k].norm() ** 2) for k in keys)
    tot = sum(float(base[k].float().norm() ** 2) for k in keys)
    masses[tuple(b)] = m / tot
print("band masses:", {str(b): round(v, 4)
                       for b, v in masses.items()}, flush=True)

order = sorted(masses, key=masses.get, reverse=True)
tok = MathTokenizer()
dev = "mps" if torch.backends.mps.is_available() else "cpu"
acc = {k: torch.zeros_like(base[k].float()) for k in keys}
for i, b in enumerate(order):
    for k in keys:
        acc[k] += comps[b][k]
    sd = dict(base)
    for k in keys:
        sd[k] = acc[k].to(base[k].dtype)
    m = build_model(len(tok.vocab), d=D, layers=LAYERS,
                    heads=HEADS, ffn=FFN).to(dev)
    m.load_state_dict({k2: v.to(dev) for k2, v in sd.items()})
    m.eval()
    with torch.no_grad():
        solves, valid = G.gate_eval(m, tok, dev)
    print(f"SPECTRUM top-{i+1} bands {[str(x) for x in order[:i+1]]}"
          f": {sum(solves.values())}/120 @ {valid:.2f}%", flush=True)
    del m
