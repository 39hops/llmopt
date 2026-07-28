"""Gauge-slack 4-crystal cell (pre-reg 2026-07-27 night, RIFF-LEDGER).

Does lattice training shrink gauge slack? Lawful form per the
ancestry verdict (07-26): fp32 seed-pair distance vs rat-Q6
seed-pair distance, same lens (raw / Hungarian-perm / rotation),
per-layer FFN gate matrices. Signal = rat pair reading BELOW the
fp32 pair (lattice canonicalization above the ancestry floor).
Skeptical prediction on record: both ~sqrt(2)-class, no closure.
Same-seed cross-arm pairs included as the ancestry control.
"""
import itertools

import torch
from scipy.optimize import linear_sum_assignment

MODELS = {
    "fp32_s1": "checkpoints/mathnative_19m_mac_fp32.pt",
    "fp32_s2": "checkpoints/mathnative_19m_mac_fp32_s2.pt",
    "rat_s1": "checkpoints/mathnative_19m_mac_ratq6_dep.pt",
    "rat_s2": "checkpoints/mathnative_19m_mac_ratq6_s2_dep.pt",
}


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(sd, dict) and "model" in sd:
        sd = sd["model"]
    return [sd[f"blocks.{li}.gate.weight"].float() for li in range(8)]


def nfro(a, b):
    return ((a - b).norm() / ((a.norm() + b.norm()) / 2)).item()


def perm_align2(a, b):
    an = torch.nn.functional.normalize(a, dim=1)
    bn = torch.nn.functional.normalize(b, dim=1)
    r, c = linear_sum_assignment(-(an @ bn.T).numpy())
    out = torch.empty_like(b)
    out[r] = b[c]
    return out


def rot_align(a, b):
    u, _, vt = torch.linalg.svd(a @ b.T)
    return (u @ vt) @ b


ws = {k: load(p) for k, p in MODELS.items()}

for m1, m2 in itertools.combinations(MODELS, 2):
    raw = perm = rot = 0.0
    for a, b in zip(ws[m1], ws[m2]):
        raw += nfro(a, b)
        perm += nfro(a, perm_align2(a, b))
        rot += nfro(a, rot_align(a, b))
    print(f"{m1:8s} {m2:8s} raw={raw/8:.4f} perm={perm/8:.4f} "
          f"rot={rot/8:.4f}", flush=True)
