"""Gauge-aligned model distance on the d256 zoo (pre-reg 2026-07-26).

15 pairs x 3 lenses (raw / permutation-aligned / rotation-aligned)
on per-layer FFN gate matrices; Spearman vs |gate delta|. The
instrument is judged against gates — never the reverse.
"""
import itertools

import torch
from scipy.optimize import linear_sum_assignment
from scipy.stats import spearmanr

MODELS = {
    "wfloor": ("checkpoints/mathnative_wfloor_d256.pt", 65),
    "s2": ("checkpoints/mathnative_wfloor_d256_s2.pt", 63),
    "s3": ("checkpoints/mathnative_wfloor_d256_s3.pt", 64),
    "pack": ("checkpoints/mathnative_wfloor_d256_pack.pt", 38),
    "stream4": ("checkpoints/mathnative_wfloor_d256_stream4.pt", 57),
    "clade2": ("checkpoints/mathnative_wfloor_d256_clade2.pt", 60),
}


def load(path):
    sd = torch.load(path, map_location="cpu", weights_only=False)
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
    # orthogonal Procrustes in NEURON space: find Q (n x n) minimizing
    # ||a - Q b||_F ; Q = U V^T from svd(a b^T)
    u, _, vt = torch.linalg.svd(a @ b.T)
    return (u @ vt) @ b


ws = {k: load(p) for k, (p, _) in MODELS.items()}
gates = {k: g for k, (_, g) in MODELS.items()}

rows = []
for m1, m2 in itertools.combinations(MODELS, 2):
    raw = perm = rot = 0.0
    for a, b in zip(ws[m1], ws[m2]):
        raw += nfro(a, b)
        perm += nfro(a, perm_align2(a, b))
        rot += nfro(a, rot_align(a, b))
    dg = abs(gates[m1] - gates[m2])
    rows.append((m1, m2, dg, raw / 8, perm / 8, rot / 8))
    print(f"{m1:8s} {m2:8s} |dgate|={dg:2d}  raw={raw/8:.4f} "
          f"perm={perm/8:.4f} rot={rot/8:.4f}", flush=True)

dgs = [r[2] for r in rows]
for i, name in ((3, "raw"), (4, "perm"), (5, "rot")):
    rho, p = spearmanr(dgs, [r[i] for r in rows])
    print(f"spearman(|dgate|, {name}) = {rho:+.3f} (p={p:.3f})")

seed_pairs = [r for r in rows if {r[0], r[1]} <= {"wfloor", "s2", "s3"}]
other = [r for r in rows if {r[0], r[1]} & {"pack", "stream4", "clade2"}]
for i, name in ((3, "raw"), (4, "perm"), (5, "rot")):
    sp = sum(r[i] for r in seed_pairs) / len(seed_pairs)
    ot = sum(r[i] for r in other) / len(other)
    print(f"{name}: seed-pair mean {sp:.4f} vs cross-schedule mean {ot:.4f}")
