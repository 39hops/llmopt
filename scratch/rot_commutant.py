"""Rotational snap R1 (pre-reg 2026-07-28): anti-commutant mass of
FFN gate matrices under channel-pairing complex structures.
W_a = (W + J_out W J_in)/2; mass = ||W_a||^2/||W||^2 (0.5 = no
rotational structure; 0 = fully complex-linear). Real crystals:
adjacent pairing + 20 random-pairing nulls. Complex-FFN arms:
native half-split pairing (positive control, expect ~0).
"""
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

REAL = {
    "wfloor": ("checkpoints/mathnative_wfloor_d256.pt", 8),
    "s2": ("checkpoints/mathnative_wfloor_d256_s2.pt", 8),
    "muon": ("checkpoints/mathnative_wfloor_d256_muon.pt", 8),
    "19m": ("checkpoints/mathnative_19m.pt", 8),
}
CPLX = {
    "cplx_none": "checkpoints/cplx_none.pt",
    "cplx_G5": "checkpoints/cplx_G5.pt",
}


def J_perm(n, perm):
    """Block rotation: pairs (perm[2k], perm[2k+1]); J e_a = e_b,
    J e_b = -e_a."""
    J = torch.zeros(n, n)
    for k in range(n // 2):
        a, b = perm[2 * k], perm[2 * k + 1]
        J[b, a] = 1.0
        J[a, b] = -1.0
    return J


def J_half(n):
    J = torch.zeros(n, n)
    h = n // 2
    J[h:, :h] = torch.eye(h)
    J[:h, h:] = -torch.eye(h)
    return J


def anti_mass(W, Jo, Ji):
    Wa = 0.5 * (W + Jo @ W @ Ji)
    return float((Wa.norm() ** 2 / W.norm() ** 2))


def gates(sd, layers):
    return [sd[f"blocks.{li}.gate.weight"].float()
            for li in range(layers)]


for name, (path, L) in REAL.items():
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ws = gates(sd, L)
    n_out, n_in = ws[0].shape
    Jo_adj = J_perm(n_out, list(range(n_out)))
    Ji_adj = J_perm(n_in, list(range(n_in)))
    adj = sum(anti_mass(w, Jo_adj, Ji_adj) for w in ws) / L
    nulls = []
    for s in range(20):
        g = torch.Generator().manual_seed(s)
        Jo = J_perm(n_out, torch.randperm(n_out, generator=g).tolist())
        Ji = J_perm(n_in, torch.randperm(n_in, generator=g).tolist())
        nulls.append(sum(anti_mass(w, Jo, Ji) for w in ws) / L)
    mu = sum(nulls) / len(nulls)
    sd_ = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5
    z = (adj - mu) / max(sd_, 1e-12)
    print(f"{name:10s} adj {adj:.5f} | null {mu:.5f} +- {sd_:.5f} "
          f"| z {z:+.2f}", flush=True)

for name, path in CPLX.items():
    sd = torch.load(path, map_location="cpu", weights_only=True)
    ks = [k for k in sd if k.endswith("gate.weight")]
    tot = 0.0
    for k in ks:
        W = sd[k].float()
        tot += anti_mass(W, J_half(W.shape[0]), J_half(W.shape[1]))
    print(f"{name:10s} native-half anti-mass {tot/len(ks):.6f} "
          f"({len(ks)} layers)", flush=True)
