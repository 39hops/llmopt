"""Symmetry ladder S1 cell 1 (pre-reg 2026-07-28): quaternionic
anti-commutant mass of FFN gate matrices. Structures I,J,K = left
quaternion-unit action on 4-channel groups (I^2=J^2=K^2=-1, IJ=K);
P(W) = (W - IWI - JWJ - KWK)/4; anti-mass = 1 - ||P(W)||^2/||W||^2
(0.75 = fully generic, 0 = exactly quaternionic-linear). Synthetic
controls run FIRST (must read 0.0 / ~0.75) — instrument fence.
Real crystals: adjacent 4-grouping + 20 random-grouping nulls.
"""
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

REAL = {
    "wfloor": ("checkpoints/mathnative_wfloor_d256.pt", 8),
    "s2": ("checkpoints/mathnative_wfloor_d256_s2.pt", 8),
    "19m": ("checkpoints/mathnative_19m.pt", 8),
}

# left mult by i, j, k on (a,b,c,d)
BLOCKS = {
    "I": [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 0, -1], [0, 0, 1, 0]],
    "J": [[0, 0, -1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, -1, 0, 0]],
    "K": [[0, 0, 0, -1], [0, 0, -1, 0], [0, 1, 0, 0], [1, 0, 0, 0]],
}


def quat_structs(n, perm):
    """Three anticommuting structures on 4-tuples
    (perm[4k..4k+3]); returns dict name -> (n,n) tensor."""
    out = {}
    for name, B in BLOCKS.items():
        M = torch.zeros(n, n)
        Bt = torch.tensor(B, dtype=torch.float32)
        for k in range(n // 4):
            idx = perm[4 * k:4 * k + 4]
            for r in range(4):
                for c in range(4):
                    M[idx[r], idx[c]] = Bt[r, c]
        out[name] = M
    return out


def project(W, So, Si):
    """Commutant projection: group-average over {1,-I..,-J..,-K..}."""
    P = W.clone()
    for name in ("I", "J", "K"):
        P = P - So[name] @ W @ Si[name]
    return P / 4.0


def anti_mass(W, So, Si):
    P = project(W, So, Si)
    return float(1.0 - (P.norm() ** 2 / W.norm() ** 2))


def gates(sd, layers):
    return [sd[f"blocks.{li}.gate.weight"].float()
            for li in range(layers)]


if __name__ == "__main__":
    # --- synthetic controls (instrument fence) ---
    g = torch.Generator().manual_seed(0)
    n = 64
    S = quat_structs(n, list(range(n)))
    # algebra check: I^2 = -1, IJ = K, anticommutation
    assert torch.allclose(S["I"] @ S["I"], -torch.eye(n))
    assert torch.allclose(S["I"] @ S["J"], S["K"])
    assert torch.allclose(S["I"] @ S["J"], -S["J"] @ S["I"])
    Wr = torch.randn(n, n, generator=g)
    member = project(Wr, S, S)  # P is idempotent -> member commutes
    print(f"control: synthetic member anti-mass "
          f"{anti_mass(member, S, S):.6f} (must be ~0)")
    print(f"control: random matrix anti-mass "
          f"{anti_mass(Wr, S, S):.6f} (expect ~0.75)")

    for name, (path, L) in REAL.items():
        sd = torch.load(path, map_location="cpu", weights_only=True)
        ws = gates(sd, L)
        n_out, n_in = ws[0].shape
        So = quat_structs(n_out, list(range(n_out)))
        Si = quat_structs(n_in, list(range(n_in)))
        adj = sum(anti_mass(w, So, Si) for w in ws) / L
        nulls = []
        for s in range(20):
            gg = torch.Generator().manual_seed(s)
            po = torch.randperm(n_out, generator=gg).tolist()
            pi = torch.randperm(n_in, generator=gg).tolist()
            nulls.append(sum(anti_mass(w, quat_structs(n_out, po),
                                       quat_structs(n_in, pi))
                             for w in ws) / L)
        mu = sum(nulls) / len(nulls)
        sd_ = (sum((x - mu) ** 2 for x in nulls) / len(nulls)) ** 0.5
        z = (adj - mu) / max(sd_, 1e-12)
        print(f"{name:8s} adj {adj:.5f} | null {mu:.5f} +- {sd_:.5f}"
              f" | z {z:+.2f}", flush=True)
