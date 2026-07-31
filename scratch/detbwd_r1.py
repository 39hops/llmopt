"""Deterministic-birth R1a (pre-reg 2026-07-31 night): integer
FFN forward + BACKWARD, the first training-side rung.

Everything is fixed-point int64 tensor ops (elementwise mul + sum
reductions — integer adds are order-independent, so any backend
that runs this computes the SAME BITS; no matmul kernels needed).
Nonlinearity: SiLU via int16 table (P3 pattern) + a dSiLU
derivative table built the same way, both sha-printed (instrument
fences travel with instruments). Rounding: round-half-away rdiv
(pack_decode convention). Backward is the STE-style deterministic
gradient: table-derivative through the quantized forward.

Checks:
  (1) DETERMINISM: two runs -> identical sha over fwd+grads;
  (2) FIDELITY: integer grads v fp64 autograd of the smooth twin,
      cosine >= 0.999 per tensor;
  (3) CROSS-DEVICE: run on Mac and 3080 — the printed shas must
      MATCH (the pass criterion; integer ops are backend-exact).
Usage: python scratch/detbwd_r1.py   (both machines, compare shas)
"""
import hashlib
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

Q = 512          # fixed-point scale (fq512, the integer-twin rung)
TS = 4096        # SiLU table half-range in x*Q units
D, F = 64, 256
SEED = 7


def rdiv(x, d):
    """Round-half-away integer divide (pack_decode convention)."""
    return torch.sign(x) * ((x.abs() + d // 2) // d)


def build_tables():
    xs = torch.arange(-TS, TS + 1, dtype=torch.float64) / Q
    sig = 1 / (1 + torch.exp(-xs))
    silu = xs * sig
    dsilu = sig * (1 + xs * (1 - sig))
    t_silu = torch.round(silu * Q).to(torch.int64)
    t_dsilu = torch.round(dsilu * Q).to(torch.int64)
    for name, t in (("silu", t_silu), ("dsilu", t_dsilu)):
        h = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
        print(f"[r1a] table {name} sha {h}")
    return t_silu, t_dsilu


def lut(t, xq, hi_pos):
    """Table lookup with per-table saturation: beyond +TS the
    value is hi_pos(x) (x for silu, Q for dsilu); beyond -TS, 0."""
    idx = torch.clamp(xq + TS, 0, 2 * TS)
    lo = t[idx]
    hi = torch.where(xq > TS, hi_pos(xq), torch.zeros_like(xq))
    return torch.where(xq.abs() <= TS, lo, hi)


def int_mm(a, w):
    """[..., K] x [N, K] -> [..., N] in int64 exact (sum-reduce)."""
    return (a.unsqueeze(-2) * w).sum(-1)


def ffn_fwd(xq, wg, wu, wd, t_silu):
    g = rdiv(int_mm(xq, wg), Q)          # [T, F] at Q scale
    u = rdiv(int_mm(xq, wu), Q)
    s = lut(t_silu, g, lambda x: x)      # silu(x)->x above TS
    p = rdiv(s * u, Q)
    y = rdiv(int_mm(p, wd.transpose(0, 1)), Q)
    return y, (g, u, s, p)


def ffn_bwd(dy, xq, wg, wu, wd, cache, t_dsilu):
    g, u, s, p = cache
    dp = rdiv(int_mm(dy, wd), Q)             # [T, F]
    dwd = int_mm(p.transpose(0, 1), dy.transpose(0, 1))
    du = rdiv(dp * s, Q)
    ds = rdiv(dp * u, Q)
    dg = rdiv(ds * lut(t_dsilu, g, lambda x: torch.full_like(x, Q)), Q)
    dwg = int_mm(dg.transpose(0, 1), xq.transpose(0, 1))
    dwu = int_mm(du.transpose(0, 1), xq.transpose(0, 1))
    dx = rdiv(int_mm(dg, wg.transpose(0, 1))
              + int_mm(du, wu.transpose(0, 1)), Q)
    return dx, dwg, dwu, dwd


def main():
    dev = ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(SEED)
    t_silu, t_dsilu = build_tables()
    # fixed-point weights/activations (fq512 range, well in-bounds)
    wg = torch.randint(-Q, Q + 1, (F, D), dtype=torch.int64)
    wu = torch.randint(-Q, Q + 1, (F, D), dtype=torch.int64)
    wd = torch.randint(-Q, Q + 1, (F, D), dtype=torch.int64)
    xq = torch.randint(-Q, Q + 1, (32, D), dtype=torch.int64)
    dy = torch.randint(-Q, Q + 1, (32, D), dtype=torch.int64)
    args = [t.to(dev) for t in (xq, wg, wu, wd)]
    ts, td = t_silu.to(dev), t_dsilu.to(dev)

    y, cache = ffn_fwd(*args, ts)
    grads = ffn_bwd(dy.to(dev), *args, cache, td)
    h = hashlib.sha256()
    for t in (y, *grads):
        h.update(t.cpu().numpy().tobytes())
    sha = h.hexdigest()
    print(f"[r1a] dev {dev} fwd+bwd sha {sha[:32]}")

    # (1) determinism: rerun
    y2, c2 = ffn_fwd(*args, ts)
    g2 = ffn_bwd(dy.to(dev), *args, c2, td)
    h2 = hashlib.sha256()
    for t in (y2, *g2):
        h2.update(t.cpu().numpy().tobytes())
    print(f"[r1a] rerun identical: {h2.hexdigest() == sha}")

    # (2) fidelity v fp64 autograd of the smooth twin
    xf = (xq.double() / Q).requires_grad_(True)
    wgf = (wg.double() / Q).requires_grad_(True)
    wuf = (wu.double() / Q).requires_grad_(True)
    wdf = (wd.double() / Q).requires_grad_(True)
    yf = torch.nn.functional.silu(xf @ wgf.T) * (xf @ wuf.T) @ wdf
    yf.backward((dy.double() / Q))
    names = ("dx", "dwg", "dwu", "dwd")
    refs = (xf.grad, wgf.grad, wuf.grad, wdf.grad)
    for name, gi, gf in zip(names, grads, refs):
        a = gi.cpu().double().flatten()
        b = gf.flatten() * Q  # integer grads carry one Q factor
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        print(f"[r1a] cos({name}) = {cos:.6f}")


if __name__ == "__main__":
    main()
