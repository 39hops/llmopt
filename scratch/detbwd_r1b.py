"""Deterministic-birth R1b (pre-reg 2026-07-31 night): integer
ATTENTION forward + backward — softmax via exp table, jacobian in
fixed point.

Single-head causal attention core (q/k/v/o projections + scaled
dot + causal integer softmax). The softmax backward needs NO new
derivative table: ds = p * (dp - sum(p*dp)) uses only the forward
probabilities — the exp table (sha-pinned) is the sole new
artifact. Rope deferred to R1c (fixed per-position rotation;
backward is its transpose). Same checks as R1a: determinism sha,
fp64-twin gradient cosines, cross-device sha match.
Usage: python scratch/detbwd_r1b.py   (both machines, compare)
"""
import hashlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import rdiv, int_mm  # noqa: E402

Q = 512
TSE = 8 * Q                  # exp table range: [-8, 0] in Q units
T, D, DH = 32, 64, 16        # seq len, model dim, head dim
SEED = 11
SCALE = round(Q * DH ** 0.5)  # one rdiv folds /sqrt(dh) and Q


def build_exp_table():
    xs = torch.arange(-TSE, 1, dtype=torch.float64) / Q
    t = torch.round(torch.exp(xs) * Q).to(torch.int64)
    h = hashlib.sha256(t.numpy().tobytes()).hexdigest()[:16]
    print(f"[r1b] table exp sha {h}")
    return t


def exp_lut(t, x):
    """x <= 0 in Q units; below -TSE -> 0 (exp(-8) < 1/2Q)."""
    return torch.where(x < -TSE, torch.zeros_like(x), t[x + TSE])


def attn_fwd(xq, wq, wk, wv, wo, t_exp):
    q = rdiv(int_mm(xq, wq), Q)              # [T, DH] at Q
    k = rdiv(int_mm(xq, wk), Q)
    v = rdiv(int_mm(xq, wv), Q)
    s = rdiv(int_mm(q, k), SCALE)            # [T, T] at Q
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool,
                                   device=s.device))
    s = torch.where(causal, s, torch.full_like(s, -(1 << 40)))
    m = s.max(-1, keepdim=True).values       # exact integer max
    e = exp_lut(t_exp, torch.clamp(s - m, min=-TSE - 1))
    z = e.sum(-1, keepdim=True)              # exact
    p = rdiv(e * Q, z)                       # [T, T] at Q
    a = rdiv(int_mm(p, v.transpose(0, 1)), Q)   # [T, DH]
    y = rdiv(int_mm(a, wo), Q)               # [T, D]
    return y, (q, k, v, p, a)


def attn_bwd(dy, xq, wq, wk, wv, wo, cache):
    q, k, v, p, a = cache
    dwo = int_mm(a.transpose(0, 1), dy.transpose(0, 1))  # [DH, D]->wo layout [D? ] see fidelity
    da = rdiv(int_mm(dy, wo.transpose(0, 1)), Q)         # [T, DH]
    dp = rdiv(int_mm(da, v), Q)                          # [T, T]
    dv = rdiv(int_mm(p.transpose(0, 1), da.transpose(0, 1)), Q)
    # softmax jacobian, fixed point: ds = p*(dp - <p, dp>) / Q
    inner = rdiv((p * dp).sum(-1, keepdim=True), Q)
    ds = rdiv(p * (dp - inner), Q)                       # [T, T]
    dq = rdiv(int_mm(ds, k.transpose(0, 1)), SCALE)      # [T, DH]
    dk = rdiv(int_mm(ds.transpose(0, 1), q.transpose(0, 1)), SCALE)
    dwq = int_mm(xq.transpose(0, 1), dq.transpose(0, 1))
    dwk = int_mm(xq.transpose(0, 1), dk.transpose(0, 1))
    dwv = int_mm(xq.transpose(0, 1), dv.transpose(0, 1))
    dx = rdiv(int_mm(dq, wq.transpose(0, 1))
              + int_mm(dk, wk.transpose(0, 1))
              + int_mm(dv, wv.transpose(0, 1)), Q)
    return dx, dwq, dwk, dwv, dwo


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(SEED)
    t_exp = build_exp_table()
    wq = torch.randint(-Q, Q + 1, (DH, D), dtype=torch.int64)
    wk = torch.randint(-Q, Q + 1, (DH, D), dtype=torch.int64)
    wv = torch.randint(-Q, Q + 1, (DH, D), dtype=torch.int64)
    wo = torch.randint(-Q, Q + 1, (D, DH), dtype=torch.int64)
    xq = torch.randint(-Q, Q + 1, (T, D), dtype=torch.int64)
    dy = torch.randint(-Q, Q + 1, (T, D), dtype=torch.int64)
    args = [t.to(dev) for t in (xq, wq, wk, wv, wo)]
    te = t_exp.to(dev)

    y, cache = attn_fwd(*args, te)
    grads = attn_bwd(dy.to(dev), *args, cache)
    h = hashlib.sha256()
    for t in (y, *grads):
        h.update(t.cpu().numpy().tobytes())
    sha = h.hexdigest()
    print(f"[r1b] dev {dev} fwd+bwd sha {sha[:32]}")
    y2, c2 = attn_fwd(*args, te)
    g2 = attn_bwd(dy.to(dev), *args, c2)
    h2 = hashlib.sha256()
    for t in (y2, *g2):
        h2.update(t.cpu().numpy().tobytes())
    print(f"[r1b] rerun identical: {h2.hexdigest() == sha}")

    # fidelity: fp64 autograd of the smooth causal twin
    xf = (xq.double() / Q).requires_grad_(True)
    wqf = (wq.double() / Q).requires_grad_(True)
    wkf = (wk.double() / Q).requires_grad_(True)
    wvf = (wv.double() / Q).requires_grad_(True)
    wof = (wo.double() / Q).requires_grad_(True)
    qf, kf, vf = xf @ wqf.T, xf @ wkf.T, xf @ wvf.T
    sf = (qf @ kf.T) / DH ** 0.5
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    sf = sf.masked_fill(~mask, float("-inf"))
    pf = torch.softmax(sf, -1)
    yf = (pf @ vf) @ wof.T
    yf.backward(dy.double() / Q)
    names = ("dx", "dwq", "dwk", "dwv", "dwo")
    # integer dw* are [in, out]-transposed relative to autograd
    refs = (xf.grad, wqf.grad.T, wkf.grad.T, wvf.grad.T, wof.grad.T)
    for name, gi, gf in zip(names, grads, refs):
        a = gi.cpu().double().flatten()
        b = gf.flatten() * Q
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        print(f"[r1b] cos({name}) = {cos:.6f}")


if __name__ == "__main__":
    main()
