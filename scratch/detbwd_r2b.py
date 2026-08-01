"""Deterministic-birth R2b (pre-reg 2026-08-01 pre-dawn): FULL
transformer block trained end-to-end in int64 — adds the three
missing integer pieces: rmsnorm backward, rope backward, CE
gradient at the head. One block (n1 -> single-head causal attn ->
residual -> n2 -> FFN -> residual -> n3 -> head), fixed random
next-token targets, IntAdamW at the R3a pin (Q_w = Q<<8, lr
1/1000). Checks: fp64-twin cosines on every param grad (composite,
the R1b lesson), rerun determinism sha, falling loss, trajectory
sha for the cross-device/cross-lab legs.
Usage: python scratch/detbwd_r2b.py
"""
import hashlib
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import Q, int_mm, rdiv  # noqa: E402
from detbwd_r1 import build_tables as build_silu_tables  # noqa: E402
from detbwd_r1 import lut  # noqa: E402
from detbwd_r1b import TSE, build_exp_table, exp_lut  # noqa: E402
from detbwd_r2_adamw import isqrt_newton  # noqa: E402
from detbwd_r3_qw import IntAdamWQw  # noqa: E402

T, D, DH, F, V = 32, 64, 16, 128, 64
ACT_CLAMP = 32 * 512               # P3 residual clamp, Q units
SEED = 17
SCALE = round(Q * DH ** 0.5)
RS = 1 << 14                      # rope table scale
R16 = 1 << 16                     # rmsnorm rsqrt scale
EPS32 = 42950                     # round(1e-5 * 2^32)
SHIFT = int(__import__("os").environ.get("SHIFT", "8"))  # R3a pin default
PQ = Q * 16                       # attention probs carried finer:
                                  # Q-resolution p is the fidelity
                                  # floor of the composite chain
GBOOST = 256                      # backward runs at Q<<8: CE-scale
                                  # grads underflow Q through the
                                  # deep chain (linear -> lossless
                                  # up to the final unboost rdiv);
                                  # 256 is the SHIPPED value (the
                                  # 64->256 sweep showed fidelity
                                  # is boost-invariant)
STEPS = int(__import__("os").environ.get("STEPS", "200"))


def rope_tables():
    import math
    half = DH // 2
    freq = torch.exp(-math.log(10000.0)
                     * torch.arange(half, dtype=torch.float64) / half)
    ang = torch.arange(T, dtype=torch.float64)[:, None] * freq[None, :]
    cos = torch.round(ang.cos() * RS).to(torch.int64)
    sin = torch.round(ang.sin() * RS).to(torch.int64)
    return cos, sin


def rope_fwd(x, cos, sin):
    half = DH // 2
    v1, v2 = x[:, :half], x[:, half:]
    return torch.cat([rdiv(v1 * cos - v2 * sin, RS),
                      rdiv(v1 * sin + v2 * cos, RS)], -1)


def rope_bwd(dx, cos, sin):
    half = DH // 2
    d1, d2 = dx[:, :half], dx[:, half:]
    return torch.cat([rdiv(d1 * cos + d2 * sin, RS),
                      rdiv(-d1 * sin + d2 * cos, RS)], -1)


def rms_fwd(x, g):
    s2 = (x * x).sum(-1, keepdim=True)
    m40 = (s2 // D) * (1 << 32) // (Q * Q) + EPS32
    isq = isqrt_newton(m40)                    # ~ sqrt(mean)*2^16
    y = rdiv(rdiv(x * g, Q) * R16, isq)
    return y, isq


def rms_bwd(dy, x, g, isq):
    t = rdiv(g * dy, Q)                        # g.dy at Q
    inner = rdiv(t * x, Q).sum(-1, keepdim=True)   # <t, x> at Q
    term1 = rdiv(t * R16, isq)
    c = rdiv(x * inner, D * Q)
    for _ in range(3):                         # x (r^3) : three r-mults
        c = rdiv(c * R16, isq)
    dx = term1 - c
    dg = rdiv(rdiv(dy * x, Q) * R16, isq)      # per-position; sum below
    return dx, dg.sum(0)


def softmax_rows(s, t_exp, scale=None):
    m = s.max(-1, keepdim=True).values
    e = exp_lut(t_exp, torch.clamp(s - m, min=-TSE - 1))
    z = e.sum(-1, keepdim=True)
    return rdiv(e * (scale or Q), z)


def softmax_bwd(p, dp, scale=None):
    inner = rdiv(p * dp, scale or Q).sum(-1, keepdim=True)
    return rdiv(p * (dp - inner), scale or Q)


class Block:
    KEYS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "wh",
            "g1", "g2", "g3")

    def __init__(self):
        mk = lambda *sh: torch.randint(-Q, Q + 1, sh,
                                       dtype=torch.int64)
        self.w = {"wq": mk(DH, D), "wk": mk(DH, D), "wv": mk(DH, D),
                  "wo": mk(D, DH), "wg": mk(F, D), "wu": mk(F, D),
                  "wd": mk(D, F), "wh": mk(V, D),
                  "g1": torch.full((D,), Q, dtype=torch.int64),
                  "g2": torch.full((D,), Q, dtype=torch.int64),
                  "g3": torch.full((D,), Q, dtype=torch.int64)}

    def fwd(self, x, tab):
        w, c = self.w, {}
        cos, sin, t_exp, ts = tab["cos"], tab["sin"], tab["exp"], tab["silu"]
        c["x"] = x
        h1, c["i1"] = rms_fwd(x, w["g1"])
        c["h1"] = h1
        q = rdiv(int_mm(h1, w["wq"]), Q)
        k = rdiv(int_mm(h1, w["wk"]), Q)
        v = rdiv(int_mm(h1, w["wv"]), Q)
        c["q0"], c["k0"], c["v"] = q, k, v
        qr, kr = rope_fwd(q, cos, sin), rope_fwd(k, cos, sin)
        c["qr"], c["kr"] = qr, kr
        s = rdiv(int_mm(qr, kr), SCALE)
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        s = torch.where(causal, s, torch.full_like(s, -(1 << 40)))
        p = softmax_rows(s, t_exp, PQ)
        c["p"] = p
        a = rdiv(int_mm(p, v.transpose(0, 1)), PQ)
        c["a"] = a
        pre1 = x + rdiv(int_mm(a, w["wo"]), Q)
        c["m1"] = (pre1.abs() <= ACT_CLAMP).to(torch.int64)
        x1 = torch.clamp(pre1, -ACT_CLAMP, ACT_CLAMP)
        c["x1"] = x1
        h2, c["i2"] = rms_fwd(x1, w["g2"])
        c["h2"] = h2
        gp = rdiv(int_mm(h2, w["wg"]), Q)
        u = rdiv(int_mm(h2, w["wu"]), Q)
        c["gp"], c["u"] = gp, u
        sg = lut(ts, gp, lambda z: z)          # silu -> x above TS
        c["sg"] = sg
        f = rdiv(sg * u, Q)
        c["f"] = f
        pre2 = x1 + rdiv(int_mm(f, w["wd"]), Q)
        c["m2"] = (pre2.abs() <= ACT_CLAMP).to(torch.int64)
        x2 = torch.clamp(pre2, -ACT_CLAMP, ACT_CLAMP)
        c["x2"] = x2
        h3, c["i3"] = rms_fwd(x2, w["g3"])
        c["h3"] = h3
        logits = rdiv(int_mm(h3, w["wh"]), Q)
        return logits, c

    def bwd(self, dlogits, c, tab):
        w = self.w
        cos, sin, td = tab["cos"], tab["sin"], tab["dsilu"]
        G = {}
        G["wh"] = rdiv(int_mm(dlogits.transpose(0, 1),
                              c["h3"].transpose(0, 1)), Q)
        dh3 = rdiv(int_mm(dlogits, w["wh"].transpose(0, 1)), Q)
        dx2, G["g3"] = rms_bwd(dh3, c["x2"], w["g3"], c["i3"])
        dx2 = dx2 * c["m2"]                   # clamp backward
        # FFN branch
        df = rdiv(int_mm(dx2, w["wd"].transpose(0, 1)), Q)
        G["wd"] = rdiv(int_mm(dx2.transpose(0, 1),
                              c["f"].transpose(0, 1)), Q)
        du = rdiv(c["sg"] * df, Q)
        dgp = rdiv(rdiv(c["u"] * df, Q)
                   * lut(td, c["gp"],
                         lambda z: torch.full_like(z, Q)), Q)
        dh2 = rdiv(int_mm(du, w["wu"].transpose(0, 1))
                   + int_mm(dgp, w["wg"].transpose(0, 1)), Q)
        G["wu"] = rdiv(int_mm(du.transpose(0, 1),
                              c["h2"].transpose(0, 1)), Q)
        G["wg"] = rdiv(int_mm(dgp.transpose(0, 1),
                              c["h2"].transpose(0, 1)), Q)
        dx1, G["g2"] = rms_bwd(dh2, c["x1"], w["g2"], c["i2"])
        dx1 = (dx1 + dx2) * c["m1"]           # residual + clamp bwd
        # attention branch
        da = rdiv(int_mm(dx1, w["wo"].transpose(0, 1)), Q)
        G["wo"] = rdiv(int_mm(dx1.transpose(0, 1),
                              c["a"].transpose(0, 1)), Q)
        dp = rdiv(int_mm(da, c["v"]), Q)
        dv = rdiv(int_mm(c["p"].transpose(0, 1),
                         da.transpose(0, 1)), PQ)
        ds = softmax_bwd(c["p"], dp, PQ)
        dqr = rdiv(int_mm(ds, c["kr"].transpose(0, 1)), SCALE)
        dkr = rdiv(int_mm(ds.transpose(0, 1),
                          c["qr"].transpose(0, 1)), SCALE)
        dq = rope_bwd(dqr, cos, sin)
        dk = rope_bwd(dkr, cos, sin)
        G["wq"] = rdiv(int_mm(dq.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        G["wk"] = rdiv(int_mm(dk.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        G["wv"] = rdiv(int_mm(dv.transpose(0, 1),
                              c["h1"].transpose(0, 1)), Q)
        dh1 = rdiv(int_mm(dq, w["wq"].transpose(0, 1))
                   + int_mm(dk, w["wk"].transpose(0, 1))
                   + int_mm(dv, w["wv"].transpose(0, 1)), Q)
        dx0, G["g1"] = rms_bwd(dh1, c["x"], w["g1"], c["i1"])
        dx0 = dx0 + dx1                       # residual to input
        # NOTE m1 gates the pre1 sum, whose x-term reaches dx0 via
        # dx1 (already masked); dh1's path bypasses the clamp.
        return G, dx0


def twin_fp64(blk, x, tgt):
    """Smooth fp64 autograd twin; returns grads dict keyed like w."""
    w = {k: (v.double() / Q).requires_grad_(True)
         for k, v in blk.w.items()}
    import math
    xf = x.double() / Q
    eps = 1e-5

    def rms(h, g):
        return g * h / torch.sqrt((h * h).mean(-1, keepdim=True)
                                  + eps)

    half = DH // 2
    freq = torch.exp(-math.log(10000.0)
                     * torch.arange(half, dtype=torch.float64) / half)
    ang = (torch.arange(T, dtype=torch.float64)[:, None]
           * freq[None, :])
    cos, sin = ang.cos(), ang.sin()

    def rope(v):
        v1, v2 = v[:, :half], v[:, half:]
        return torch.cat([v1 * cos - v2 * sin,
                          v1 * sin + v2 * cos], -1)

    tw = {}
    h1 = rms(xf, w["g1"])
    tw["h1"] = h1
    q, k, v = h1 @ w["wq"].T, h1 @ w["wk"].T, h1 @ w["wv"].T
    qr, kr = rope(q), rope(k)
    s = (qr @ kr.T) / DH ** 0.5
    mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
    tw["p"] = p
    x1 = torch.clamp(xf + (p @ v) @ w["wo"].T, -32.0, 32.0)
    tw["x1"] = x1
    h2 = rms(x1, w["g2"])
    tw["h2"] = h2
    f = torch.nn.functional.silu(h2 @ w["wg"].T) * (h2 @ w["wu"].T)
    tw["f"] = f
    x2 = torch.clamp(x1 + f @ w["wd"].T, -32.0, 32.0)
    tw["x2"] = x2
    logits = rms(x2, w["g3"]) @ w["wh"].T
    tw["logits"] = logits
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    loss.backward()
    return {k: v.grad for k, v in w.items()}, tw


def main():
    torch.manual_seed(SEED)
    ts, td = build_silu_tables()
    t_exp = build_exp_table()
    cos, sin = rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    blk = Block()
    x = torch.randint(-Q, Q + 1, (T, D), dtype=torch.int64)
    tgt = torch.randint(0, V, (T,))
    onehot = torch.nn.functional.one_hot(tgt, V).to(torch.int64)

    # --- (a) fidelity: composite param-grad cosines vs fp64 twin
    logits, c = blk.fwd(x, tab)
    p = softmax_rows(logits, t_exp)
    dlogits = (p - Q * onehot) * GBOOST       # CE grad, boosted
    G, _ = blk.bwd(dlogits, c, tab)
    G = {k: v for k, v in G.items()}          # boosted-scale grads
    ref, tw = twin_fp64(blk, x, tgt)
    fwd_map = {"h1": c["h1"], "p": c["p"], "x1": c["x1"],
               "h2": c["h2"], "f": c["f"], "x2": c["x2"],
               "logits": logits}
    for k, iv in fwd_map.items():
        a = iv.double().flatten()
        b = tw[k].double().flatten() * Q
        cosv = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        print(f"[r2b] fwd cos({k}) = {cosv:.6f}")
    worst = 1.0
    for k in Block.KEYS:
        a = G[k].double().flatten()
        b = ref[k].flatten()
        cosv = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        worst = min(worst, cosv)
        print(f"[r2b] cos({k}) = {cosv:.6f}")
    print(f"[r2b] worst cosine {worst:.6f}")

    # --- (b) determinism: full fwd+bwd sha, rerun
    def fb_sha():
        lg, cc = blk.fwd(x, tab)
        pp = softmax_rows(lg, t_exp)
        GG, dx = blk.bwd((pp - Q * onehot) * GBOOST, cc, tab)
        h = hashlib.sha256()
        for k in Block.KEYS:
            h.update(GG[k].numpy().tobytes())
        h.update(dx.numpy().tobytes())
        return h.hexdigest()
    s1, s2 = fb_sha(), fb_sha()
    print(f"[r2b] rerun identical: {s1 == s2}  sha {s1[:32]}")

    # --- (c) training at the R3a pin
    params = [blk.w[k] for k in Block.KEYS]
    for k in Block.KEYS:                       # lift to Q_w
        blk.w[k] = blk.w[k] << SHIFT
    wide = {k: blk.w[k] for k in Block.KEYS}
    import os
    sched = os.environ.get("SCHED") == "1"
    opt = IntAdamWQw([wide[k] for k in Block.KEYS], SHIFT, lrd=1000)
    losses = []
    th = hashlib.sha256()
    for step in range(1, STEPS + 1):
        if sched and step in (250, 500, 750):
            opt.lrd *= 2                      # integer lr decay
        blk.w = {k: rdiv(wide[k], 1 << SHIFT) for k in Block.KEYS}
        lg, cc = blk.fwd(x, tab)
        pp = softmax_rows(lg, t_exp)
        # monitor: sum(Q - p_target) (no log table needed)
        losses.append(int((Q - pp[torch.arange(T), tgt]).sum()))
        GG, _ = blk.bwd((pp - Q * onehot) * GBOOST, cc, tab)
        opt.step([rdiv(GG[k], Q * GBOOST) for k in Block.KEYS])
        if step % max(50, STEPS // 8) == 0:
            for k in Block.KEYS:
                th.update(wide[k].numpy().tobytes())
            print(f"[r2b] step {step} loss {losses[-1]} "
                  f"nz {opt.nz_last:.3f} "
                  f"traj-sha {th.hexdigest()[:16]}", flush=True)
    fell = losses[-1] < losses[len(losses) // 2] < losses[0]
    print(f"[r2b] loss {losses[0]} -> {losses[len(losses) // 2]} -> "
          f"{losses[-1]}  falling: {fell}")
    print(f"[r2b] FINAL trajectory sha {th.hexdigest()}")
    _ = params


if __name__ == "__main__":
    main()
