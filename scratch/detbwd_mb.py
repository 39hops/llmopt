"""Deterministic-birth MULTI-BLOCK reference (queued 2026-08-01):
N transformer bodies chained by dx0 + embedding and TIED head at
the ends — the full mini-LM anatomy, all int64, at the R2b
contract (SHIFT=12 default here, GBOOST=256, PQ, ACT_CLAMP,
constant lr 1/1000).

Anatomy vs the certified single-block Block (detbwd_r2b):
  emb[tok] -> Body_1 -> ... -> Body_N -> rmsnorm(g_f) -> emb^T
Body = n1 -> attn -> +res/clamp -> n2 -> FFN -> +res/clamp
(9 params: wq wk wv wo wg wu wd g1 g2 — the Block minus g3/wh).
The head is TIED to the embedding: logits = rdiv(h_f @ emb^T, Q).
Embedding grad = head part (rdiv per the wh convention) + EXACT
integer scatter-add of dx0 rows by token — summed AFTER each
part's own rounding (placement is part of the spec).

Checks: composite fp64-twin cosines on every param grad, rerun
determinism sha, training with milestone trajectory shas.
Env: STEPS (default 200), SHIFT (default 12), NBLK (default 2).
Usage: python scratch/detbwd_mb.py
"""
import hashlib
import os
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from detbwd_r1 import Q, int_mm, rdiv  # noqa: E402
from detbwd_r1 import lut  # noqa: E402
from detbwd_r2b import (  # noqa: E402
    ACT_CLAMP, D, DH, F, GBOOST, PQ, SCALE, T, V,
    build_exp_table, build_silu_tables, rms_bwd, rms_fwd,
    rope_bwd, rope_fwd, rope_tables, softmax_bwd, softmax_rows)
from detbwd_r3_qw import IntAdamWQw  # noqa: E402

SEED = 17
NBLK = int(os.environ.get("NBLK", "2"))
SHIFT = int(os.environ.get("SHIFT", "12"))
STEPS = int(os.environ.get("STEPS", "200"))


class Body:
    """Block minus final norm + head: x -> x2, chainable."""
    KEYS = ("wq", "wk", "wv", "wo", "wg", "wu", "wd", "g1", "g2")

    def __init__(self):
        mk = lambda *sh: torch.randint(-Q, Q + 1, sh,
                                       dtype=torch.int64)
        self.w = {"wq": mk(DH, D), "wk": mk(DH, D), "wv": mk(DH, D),
                  "wo": mk(D, DH), "wg": mk(F, D), "wu": mk(F, D),
                  "wd": mk(D, F),
                  "g1": torch.full((D,), Q, dtype=torch.int64),
                  "g2": torch.full((D,), Q, dtype=torch.int64)}

    def fwd(self, x, tab):
        w, c = self.w, {}
        cos, sin, t_exp, ts = (tab["cos"], tab["sin"],
                               tab["exp"], tab["silu"])
        c["x"] = x
        h1, c["i1"] = rms_fwd(x, w["g1"])
        c["h1"] = h1
        q = rdiv(int_mm(h1, w["wq"]), Q)
        k = rdiv(int_mm(h1, w["wk"]), Q)
        v = rdiv(int_mm(h1, w["wv"]), Q)
        c["v"] = v
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
        sg = lut(tab["silu"], gp, lambda z: z)
        c["sg"] = sg
        f = rdiv(sg * u, Q)
        c["f"] = f
        pre2 = x1 + rdiv(int_mm(f, w["wd"]), Q)
        c["m2"] = (pre2.abs() <= ACT_CLAMP).to(torch.int64)
        x2 = torch.clamp(pre2, -ACT_CLAMP, ACT_CLAMP)
        _ = ts
        return x2, c

    def bwd(self, dxin, c, tab):
        """dxin = grad wrt this body's OUTPUT x2 (post-clamp)."""
        w = self.w
        cos, sin, td = tab["cos"], tab["sin"], tab["dsilu"]
        G = {}
        dx2 = dxin * c["m2"]                  # clamp backward
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
        return G, dx0


class MB:
    """emb + NBLK bodies + final norm + tied head."""

    def __init__(self):
        # Draw order is the CONTRACT: emb, then bodies in order
        # (each body draws its 7 matrices), then g_f. Init export
        # must serialize in exactly this order.
        self.emb = torch.randint(-Q, Q + 1, (V, D),
                                 dtype=torch.int64)
        self.bodies = [Body() for _ in range(NBLK)]
        self.g_f = torch.full((D,), Q, dtype=torch.int64)

    def param_items(self):
        yield "emb", self.emb
        for i, b in enumerate(self.bodies):
            for k in Body.KEYS:
                yield f"b{i}.{k}", b.w[k]
        yield "g_f", self.g_f

    def fwd(self, tok, tab):
        c = {"tok": tok, "bodies": []}
        x = self.emb[tok]                      # rows at Q scale
        for b in self.bodies:
            x, cb = b.fwd(x, tab)
            c["bodies"].append(cb)
        c["xf"] = x
        hf, c["i_f"] = rms_fwd(x, self.g_f)
        c["hf"] = hf
        logits = rdiv(int_mm(hf, self.emb), Q)   # tied head
        return logits, c

    def bwd(self, dlogits, c, tab):
        G = {}
        # tied head: wh-convention grad wrt emb + dh into the norm
        g_head = rdiv(int_mm(dlogits.transpose(0, 1),
                             c["hf"].transpose(0, 1)), Q)
        dhf = rdiv(int_mm(dlogits, self.emb.transpose(0, 1)), Q)
        dx, G["g_f"] = rms_bwd(dhf, c["xf"], self.g_f, c["i_f"])
        for i in range(NBLK - 1, -1, -1):
            Gb, dx = self.bodies[i].bwd(dx, c["bodies"][i], tab)
            for k in Body.KEYS:
                G[f"b{i}.{k}"] = Gb[k]
        # embedding lookup: EXACT scatter-add of dx0 rows by token,
        # summed with the (already-rounded) head part — placement
        # is part of the spec (rdiv-grouping rule)
        g_tok = torch.zeros(V, D, dtype=torch.int64)
        g_tok.index_add_(0, c["tok"], dx)
        G["emb"] = g_head + g_tok
        return G


def twin_fp64(m, tok, tgt):
    import math
    ps = {n: (p.double() / Q).requires_grad_(True)
          for n, p in m.param_items()}
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

    x = ps["emb"][tok]
    for i in range(NBLK):
        w = {k: ps[f"b{i}.{k}"] for k in Body.KEYS}
        h1 = rms(x, w["g1"])
        q, k_, v = h1 @ w["wq"].T, h1 @ w["wk"].T, h1 @ w["wv"].T
        qr, kr = rope(q), rope(k_)
        s = (qr @ kr.T) / DH ** 0.5
        mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        p = torch.softmax(s.masked_fill(~mask, float("-inf")), -1)
        x1 = torch.clamp(x + (p @ v) @ w["wo"].T, -32.0, 32.0)
        h2 = rms(x1, w["g2"])
        f = (torch.nn.functional.silu(h2 @ w["wg"].T)
             * (h2 @ w["wu"].T))
        x = torch.clamp(x1 + f @ w["wd"].T, -32.0, 32.0)
    logits = rms(x, ps["g_f"]) @ ps["emb"].T
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    loss.backward()
    return {n: p.grad for n, p in ps.items()}


def main():
    torch.manual_seed(SEED)
    ts, td = build_silu_tables()
    t_exp = build_exp_table()
    cos, sin = rope_tables()
    tab = {"silu": ts, "dsilu": td, "exp": t_exp,
           "cos": cos, "sin": sin}
    m = MB()
    tok = torch.randint(0, V, (T,))
    tgt = torch.randint(0, V, (T,))
    onehot = torch.nn.functional.one_hot(tgt, V).to(torch.int64)
    names = [n for n, _ in m.param_items()]
    print(f"[mb] NBLK={NBLK} SHIFT={SHIFT} params "
          f"{sum(p.numel() for _, p in m.param_items())}")

    # --- (a) composite fp64-twin cosines
    logits, c = m.fwd(tok, tab)
    p = softmax_rows(logits, t_exp)
    G = m.bwd((p - Q * onehot) * GBOOST, c, tab)
    ref = twin_fp64(m, tok, tgt)
    worst, argw = 1.0, ""
    for n in names:
        a = G[n].double().flatten()
        b = ref[n].flatten()
        cosv = float((a @ b) / (a.norm() * b.norm() + 1e-12))
        if cosv < worst:
            worst, argw = cosv, n
        print(f"[mb] cos({n}) = {cosv:.6f}")
    print(f"[mb] worst cosine {worst:.6f} ({argw})")

    # --- (b) rerun determinism
    def fb_sha():
        lg, cc = m.fwd(tok, tab)
        pp = softmax_rows(lg, t_exp)
        GG = m.bwd((pp - Q * onehot) * GBOOST, cc, tab)
        h = hashlib.sha256()
        for n in names:
            h.update(GG[n].numpy().tobytes())
        return h.hexdigest()
    s1, s2 = fb_sha(), fb_sha()
    print(f"[mb] rerun identical: {s1 == s2}  sha {s1[:32]}")

    # --- (c) training at the R2b contract
    flat = dict(m.param_items())
    wide = {n: flat[n] << SHIFT for n in names}
    opt = IntAdamWQw([wide[n] for n in names], SHIFT, lrd=1000)
    sched = os.environ.get("SCHED") == "1"
    losses, th = [], hashlib.sha256()
    for step in range(1, STEPS + 1):
        if sched and step in (250, 500, 750):
            opt.lrd *= 2                      # integer lr decay
        nar = {n: rdiv(wide[n], 1 << SHIFT) for n in names}
        m.emb = nar["emb"]
        m.g_f = nar["g_f"]
        for i, b in enumerate(m.bodies):
            b.w = {k: nar[f"b{i}.{k}"] for k in Body.KEYS}
        lg, cc = m.fwd(tok, tab)
        pp = softmax_rows(lg, t_exp)
        losses.append(int((Q - pp[torch.arange(T), tgt]).sum()))
        GG = m.bwd((pp - Q * onehot) * GBOOST, cc, tab)
        opt.step([rdiv(GG[n], Q * GBOOST) for n in names])
        if step % max(25, STEPS // 8) == 0:
            for n in names:
                th.update(wide[n].numpy().tobytes())
            print(f"[mb] step {step} loss {losses[-1]} "
                  f"nz {opt.nz_last:.3f} "
                  f"traj-sha {th.hexdigest()[:16]}", flush=True)
    fell = losses[-1] < losses[len(losses) // 2] < losses[0]
    print(f"[mb] loss {losses[0]} -> {losses[len(losses) // 2]} -> "
          f"{losses[-1]}  falling: {fell}")
    print(f"[mb] FINAL trajectory sha {th.hexdigest()}")


if __name__ == "__main__":
    main()
