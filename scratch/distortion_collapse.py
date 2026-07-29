"""THE DISTORTION COLLAPSE (pre-reg 2026-07-29 eve): one curve for
the quantization axis. For every logged snap cell, recompute the
induced normalized distortion x = param-weighted mean of
(W - Wq)^2 / sigma_t^2 over ALL params (unsnapped tensors
contribute 0), and pair with the BOOKED solves (y = solves /
control). Claim: y = f(x), geometry/location/width-blind, knee at
x ~ (0.5-1.0 sigma)^2 / 12 = 0.02-0.08. Desk only, no gates.
Solves below are transcribed from logs/ (snap_alloc*.log,
polar_snap*.log, snap_q*_gate.log). __main__-guarded.
"""
import math
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402


def load(p):
    return torch.load(p, map_location="cpu", weights_only=True)


def rat_snap(w, qm):  # best rational, denom <= qm (snap_alloc rule)
    wf = w.float()
    best = torch.round(wf)
    err = (wf - best).abs()
    for q in range(2, qm + 1):
        cand = torch.round(wf * q) / q
        e = (wf - cand).abs()
        m = e < err
        best = torch.where(m, cand, best)
        err = torch.where(m, e, err)
    return best


def x_of(sd, snapped):  # snapped: {key: Wq}
    num = den = 0.0
    tot = sum(v.numel() for v in sd.values() if v.is_floating_point())
    for k, wq in snapped.items():
        w = sd[k].float()
        var = float(w.var())
        num += float(((w - wq) ** 2).sum()) / var
    return num / tot


def polar_q(W, mstep, na):
    h = W.shape[0] // 2
    wr, wi = W[:h].float(), W[h:].float()
    s = mstep * W.float().std()
    m = torch.sqrt(wr * wr + wi * wi)
    th = torch.atan2(wi, wr)
    mq = torch.round(m / s) * s
    thq = torch.round(th / (2 * math.pi / na)) * (2 * math.pi / na)
    return torch.cat([mq * torch.cos(thq), mq * torch.sin(thq)])


def uni_q(W, u):
    s = u * W.float().std()
    return torch.round(W.float() / s) * s


def main():
    rows = []
    # --- cplx_none d384 (control 63); fence gate+up ---
    sd = load("checkpoints/cplx_none.pt")
    keys = [f"blocks.{li}.{nm}.weight" for li in range(8)
            for nm in ("gate", "up")]
    for tag, fn, y in (
            ("cplx uni 0.5s", lambda W: uni_q(W, 0.5), 63),
            ("cplx uni 1.0s", lambda W: uni_q(W, 1.0), 63),
            ("cplx uni 2.0s", lambda W: uni_q(W, 2.0), 60),
            ("cplx pol 1sx64", lambda W: polar_q(W, 1.0, 64), 63),
            ("cplx pol 1sx16", lambda W: polar_q(W, 1.0, 16), 63),
            ("cplx pol 2sx64", lambda W: polar_q(W, 2.0, 64), 63),
            ("cplx pol .5sx8", lambda W: polar_q(W, 0.5, 8), 63),
            ("cplx pol 2sx8", lambda W: polar_q(W, 2.0, 8), 61),
            ("cplx pol 3sx6", lambda W: polar_q(W, 3.0, 6), 58)):
        rows.append((x_of(sd, {k: fn(sd[k]) for k in keys}),
                     y / 63, tag))
    # --- d56 EMA (control 63); snap_alloc cells ---
    sd = load("checkpoints/sym_birth_dense_w56_ema.pt")
    attn = [k for li in range(8)
            for k in (f"blocks.{li}.qkv.weight",
                      f"blocks.{li}.o.weight")]
    mlp = [k for li in range(8)
           for k in (f"blocks.{li}.gate.weight",
                     f"blocks.{li}.up.weight",
                     f"blocks.{li}.down.weight")]
    for tag, keys2, qm, y in (
            ("d56 Q16 attn", attn, 16, 63),
            ("d56 Q16 mlp", mlp, 16, 63),
            ("d56 Q16 both", attn + mlp, 16, 63),
            ("d56 Q8 attn", attn, 8, 59),
            ("d56 Q8 mlp", mlp, 8, 59),
            ("d56 Q8 both", attn + mlp, 8, 56)):
        rows.append((x_of(sd, {k: rat_snap(sd[k], qm) for k in keys2}),
                     y / 63, tag))
    # --- 19M d384 (control 49); snapped ckpts on disk ---
    base = load("checkpoints/mathnative_19m.pt")
    for q, y in ((4, 0), (16, 26), (24, 45), (32, 43),
                 (48, 47), (64, 48)):
        sq = load(f"checkpoints/snap19m_q{q}.pt")
        snapped = {k: sq[k].float() for k in sq
                   if sq[k].is_floating_point()
                   and not torch.equal(sq[k], base[k])}
        rows.append((x_of(base, snapped), y / 49, f"19M Q{q}"))
    rows.sort()
    print(f"{'x = D/sigma^2':>14}  {'y = kept':>8}  cell")
    for x, y, tag in rows:
        print(f"{x:14.5f}  {y:8.3f}  {tag}", flush=True)


if __name__ == "__main__":
    main()
