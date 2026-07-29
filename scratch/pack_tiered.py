"""PACKED CRYSTAL C5 (pre-reg 2026-07-29 night): the tiered pack.
matryoshka_d56_3tier.pt -> nested artifact: non-gate tensors packed
once (C0 rule); gate.weight payloads nested — tier-8 base =
numel/8 orbit representatives of P_C8, tier-2 payload = numel/2
delta v the QUANTIZED tier-8 prediction, dense payload = full
delta v reconstructed tier-2. Each payload sigma-law-quantized on
its own sigma. Desk identity check, then full gates on all three
packed tiers (booked fp tiers: 57/57/48). __main__-guarded.
"""
import math
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/matryoshka_d56_3tier.pt"
CFG = dict(d=56, layers=8, heads=4, ffn=224)
GATES = [f"blocks.{li}.gate.weight" for li in range(8)]


def perm(n, nb, s):
    return torch.tensor([nb * (r // nb) + (r % nb - s) % nb
                         for r in range(n)])


def project(W, nb):
    acc = torch.zeros_like(W)
    for s in range(nb):
        acc = acc + W[perm(W.shape[0], nb, s)][:, perm(W.shape[1], nb, s)]
    return acc / nb


def recon(rep, nb, n_out):
    """rep [n_out/nb, n_in] (rows i%nb==0 of a C_nb-invariant W)."""
    n_in = rep.shape[1]
    W = torch.zeros(n_out, n_in)
    for s in range(nb):
        rows = torch.arange(n_out) % nb == s
        W[rows] = rep[:, perm(n_in, nb, s)][torch.arange(n_out)[rows] // nb]
    return W


def squant(x):
    """sigma-law quantize; -> (xq, q, bits, nbits_total)"""
    s = float(x.float().std())
    q = math.ceil(2.0 / max(s, 1e-8))
    codes = torch.round(x.float() * q).to(torch.int64)
    span = int(codes.max()) - int(codes.min()) + 1
    bits = max(1, math.ceil(math.log2(span)))
    return codes.float() / q, q, bits, bits * x.numel()


def main():
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base = torch.load(CKPT, map_location="cpu", weights_only=True)

    # desk identity: recon(reps of P_C8) == P_C8 exactly
    W0 = base[GATES[0]].float()
    for nb in (2, 8):
        P = project(W0, nb)
        err = float((recon(P[::nb].clone(), nb, P.shape[0]) - P)
                    .abs().max())
        print(f"C5 identity C{nb}: max|recon-proj| = {err:.2e}",
              flush=True)
        assert err < 1e-5

    shared_bits = 0
    tiers = {8: {}, 2: {}, 0: {}}  # per-gate-tensor effective weights
    pay = {8: 0, 2: 0, 0: 0}  # payload bits
    for k, v in base.items():
        if k in GATES:
            W = v.float()
            n = W.shape[0]
            p8, p2 = project(W, 8), project(W, 2)
            rep8q, _, _, b8 = squant(p8[::8].clone())
            T8 = recon(rep8q, 8, n)
            d2q, _, _, b2 = squant(p2[::2].clone() - T8[::2])
            T2 = recon(T8[::2] + d2q, 2, n)
            dfq, _, _, bf = squant(W - T2)
            tiers[8][k], tiers[2][k], tiers[0][k] = T8, T2, T2 + dfq
            pay[8] += b8
            pay[2] += b2
            pay[0] += bf
        elif v.ndim == 2 and k.startswith("blocks."):
            wq, _, _, b = squant(v)
            for t in tiers:
                tiers[t][k] = wq
            shared_bits += b
        else:
            for t in tiers:
                tiers[t][k] = v.float()

    ng = sum(base[k].numel() for k in GATES)
    tot = sum(v.numel() for v in base.values())
    cum = 0
    for t, extra in ((8, pay[8]), (2, pay[2]), (0, pay[0])):
        cum += extra
        print(f"C5 bytes tier{t or 'D'}: payload {extra / 8:.0f} B "
              f"(cum gate {cum / 8:.0f} B / fp32 {ng * 4} B) | "
              f"artifact {(shared_bits + cum) / 8:.0f} B "
              f"v fp32 {tot * 4} B", flush=True)

    for t, label in ((8, "EIGHTH"), (2, "HALF"), (0, "DENSE")):
        sd = dict(base)
        sd.update(tiers[t])
        m = build_model(len(tok.vocab), **CFG).to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        m.eval()
        with torch.no_grad():
            solves, valid = G.gate_eval(m, tok, dev)
        print(f"C5 packed {label}: {sum(solves.values())}/120 "
              f"@ {valid:.2f}%", flush=True)
        del m


if __name__ == "__main__":
    main()
