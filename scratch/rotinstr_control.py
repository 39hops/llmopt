"""A3 (revival-sweep Tier A, 2026-07-31): rotation-instrument
POSITIVE CONTROL. Run the weight-side rotation instruments
(weight-FFT euler lenses + anti-commutant mass) on the FOURIER-2b
crystal, the one substrate with a CONFIRMED activation clock
(276/512 periodic neurons at k=5). Two informative outcomes:
  - instruments read NULL here too -> weight-side lenses are blind
    to activation clocks (the old spontaneous-rotation nulls said
    nothing about representations);
  - instruments FIRE -> the old nulls were DIET statements (no
    forced periodic computation), per the clock-placement law.
CPU, minutes. Usage: python scratch/rotinstr_control.py
"""
import sys

sys.path.insert(0, ".")
import torch  # noqa: E402

CKPT = "checkpoints/fourier2b_widemod.pt"


def ks_uniform(theta):
    t, _ = torch.sort((theta + torch.pi) / (2 * torch.pi))
    n = t.numel()
    grid = torch.arange(1, n + 1, dtype=torch.float32) / n
    return float(torch.max(torch.abs(t - grid)))


def phase_stat(W):
    C = W.shape[1] - (W.shape[1] % 2)
    re, im = W[:, 0:C:2], W[:, 1:C:2]
    theta = torch.atan2(im.reshape(-1), re.reshape(-1))
    return ks_uniform(theta)


def fft_stat(W):
    sp = torch.fft.rfft(W, dim=1).abs() ** 2
    top = torch.topk(sp, k=min(8, sp.shape[1]), dim=1).values.sum(1)
    return float((top / sp.sum(1).clamp(min=1e-12)).mean())


def J_perm(n, perm):
    J = torch.zeros(n, n)
    for k in range(n // 2):
        a, b = perm[2 * k], perm[2 * k + 1]
        J[b, a] = 1.0
        J[a, b] = -1.0
    return J


def anti_mass(W, Jo, Ji):
    Wa = 0.5 * (W + Jo @ W @ Ji)
    return float((Wa.norm() / W.norm()) ** 2)


def main():
    sd = torch.load(CKPT, map_location="cpu", weights_only=True)["sd"]
    mats = {k: v.float() for k, v in sd.items()
            if v.dim() == 2 and any(t in k for t in
                                    ("gate", "up", "down"))}
    print(f"[rot-ctl] {len(mats)} FFN matrices from {CKPT}")

    # Lens 1+2: phase-pair KS + rFFT top-8, v 20 matched shuffles
    pz_all, fz_all = [], []
    for k, W in mats.items():
        pr, fr = phase_stat(W), fft_stat(W)
        pcs, fcs = [], []
        for s in range(20):
            g = torch.Generator().manual_seed(s)
            Wc = W[:, torch.randperm(W.shape[1], generator=g)]
            pcs.append(phase_stat(Wc))
            idx = torch.argsort(torch.rand(W.shape, generator=g),
                                dim=1)
            fcs.append(fft_stat(torch.gather(W, 1, idx)))
        pm, ps = torch.tensor(pcs).mean(), torch.tensor(pcs).std()
        fm, fs = torch.tensor(fcs).mean(), torch.tensor(fcs).std()
        pz_all.append(float((pr - pm) / max(float(ps), 1e-9)))
        fz_all.append(float((fr - fm) / max(float(fs), 1e-9)))
    pz = torch.tensor(pz_all)
    fz = torch.tensor(fz_all)
    print(f"[rot-ctl euler] phase-pair z mean {pz.mean():+.2f} "
          f"max|z| {pz.abs().max():.2f} | fft-order z mean "
          f"{fz.mean():+.2f} max|z| {fz.abs().max():.2f} | bar 3")

    # Lens 3: anti-commutant mass, adjacent pairing v 20 random
    g = torch.Generator().manual_seed(7)
    masses, nulls = [], []
    for k, W in mats.items():
        if "gate" not in k:
            continue
        no, ni = W.shape
        Jo = J_perm(no, list(range(no)))
        Ji = J_perm(ni, list(range(ni)))
        masses.append(anti_mass(W, Jo, Ji))
        for _ in range(20):
            po = torch.randperm(no, generator=g).tolist()
            pi = torch.randperm(ni, generator=g).tolist()
            nulls.append(anti_mass(W, J_perm(no, po), J_perm(ni, pi)))
    m = torch.tensor(masses)
    z = torch.tensor(nulls)
    zscore = float((m.mean() - z.mean()) / max(float(z.std()), 1e-9))
    print(f"[rot-ctl commutant] adjacent anti-mass mean "
          f"{m.mean():.4f} (0.5 = none) | random-pairing null "
          f"{z.mean():.4f}±{z.std():.4f} | z {zscore:+.2f}")
    fire = max(float(pz.abs().max()), float(fz.abs().max())) > 3 \
        or abs(zscore) > 3
    print(f"[rot-ctl verdict-input] instruments "
          f"{'FIRE' if fire else 'NULL'} on the clock crystal")


if __name__ == "__main__":
    main()
