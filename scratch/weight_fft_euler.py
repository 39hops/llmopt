"""Weight-FFT euler read (pre-reg 2026-07-26, RESULTS.md).

Free prologue instrument for the complex bracket: do real-born
crystals already carry phase-pair / rotational structure in FFN
rows? Two lenses vs matched shuffle controls (20 seeds each):
  (a) adjacent-channel (2k,2k+1) phase-angle uniformity (KS stat)
  (b) per-row rFFT top-8 energy fraction (positional order)
House prediction: NULL on both (gauge law — channel order is
meaningless to SGD). Structure bar: crystal > control by 3 sigma.
"""
import sys

import torch

CKPT = "checkpoints/merged_grown_latent.pt"
sd = torch.load(CKPT, map_location="cpu")
mats = {k: W.float() for k, W in sd.items()
        if W.dim() == 2 and "ffn" in k.lower() or
        (W.dim() == 2 and ("w1" in k or "w2" in k or "up" in k
                           or "down" in k or "gate" in k))}
if not mats:  # fall back: all 2D interior
    mats = {k: W.float() for k, W in sd.items()
            if W.dim() == 2 and "emb" not in k and W.shape[0] != 40}
print(f"{len(mats)} matrices from {CKPT}")


def ks_uniform(theta):
    # KS statistic against uniform on [-pi, pi]
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


tot_p_real, tot_f_real = [], []
tot_p_ctl, tot_f_ctl = [], []
for k, W in mats.items():
    pr, fr = phase_stat(W), fft_stat(W)
    pcs, fcs = [], []
    for s in range(20):
        g = torch.Generator().manual_seed(s)
        # column shuffle kills pairing; within-row shuffle kills
        # positional order at identical magnitude statistics
        Wc = W[:, torch.randperm(W.shape[1], generator=g)]
        pcs.append(phase_stat(Wc))
        idx = torch.argsort(torch.rand(W.shape, generator=g), dim=1)
        fcs.append(fft_stat(torch.gather(W, 1, idx)))
    pm, ps = torch.tensor(pcs).mean(), torch.tensor(pcs).std()
    fm, fs = torch.tensor(fcs).mean(), torch.tensor(fcs).std()
    pz = (pr - pm) / max(float(ps), 1e-9)
    fz = (fr - fm) / max(float(fs), 1e-9)
    tot_p_real.append(pz)
    tot_f_real.append(fz)
    print(f"{k:55s} phaseKS {pr:.5f} (ctl {pm:.5f}±{ps:.5f}, "
          f"z={pz:+.1f}) | fft8 {fr:.4f} (ctl {fm:.4f}±{fs:.4f}, "
          f"z={fz:+.1f})")

pz_all = torch.tensor(tot_p_real)
fz_all = torch.tensor(tot_f_real)
print(f"\nSUMMARY over {len(mats)} matrices:")
print(f"  phase-pair z: mean {pz_all.mean():+.2f}, "
      f"max |z| {pz_all.abs().max():.2f}")
print(f"  fft-order  z: mean {fz_all.mean():+.2f}, "
      f"max |z| {fz_all.abs().max():.2f}")
print("  bar: 3 sigma. VERDICT:",
      "STRUCTURE" if max(pz_all.abs().max(),
                         fz_all.abs().max()) > 3 else "NULL")
