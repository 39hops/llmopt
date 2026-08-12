"""d2 endpoint verification (amendment 2026-07-28): are the fp64-
masters and exact-dd arms' endpoints WEIGHT-identical, or only
count/outcome-identical as booked? Three reads: (1) element-wise
state_dict equality; (2) deployed ternary sign-map (flip-SET)
equality; (3) calib_probe fingerprints on both.
Runs on the 3080 (checkpoints live there); CPU-safe.
"""
import sys

import torch  # noqa: E402

A = "checkpoints/metab_d2_fp64.pt"
B = "checkpoints/metab_d2_dd.pt"
sa = torch.load(A, map_location="cpu", weights_only=True)
sb = torch.load(B, map_location="cpu", weights_only=True)

assert set(sa) == set(sb), "key mismatch"
neq_t = neq_e = 0
max_abs = 0.0
for k in sa:
    d = (sa[k] != sb[k])
    if d.any():
        neq_t += 1
        neq_e += int(d.sum())
        max_abs = max(max_abs,
                      float((sa[k] - sb[k]).abs().max()))
print(f"[1] state_dict: {neq_t} tensors differ, {neq_e} elements, "
      f"max |delta| {max_abs:.3e}"
      f"{' — BIT-IDENTICAL' if neq_t == 0 else ''}", flush=True)


def tern_sign(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5 * s, torch.zeros_like(w),
                       torch.sign(w))


flipdiff = 0
for k in sa:
    if sa[k].dim() == 2 and sa[k].shape[-1] != 40:
        flipdiff += int((tern_sign(sa[k].float())
                         != tern_sign(sb[k].float())).sum())
print(f"[2] deployed lattice: {flipdiff} sign cells differ"
      f"{' — FLIP-SET IDENTICAL' if flipdiff == 0 else ''}",
      flush=True)

from calib_probe import flips_per_token  # noqa: E402
for name, ck in (("fp64", A), ("dd", B)):
    r = flips_per_token(ck, 768, 8, 3840, 12, 16)
    print(f"[3] probe {name}: flips/tok {r['flips_per_token']:.5f} "
          f"margin_med {r['margin_median']:.3f} "
          f"at_flips {r['margin_at_flips']:.2e}", flush=True)
