"""Metabolic-vs-champion confluence: where did 471 signed rows land?
Per-matrix ||dW||, effective rank of delta, top-layer localization,
ternary flip census (would the 1.58-bit deployment even change?)."""
import sys, torch
sys.path.insert(0, "."); sys.path.insert(0, "scripts")

A = torch.load("checkpoints/mathnative_gen6_grown.pt", map_location="cpu")
B = torch.load("checkpoints/metabolic_live.pt", map_location="cpu")

def ternary(w):
    s = w.abs().mean(dim=1, keepdim=True).clamp(min=1e-8)
    return torch.where(w.abs() < 0.5*s, torch.zeros_like(w),
                       torch.sign(w)*s)

tot_d = tot_w = flips = nz = 0
rows = []
for k in A:
    d = (B[k].float() - A[k].float())
    nd, nw = float(d.norm()), float(A[k].float().norm())
    tot_d += nd**2; tot_w += nw**2
    if d.dim() == 2 and min(d.shape) > 8:
        try:
            sv = torch.linalg.svdvals(d)
            er = float((sv.sum()/sv.max())) if sv.max() > 0 else 0
        except Exception:
            er = -1
        f = int((torch.sign(ternary(B[k].float()))
                 != torch.sign(ternary(A[k].float()))).sum())
        flips += f; nz += d.numel()
        rows.append((k, nd, nd/max(nw,1e-9), er, f))
rows.sort(key=lambda r: -r[1])
print(f"TOTAL ||dW|| = {tot_d**0.5:.4f} (champion norm {tot_w**0.5:.1f}, "
      f"ratio {100*tot_d**0.5/tot_w**0.5:.3f}%)")
print(f"TERNARY FLIP CENSUS: {flips} flips / {nz} weights "
      f"({100*flips/max(nz,1):.4f}%)")
print("\ntop-12 moved matrices (name, ||d||, rel, eff-rank, flips):")
for k, nd, rel, er, f in rows[:12]:
    print(f"  {k:34s} {nd:8.4f} {100*rel:6.3f}% er={er:6.1f} f={f}")
mass = sum(r[1]**2 for r in rows)
top = sorted(rows, key=lambda r: -r[1])
c = 0
for i, r in enumerate(top):
    c += r[1]**2
    if c > 0.5 * mass:
        print(f"\n50% of delta mass in top {i+1}/{len(rows)} matrices")
        break
lay = {}
for k, nd, rel, er, f in rows:
    if "blocks." in k:
        li = int(k.split(".")[1]); lay[li] = lay.get(li, 0) + nd**2
print("per-layer delta mass:", {k: round(v**0.5, 3)
      for k, v in sorted(lay.items())})
