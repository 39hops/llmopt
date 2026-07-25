"""Boundary-or-bulk regression on the completed 0.5M->400M grid.

Pre-reg: RESULTS.md 2026-07-25 pre-dawn. No new runs — analysis of
measured cells only. Three sectors, never pooled (device fences):
  (a) rising arm <= W* (Mac lineage): collinearity exhibit, affine fits.
  (b) bits axis @ fixed 19M (3080/TF32): ordering test, volume vs bits.
  (c) width-bits crossover grid (3080/TF32): ordering test, the decider.
Param counts read from the actual checkpoints (never trust a label).
"""
import torch

CKPT = "checkpoints"


def n_params(path):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "model" in sd:
        sd = sd["model"]
    return sum(v.numel() for v in sd.values() if hasattr(v, "numel"))


def spearman(xs, ys):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx * dy else float("nan")


def affine_r2(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    if sxx == 0:
        return float("nan")
    b = sxy / sxx
    a = my - b * mx
    ss_res = sum((y - (a + b * x)) ** 2 for x, y in zip(xs, ys))
    ss_tot = sum((y - my) ** 2 for y in ys)
    return 1 - ss_res / ss_tot if ss_tot else float("nan")


import math

# ---- sector (a): rising arm <= W*, Mac lineage, gen-4 3ep, fp32 ----
# (name, ckpt, d, L, gate/120)
RISING = [
    ("d64", f"{CKPT}/mathnative_wfloor_d64.pt", 64, 8, 38),
    ("d128", f"{CKPT}/mathnative_wfloor_d128.pt", 128, 8, 57),
    ("d256", f"{CKPT}/mathnative_wfloor_d256.pt", 256, 8, 65),
    ("d384-19M", f"{CKPT}/mathnative_19m.pt", 384, 8, 64),
    ("d512-45M", f"{CKPT}/mathnative_45m_gen4_std.pt", 512, 12, 69),
]
UNDERFED = [  # shown, never fit (113M 3ep; 200M/400M 1-epoch license)
    ("d768-113M", f"{CKPT}/mathnative_110m_gen4_std_3ep.pt", 768, 12, 65),
    ("d1024-200M", f"{CKPT}/mathnative_200m_gen4_ep1.pt", 1024, 12, 49),
    ("d1280-400M", f"{CKPT}/mathnative_400m_gen4_ep1.pt", 1280, 12, 30),
]

# ---- sector (b): alphabet tournament @19M/d384, 3080/TF32, gen-4 3ep ----
BITS_AXIS = [("B", 1.00, 54), ("T", 1.58, 60), ("M4", 2.00, 61),
             ("M5", 2.32, 62), ("P2", 3.17, 66), ("fp32", 32.0, 64)]

# ---- sector (c): crossover grid, 3080/TF32, gen-4 3ep from-birth ----
# (d, bits, gate). N taken from same-width checkpoint counts.
CROSSOVER = [(256, 32.0, 61), (384, 32.0, 64), (768, 32.0, 58),
             (256, 1.58, 60), (384, 1.58, 60), (768, 1.58, 65)]

MEASURES = {
    "N*b (volume)": lambda N, d, L, b: N * b,
    "N (bulk)": lambda N, d, L, b: N,
    "sqrt(N)": lambda N, d, L, b: math.sqrt(N),
    "d (boundary)": lambda N, d, L, b: d,
    "d*L": lambda N, d, L, b: d * L,
    "b+0.5*log2(d)": lambda N, d, L, b: b + 0.5 * math.log2(d),
}


def main():
    print("== param counts (from checkpoints) ==")
    Ns = {}
    for name, path, d, L, gate in RISING + UNDERFED:
        Ns[name] = n_params(path)
        print(f"  {name:12s} N={Ns[name]:>12,}  gate={gate}")
    Nd = {d: Ns[n] for n, _, d, _, _ in [(r[0], 0, r[2], 0, 0) for r in RISING + UNDERFED]}

    print("\n== sector (a): rising arm (collinearity exhibit, n=1/cell) ==")
    gates = [g for *_, g in RISING]
    for mname, f in MEASURES.items():
        xs = [math.log10(max(f(Ns[n], d, L, 32.0), 1e-9))
              for n, _, d, L, _ in RISING]
        print(f"  R^2 vs log10({mname:16s}) = {affine_r2(xs, gates):.3f}")
    print("  (pre-reg: dR^2 < 0.05 = cannot discriminate here)")

    print("\n== sector (b): bits axis @19M — ordering test ==")
    n19 = Ns["d384-19M"]
    for mname, f in MEASURES.items():
        xs = [f(n19, 384, 8, b) for _, b, _ in BITS_AXIS]
        ys = [g for *_, g in BITS_AXIS]
        rho = spearman(xs, ys)
        viol = [(a[0], c[0]) for i, a in enumerate(BITS_AXIS)
                for c in BITS_AXIS[i + 1:]
                if (f(n19, 384, 8, a[1]) - f(n19, 384, 8, c[1]))
                * (a[2] - c[2]) < 0]
        print(f"  {mname:16s} rho={rho:+.3f}  violations={viol or 'none'}")

    print("\n== sector (c): crossover grid — the decider ==")
    for mname, f in MEASURES.items():
        cells = [(f(Nd[d], d, 12 if d >= 512 else 8, b), g, (d, b))
                 for d, b, g in CROSSOVER]
        rho = spearman([c[0] for c in cells], [c[1] for c in cells])
        viol = [(a[2], c[2]) for i, a in enumerate(cells)
                for c in cells[i + 1:]
                if (a[0] - c[0]) * (a[1] - c[1]) < 0]
        top_pred = max(cells)[2]
        print(f"  {mname:16s} rho={rho:+.3f}  predicts-top={top_pred}"
              f"  pair-violations={len(viol)}")

    print("\n(underfed fence: 113M=65(3ep), 200M=49(1ep), 400M=30(1ep) "
          "excluded from every fit)")


if __name__ == "__main__":
    main()
