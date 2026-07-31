"""A2 (revival-sweep Tier A, 2026-07-31): graph-modularity Q
dispersion on the three same-diet wfloor_d256 seed births — the
"free sigma" the 07-26 NULL entry named but never ran. The +0.030
dQ verdict was a BAR-based null with unmeasured dispersion; this
cell measures it and re-adjudicates. CPU, minutes.
Usage: python scratch/graph_mod_sigma.py
"""
import sys

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
from graph_modularity_gen8 import read  # noqa: E402

SEEDS = {
    "s1": "checkpoints/mathnative_wfloor_d256.pt",
    "s2": "checkpoints/mathnative_wfloor_d256_s2.pt",
    "s3": "checkpoints/mathnative_wfloor_d256_s3.pt",
}


def main():
    qs = {}
    for name, path in SEEDS.items():
        print(f"== {name}: {path}", flush=True)
        q, c = read(path)
        qs[name] = q
        print(f"{name}: mean Q={q:.4f} clustering={c:.4f}", flush=True)
    vals = list(qs.values())
    mean = sum(vals) / len(vals)
    var = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
    sd = var ** 0.5
    print(f"\n[graph-mod sigma] Q per seed {['%.4f' % v for v in vals]}"
          f" | mean {mean:.4f} | sd {sd:.4f}")
    dq = 0.030  # the 07-26 gen-8 vs 19m delta under re-adjudication
    z = dq / max(sd * (2 ** 0.5), 1e-9)   # delta of two draws
    print(f"[graph-mod sigma] dQ +0.030 in seed-sigma units: "
          f"z = {z:.2f} (delta-sigma = sd*sqrt(2))")


if __name__ == "__main__":
    main()
