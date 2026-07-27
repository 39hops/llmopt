"""Aggregate logs/pp_dist_probe.jsonl (pincer distribution
readout). Every dimension the sidecar carries, reported against
chance: per-model calibration (mass-on-solving v uniform
baseline, top-1 v chance, Spearman, entropy), per-level split,
per-rule-family solve rates + model mass, calibration deciles
(pooled children: predicted mass v realized solve freq), length
bias. Pure read — no model, no oracle.

    python scratch/pincer_dist_report.py [logs/pp_dist_probe.jsonl]
"""
import json
import math
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "logs/pp_dist_probe.jsonl"
rows = [json.loads(l) for l in open(path)]
models = sorted(k[2:] for k in rows[0] if k.startswith("d_"))
print(f"{len(rows)} states; models: {models}\n")

# per-state chance baselines
base_mass = sum(r["n_solving"] / r["n_legal"] for r in rows) / len(rows)
base_top1 = base_mass
print(f"chance baseline (uniform): mass/top1 {base_mass:.3f}; "
      f"mean legal set {sum(r['n_legal'] for r in rows)/len(rows):.1f}, "
      f"mean solving {sum(r['n_solving'] for r in rows)/len(rows):.1f}, "
      f"value-skips {sum(r['n_val_skip'] for r in rows)}")

print(f"\n{'model':<14}{'mass_solv':>10}{'top1_solv':>10}"
      f"{'spearman':>10}{'ent_norm':>9}{'len_corr':>9}")
for m in models:
    d = [r[f"d_{m}"] for r in rows]
    sp_ = [x["spearman_v_solved"] for x in d
           if x["spearman_v_solved"] is not None]
    lc = [x["len_corr"] for x in d if x["len_corr"] is not None]
    print(f"{m:<14}"
          f"{sum(x['mass_solving'] for x in d)/len(d):>10.3f}"
          f"{sum(x['top1_solves'] for x in d)/len(d):>10.3f}"
          f"{sum(sp_)/max(len(sp_),1):>10.3f}"
          f"{sum(x['entropy_norm'] for x in d)/len(d):>9.3f}"
          f"{sum(lc)/max(len(lc),1):>9.3f}")

print("\nper-level top1_solves (n states):")
by_lv = defaultdict(list)
for r in rows:
    by_lv[r["level"]].append(r)
for lv in sorted(by_lv):
    rs = by_lv[lv]
    cells = " ".join(
        f"{m}:{sum(r[f'd_{m}']['top1_solves'] for r in rs)/len(rs):.2f}"
        for m in models)
    print(f"  L{lv} (n={len(rs)}, chance "
          f"{sum(r['n_solving']/r['n_legal'] for r in rs)/len(rs):.2f}): "
          f"{cells}")

print("\nper-rule-family (children pooled): solve rate + mean model mass")
fam = defaultdict(lambda: defaultdict(list))
for r in rows:
    for m in models:
        lps = [c["logp"][m] for c in r["children"]]
        mx = max(lps)
        ps = [math.exp(l - mx) for l in lps]
        z = sum(ps)
        for c, p in zip(r["children"], ps):
            f = c["rule"].split("@")[0]
            fam[f][m].append(p / z)
            if m == models[0]:
                fam[f]["_solved"].append(1.0 if c["solved"] else 0.0)
                fam[f]["_chance"].append(1.0 / r["n_legal"])
for f in sorted(fam, key=lambda f: -len(fam[f]["_solved"])):
    n = len(fam[f]["_solved"])
    sv = sum(fam[f]["_solved"]) / n
    ch = sum(fam[f]["_chance"]) / n
    cells = " ".join(f"{m}:{sum(fam[f][m])/n:.3f}" for m in models)
    print(f"  {f:<16} n={n:<5} solve {sv:.2f}  chance-mass {ch:.3f}  {cells}")

print("\ncalibration deciles (pooled children: predicted mass -> "
      "realized solve freq):")
for m in models:
    pool = []
    for r in rows:
        lps = [c["logp"][m] for c in r["children"]]
        mx = max(lps)
        ps = [math.exp(l - mx) for l in lps]
        z = sum(ps)
        pool += [(p / z, c["solved"]) for p, c in zip(ps, r["children"])]
    pool.sort()
    k = len(pool) // 10
    line = []
    for i in range(10):
        seg = pool[i * k:(i + 1) * k] if i < 9 else pool[9 * k:]
        line.append(f"{sum(p for p,_ in seg)/len(seg):.2f}/"
                    f"{sum(s for _,s in seg)/len(seg):.2f}")
    print(f"  {m:<14} {' '.join(line)}")
