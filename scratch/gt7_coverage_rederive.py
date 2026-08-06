"""GT-7 precursor: re-derive MOE-GT-6's exploratory coverage lenses
from committed artifacts (reviewer-scan gap 2026-08-06: the 0.755
Spearman was a desk cell with no committed derivation — GT-7 cannot
register coverage as its ladder variable until the number has a
script, the GT-3 discipline).

Inputs (all committed or regenerable):
  checkpoints/gt6_ladder_r{60..80}_d{0,1}.json + gt6_novrb_d{0,1}
  checkpoints/moe_gt1_arm0.json          (the arm0 recall axis)
  checkpoints/gt3_core_keep.json         (symbolic core)
  logs/opus/gt3_prose_traj.jsonl + gt4_dialog_traj.jsonl
                                         (verbal core, via
                                          gt2_jaccard keep rule)
Gate scores are BOOKED data (VERDICT MOE-GT-6, dicts sum-verified
there) and are transcribed here with their arm names.

Definitions under test (the desk cell's, made explicit):
  verbal-only experts(li) = (prose-keep & dialog-keep) - core
  verbal coverage  = mean_li |keep & vonly| / |vonly|
  math-ext experts(li) = gt1 top-demand 45.3% keep - core
  math-ext coverage = mean_li |keep & mext| / |mext|
  global recall    = arm0 count-weighted open-loop recall
  min-layer recall = min_li per-layer recall
Spearman by hand (mean-rank ties), no scipy.

OUTCOME (2026-08-06, booked as AMENDMENT MOE-GT-6-LENSES): the
three quoted per-arm verbal coverages reproduce EXACTLY (novrb
0.000/0.000, r75_d0 0.430, r80_d0 0.538) — the lens definition and
artifacts are sound — but NONE of the four booked Spearmans
(0.755/0.532/0.502/0.434) reproduce under any gate-pair ordering
consistent with the verdict text (full 16-permutation sweep, both
set-fraction and demand-weighted lenses). The desk cell's
correlation inputs are not reconstructable; the four Spearmans are
demoted to UNVERIFIED-EXPLORATORY. THIS SCRIPT is the lens
authority for GT-7: the registered variable is what this file
computes. Gate totals below use the verdict's quotable anchors
(r70_d0=56 at 0.703 recall, r75_d0=16, r80 d0/d1 = 9/66, novrb
0/7); r60/r65 draw order is the unreconstructable residue.
Asserts: per-arm coverages (hard); Spearmans print as REPORT.

Usage: .venv/bin/python scratch/gt7_coverage_rederive.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")

# BOOKED gate totals per arm (VERDICT MOE-GT-6; 2-draw pairs
# reconstructed from the booked means/spreads and quoted pairs).
GATE = {
    "gt6_ladder_r60_d0": 0, "gt6_ladder_r60_d1": 4,
    "gt6_ladder_r65_d0": 4, "gt6_ladder_r65_d1": 30,
    "gt6_ladder_r70_d0": 56, "gt6_ladder_r70_d1": 9,
    "gt6_ladder_r75_d0": 16, "gt6_ladder_r75_d1": 51,
    "gt6_ladder_r80_d0": 9, "gt6_ladder_r80_d1": 66,
    "gt6_novrb_d0": 0, "gt6_novrb_d1": 7,
}
BOOKED = {"verbal": 0.755, "recall": 0.532,
          "minlayer": 0.502, "mathext": 0.434}
BOOKED_ARM_VERBAL = {"gt6_novrb_d0": 0.000, "gt6_novrb_d1": 0.000,
                     "gt6_ladder_r75_d0": 0.430,
                     "gt6_ladder_r80_d0": 0.538}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            mean_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                r[order[k]] = mean_rank
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy)


def main():
    from gt2_jaccard import decode_counts, keep as keeprule

    arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
    counts = {int(li): row for li, row in arm0["counts"].items()}
    core = {int(li): set(v) for li, v in
            json.load(open("checkpoints/gt3_core_keep.json")).items()}
    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vonly = {li: (kp[li] & kd[li]) - core[li] for li in core}
    kmath = keeprule({li: {e: c for e, c in enumerate(row)}
                      for li, row in counts.items()})
    mext = {li: kmath[li] - core[li] for li in core}

    def recall(keepset):
        hit = tot = 0
        per = []
        for li, row in counts.items():
            h = t = 0
            for e, c in enumerate(row):
                t += c
                if e in keepset[li]:
                    h += c
            hit, tot = hit + h, tot + t
            per.append(h / t)
        return hit / tot, min(per)

    def setcov(keepset, ref):
        vals = [len(keepset[li] & ref[li]) / len(ref[li])
                for li in ref if ref[li]]
        return sum(vals) / len(vals)

    arms = sorted(GATE)
    rows = {}
    for a in arms:
        ks = {int(li): set(v) for li, v in
              json.load(open(f"checkpoints/{a}.json")).items()}
        r, rmin = recall(ks)
        rows[a] = {"gate": GATE[a], "recall": r, "minlayer": rmin,
                   "verbal": setcov(ks, vonly),
                   "mathext": setcov(ks, mext)}
        print(f"{a}: gate {GATE[a]:3d} | recall {r:.3f} | "
              f"verbal {rows[a]['verbal']:.3f} | "
              f"mathext {rows[a]['mathext']:.3f}", flush=True)
    gate = [rows[a]["gate"] for a in arms]
    for lens in ("verbal", "recall", "minlayer", "mathext"):
        s = spearman(gate, [rows[a][lens] for a in arms])
        print(f"Spearman(gate, {lens}) = {s:.3f}  (booked "
              f"{BOOKED[lens]:.3f} — UNVERIFIED-EXPLORATORY, see "
              f"AMENDMENT MOE-GT-6-LENSES)")
    fails = []
    for a, v in BOOKED_ARM_VERBAL.items():
        got = rows[a]["verbal"]
        if abs(got - v) > 0.005:
            fails.append(f"{a} verbal {got:.3f} != {v:.3f}")
    if fails:
        print(f"PER-ARM RE-DERIVATION MISS: {fails}", flush=True)
        sys.exit(1)
    print("PER-ARM COVERAGES REPRODUCE; this script is the GT-7 "
          "lens authority.")


if __name__ == "__main__":
    main()
