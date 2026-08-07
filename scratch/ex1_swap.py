"""EX-ANAT-1 swap builder (IDENTITY battery rung 1, spec
2026-08-06-identity-battery.md, frozen design a-d; Artin GO).

For each GT-7 bin pair (hi, lo), builds the symmetric
CLASS-PRESERVING swap at k=KSWAP (default 4): per layer, the
set-difference partitions into verbal-only and non-verbal-fill
classes; within each class exclusives rank by arm0 demand count
(ties by ascending expert id); the top q = min(|A_c|, |B_c|, k)
exchange sides. Coverage is structurally invariant; recall is
asserted within +-0.01 of the bin band and the builder ABORTS
loudly if not (no repair path, per the grounded design).

Pairs (hi = higher-scoring draw): c15 hi=d1(53) lo=d0(15);
c30 hi=d1(54) lo=d0(5); c45 hi=d0(46) lo=d1(10).

Outputs: checkpoints/ex1_{bin}_{hi,lo}s.json (the swapped sets)
+ printed receipts (recall/coverage/swap counts per arm).

Usage: .venv/bin/python scratch/ex1_swap.py   [KSWAP=4]
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

K = int(os.environ.get("KSWAP", "4"))
# PREFIX names the output family (ex1 = the booked EX-ANAT-1
# subjects; the bisection ladder uses ex1b_k{K}). Existing outputs
# REFUSE to overwrite (cited-evidence guard, handoff 2026-08-06-2).
PREFIX = os.environ.get("PREFIX", "ex1")
RTARGET = 0.72
PAIRS = {  # bin -> (hi draw file, lo draw file)
    "c15": ("gt7_ladder_c15_d1", "gt7_ladder_c15_d0"),
    "c30": ("gt7_ladder_c30_d1", "gt7_ladder_c30_d0"),
    "c45": ("gt7_ladder_c45_d0", "gt7_ladder_c45_d1"),
}

arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
counts = {int(l): r for l, r in arm0["counts"].items()}
core = {int(l): set(v) for l, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}


def recall(kset):
    h = t = 0
    for l, row in counts.items():
        for e, c in enumerate(row):
            t += c
            if e in kset[l]:
                h += c
    return h / t


def setcov(kset, ref):
    vs = [len(kset[l] & ref[l]) / len(ref[l]) for l in ref if ref[l]]
    return sum(vs) / len(vs)


def main():
    from gt2_jaccard import decode_counts, keep as keeprule

    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vonly = {l: (kp[l] & kd[l]) - core[l] for l in core}

    for binname, (hi_f, lo_f) in PAIRS.items():
        hi = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{hi_f}.json")).items()}
        lo = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{lo_f}.json")).items()}
        cov_band = setcov(hi, vonly)
        s_hi, s_lo, n_swap = {}, {}, 0
        for l in core:
            A, B = hi[l] - lo[l], lo[l] - hi[l]
            nh, nl = set(hi[l]), set(lo[l])
            for is_v in (True, False):
                Ac = sorted((e for e in A if (e in vonly[l]) == is_v),
                            key=lambda e: (-counts[l][e], e))
                Bc = sorted((e for e in B if (e in vonly[l]) == is_v),
                            key=lambda e: (-counts[l][e], e))
                q = min(len(Ac), len(Bc), K)
                nh -= set(Ac[:q]); nh |= set(Bc[:q])
                nl -= set(Bc[:q]); nl |= set(Ac[:q])
                n_swap += q
            s_hi[l], s_lo[l] = nh, nl
        for name, orig, sw in ((f"{PREFIX}_{binname}_his", hi, s_hi),
                               (f"{PREFIX}_{binname}_los", lo, s_lo)):
            out = Path(f"checkpoints/{name}.json")
            if out.exists() and os.environ.get("OVERWRITE") != "1":
                raise SystemExit(f"REFUSING to overwrite {out} "
                                 "(cited-evidence guard)")
            r, c = recall(sw), setcov(sw, vonly)
            assert abs(r - RTARGET) <= 0.01, (
                f"{name} recall {r:.4f} OUT OF BAND — ABORT (no repair)")
            assert abs(c - cov_band) <= 0.001, (
                f"{name} coverage {c:.4f} moved from {cov_band:.4f}")
            json.dump({str(l): sorted(v) for l, v in sw.items()},
                      open(f"checkpoints/{name}.json", "w"))
            moved = sum(len(orig[l] - sw[l]) for l in core)
            print(f"{name}: recall {r:.4f} | cov {c:.4f} "
                  f"(band {cov_band:.4f}) | experts moved {moved}",
                  flush=True)


if __name__ == "__main__":
    main()
