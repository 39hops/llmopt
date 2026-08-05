"""MOE-GT-6 keep-sets: the recall ladder + the verbal-excluded arm.

Measures the SHAPE of the capability shoulder GT-5c established.
Every keep-set = frozen D3 symbolic core + a uniform RANDOM per-layer
fill, with the per-layer fill count k tuned (same k every layer, k
searched per draw) so the arm0-axis open-loop recall lands within
±0.01 of the target. Fill stays random — never demand-ranked — so
recall is the only moving variable above the core.

Arms written to checkpoints/:
  gt6_ladder_r{target}_d{0,1}.json   targets 0.60 0.65 0.70 0.75 0.80
  gt6_novrb_d{0,1}.json              fill drawn from 128 - core -
                                     VERBAL-core (the F7 audit arm),
                                     tuned to ~0.73 (the union mask's
                                     recall) — "not verbal-specific"
                                     is currently measured against
                                     partly-verbal fills; this closes
                                     that.

Recall axis: checkpoints/moe_gt1_arm0.json counts (the arm0-json axis,
same as moe_gt1_arm2's open_loop_recall — never mix axes, GT2-EXCLUSION
4c).
"""

import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
counts = {int(li): row for li, row in arm0["counts"].items()}
core = {int(li): set(v) for li, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}
kp_prose = None  # loaded lazily for the verbal-excluded pool


def recall(keep):
    hit = tot = 0
    for li, row in counts.items():
        kept = keep[li]
        for e, c in enumerate(row):
            tot += c
            if e in kept:
                hit += c
    return hit / tot


def build(k_fill, rng, pool_fn):
    keep = {}
    for li in core:
        pool = sorted(pool_fn(li))
        keep[li] = core[li] | set(rng.sample(pool, min(k_fill, len(pool))))
    return keep


def tune(target, seed_tag, pool_fn, k_max=91):
    """Search the per-layer fill count whose drawn keep-set lands
    within ±0.01 of target recall; the draw is re-made per k with the
    SAME string seed so the accepted set is reproducible."""
    for k in range(0, k_max + 1):
        rng = random.Random(f"{seed_tag}-k{k}")
        keep = build(k, rng, pool_fn)
        r = recall(keep)
        if r >= target - 0.01:
            return keep, r, k
    raise RuntimeError(f"target {target} unreachable")


def dump(keep, r, k, name):
    out = f"checkpoints/{name}.json"
    json.dump({str(li): sorted(v) for li, v in keep.items()}, open(out, "w"))
    sizes = [len(v) for v in keep.values()]
    print(f"{name}: recall {r:.4f} | fill k={k} | mean keep "
          f"{sum(sizes)/len(sizes):.1f}/128")


def main():
    noncore = lambda li: set(range(128)) - core[li]
    for target in (0.60, 0.65, 0.70, 0.75, 0.80):
        for d in (0, 1):
            keep, r, k = tune(target, f"gt6-ladder-{target}-{d}", noncore)
            dump(keep, r, k, f"gt6_ladder_r{int(target*100)}_d{d}")

    from gt2_jaccard import decode_counts, keep as keeprule
    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vcore = {li: kp[li] & kd[li] for li in kp}
    no_verbal = lambda li: set(range(128)) - core[li] - vcore[li]
    for d in (0, 1):
        keep, r, k = tune(0.73, f"gt6-novrb-{d}", no_verbal, k_max=60)
        dump(keep, r, k, f"gt6_novrb_d{d}")


if __name__ == "__main__":
    main()
