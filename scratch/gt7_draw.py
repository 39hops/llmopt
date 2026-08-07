"""MOE-GT-7 keep-set draws (PRE-REG MOE-GT-7, fired on Artin GO
2026-08-06). The verbal-coverage ladder at FIXED recall.

Construction per draw (rtarget, ctarget, string seed):
  keep(li) = symbolic core(li)
           | seeded sample of round(ctarget * |vonly(li)|) from
             vonly(li)                       [pins coverage by
                                              construction]
           | k-tuned random fill from 128 - core - vonly
                                             [pins recall WITHOUT
                                              moving coverage]
  accept iff recall in rtarget +- 0.01 AND coverage in
  ctarget +- 0.03; on a k-scan that jumps over the recall window,
  retry with the next attempt-suffixed string seed (reproducible).

LENS DISCIPLINE: vonly and the coverage number use the EXACT code
path of the lens authority (scratch/gt7_coverage_rederive.py):
gt2_jaccard.decode_counts/keep at their DEFAULTS on the committed
gt3_prose/gt4_dialog trajs. Those files are v3-tagged, so the
defaults include DROP_TAIL=1 (per VERDICT LAB-TRAJ's fence that is
the v2 convention over-applied) — but the pre-reg freezes the lens
AS THE SCRIPT COMPUTES IT, and the booked per-arm coverages
(0.430/0.538) reproduce only under it. Frozen means frozen.

Arms written to checkpoints/:
  gt7_ladder_c{15,30,45}_d{0,1}.json   recall 0.72, coverage bins
  gt7_anom_r75_{0,1}.json              (0.75, 0.430) fresh seeds
  gt7_anom_r80_{0,1}.json              (0.80, 0.538) fresh seeds
The 0.00 bin is the existing gt6_novrb pair, REUSED per pre-reg
(gates already booked 0/7 — not re-run).

Recall axis: checkpoints/moe_gt1_arm0.json counts (arm0-json axis,
never mixed — GT2-EXCLUSION 4c).

Usage: .venv/bin/python scratch/gt7_draw.py   (desk-only, no model)
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
counts = {int(li): row for li, row in arm0["counts"].items()}
core = {int(li): set(v) for li, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}


def _assert_lens_env_clean():
    # Cited-evidence guard (code review 2026-08-07): the lens resolves
    # FRAC/GATE_ONLY/DROP_TAIL from env at CALL time — a polluted shell
    # silently changes vonly and every drawn set. Refuse, never adapt.
    import os as _os
    bad = [k for k in ("FRAC", "GATE_ONLY", "DROP_TAIL") if k in _os.environ]
    if bad:
        raise SystemExit(f"lens env polluted ({bad}) — unset before drawing")


def recall(keep):
    hit = tot = 0
    for li, row in counts.items():
        kept = keep[li]
        for e, c in enumerate(row):
            tot += c
            if e in kept:
                hit += c
    return hit / tot


def setcov(keepset, ref):
    vals = [len(keepset[li] & ref[li]) / len(ref[li])
            for li in ref if ref[li]]
    return sum(vals) / len(vals)


def draw(rtarget, ctarget, seed_tag, vonly, k_max=91, attempts=20):
    pool = {li: sorted(set(range(128)) - core[li] - vonly[li])
            for li in core}
    vsorted = {li: sorted(vonly[li]) for li in core}
    for a in range(attempts):
        tag = seed_tag if a == 0 else f"{seed_tag}-a{a}"
        # scan assumption: accepts the FIRST k landing in window;
        # per-k reseeding makes recall non-monotone in k, so a
        # coverage-miss break may skip other viable k (reachability
        # only — accepted arms recompute r/c exactly).
        for k in range(0, k_max + 1):
            rng = random.Random(f"{tag}-k{k}")
            keep = {}
            for li in sorted(core):  # RNG stream pinned to numeric
                nv = round(ctarget * len(vsorted[li]))  # order (== file order today, asserted no-op at adoption)
                vpick = set(rng.sample(vsorted[li], nv)) if nv else set()
                fill = set(rng.sample(pool[li], min(k, len(pool[li]))))
                keep[li] = core[li] | vpick | fill
            r = recall(keep)
            if r > rtarget + 0.01:
                break  # overshot the window on this seed; next attempt
            if r >= rtarget - 0.01:
                c = setcov(keep, vonly)
                if abs(c - ctarget) <= 0.03:
                    return keep, r, c, k, tag
                break  # recall landed but coverage off; reseed
    raise RuntimeError(f"({rtarget},{ctarget}) unreachable in "
                       f"{attempts} attempts")


def dump(keep, name, r, c, k, tag):
    out = f"checkpoints/{name}.json"
    if Path(out).exists() and os.environ.get("OVERWRITE") != "1":
        raise SystemExit(f"REFUSING to overwrite {out} "
                         "(cited-evidence guard, handoff 2026-08-06-2)")
    json.dump({str(li): sorted(v) for li, v in keep.items()},
              open(out, "w"))
    sizes = [len(v) for v in keep.values()]
    print(f"{name}: recall {r:.4f} | verbal-cov {c:.4f} | fill k={k} "
          f"| seed {tag} | mean keep {sum(sizes)/len(sizes):.1f}/128",
          flush=True)


def main():
    _assert_lens_env_clean()
    from gt2_jaccard import decode_counts, keep as keeprule

    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vonly = {li: (kp[li] & kd[li]) - core[li] for li in core}
    vsz = [len(vonly[li]) for li in sorted(vonly)]
    print(f"[gt7] vonly sizes: min {min(vsz)} max {max(vsz)} "
          f"mean {sum(vsz)/len(vsz):.1f}", flush=True)

    for c in (0.15, 0.30, 0.45):
        for d in (0, 1):
            keep, r, cov, k, tag = draw(
                0.72, c, f"gt7-ladder-c{c}-d{d}", vonly)
            dump(keep, f"gt7_ladder_c{int(c*100)}_d{d}", r, cov, k, tag)
    for name, rt, ct in (("r75", 0.75, 0.430), ("r80", 0.80, 0.538)):
        for j in (0, 1):
            keep, r, cov, k, tag = draw(
                rt, ct, f"gt7-anom-{name}-{j}", vonly)
            dump(keep, f"gt7_anom_{name}_{j}", r, cov, k, tag)


if __name__ == "__main__":
    main()
