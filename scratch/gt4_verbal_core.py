"""MOE-GT-4 readouts: does the verbal branch have its own core?

Reuses gt2_jaccard's certified decode-only pipeline (DROP_TAIL keep-
rule, 45.3% keep-sets, size-matched first-80-gate-prompt references —
the GT-3 controls). Prints, per the pre-reg:
  (i)   Jaccard(prose, dialog) + split-half nulls for both verbal
        corpora (the registered discriminator line);
  (ii)  Jaccard(dialog, math/phys/code/proofs);
  (iii) GT2-CORE-0 containment in dialog (calibration: prose 0.250),
        against the D3-frozen core (checkpoints/gt3_core_keep.json);
  (iv)  VERBAL-CORE candidate: per-layer prose&dialog size vs the
        independence null, and its containment in each verbal
        coalition.

Usage: .venv/bin/python scratch/gt4_verbal_core.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gt2_jaccard import FRAC, decode_counts, jmean, keep

TRAJ = {
    "math": ("logs/opus/moe_gt1_traj_v2.jsonl", True),
    "phys": ("logs/opus/gt2_phys_traj.jsonl", True),
    "code": ("logs/opus/gt2_code_traj.jsonl", True),
    "proofs": ("logs/opus/gt3_proofs_traj.jsonl", False),
    "prose": ("logs/opus/gt3_prose_traj.jsonl", False),
    "dialog": ("logs/opus/gt4_dialog_traj.jsonl", False),
}


def main():
    counts, keeps = {}, {}
    for d, (p, trim) in TRAJ.items():
        pred = ((lambda r: isinstance(r["prompt"], int) and r["prompt"] < 80)
                if trim else (lambda r: True))
        counts[d] = decode_counts(p, pred)
        keeps[d] = keep(counts[d])
    k = len(next(iter(keeps["math"].values())))
    print(f"frac {FRAC} | keep {k}/128 | 80-prompt refs")

    # (i) the discriminator line
    m, lo = jmean(keeps["prose"], keeps["dialog"])
    print(f"(i) Jaccard(prose, dialog): mean {m:.4f} min {lo:.4f}")
    for d in ("prose", "dialog"):
        p, _ = TRAJ[d]
        half = lambda par: (lambda r: r["prompt"] % 2 == par)
        mh, loh = jmean(keep(decode_counts(p, half(0))),
                        keep(decode_counts(p, half(1))))
        print(f"    {d} split-half null: mean {mh:.4f} min {loh:.4f}")

    # (ii) dialog vs the symbolic domains
    for d in ("math", "phys", "code", "proofs"):
        m, lo = jmean(keeps["dialog"], keeps[d])
        print(f"(ii) Jaccard(dialog, {d}): mean {m:.4f} min {lo:.4f}")

    # (iii) GT2-CORE-0 containment (frozen D3 core)
    core = {int(li): set(v) for li, v in
            json.load(open("checkpoints/gt3_core_keep.json")).items()}
    for d in ("dialog", "prose"):
        cont = [len(core[li] & keeps[d][li]) / len(core[li])
                for li in core if core[li]]
        print(f"(iii) GT2-CORE-0 containment in {d}: "
              f"{sum(cont) / len(cont):.4f}")

    # (iv) the verbal-core candidate
    vcore = {li: keeps["prose"][li] & keeps["dialog"][li]
             for li in keeps["prose"]}
    sizes = [len(v) for v in vcore.values()]
    print(f"(iv) VERBAL core: mean {sum(sizes)/len(sizes):.1f}/{k} per "
          f"layer (min {min(sizes)} max {max(sizes)}; independence "
          f"null {k*(k/128)**2:.1f})")
    for d in ("prose", "dialog"):
        cont = [len(vcore[li] & keeps[d][li]) / len(vcore[li])
                for li in vcore if vcore[li]]
        print(f"    verbal-core containment in {d}: "
              f"{sum(cont) / len(cont):.4f}")
    # symmetry check vs the symbolic base: overlap of the two cores
    ov = [len(vcore[li] & core[li]) / len(vcore[li] | core[li])
          for li in core if vcore[li] | core[li]]
    print(f"    Jaccard(verbal core, GT2-CORE-0): "
          f"{sum(ov) / len(ov):.4f}")


if __name__ == "__main__":
    main()
