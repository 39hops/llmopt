"""EX-ANAT-3 subject builder (provenance repair, 2026-08-07 review:
the cited keep-sets were built by inline heredocs — this commits the
exact recipe; BYTE-IDENTITY against the existing cited artifacts is
asserted before this file may serve as their provenance).

Builds, in order:
  checkpoints/ex3_inv_pooled.json   80 tri-bin invariant carriers,
                                    pooled arm0 demand ranking
  checkpoints/ex3_inv_decode.json   83 invariants, decode-only
                                    ranking (moe_gt1_traj_v2 with
                                    DROP_TAIL=1 per the v2 fence)
  checkpoints/ex3_del_invp.json     full-128 minus the pooled 80
  checkpoints/ex3_del_invd.json     full-128 minus the decode 83
  checkpoints/ex3_del_rand{0,1}.json  matched-rank random deletions
      (per pooled invariant: one non-invariant expert sampled from
      its +-8 demand-rank window, string seeds "ex3-rand-{j}",
      draw order = sorted invariant list)
  checkpoints/ex1_full128.json      the all-128 paired-full keepset

Carrier rule (frozen, = scratch/ex1_swap.py): per GT-7 bin pair,
per side, per class (vonly v fill), top-4 exclusives by demand
count (ties ascending id), q capped by the other side's class pool;
invariants = intersection over the three bins. vonly per the lens
authority (gt2_jaccard defaults on the gt3/gt4 trajs, minus core).

Usage: .venv/bin/python scratch/ex3_build.py        (build + verify)
       VERIFY_ONLY=1 ... (assert byte-identity, write nothing)
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

PAIRS = {"c15": ("gt7_ladder_c15_d1", "gt7_ladder_c15_d0"),
         "c30": ("gt7_ladder_c30_d1", "gt7_ladder_c30_d0"),
         "c45": ("gt7_ladder_c45_d0", "gt7_ladder_c45_d1")}

arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
pooled = {int(l): r for l, r in arm0["counts"].items()}
core = {int(l): set(v) for l, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}


def invariants(counts, vonly):
    per_bin = {}
    for b, (hf, lf) in PAIRS.items():
        hi = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{hf}.json")).items()}
        lo = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{lf}.json")).items()}
        cs = set()
        for l in core:
            for own, other in ((hi, lo), (lo, hi)):
                ex = own[l] - other[l]
                for is_v in (True, False):
                    ranked = sorted(
                        (e for e in ex if (e in vonly[l]) == is_v),
                        key=lambda e: (-counts[l][e], e))
                    q = min(len(ranked), 4,
                            len([e for e in (other[l] - own[l])
                                 if (e in vonly[l]) == is_v]))
                    cs |= {(l, e) for e in ranked[:q]}
        per_bin[b] = cs
    return per_bin["c15"] & per_bin["c30"] & per_bin["c45"]


def emit(name, obj):
    out = Path(f"checkpoints/{name}.json")
    text = json.dumps(obj)
    if out.exists():
        if out.read_text() == text:
            print(f"{name}: BYTE-IDENTICAL to cited artifact")
            return
        raise SystemExit(f"{name}: MISMATCH with cited artifact — "
                         "provenance claim would be FALSE, aborting")
    if os.environ.get("VERIFY_ONLY") == "1":
        raise SystemExit(f"{name}: missing under VERIFY_ONLY")
    out.write_text(text)
    print(f"{name}: written")


def main():
    from gt2_jaccard import decode_counts, keep as keeprule

    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vonly = {l: (kp[l] & kd[l]) - core[l] for l in core}
    dec = decode_counts("logs/opus/moe_gt1_traj_v2.jsonl",
                        drop_tail=True)
    dec_full = {l: [dec.get(l, {}).get(e, 0) for e in range(128)]
                for l in core}

    inv_p = invariants(pooled, vonly)
    inv_d = invariants(dec_full, vonly)
    emit("ex3_inv_pooled", sorted([list(x) for x in inv_p]))
    emit("ex3_inv_decode", sorted([list(x) for x in inv_d]))

    def del_set(deleted):
        return {str(l): sorted(set(range(128))
                               - {e for (ll, e) in deleted if ll == l})
                for l in range(48)}
    emit("ex3_del_invp", del_set(inv_p))
    emit("ex3_del_invd", del_set(inv_d))
    for j in (0, 1):
        rng = random.Random(f"ex3-rand-{j}")
        deleted = set()
        for (l, e) in sorted(inv_p):
            order = sorted(range(128),
                           key=lambda x: (-pooled[l][x], x))
            rank = order.index(e)
            window = [x for x in order[max(0, rank - 8):rank + 9]
                      if (l, x) not in inv_p
                      and (l, x) not in deleted and x != e]
            deleted.add((l, rng.choice(window)))
        emit(f"ex3_del_rand{j}", del_set(deleted))
    emit("ex1_full128", {str(l): list(range(128)) for l in range(48)})


if __name__ == "__main__":
    main()
