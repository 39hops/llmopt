"""EX-ANAT-2 one-sided arm builder v2 (IDENTITY battery rung 2,
sharpened by VERDICT EX-ANAT-1B; Artin GO, Mac).

v1 (replace-with-bottom) ABORTED on its own band assertion — the
rank asymmetry between one side's top and the other's bottom
exclusives carries ~0.06 recall, structurally unpinnable. v2 uses
PURE one-sided arms with MATCHED CONTROLS: the control carries the
same count/class change, so identity reads as (arm - control) and
no pinning is needed. Per bin pair, per side, k=4 per class:

  ADD    own | other side's TOP-4-per-class exclusives (pure add)
  ADDC   own | count-equal NEAREST-DEMAND-matched experts from the
         non-kept class pool (greedy per added expert) — matches
         count, class, AND approximate mass
  REM    own - own TOP-2-per-class exclusives (k=2 for REM: at k=4
         the vonly-exclusive class is mostly consumed and the arm
         becomes a verbal-exclusion experiment — v2 receipt showed
         cov 0.149->0.052 with the control pool exhausted; k=2
         also aims REM at the sharpest 1B question: was the k=2
         "redundancy" just the masked add?)
  REMC   own - count-equal random from own NON-TOP-2 exclusives
         (pool sufficient at k=2; coverage matches exactly)

Class-preserving counts: coverage moves identically in arm and
control. Recall is RECORDED per arm (receipts), not pinned —
registered grounding: GT-6/GT-7 measured that recall in this band
does not organize capability (draws 0.71-0.80 scored 5-75), so
band-scale recall mismatch cannot explain 20-solve deltas.

Outputs: checkpoints/ex2_{bin}_{hi|lo}_{add|addc|rem|remc}.json
(24 sets). Overwrite-guarded. Ranking: arm0 demand count within
class, ties ascending id.

Usage: .venv/bin/python scratch/ex2_build.py
"""

import json
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

K = 4      # ADD leg
KREM = 2   # REM leg (see docstring)
PAIRS = {
    "c15": ("gt7_ladder_c15_d1", "gt7_ladder_c15_d0"),
    "c30": ("gt7_ladder_c30_d1", "gt7_ladder_c30_d0"),
    "c45": ("gt7_ladder_c45_d0", "gt7_ladder_c45_d1"),
}

arm0 = json.loads(Path("checkpoints/moe_gt1_arm0.json").read_text())
counts = {int(l): r for l, r in arm0["counts"].items()}
core = {int(l): set(v) for l, v in
        json.load(open("checkpoints/gt3_core_keep.json")).items()}


def _assert_lens_env_clean():
    # Cited-evidence guard (code review 2026-08-07): the lens resolves
    # FRAC/GATE_ONLY/DROP_TAIL from env at CALL time — a polluted shell
    # silently changes vonly and every drawn set. Refuse, never adapt.
    import os as _os
    bad = [k for k in ("FRAC", "GATE_ONLY", "DROP_TAIL") if k in _os.environ]
    if bad:
        raise SystemExit(f"lens env polluted ({bad}) — unset before drawing")


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


def ranked(pool, l, is_v, vonly):
    return sorted((e for e in pool if (e in vonly[l]) == is_v),
                  key=lambda e: (-counts[l][e], e))


def main():
    _assert_lens_env_clean()
    from gt2_jaccard import decode_counts, keep as keeprule

    kp = keeprule(decode_counts("logs/opus/gt3_prose_traj.jsonl"))
    kd = keeprule(decode_counts("logs/opus/gt4_dialog_traj.jsonl"))
    vonly = {l: (kp[l] & kd[l]) - core[l] for l in core}

    for binname, (hi_f, lo_f) in PAIRS.items():
        hi = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{hi_f}.json")).items()}
        lo = {int(l): set(v) for l, v in
              json.load(open(f"checkpoints/{lo_f}.json")).items()}
        for side, own, other in (("hi", hi, lo), ("lo", lo, hi)):
            rng = random.Random(f"ex2-{binname}-{side}")
            arms = {m: {} for m in ("add", "addc", "rem", "remc")}
            for l in sorted(core):  # RNG stream pinned
                own_ex = own[l] - other[l]
                oth_ex = other[l] - own[l]
                nonkept = set(range(128)) - own[l]
                a, ac = set(own[l]), set(own[l])
                r, rc = set(own[l]), set(own[l])
                for is_v in (True, False):
                    top_oth = ranked(oth_ex, l, is_v, vonly)[:K]
                    a |= set(top_oth)
                    # ADDC: greedy nearest-demand match per added
                    # expert from the non-kept class pool
                    pool = [e for e in sorted(nonkept)
                            if ((e in vonly[l]) == is_v)
                            and e not in top_oth]
                    taken = set()
                    for e in top_oth:
                        cand = min((c for c in pool if c not in taken),
                                   key=lambda c: (abs(counts[l][c]
                                                      - counts[l][e]), c),
                                   default=None)
                        if cand is not None:
                            taken.add(cand)
                    ac |= taken
                    top_own = ranked(own_ex, l, is_v, vonly)[:KREM]
                    r -= set(top_own)
                    rest = [e for e in sorted(own_ex)
                            if ((e in vonly[l]) == is_v)
                            and e not in top_own]
                    rc -= set(rng.sample(rest, min(len(top_own),
                                                   len(rest))))
                arms["add"][l], arms["addc"][l] = a, ac
                arms["rem"][l], arms["remc"][l] = r, rc
            for mode, kset in arms.items():
                name = f"ex2_{binname}_{side}_{mode}"
                out = Path(f"checkpoints/{name}.json")
                if out.exists() and os.environ.get("OVERWRITE") != "1":
                    raise SystemExit(f"REFUSING to overwrite {out}")
                json.dump({str(l): sorted(v) for l, v in kset.items()},
                          open(out, "w"))
                delta = sum(len(kset[l] ^ own[l]) for l in core)
                print(f"{name}: recall {recall(kset):.4f} | "
                      f"cov {setcov(kset, vonly):.4f} | "
                      f"delta-slots {delta}", flush=True)


if __name__ == "__main__":
    main()
