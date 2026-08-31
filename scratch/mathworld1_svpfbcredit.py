"""MATH-CYBER-1 SVP-FIELD-BLOCK-CREDIT-DESK-0 — POSTHOC
DESCRIPTIVE mechanism desk over the two EXISTING raw stage-3
token-score artifacts (seed 19001 svpfoheld + seed 20001
svpfoheld20). No model inference, no training, no new scoring,
no new population; seeds never pooled. Families:

  A FIELD-BOUNDARY TOP1 — pessimistic top-1 at k=0 and at each
    semantic field boundary in serialization order.
    Blocks (positions 1-indexed): CANONICAL RULE=1-2, SITE=3-5,
    PARAM=6-8, EOS=9; PARAM_FIRST PARAM=1-3, RULE=4-5,
    SITE=6-8, EOS=9. (The GO's expected tables lead with the
    k=1 value as 'start'; k=0 is 0 by the pessimistic
    convention — both are reported.)
  B DYNAMIC GOLD MARGIN M(k) = S_gold(k) - max_nongold S_j(k)
    at each boundary: median, quartiles (lower/upper via
    sorted[n//4], sorted[n//2], sorted[3n//4] — index
    convention disclosed), fraction M(k)>0, boundary deltas of
    the median.
  C FIXED-FINAL-RIVAL ADDITIVE CREDIT — r* chosen ONCE per
    state as the strongest non-gold at k=9 (tie toward lower
    index, the booked scorer's convention); per block
    Delta_block = D(end) - D(start-1) with D(k) = S_gold(k) -
    S_r*(k), D(0)=0; blocks partition k=1..9 so the block
    deltas sum exactly to the final gold-v-r* margin. Medians
    primary, means riders, sign counts.
  D RESCUE/DAMAGE STATE COUNTS at block boundaries (exact
    definitions in code, disclosed in the receipt), overall and
    within the frozen EARLY-29 / RELOCATED-67 groups.
  E cross-seed routing questions answered descriptively from
    A-D only.

Everything descriptive; no inferential statistics; raw
artifacts are byte-pinned and re-hashed pre and post.

Output logs/mathworld1/svpfbcredit/desk.json (refuse-if-exists).

    .venv/bin/python scratch/mathworld1_svpfbcredit.py   (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.mathworld1_svpbirth import gate  # noqa: E402

RAWS = {
    19001: ("logs/mathworld1/svpfoheld/raw_token_scores.jsonl",
            "ee0319f4c63396f7998cdffa5dee5b3d8c3bcf4b35866a3d"
            "cf499efb9c673111"),
    20001: ("logs/mathworld1/svpfoheld20/raw_token_scores"
            ".jsonl",
            "18f27b666ce54c5402abff746e696f9b8c35c94235d096a0"
            "60f97cf015aae17f"),
}
CENSUS_JSON = "logs/mathworld1/svpforder_census/census.json"
CENSUS_SHA = ("7d8343b3f328b00a1147b675441181e897d3d96eaff33f"
              "e33d403aa482dd71d5")
ARMS = ["CANONICAL", "PARAM_FIRST"]
# semantic blocks, 1-indexed inclusive (k = end position)
BLOCKS = {
    "CANONICAL": [("RULE", 1, 2), ("SITE", 3, 5),
                  ("PARAM", 6, 8), ("EOS", 9, 9)],
    "PARAM_FIRST": [("PARAM", 1, 3), ("RULE", 4, 5),
                    ("SITE", 6, 8), ("EOS", 9, 9)],
}
OUTDIR = Path("logs/mathworld1/svpfbcredit")


def fsha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def cums(vecs):
    out = []
    for v in vecs:
        c, s = [], 0.0
        for x in v:
            s += x
            c.append(s)
        out.append(c)
    return out


def gold_top(cum, li, k):
    g = cum[li][k - 1]
    return all(cum[j][k - 1] < g for j in range(len(cum))
               if j != li)


def margin(cum, li, k):
    g = cum[li][k - 1]
    return g - max(cum[j][k - 1] for j in range(len(cum))
                   if j != li)


def q(vals):
    sv = sorted(vals)
    n = len(sv)
    return {"q1": round(sv[n // 4], 4),
            "median": round(sv[n // 2], 4),
            "q3": round(sv[3 * n // 4], 4),
            "frac_pos": round(sum(1 for x in sv if x > 0) / n,
                              4)}


def med(vals):
    sv = sorted(vals)
    return round(sv[len(sv) // 2], 4)


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    gate(fsha(CENSUS_JSON) == CENSUS_SHA, "CENSUS PIN")
    census = json.loads(Path(CENSUS_JSON).read_text())
    ps = census["structural"]["CANONICAL"]["per_state"]
    early = {b for b, k in ps.items() if k == 1}
    late = {b for b, k in ps.items() if k == 8}
    gate(len(early) == 29 and len(late) == 67, "GROUPS")

    data = {}   # seed -> arm -> block_id -> (cum, li)
    for seed, (path, sha) in RAWS.items():
        gate(fsha(path) == sha, f"RAW PIN {seed}")
        d = {a: {} for a in ARMS}
        for line in open(path):
            row = json.loads(line)
            cum = cums(row["token_lps"])
            gate(all(len(c) == 9 for c in cum), "T!=9")
            d[row["arm"]][row["block_id"]] = (cum,
                                              row["label_index"])
        for a in ARMS:
            gate(len(d[a]) == 96, f"ROWS {seed} {a}")
            gate(set(d[a]) == early | late, f"IDS {seed} {a}")
        data[seed] = d

    out = {"blocks": {a: [[n, s, e] for n, s, e in BLOCKS[a]]
                      for a in ARMS},
           "pins": {str(s): RAWS[s][1] for s in RAWS},
           "census_pin": CENSUS_SHA,
           "definitions": {
               "famA": "pessimistic top-1 count/96 at k=0 "
                       "(=0 by convention) and at each block "
                       "end; k=1 also reported (the GO tables' "
                       "leading value)",
               "famB": "M(k)=S_gold(k)-max_nongold S_j(k); "
                       "quartiles at sorted[n//4], [n//2], "
                       "[3n//4] (upper-type indices)",
               "famC": "r* fixed at k=9 (max cum, tie lower "
                       "index); D(k)=S_gold(k)-S_r*(k), "
                       "D(0)=0; Delta_block=D(end)-D(start-1); "
                       "blocks partition 1..9 so sum = D(9)",
               "famD": "per arm in ITS serialization: "
                       "PARAM_CORRECT = top1 at end PARAM; "
                       "PARAM_DAMAGED = top1 at some k inside "
                       "an EARLIER position (k < PARAM start; "
                       "for PARAM_FIRST, PARAM is first so "
                       "damaged uses top1 at k in 1..2 i.e. "
                       "inside the block before its end) AND "
                       "not top1 at end PARAM; RULE_RESCUED = "
                       "not top1 at end PARAM, top1 at end "
                       "RULE (PARAM_FIRST only: RULE follows "
                       "PARAM); RULE_LOST = top1 at end PARAM, "
                       "not at end RULE (PARAM_FIRST only); "
                       "FINAL_RESCUED = not top1 at end PARAM, "
                       "top1 at k=9"},
           "families": {}}

    for seed in RAWS:
        sd = {}
        for a in ARMS:
            rows = data[seed][a]
            bl = BLOCKS[a]
            # FAMILY A
            acc = {"k0": 0,
                   "k1": sum(1 for c, li in rows.values()
                             if gold_top(c, li, 1))}
            for name, _, e in bl:
                acc[f"end_{name}_k{e}"] = sum(
                    1 for c, li in rows.values()
                    if gold_top(c, li, e))
            seq = [0] + [acc[f"end_{n}_k{e}"]
                         for n, _, e in bl]
            famA = {"top1": acc,
                    "boundary_deltas": {
                        bl[i][0]: seq[i + 1] - seq[i]
                        for i in range(len(bl))}}
            # FAMILY B
            famB = {}
            prev_med = 0.0
            for name, _, e in bl:
                vals = [margin(c, li, e)
                        for c, li in rows.values()]
                stats = q(vals)
                stats["median_delta_from_prev_boundary"] = \
                    round(stats["median"] - prev_med, 4)
                prev_med = stats["median"]
                famB[f"end_{name}_k{e}"] = stats
            # FAMILY C
            famC = {}
            per_block_vals = {n: [] for n, _, _ in bl}
            finals = []
            for c, li in rows.values():
                rivals = [j for j in range(len(c)) if j != li]
                rstar = max(rivals,
                            key=lambda j: (c[j][8], -j))
                def D(k):
                    if k == 0:
                        return 0.0
                    return c[li][k - 1] - c[rstar][k - 1]
                for name, s0, e in bl:
                    per_block_vals[name].append(D(e) - D(s0 - 1))
                finals.append(D(9))
            for name, _, _ in bl:
                v = per_block_vals[name]
                famC[name] = {
                    "median": med(v),
                    "mean": round(sum(v) / len(v), 4),
                    "n_pos": sum(1 for x in v if x > 0),
                    "n_neg": sum(1 for x in v if x < 0),
                    "n_zero": sum(1 for x in v if x == 0)}
            famC["additivity_check_max_abs"] = round(
                max(abs(sum(per_block_vals[n][i]
                            for n, _, _ in bl) - finals[i])
                    for i in range(len(finals))), 10)
            # FAMILY D
            pe = {n: e for n, _, e in bl}
            pstart = {n: s0 for n, s0, _ in bl}
            famD = {}
            for grp_name, grp in (("all", set(rows)),
                                  ("early29", early),
                                  ("late67", late)):
                cnt = Counter()
                for b in grp:
                    c, li = rows[b]
                    tp = {k: gold_top(c, li, k)
                          for k in range(1, 10)}
                    after_param = tp[pe["PARAM"]]
                    before_ks = range(1, pe["PARAM"]) \
                        if a == "PARAM_FIRST" \
                        else range(1, pstart["PARAM"])
                    was_before = any(tp[k] for k in before_ks)
                    if after_param:
                        cnt["PARAM_CORRECT"] += 1
                    if was_before and not after_param:
                        cnt["PARAM_DAMAGED"] += 1
                    if a == "PARAM_FIRST":
                        if not after_param and tp[pe["RULE"]]:
                            cnt["RULE_RESCUED"] += 1
                        if after_param and not tp[pe["RULE"]]:
                            cnt["RULE_LOST"] += 1
                    if not after_param and tp[9]:
                        cnt["FINAL_RESCUED"] += 1
                famD[grp_name] = dict(cnt)
            sd[a] = {"A": famA, "B": famB, "C": famC,
                     "D": famD}
        out["families"][str(seed)] = sd

    for seed, (path, sha) in RAWS.items():
        gate(fsha(path) == sha, f"POST RAW PIN {seed}")
    OUTDIR.mkdir(parents=True)
    (OUTDIR / "desk.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out["families"], indent=1), flush=True)
    print("[svpfbcredit] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
