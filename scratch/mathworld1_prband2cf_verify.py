"""Independent verifier for MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-
NUISANCE-COUNTERFACTUAL-0. Reconstructs every aggregate from the raw
per-token lps in logs/mathworld1/prband2cf/cf_scores.jsonl with its
own code (no scorer aggregate function imported), and checks:

  RAW replay exact v the tracked booked stream; view prompt mapping
  (cur_sha per row == sha of the pinned views.jsonl cur); invariant
  state / gold / candidates / continuations across views; every SUM;
  top-1; target margins d / m; Delta_K / Delta_LOW; epsilon classes
  and polarities; pair labels; top-action transitions; MASK0
  cross-view spread; SURVIVES / COLLAPSES / REVERSES / PARTIAL; all
  checkpoint summaries — each against the scorer's receipts.

Usage: .venv/bin/python scratch/mathworld1_prband2cf_verify.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

D = Path("logs/mathworld1/prband2cf")
OUT = Path("logs/mathworld1/prband2cf_verify")
VIEWS = "logs/mathworld1/prband2nuis/views.jsonl"
OLD = "logs/mathworld1/prband2score/raw_scores.jsonl"
PRIMARY = "logs/mathworld1/prband2prod/primary.jsonl"
EPS_SCORE, EPS_D, EPS_DELTA = 1e-05, 2e-05, 4e-05
VN = ["RAW", "K_FIRST", "LOW_PAIR_FIRST"]
SEM = [("i_sum", "I", 0, "none", -1), ("i_unprod", "I", 0, "term_index", 1),
       ("i_unprod", "I", 0, "term_index", 3), ("i_unprod", "I", 0, "term_index", 5)]
NAME = {SEM[0]: "i_sum", SEM[1]: "A0", SEM[2]: "B0", SEM[3]: "I0/t5"}
A0, B0 = SEM[1], SEM[2]


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def main():
    dis = []
    rec = json.loads((D / "prband2cf_receipt.json").read_text())
    cells = json.loads((D / "cells.json").read_text())
    per_state = json.loads((D / "per_state.json").read_text())
    pairs = json.loads((D / "pairs.json").read_text())
    mask0 = json.loads((D / "mask0.json").read_text())
    replay = json.loads((D / "replay.json").read_text())
    P = [json.loads(l) for l in open(PRIMARY)]
    views = {}
    for l in open(VIEWS):
        r = json.loads(l)
        views[(r["pair_id"], r["theta"], r["view"])] = r["cur"]
    rows = [json.loads(l) for l in open(D / "cf_scores.jsonl")]
    if len(rows) != 4 * 3 * 2 * 96 * 4:
        dis.append(f"row count {len(rows)}")
    # --- sums, view mapping, invariants ---
    S = defaultdict(dict)
    conts = defaultdict(set)
    for r in rows:
        if abs(sum(r["lps"]) - r["sum"]) > 0 or len(r["lps"]) != 9:
            dis.append(f"sum/len row {r['seed']} {r['view']} {r['state']}")
        k = (r["pair_id"], r["theta"], r["view"])
        if sha(views[k]) != r["cur_sha"]:
            dis.append(f"view mapping {k}")
        p = P[r["state"]]
        if p["pair_id"] != r["pair_id"] or p["theta"] != r["theta"] \
                or p["cur_sha"] != r["raw_cur_sha"] or p["gold_tuple"] != r["gold"]:
            dis.append(f"state identity {r['state']}")
        conts[(r["representation"], r["state"], tuple(r["candidate"]))].add(
            tuple(r["continuation"]))
        S[(r["seed"], r["representation"], r["arm"], r["view"], r["state"])][
            tuple(r["candidate"])] = r["sum"]
    if any(len(v) != 1 for v in conts.values()):
        dis.append("continuation varies across views/arms")
    # --- RAW replay ---
    old = {}
    for l in open(OLD):
        r = json.loads(l)
        old[(r["seed"], r["representation"], r["arm"], r["state"],
             tuple(r["candidate"]))] = (r["lps"], r["sum"])
    n_cmp = n_ok = 0
    drift = 0.0
    for r in rows:
        if r["view"] != "RAW":
            continue
        o = old[(r["seed"], r["representation"], r["arm"], r["state"],
                 tuple(r["candidate"]))]
        n_cmp += 1
        n_ok += (o[0] == r["lps"] and o[1] == r["sum"])
        drift = max(drift, max(abs(a - b) for a, b in zip(o[0], r["lps"])))
    if n_cmp != 3072 or n_ok != n_cmp or not replay["exact"] \
            or replay["rows_exact"] != n_ok:
        dis.append(f"replay {n_ok}/{n_cmp} receipt {replay['rows_exact']}")
    # --- per cell ---
    gold = {i: tuple(p["gold_tuple"]) for i, p in enumerate(P)}
    g = {i: 1 if p["theta"] == "SIN_LOW" else -1 for i, p in enumerate(P)}
    summary = {}
    for seed in ("19001", "20001"):
        for rep in ("CANONICAL", "PARAM_FIRST"):
            ck = f"{seed}|{rep}"
            for arm in ("FULL", "MASK0"):
                tops, d = {}, {}
                for vn in VN:
                    tp, corr = [], []
                    for i in range(96):
                        sc = S[(seed, rep, arm, vn, i)]
                        best = max(sc.values())
                        ws = [k for k, v in sc.items() if v == best]
                        w = ws[0] if len(ws) == 1 else None
                        tp.append((NAME[w] if w else None,
                                   best - sorted(sc.values())[-2]))
                        corr.append(w == gold[i])
                        d[(vn, i)] = sc[A0] - sc[B0]
                    tops[vn] = tp
                    c = cells[f"{ck}|{arm}|{vn}"]
                    mine = {"top1": sum(corr),
                            "A0_correct": sum(x for x, i in zip(corr, range(96))
                                              if gold[i] == A0),
                            "B0_correct": sum(x for x, i in zip(corr, range(96))
                                              if gold[i] == B0),
                            "both_correct_pairs": sum(corr[k] and corr[k + 1]
                                                      for k in range(0, 96, 2)),
                            "correct_direction_pairs": sum(
                                g[k] * d[(vn, k)] >= EPS_D
                                and g[k + 1] * d[(vn, k + 1)] >= EPS_D
                                for k in range(0, 96, 2)),
                            "reversed_direction_pairs": sum(
                                g[k] * d[(vn, k)] <= -EPS_D
                                and g[k + 1] * d[(vn, k + 1)] <= -EPS_D
                                for k in range(0, 96, 2))}
                    for key, v in mine.items():
                        if c[key] != v:
                            dis.append(f"{ck} {arm} {vn} {key} {c[key]} v {v}")
                    summary[(ck, arm, vn)] = mine
                # classes
                ps = per_state[f"{ck}|{arm}"]
                cls_cnt, pol_cnt = Counter(), Counter()
                mine_states = []
                for i in range(96):
                    dR, dK, dL = d[("RAW", i)], d[("K_FIRST", i)], d[("LOW_PAIR_FIRST", i)]
                    mR, mL = g[i] * dR, g[i] * dL
                    rob = lambda a, b: sgn(a) != sgn(b) and abs(a) >= EPS_D and abs(b) >= EPS_D  # noqa: E731
                    fRL, fRK = rob(dR, dL), rob(dR, dK)
                    if min(abs(dR), abs(dK), abs(dL)) < EPS_D:
                        cls, pol = "SUBNOISE", None
                    elif fRL and abs(dL - dR) >= EPS_DELTA:
                        cls = "CUE-FOLLOWING"
                        pol = ("CORRECT-POLARITY" if mR >= EPS_D and mL <= -EPS_D
                               else "REVERSED-POLARITY" if mR <= -EPS_D and mL >= EPS_D
                               else None)
                    elif fRK:
                        cls, pol = "NEUTRAL-SHIFT", None
                    else:
                        cls, pol = "RENDER-SIGN-INVARIANT", None
                    s = ps[i]
                    if (s["class"], s["polarity"]) != (cls, pol) \
                            or abs(s["d_RAW"] - dR) > 0 or abs(s["Delta_LOW"] - (dL - dR)) > 0 \
                            or abs(s["Delta_K"] - (dK - dR)) > 0 \
                            or s["flags"]["ROBUST_RAW_LOW_FLIP"] != fRL \
                            or s["flags"]["ROBUST_RAW_K_FLIP"] != fRK:
                        dis.append(f"{ck} {arm} state {i} class")
                    cls_cnt[cls] += 1
                    if pol:
                        pol_cnt[f"{cls} {pol}"] += 1
                    mine_states.append((dR, dK, dL, mR, mL, cls))
                con = cells[f"{ck}|{arm}|CONTRASTS"]
                if con["class_census"] != dict(cls_cnt) or con["polarity_census"] != dict(pol_cnt):
                    dis.append(f"{ck} {arm} census {con['class_census']} v {dict(cls_cnt)}")
                # transitions + changes
                for a, b in (("RAW", "K_FIRST"), ("RAW", "LOW_PAIR_FIRST"),
                             ("K_FIRST", "LOW_PAIR_FIRST")):
                    ch = [tops[a][i][0] != tops[b][i][0] for i in range(96)]
                    rob = sum(c and tops[a][i][1] >= EPS_SCORE and tops[b][i][1] >= EPS_SCORE
                              for i, c in enumerate(ch))
                    tr = Counter(f"{tops[a][i][0]}->{tops[b][i][0]}" for i in range(96))
                    mine_ch = {"states_changed": sum(ch), "robust_changes": rob,
                               "pairs_with_change": sum(ch[k] or ch[k + 1]
                                                        for k in range(0, 96, 2)),
                               "transition": dict(tr)}
                    got = con["changes"][f"{a}->{b}"]
                    if any(got[k] != v for k, v in mine_ch.items()):
                        dis.append(f"{ck} {arm} changes {a}->{b}")
                # pair labels
                pc = Counter()
                for k in range(0, 96, 2):
                    a, b = mine_states[k], mine_states[k + 1]
                    corrf = {vn: (tops[vn][k][0] == NAME[gold[k]]) and
                             (tops[vn][k + 1][0] == NAME[gold[k + 1]]) for vn in VN}
                    lab = {
                        "PAIR-BUNDLE-FOLLOWING CORRECT-POLARITY": all(
                            x[3] >= EPS_D and x[4] <= -EPS_D for x in (a, b)),
                        "PAIR-BUNDLE-FOLLOWING REVERSED-POLARITY": all(
                            x[3] <= -EPS_D and x[4] >= EPS_D for x in (a, b)),
                        "PAIR-REGISTERED-CUE-REMOVAL-STABLE": all(
                            sgn(x[0]) == sgn(x[1]) and abs(x[0]) >= EPS_D and abs(x[1]) >= EPS_D
                            for x in (a, b)),
                        "PAIR-BUNDLE-REVERSAL-STABLE": all(
                            sgn(x[0]) == sgn(x[2]) and abs(x[0]) >= EPS_D and abs(x[2]) >= EPS_D
                            for x in (a, b)),
                        "raw_both_correct_with_cue_following": corrf["RAW"] and any(
                            x[5] == "CUE-FOLLOWING" for x in (a, b)),
                        "raw_both_correct_with_neutral_shift": corrf["RAW"] and any(
                            x[5] == "NEUTRAL-SHIFT" for x in (a, b))}
                    got = pairs[f"{ck}|{arm}"][k // 2]
                    for key, v in lab.items():
                        pc[key] += v
                        if got[key] != v:
                            dis.append(f"{ck} {arm} pair {k // 2} {key}")
                    if got["both_correct"] != corrf:
                        dis.append(f"{ck} {arm} pair {k // 2} both_correct")
                for key, v in pc.items():
                    if con["pair_census"][key] != v:
                        dis.append(f"{ck} {arm} pair census {key}")
                # reading law
                cR = summary[(ck, arm, "RAW")]
                for vn, sk in (("K_FIRST", "raw_both_correct_with_neutral_shift"),
                               ("LOW_PAIR_FIRST", "raw_both_correct_with_cue_following")):
                    cv = summary[(ck, arm, vn)]
                    rd = {"SURVIVES": cv["both_correct_pairs"] >= cR["both_correct_pairs"] - 2
                          and cv["top1"] >= cR["top1"] - 4 and pc[sk] <= 2}
                    if vn == "LOW_PAIR_FIRST":
                        rd["COLLAPSES"] = cv["both_correct_pairs"] <= cR["both_correct_pairs"] // 2
                        rd["REVERSES"] = (pc["PAIR-BUNDLE-FOLLOWING CORRECT-POLARITY"]
                                          + pc["PAIR-BUNDLE-FOLLOWING REVERSED-POLARITY"]) >= 24
                    rd["PARTIAL"] = not any(rd.values())
                    if con["reading"][vn] != rd:
                        dis.append(f"{ck} {arm} reading {vn} {con['reading'][vn]} v {rd}")
                # MASK0 spread
                if arm == "MASK0":
                    sp_max = 0.0
                    for i in range(96):
                        for s in SEM:
                            vals = [S[(seed, rep, arm, vn, i)][s] for vn in VN]
                            sp_max = max(sp_max, max(vals) - min(vals))
                    same = all(len({tops[vn][i][0] for vn in VN}) == 1 for i in range(96))
                    m0 = mask0[ck]
                    if m0["max_cross_view_spread"] != sp_max or m0["same_top_all_views"] != same \
                            or m0["sane"] != (sp_max <= EPS_SCORE and same
                                              and m0["robust_top_switches"] == 0
                                              and m0["robust_margin_flips"] == 0):
                        dis.append(f"{ck} mask0 spread {sp_max} v {m0}")
    # receipt summary
    for ck in rec["summary"]:
        for arm in rec["summary"][ck]:
            for vn in VN:
                if rec["summary"][ck][arm][vn] != summary[(ck, arm, vn)]:
                    dis.append(f"receipt summary {ck} {arm} {vn}")
    if rec.get("semantic_beyond_all_surface_identifiable") is not False:
        dis.append("semantic flag not false")
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"discrepancies": dis, "verdict": "VERIFIED" if not dis else "DISCREPANCIES",
           "replay_rows": n_cmp, "replay_exact": n_ok, "replay_max_abs_lp_drift": drift,
           "rows": len(rows),
           "summary": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in summary.items()},
           "scorer_receipt_sha": hashlib.sha256(
               (D / "prband2cf_receipt.json").read_bytes()).hexdigest(),
           "cf_scores_sha": hashlib.sha256((D / "cf_scores.jsonl").read_bytes()).hexdigest()}
    (OUT / "verify_receipt.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "summary"}, indent=1))


if __name__ == "__main__":
    main()
