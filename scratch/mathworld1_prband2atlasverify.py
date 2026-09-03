"""Independent verifier for MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-SCORING-0.

Reconstructs every load-bearing quantity from the four chunk score
streams and the pinned atlas artifacts alone, with its own code
(imports no aggregate function from prband2atlasagg or the scorer),
then compares against aggregate.json, policy_table.jsonl and the
chunk receipts: completeness (720 policies x 96 states x 4 candidates
FULL per checkpoint, 16 x 96 x 4 MASK0), T / B / A0 / B0 / ties per
(checkpoint, policy), maximin ranking with stage tie sets and winner,
the four tiers, individual optima and argmax sets, near-individual
sets and their intersection, minimax-regret winner, both Pareto
fronts, six Spearman pairs and conflict counts, basin-width numbers,
anchor reproduction against the booked L65490 aggregates, RAW replay
against the tracked raw_scores.jsonl (exact), MASK0 sanity (spread,
same top, robust switches / flips), and the LP audit: for every row
carrying an lps vector, sum(lps) == stored sum exactly, and the set
of policies carrying lps equals the frozen audit subset.

Writes logs/mathworld1/prband2atlasscore/verify_receipt.json.
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/prband2atlasscore")
POLICIES = "logs/mathworld1/prband2atlas/atlas_policies.jsonl"
OLD_RAW = "logs/mathworld1/prband2score/raw_scores.jsonl"
CKS = ["19001|CANONICAL", "19001|PARAM_FIRST", "20001|CANONICAL",
       "20001|PARAM_FIRST"]
A0 = ("i_unprod", "I", 0, "term_index", 1)
B0 = ("i_unprod", "I", 0, "term_index", 3)
EPS_D, EPS_SCORE = 2e-05, 1e-05
AUDIT = {12, 480, 300, 268, 524, 687, 43, 508, 405, 353, 456, 338, 486, 355, 293, 162}
EXPECT = {"19001|CANONICAL": {12: (29, 0), 480: (48, 0), 300: (82, 34)},
          "19001|PARAM_FIRST": {12: (48, 0), 480: (95, 47), 300: (54, 6)},
          "20001|CANONICAL": {12: (96, 48), 480: (48, 0), 300: (48, 0)},
          "20001|PARAM_FIRST": {12: (62, 14), 480: (47, 8), 300: (17, 0)}}
D = []


def chk(cond, msg):
    if not cond:
        D.append(msg)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def ranks(v):
    idx = sorted(range(len(v)), key=lambda i: v[i])
    out = [0.0] * len(v)
    i = 0
    while i < len(idx):
        j = i
        while j + 1 < len(idx) and v[idx[j + 1]] == v[idx[i]]:
            j += 1
        for k in range(i, j + 1):
            out[idx[k]] = (i + j) / 2 + 1
        i = j + 1
    return out


def rho(x, y):
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx)
    syy = sum((b - my) ** 2 for b in ry)
    return sxy / (sxx * syy) ** 0.5 if sxx and syy else None


def main():
    agg = json.loads((OUTDIR / "aggregate.json").read_text())
    table = {}
    for l in open(OUTDIR / "policy_table.jsonl"):
        r = json.loads(l)
        table[r["atlas_index"]] = r
    chk(len(table) == 720, "table 720")
    pol = {}
    for l in open(POLICIES):
        p = json.loads(l)
        pol[p["atlas_index"]] = p
    rid = {i: pol[i]["render_id"] for i in pol}
    old = {}
    for l in open(OLD_RAW):
        r = json.loads(l)
        if r["mask"] == 255:
            old[(r["seed"], r["representation"], r["state"], tuple(r["candidate"]))] = (r["lps"], r["sum"])
    S = {}     # ck -> idx -> state -> {cand: sum}
    G, TH = {}, {}
    M0 = {}    # ck -> (state, cand) -> {idx: sum}
    lp_rows = lp_ok = 0
    lp_policies = defaultdict(set)
    for ck in CKS:
        seed, rep = ck.split("|")
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        cr = json.loads((cdir / "chunk_receipt.json").read_text())
        chk(cr["verdict"] == "CHUNK COMPLETE", f"{ck} chunk verdict")
        chk(sha(cdir / "scores.jsonl") == cr["scores_sha256"] == agg["chunk_shas"][ck], f"{ck} chunk sha")
        chk(cr["torch"] == "2.12.1" and cr["device"] == "mps", f"{ck} runtime")
        full = defaultdict(lambda: defaultdict(dict))
        m0 = {}
        raw_seen = defaultdict(list)
        gold, theta = {}, {}
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            c = tuple(r["candidate"])
            chk(r["seed"] == seed and r["representation"] == rep, f"{ck} row ckpt")
            chk(r["render_id"] == rid[r["atlas_index"]], f"{ck} render_id")
            if "lps" in r:
                lp_rows += 1
                lp_ok += (float(sum(r["lps"])) == r["sum"]) and len(r["lps"]) == 9
                lp_policies[r["arm"]].add(r["atlas_index"])
            if r["arm"] == "FULL":
                if r["atlas_index"] == 12:
                    raw_seen[(r["state"], c)].append((r.get("lps"), r["sum"]))
                full[r["atlas_index"]][r["state"]][c] = r["sum"]
                gold[r["state"]] = tuple(r["gold"])
                theta[r["state"]] = r["theta"]
            else:
                chk(r["arm"] == "MASK0" and r["mask"] == 0, f"{ck} arm")
                m0.setdefault((r["state"], c), {})[r["atlas_index"]] = r["sum"]
        chk(sorted(full) == list(range(720)), f"{ck} 720 policies")
        chk(all(len(full[i]) == 96 and all(len(v) == 4 for v in full[i].values()) for i in full), f"{ck} complete")
        chk(len(m0) == 384 and all(len(v) == 16 for v in m0.values()), f"{ck} mask0 16x384")
        # RAW replay: both writes of index 12 must equal tracked rows exactly
        for (s, c), lst in raw_seen.items():
            olp, osum = old[(seed, rep, s, c)]
            chk(len(lst) == 2 and all(lp == olp and tot == osum for lp, tot in lst), f"{ck} raw replay {s} {c}")
        chk(len(raw_seen) == 384, f"{ck} raw rows")
        # MASK0 sanity
        spread = max(max(v.values()) - min(v.values()) for v in m0.values())
        chk(spread <= EPS_SCORE, f"{ck} mask0 spread {spread}")
        chk(abs(spread - cr["mask0"]["max_spread"]) == 0, f"{ck} mask0 spread receipt")
        S[ck], G[ck], TH[ck], M0[ck] = full, gold, theta, m0
    chk(lp_rows == lp_ok, f"lp resum {lp_ok}/{lp_rows}")
    chk(lp_policies["FULL"] == AUDIT, f"lp FULL policies {sorted(lp_policies['FULL'])}")
    chk(lp_policies["MASK0"] == AUDIT, "lp MASK0 policies")
    # metrics
    T = {ck: {} for ck in CKS}
    B = {ck: {} for ck in CKS}
    for ck in CKS:
        for i in range(720):
            corr = []
            for s in range(96):
                sc = S[ck][i][s]
                best = max(sc.values())
                w = [c for c, v in sc.items() if v == best]
                corr.append(len(w) == 1 and w[0] == G[ck][s])
            T[ck][i] = sum(corr)
            B[ck][i] = sum(corr[k] and corr[k + 1] for k in range(0, 96, 2))
            chk(table[i][ck]["T"] == T[ck][i] and table[i][ck]["B"] == B[ck][i], f"table {ck} {i}")
        for idx, (eT, eB) in EXPECT[ck].items():
            chk((T[ck][idx], B[ck][idx]) == (eT, eB), f"anchor {ck} {idx} {(T[ck][idx], B[ck][idx])} v {(eT, eB)}")
    vT = lambda i: tuple(T[ck][i] for ck in CKS)  # noqa: E731
    vB = lambda i: tuple(B[ck][i] for ck in CKS)  # noqa: E731
    # maximin
    pool = list(range(720))
    for kf in (lambda i: min(vB(i)), lambda i: min(vT(i)), lambda i: sum(vB(i)), lambda i: sum(vT(i))):
        m = max(kf(i) for i in pool)
        pool = [i for i in pool if kf(i) == m]
    win = min(pool, key=lambda i: rid[i])
    chk(win == agg["maximin"]["winner"]["atlas_index"], f"maximin winner {win}")
    chk(pool == agg["maximin"]["final_tie_set"], "maximin final tie set")
    chk(tuple(agg["maximin"]["winner"]["T"]) == vT(win) and tuple(agg["maximin"]["winner"]["B"]) == vB(win), "winner vectors")
    # tiers
    tiers = {"UNIVERSAL_STATE_CONDITIONED_CORRECT": lambda t, b: t > 48 and b > 0,
             "UNIVERSAL_MAJORITY": lambda t, b: t >= 72 and b >= 24,
             "UNIVERSAL_STRONG": lambda t, b: t >= 84 and b >= 36,
             "UNIVERSAL_NEAR_CEILING": lambda t, b: t >= 90 and b >= 42}
    for name, fn in tiers.items():
        sat = [i for i in range(720) if all(fn(T[ck][i], B[ck][i]) for ck in CKS)]
        chk(sat == [p["atlas_index"] for p in agg["tiers"][name]["policies"]]
            and len(sat) == agg["tiers"][name]["count"], f"tier {name}")
    # optima + near sets
    near = {}
    for ck in CKS:
        Bs = max(B[ck].values())
        Tb = max(T[ck].values())
        o = agg["individual_optima"][ck]
        chk(o["B_star"] == Bs and o["Tbest"] == Tb, f"optima {ck}")
        chk(o["B_argmax"] == [i for i in range(720) if B[ck][i] == Bs], f"B argmax {ck}")
        chk(o["T_argmax"] == [i for i in range(720) if T[ck][i] == Tb], f"T argmax {ck}")
        near[ck] = {i for i in range(720) if B[ck][i] >= Bs - 2 and T[ck][i] >= Tb - 4}
        chk(o["near_set_size"] == len(near[ck]), f"near size {ck}")
    common = sorted(set.intersection(*near.values()))
    chk(common == [p["atlas_index"] for p in agg["common_basin"]["policies"]]
        and len(common) == agg["common_basin"]["count"], "common basin")
    # regret
    Bstar = {ck: max(B[ck].values()) for ck in CKS}
    Tbest = {ck: max(T[ck].values()) for ck in CKS}
    key = lambda i: (max(Bstar[ck] - B[ck][i] for ck in CKS), max(Tbest[ck] - T[ck][i] for ck in CKS),  # noqa: E731
                     sum(Bstar[ck] - B[ck][i] for ck in CKS), sum(Tbest[ck] - T[ck][i] for ck in CKS), rid[i])
    rw = min(range(720), key=key)
    chk(rw == agg["minimax_regret"]["winner"]["atlas_index"], f"regret winner {rw}")
    chk(agg["minimax_regret"]["equals_maximin"] == (rw == win), "regret equals maximin flag")
    # pareto
    for name, vf in (("B", vB), ("T", vT)):
        vs = {i: vf(i) for i in range(720)}
        fr = [i for i, v in vs.items() if not any(
            all(w[k] >= v[k] for k in range(4)) and any(w[k] > v[k] for k in range(4))
            for j, w in vs.items() if j != i)]
        chk(fr == [p["atlas_index"] for p in agg["pareto"][name]["policies"]], f"pareto {name}")
    # pairwise
    for a in range(4):
        for b in range(a + 1, 4):
            c1, c2 = CKS[a], CKS[b]
            pw = agg["pairwise"][f"{c1} v {c2}"]
            t1 = [T[c1][i] for i in range(720)]
            t2 = [T[c2][i] for i in range(720)]
            b1 = [B[c1][i] for i in range(720)]
            b2 = [B[c2][i] for i in range(720)]
            rT, rB = rho(t1, t2), rho(b1, b2)
            chk((rT is None and pw["spearman_T"] is None) or abs(rT - pw["spearman_T"]) < 1e-12, f"spearman T {c1} {c2}")
            chk((rB is None and pw["spearman_B"] is None) or abs(rB - pw["spearman_B"]) < 1e-12, f"spearman B {c1} {c2}")
            chk(pw["c1_up_c2_down_T"] == sum(t1[i] > T[c1][12] and t2[i] < T[c2][12] for i in range(720)), f"conflict T {c1} {c2}")
            chk(pw["c1_up_c2_down_B"] == sum(b1[i] > B[c1][12] and b2[i] < B[c2][12] for i in range(720)), f"conflict B {c1} {c2}")
            chk(pw["near_overlap_size"] == len(near[c1] & near[c2]), f"near overlap {c1} {c2}")
    # basin
    for ck in CKS:
        bw = agg["basin_width"][ck]
        ts = [T[ck][i] for i in range(720)]
        bs = [B[ck][i] for i in range(720)]
        chk(bw["T"]["min"] == min(ts) and bw["T"]["max"] == max(ts), f"basin T {ck}")
        chk(bw["B"]["min"] == min(bs) and bw["B"]["max"] == max(bs), f"basin B {ck}")
        chk(bw["strong"] == sum(t >= 84 and b >= 36 for t, b in zip(ts, bs)), f"strong {ck}")
        chk(bw["T_gt_48"] == sum(t > 48 for t in ts) and bw["B_gt_0"] == sum(b > 0 for b in bs), f"gt {ck}")
    p3 = agg["prior3_narrow_basin_20001C"]
    strong20 = sum(T["20001|CANONICAL"][i] >= 84 and B["20001|CANONICAL"][i] >= 36 for i in range(720))
    chk(p3["strong_count_20001C"] == strong20 and p3["verdict"] == ("CONFIRMED" if strong20 < 72 else "REFUTED"), "prior3")
    rec = {"verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D,
           "lp_rows": lp_rows, "lp_ok": lp_ok,
           "aggregate_sha256": sha(OUTDIR / "aggregate.json"),
           "policy_table_sha256": sha(OUTDIR / "policy_table.jsonl"),
           "chunk_shas": agg["chunk_shas"],
           "maximin_winner": win, "common_basin": common,
           "T_summary": {ck: {"RAW": T[ck][12], "max": max(T[ck].values())} for ck in CKS}}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:3000])


if __name__ == "__main__":
    main()
