"""Independent verifier for RENDER-ATLAS-FRESH-SEED-PREVALENCE-0
(adapted from scratch/mathworld1_prband2atlasverify.py; imports no
aggregate function). From the eight chunk streams, the discovery
policy table, the freeze receipt and the Stage-A stream alone:
completeness (720 x 96 x 4 FULL + 16 x 96 x 4 MASK0 per chunk),
every T / B in the fresh policy table, Stage-A replay of index 12
(both writes) exact, MASK0 spread, LP re-sum on every lps-bearing
row and the lps policy set == frozen 16, anchor reproduction of 12
and 488 against the Stage-A aggregate, fresh maximin winner and tie
set, tiers over 8, optima / near sets / 8-way intersection, R488
ranks, regret winner, Pareto fronts, all 28 Spearman pairs, basin
counts, twelve-way tier counts and near intersection, priors.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/prband2atlasfresh")
POLICIES = "logs/mathworld1/prband2atlas/atlas_policies.jsonl"
DISC_TABLE = "logs/mathworld1/prband2atlasscore/policy_table.jsonl"
FREEZE = "logs/mathworld1/prband2fresh_train/fresh_checkpoint_freeze.json"
STAGEA = "logs/mathworld1/prband2fresh_score/scores.jsonl"
STAGEA_AGG = "logs/mathworld1/prband2fresh_score/aggregate.json"
DISC = ["19001|CANONICAL", "19001|PARAM_FIRST", "20001|CANONICAL", "20001|PARAM_FIRST"]
A0 = ("i_unprod", "I", 0, "term_index", 1)
B0 = ("i_unprod", "I", 0, "term_index", 3)
EPS_SCORE = 1e-05
AUDIT = {12, 480, 300, 268, 524, 687, 43, 508, 405, 353, 456, 338, 486, 355, 293, 162}
TIERS = {"UNIVERSAL_STATE_CONDITIONED_CORRECT": lambda t, b: t > 48 and b > 0,
         "UNIVERSAL_MAJORITY": lambda t, b: t >= 72 and b >= 24,
         "UNIVERSAL_STRONG": lambda t, b: t >= 84 and b >= 36,
         "UNIVERSAL_NEAR_CEILING": lambda t, b: t >= 90 and b >= 42}
D = []


def chk(c, m):
    if not c:
        D.append(m)


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
    fz = json.load(open(FREEZE))
    CKS = [f"{c['seed']}|{c['representation']}" for c in fz["checkpoints"]]
    fsha_ = {f"{c['seed']}|{c['representation']}": c["sha256"] for c in fz["checkpoints"]}
    chk(agg["checkpoints"] == CKS, "ck order")
    table = {}
    for l in open(OUTDIR / "policy_table.jsonl"):
        r = json.loads(l)
        table[r["atlas_index"]] = r
    pol = {p["atlas_index"]: p for p in map(json.loads, open(POLICIES))}
    rid = {i: pol[i]["render_id"] for i in pol}
    disc = {}
    for l in open(DISC_TABLE):
        r = json.loads(l)
        disc[r["atlas_index"]] = r
    sa = json.load(open(STAGEA_AGG))
    old = {}
    for l in open(STAGEA):
        r = json.loads(l)
        if r["cohort"] == "FRESH" and r["arm"] == "FULL":
            old[(r["seed"], r["representation"], r["view"], r["state"], tuple(r["candidate"]))] = (r["lps"], r["sum"])
    T = {ck: {} for ck in CKS}
    B = {ck: {} for ck in CKS}
    lp_rows = lp_ok = 0
    for ck in CKS:
        seed, rep = ck.split("|")
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        cr = json.loads((cdir / "chunk_receipt.json").read_text())
        chk(cr["verdict"] == "CHUNK COMPLETE" and cr["sha256"] == fsha_[ck], f"{ck} chunk")
        chk(sha(cdir / "scores.jsonl") == cr["scores_sha256"] == agg["chunk_shas"][ck], f"{ck} sha")
        full = defaultdict(lambda: defaultdict(dict))
        m0, raw_seen, gold, theta, pid = {}, defaultdict(list), {}, {}, {}
        r488_seen = {}
        lp_pol = defaultdict(set)
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            c = tuple(r["candidate"])
            chk(r["ckpt_sha"] == fsha_[ck], f"{ck} row sha")
            if "lps" in r:
                lp_rows += 1
                lp_ok += len(r["lps"]) == 9 and float(sum(r["lps"])) == r["sum"]
                lp_pol[r["arm"]].add(r["atlas_index"])
            if r["arm"] == "FULL":
                if r["atlas_index"] == 12:
                    raw_seen[(r["state"], c)].append((r.get("lps"), r["sum"]))
                if r["atlas_index"] == 488:
                    r488_seen[(r["state"], c)] = (r.get("lps"), r["sum"])
                full[r["atlas_index"]][r["state"]][c] = r["sum"]
                gold[r["state"]], theta[r["state"]], pid[r["state"]] = tuple(r["gold"]), r["theta"], r["pair_id"]
            else:
                m0.setdefault((r["state"], c), {})[r["atlas_index"]] = r["sum"]
        chk(sorted(full) == list(range(720)) and all(len(full[i]) == 96 and all(len(v) == 4 for v in full[i].values()) for i in full), f"{ck} complete")
        chk(len(m0) == 384 and all(len(v) == 16 for v in m0.values()), f"{ck} mask0 cells")
        chk(lp_pol["FULL"] == AUDIT | {488} and lp_pol["MASK0"] == AUDIT, f"{ck} lp policies")
        for (s, c), lst in raw_seen.items():
            olp, osum = old[(seed, rep, "RAW", s, c)]
            chk(len(lst) == 2 and all(lp == olp and tot == osum for lp, tot in lst), f"{ck} stage-a replay {s}")
        for (s, c), (lp, tot) in r488_seen.items():
            olp, osum = old[(seed, rep, "R488", s, c)]
            chk(lp == olp and tot == osum, f"{ck} stage-a R488 replay {s}")
        chk(len(r488_seen) == 384, f"{ck} r488 rows")
        chk(len(raw_seen) == 384, f"{ck} raw rows")
        spread = max(max(v.values()) - min(v.values()) for v in m0.values())
        chk(spread <= EPS_SCORE and spread == cr["mask0"]["max_spread"], f"{ck} mask0 spread")
        pairs = defaultdict(dict)
        for s in range(96):
            pairs[pid[s]][theta[s]] = s
        chk(len(pairs) == 48, f"{ck} pairs")
        for i in range(720):
            corr = {}
            for s in range(96):
                sc = full[i][s]
                best = max(sc.values())
                w = [c for c, v in sc.items() if v == best]
                corr[s] = len(w) == 1 and w[0] == gold[s]
            T[ck][i] = sum(corr.values())
            B[ck][i] = sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]] for p in pairs.values())
            chk(table[i][ck]["T"] == T[ck][i] and table[i][ck]["B"] == B[ck][i], f"table {ck} {i}")
        for idx, key in ((12, "RAW"), (488, "R488")):
            chk((T[ck][idx], B[ck][idx]) == (sa["per_checkpoint"][ck][key]["T"], sa["per_checkpoint"][ck][key]["B"]), f"anchor {ck} {idx}")
    chk(lp_rows == lp_ok, f"lp resum {lp_ok}/{lp_rows}")
    vT = lambda i: tuple(T[ck][i] for ck in CKS)  # noqa: E731
    vB = lambda i: tuple(B[ck][i] for ck in CKS)  # noqa: E731
    pool = list(range(720))
    for kf in (lambda i: min(vB(i)), lambda i: min(vT(i)), lambda i: sum(vB(i)), lambda i: sum(vT(i))):
        m = max(kf(i) for i in pool)
        pool = [i for i in pool if kf(i) == m]
    win = min(pool, key=lambda i: rid[i])
    mx = agg["maximin_fresh"]
    chk(win == mx["winner"]["atlas_index"] and pool == mx["final_tie_set"], "maximin")
    chk(mx["B_min"] == min(vB(win)) and mx["T_min"] == min(vT(win)), "maximin values")
    chk(mx["R488_B_min"] == min(vB(488)) and mx["R488_T_min"] == min(vT(488)), "R488 minima")
    for name, fn in TIERS.items():
        sat = [i for i in range(720) if all(fn(T[ck][i], B[ck][i]) for ck in CKS)]
        chk(len(sat) == agg["tiers_fresh"][name]["count"], f"tier8 {name}")
    near = {}
    for ck in CKS:
        Bs, Tb = max(B[ck].values()), max(T[ck].values())
        o = agg["individual_optima"][ck]
        chk(o["B_star"] == Bs and o["Tbest"] == Tb and o["ceiling"] == (Bs == 48 and Tb == 96), f"opt {ck}")
        near[ck] = {i for i in range(720) if B[ck][i] >= Bs - 2 and T[ck][i] >= Tb - 4}
        chk(o["near_set_size"] == len(near[ck]) and o["R488_in_near"] == (488 in near[ck]) and o["RAW_in_near"] == (12 in near[ck]), f"near {ck}")
        order = sorted(range(720), key=lambda i: (-B[ck][i], -T[ck][i], rid[i]))
        rp = agg["R488_position"][ck]
        chk(rp["rank_1_based"] == order.index(488) + 1 and rp["RAW_rank_1_based"] == order.index(12) + 1, f"rank {ck}")
        chk(rp["regret_B"] == Bs - B[ck][488] and rp["regret_T"] == Tb - T[ck][488], f"regret488 {ck}")
    common8 = sorted(set.intersection(*near.values()))
    pv = agg["prevalence"]
    chk(pv["common_near_8"] == common8 and pv["incompatibility_within_fresh"] == (len(common8) == 0), "common8")
    chk(pv["fresh_ceiling_count"] == sum(max(B[ck].values()) == 48 and max(T[ck].values()) == 96 for ck in CKS), "ceiling count")
    chk(pv["fresh_R488_in_near"] == sum(488 in near[ck] for ck in CKS), "R488 in near count")
    # twelve-way
    ALL = CKS + DISC
    cell = lambda ck, i, k: (T if k == "T" else B)[ck][i] if ck in T else disc[i][ck][k]  # noqa: E731
    for name, fn in TIERS.items():
        chk(pv["tiers_all_12"][name] == sum(all(fn(cell(ck, i, "T"), cell(ck, i, "B")) for ck in ALL) for i in range(720)), f"tier12 {name}")
    dnear = {}
    for ck in DISC:
        Bs = max(disc[i][ck]["B"] for i in range(720))
        Tb = max(disc[i][ck]["T"] for i in range(720))
        dnear[ck] = {i for i in range(720) if disc[i][ck]["B"] >= Bs - 2 and disc[i][ck]["T"] >= Tb - 4}
    common12 = sorted(set(common8) & set.intersection(*dnear.values())) if common8 else []
    chk(pv["common_near_12"] == common12, "common12")
    wfirst = [i for i in range(720) if pol[i]["roles"][0] == "W"]
    chk(pv["W_first_48_0_cells"] == sum(T[ck][i] == 48 and B[ck][i] == 0 for ck in CKS for i in wfirst), "W-first cells")
    # regret, pareto, spearman, basin
    Bst = {ck: max(B[ck].values()) for ck in CKS}
    Tbs = {ck: max(T[ck].values()) for ck in CKS}
    key = lambda i: (max(Bst[ck] - B[ck][i] for ck in CKS), max(Tbs[ck] - T[ck][i] for ck in CKS),  # noqa: E731
                     sum(Bst[ck] - B[ck][i] for ck in CKS), sum(Tbs[ck] - T[ck][i] for ck in CKS), rid[i])
    chk(min(range(720), key=key) == agg["minimax_regret_fresh"]["winner"]["atlas_index"], "regret winner")
    for name, vf in (("B", vB), ("T", vT)):
        vs = {i: vf(i) for i in range(720)}
        fr = [i for i, v in vs.items() if not any(all(w[k] >= v[k] for k in range(8)) and any(w[k] > v[k] for k in range(8)) for j, w in vs.items() if j != i)]
        chk(len(fr) == agg["pareto_fresh"][name]["size"] and (488 in fr) == agg["pareto_fresh"][name]["R488_on_front"], f"pareto {name}")
    for a in range(8):
        for b in range(a + 1, 8):
            c1, c2 = CKS[a], CKS[b]
            pw = agg["pairwise"][f"{c1} v {c2}"]
            r1 = rho([T[c1][i] for i in range(720)], [T[c2][i] for i in range(720)])
            r2 = rho([B[c1][i] for i in range(720)], [B[c2][i] for i in range(720)])
            chk((r1 is None and pw["spearman_T"] is None) or abs(r1 - pw["spearman_T"]) < 1e-12, f"rhoT {c1} {c2}")
            chk((r2 is None and pw["spearman_B"] is None) or abs(r2 - pw["spearman_B"]) < 1e-12, f"rhoB {c1} {c2}")
            chk(pw["near_overlap_size"] == len(near[c1] & near[c2]), f"overlap {c1} {c2}")
    for ck in CKS:
        bw = agg["basin_width"][ck]
        chk(bw["strong"] == sum(T[ck][i] >= 84 and B[ck][i] >= 36 for i in range(720)) and bw["majority"] == sum(T[ck][i] >= 72 and B[ck][i] >= 24 for i in range(720)), f"basin {ck}")
    p = agg["priors"]
    chk(p["1_all_8_reach_ceiling"]["outcome"] == (pv["fresh_ceiling_count"] == 8) and p["2_common_near_8_empty"]["outcome"] == (len(common8) == 0)
        and p["3_common_near_12_empty"]["outcome"] == (len(common12) == 0) and p["5_fresh_maximin_Bmin_le_30"]["outcome"] == (min(vB(win)) <= 30)
        and p["7_R488_in_near_le_1"]["outcome"] == (sum(488 in near[ck] for ck in CKS) <= 1), "priors")
    rec = {"verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D[:50], "n_discrepancies": len(D),
           "lp_rows": lp_rows, "lp_ok": lp_ok, "aggregate_sha256": sha(OUTDIR / "aggregate.json"),
           "policy_table_sha256": sha(OUTDIR / "policy_table.jsonl"), "chunk_shas": agg["chunk_shas"],
           "verifier_sha256": sha(__file__), "maximin_winner": win, "common_near_8": common8, "common_near_12": common12}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1)[:2500])


if __name__ == "__main__":
    main()
