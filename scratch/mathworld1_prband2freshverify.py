"""Independent verifier for RENDER-ATLAS-FRESH-SEED-0 Stage A.
Reconstructs from the raw score stream alone (no scorer / aggregator
import): 16 FULL cells and 16 MASK0 cells complete, explicit pair_id
pairing, T / B / A0 / B0 / ties, RAW-R488 deltas, MAJORITY and
PAIR-MAJORITY, secondary tiers, comparator labels, priors, MASK0
sanity, old-checkpoint replay (exact) with join == adjacency, LP
re-sum on every row, checkpoint sha binding to the freeze receipt.
Writes verify_receipt.json.
"""
import hashlib
import json
from collections import defaultdict
from pathlib import Path

OUTDIR = Path("logs/mathworld1/prband2fresh_score")
FREEZE = "logs/mathworld1/prband2fresh_train/fresh_checkpoint_freeze.json"
OLD_RAW = "logs/mathworld1/prband2score/raw_scores.jsonl"
A0 = ("i_unprod", "I", 0, "term_index", 1)
B0 = ("i_unprod", "I", 0, "term_index", 3)
EPS_D, EPS_SCORE = 2e-05, 1e-05
D = []


def chk(c, m):
    if not c:
        D.append(m)


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def main():
    agg = json.loads((OUTDIR / "aggregate.json").read_text())
    rec = json.loads((OUTDIR / "prband2fresh_receipt.json").read_text())
    fz = json.loads(open(FREEZE).read())
    fresh_sha = {(c["seed"], c["representation"]): c["sha256"] for c in fz["checkpoints"]}
    chk(sha(OUTDIR / "scores.jsonl") == rec["scores_sha256"] == agg["scores_sha256"], "stream sha")
    old = {}
    for l in open(OLD_RAW):
        r = json.loads(l)
        if r["mask"] == 255:
            old[(r["seed"], r["representation"], r["state"], tuple(r["candidate"]))] = (r["lps"], r["sum"])
    cells = defaultdict(lambda: defaultdict(dict))
    gold, theta, pid = {}, {}, {}
    lp_rows = lp_ok = 0
    for l in open(OUTDIR / "scores.jsonl"):
        r = json.loads(l)
        lp_rows += 1
        lp_ok += (len(r["lps"]) == 9 and float(sum(r["lps"])) == r["sum"])
        c = tuple(r["candidate"])
        key = (r["cohort"], r["seed"], r["representation"], r["view"], r["arm"])
        chk(c not in cells[key][r["state"]], f"dup {key} {r['state']} {c}")
        cells[key][r["state"]][c] = r["sum"]
        gold[r["state"]], theta[r["state"]], pid[r["state"]] = tuple(r["gold"]), r["theta"], r["pair_id"]
        if r["cohort"] == "OLD":
            olp, osum = old[(r["seed"], r["representation"], r["state"], c)]
            chk(r["lps"] == olp and r["sum"] == osum, f"old replay {r['seed']} {r['representation']} {r['state']}")
        else:
            chk(r["ckpt_sha"] == fresh_sha[(r["seed"], r["representation"])], "fresh sha binding")
    chk(lp_rows == lp_ok, f"lp resum {lp_ok}/{lp_rows}")
    pairs = defaultdict(dict)
    for s in range(96):
        pairs[pid[s]][theta[s]] = s
    chk(len(pairs) == 48 and all(sorted(v) == ["COS_LOW", "SIN_LOW"] for v in pairs.values()), "pairs")
    fresh_full = [k for k in cells if k[0] == "FRESH" and k[4] == "FULL"]
    fresh_m0 = [k for k in cells if k[0] == "FRESH" and k[4] == "MASK0"]
    chk(len(fresh_full) == 16 and len(fresh_m0) == 16, "16+16 cells")
    chk(len([k for k in cells if k[0] == "OLD"]) == 4, "4 old replay cells")
    chk(all(len(cells[k]) == 96 and all(len(v) == 4 for v in cells[k].values()) for k in cells), "complete")

    def met(sc):
        corr, tops, ties = {}, {}, 0
        for s in range(96):
            best = max(sc[s].values())
            w = [c for c, v in sc[s].items() if v == best]
            ties += len(w) > 1
            corr[s] = len(w) == 1 and w[0] == gold[s]
            tops[s] = w[0] if len(w) == 1 else None
        B = sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]] for p in pairs.values())
        Badj = sum(corr[k] and corr[k + 1] for k in range(0, 96, 2))
        return sum(corr.values()), B, Badj, corr, tops, ties

    for k in [k for k in cells if k[0] == "OLD"]:
        T, B, Badj, *_ = met(cells[k])
        rp = agg["replay"][f"{k[1]}|{k[2]}"]
        chk(rp["exact"] == 384 and rp["pass"] and B == Badj == rp["B_pairid"] and T == rp["T"], f"replay agg {k}")
    per = agg["per_checkpoint"]
    n_pair = n_maj = n_scc = n_strong = n_imp = n_het = 0
    labels = set()
    for seed in ["21001", "22001", "23001", "24001"]:
        pm = True
        Tx, Bx = {}, {}
        for rep in ["CANONICAL", "PARAM_FIRST"]:
            ck = f"{seed}|{rep}"
            m = {vn: met(cells[("FRESH", seed, rep, vn, "FULL")]) for vn in ("RAW", "R488")}
            for vn in ("RAW", "R488"):
                T, B, Badj, corr, tops, ties = m[vn]
                chk(per[ck][vn]["T"] == T and per[ck][vn]["B"] == B and per[ck][vn]["ties"] == ties, f"metric {ck} {vn}")
                chk(per[ck][vn]["A0_correct"] == sum(corr[s] for s in range(96) if gold[s] == A0), f"A0 {ck} {vn}")
            r, x = m["RAW"], m["R488"]
            chk(per[ck]["delta"]["Delta_T"] == x[0] - r[0] and per[ck]["delta"]["Delta_B"] == x[1] - r[1], f"delta {ck}")
            chk(per[ck]["delta"]["gained"] == sum((not r[3][s]) and x[3][s] for s in range(96)), f"gained {ck}")
            chk(per[ck]["delta"]["lost"] == sum(r[3][s] and (not x[3][s]) for s in range(96)), f"lost {ck}")
            majx = x[0] >= 72 and x[1] >= 24
            chk(per[ck]["R488_MAJORITY"] == majx, f"maj {ck}")
            # mask0
            a, b = cells[("FRESH", seed, rep, "RAW", "MASK0")], cells[("FRESH", seed, rep, "R488", "MASK0")]
            spread = max(abs(a[s][c] - b[s][c]) for s in range(96) for c in a[s])
            chk(abs(spread - per[ck]["mask0"]["max_spread"]) == 0 and (spread <= EPS_SCORE) == per[ck]["mask0"]["pass"]
                or not per[ck]["mask0"]["pass"], f"mask0 {ck}")
            interp = per[ck]["interpretable"]
            n_maj += majx and interp
            pm = pm and majx and interp
            n_scc += x[0] > 48 and x[1] > 0
            n_strong += x[0] >= 84 and x[1] >= 36
            lab = "IMPROVES" if x[1] > r[1] else "TIES" if x[1] == r[1] else "WORSENS"
            chk(per[ck]["comparator_B"] == lab, f"comparator {ck}")
            labels.add(lab)
            n_imp += lab == "IMPROVES"
            Tx[rep], Bx[rep] = x[0], x[1]
        n_pair += pm
        chk(agg["seed_pair"][seed]["PAIR_MAJORITY"] == pm, f"pair maj {seed}")
        n_het += abs(Tx["CANONICAL"] - Tx["PARAM_FIRST"]) >= 8 or abs(Bx["CANONICAL"] - Bx["PARAM_FIRST"]) >= 4
    pr = agg["primary"]
    chk(pr["PAIR_MAJORITY_seeds_of_4"] == n_pair and pr["R488_MAJORITY_checkpoints_of_8"] == n_maj, "primary counts")
    p = agg["priors"]
    chk(p["1_R488_MAJORITY_ge_6_of_8"]["count"] == n_maj and p["2_PAIR_MAJORITY_ge_3_of_4"]["count"] == n_pair
        and p["3_SCC_all_8"]["count"] == n_scc and p["4_STRONG_all_8"]["count"] == n_strong
        and p["5_B_IMPROVES_ge_5_of_8"]["count"] == n_imp and p["6_arm_heterogeneity_ge_2_of_4"]["count"] == n_het
        and p["7_identical_B_label_all_8"]["outcome"] == (len(labels) == 1), "priors")
    out = {"verdict": "VERIFIED" if not D else "DISCREPANCIES", "discrepancies": D,
           "lp_rows": lp_rows, "lp_ok": lp_ok, "aggregate_sha256": sha(OUTDIR / "aggregate.json"),
           "primary": {"PAIR_MAJORITY": n_pair, "R488_MAJORITY": n_maj}}
    (OUTDIR / "verify_receipt.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
