"""Frozen Stage-A analysis for RENDER-ATLAS-FRESH-SEED-0 (RESULTS
L65695). Reads the scorer receipt and the raw score stream (FULL
rows), recomputes every metric from the sums with an explicit
pair_id join, and writes aggregate.json: per checkpoint RAW / R488
T, B, A0, B0, ties, censuses, deltas; seed-pair minima and deltas;
MAJORITY floor labels, PAIR-MAJORITY count / 4 and the frozen
transfer label; secondary tiers; sign-only RAW comparator; the
seven registered priors adjudicated; MASK0 sanity carried from the
receipt. Never pools old with fresh; never pools to N = 768.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.mathworld1_prband2score import A0, B0, fsha  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

OUTDIR = Path("logs/mathworld1/prband2fresh_score")
SEEDS = ["21001", "22001", "23001", "24001"]
REPS = ["CANONICAL", "PARAM_FIRST"]
VIEWS = ["RAW", "R488"]
LABELS = {4: "ALL-SEED-PAIR MAJORITY TRANSFER", 3: "BROAD PAIR-MAJORITY TRANSFER",
          2: "MIXED PAIR-MAJORITY TRANSFER", 1: "MIXED PAIR-MAJORITY TRANSFER",
          0: "NO PAIR-MAJORITY TRANSFER"}
maj = lambda T, B: T >= 72 and B >= 24  # noqa: E731
scc = lambda T, B: T > 48 and B > 0  # noqa: E731
strong = lambda T, B: T >= 84 and B >= 36  # noqa: E731
nearc = lambda T, B: T >= 90 and B >= 42  # noqa: E731


def main():
    rec = json.loads((OUTDIR / "prband2fresh_receipt.json").read_text())
    gate(rec["verdict"] == "STAGE A SCORED" and not rec["smoke"], "SCORER VERDICT")
    gate(fsha(str(OUTDIR / "scores.jsonl")) == rec["scores_sha256"], "STREAM SHA")
    S = defaultdict(lambda: defaultdict(dict))  # (seed,rep,view) -> state -> sem -> sum
    gold, theta, pid = {}, {}, {}
    for l in open(OUTDIR / "scores.jsonl"):
        r = json.loads(l)
        if r["cohort"] != "FRESH" or r["arm"] != "FULL":
            continue
        S[(r["seed"], r["representation"], r["view"])][r["state"]][tuple(r["candidate"])] = r["sum"]
        gold[r["state"]] = tuple(r["gold"])
        theta[r["state"]] = r["theta"]
        pid[r["state"]] = r["pair_id"]
    gate(len(S) == 16 and all(len(v) == 96 and all(len(c) == 4 for c in v.values())
                             for v in S.values()), "16 FULL CELLS x 96 x 4")
    pairs = defaultdict(dict)
    for s in range(96):
        pairs[pid[s]][theta[s]] = s
    gate(len(pairs) == 48 and all(sorted(v) == ["COS_LOW", "SIN_LOW"] for v in pairs.values()), "PAIRS")

    def met(sc):
        corr, ties, tops = {}, 0, {}
        for s in range(96):
            best = max(sc[s].values())
            w = [c for c, v in sc[s].items() if v == best]
            ties += len(w) > 1
            corr[s] = len(w) == 1 and w[0] == gold[s]
            tops[s] = w[0] if len(w) == 1 else None
        return {"T": sum(corr.values()),
                "B": sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]] for p in pairs.values()),
                "A0_correct": sum(corr[s] for s in range(96) if gold[s] == A0),
                "B0_correct": sum(corr[s] for s in range(96) if gold[s] == B0),
                "ties": ties, "_corr": corr, "_tops": tops}

    per = {}
    for seed in SEEDS:
        for rep in REPS:
            ck = f"{seed}|{rep}"
            m = {vn: met(S[(seed, rep, vn)]) for vn in VIEWS}
            r, x = m["RAW"], m["R488"]
            fr = rec["fresh"][ck]
            for vn in VIEWS:
                gate(fr["FULL"][vn]["T"] == m[vn]["T"] and fr["FULL"][vn]["B"] == m[vn]["B"],
                     f"RECEIPT/RECOUNT {ck} {vn}")
            per[ck] = {
                "seed": seed, "representation": rep, "sha256": fr["sha256"],
                **{vn: {k: v for k, v in m[vn].items() if not k.startswith("_")} for vn in VIEWS},
                "receipt_full": fr["FULL"], "delta": fr["delta"], "mask0": fr["mask0"],
                "R488_MAJORITY": maj(x["T"], x["B"]), "RAW_MAJORITY": maj(r["T"], r["B"]),
                "tiers_R488": {"SCC": scc(x["T"], x["B"]), "STRONG": strong(x["T"], x["B"]),
                               "NEAR_CEILING": nearc(x["T"], x["B"])},
                "tiers_RAW": {"SCC": scc(r["T"], r["B"]), "STRONG": strong(r["T"], r["B"]),
                              "NEAR_CEILING": nearc(r["T"], r["B"])},
                "comparator_B": "IMPROVES" if x["B"] > r["B"] else "TIES" if x["B"] == r["B"] else "WORSENS",
                "comparator_T": "IMPROVES" if x["T"] > r["T"] else "TIES" if x["T"] == r["T"] else "WORSENS",
                "interpretable": fr["mask0"]["pass"]}
    seedpair = {}
    for seed in SEEDS:
        c, p = per[f"{seed}|CANONICAL"], per[f"{seed}|PARAM_FIRST"]
        sp = {}
        for vn in VIEWS:
            sp[f"Tmin_{vn}"] = min(c[vn]["T"], p[vn]["T"])
            sp[f"Bmin_{vn}"] = min(c[vn]["B"], p[vn]["B"])
        sp["Delta_Bmin"] = sp["Bmin_R488"] - sp["Bmin_RAW"]
        sp["Delta_Tmin"] = sp["Tmin_R488"] - sp["Tmin_RAW"]
        sp["PAIR_MAJORITY"] = c["R488_MAJORITY"] and p["R488_MAJORITY"] and c["interpretable"] and p["interpretable"]
        sp["arm_heterogeneity_R488"] = {"abs_dT": abs(c["R488"]["T"] - p["R488"]["T"]),
                                        "abs_dB": abs(c["R488"]["B"] - p["R488"]["B"]),
                                        "material": abs(c["R488"]["T"] - p["R488"]["T"]) >= 8
                                        or abs(c["R488"]["B"] - p["R488"]["B"]) >= 4}
        seedpair[seed] = sp
    n_pair = sum(sp["PAIR_MAJORITY"] for sp in seedpair.values())
    n_maj = sum(per[ck]["R488_MAJORITY"] and per[ck]["interpretable"] for ck in per)
    n_scc = sum(per[ck]["tiers_R488"]["SCC"] for ck in per)
    n_strong = sum(per[ck]["tiers_R488"]["STRONG"] for ck in per)
    n_nc = sum(per[ck]["tiers_R488"]["NEAR_CEILING"] for ck in per)
    n_impB = sum(per[ck]["comparator_B"] == "IMPROVES" for ck in per)
    n_hetero = sum(sp["arm_heterogeneity_R488"]["material"] for sp in seedpair.values())
    labels_B = {per[ck]["comparator_B"] for ck in per}
    priors = {
        "1_R488_MAJORITY_ge_6_of_8": {"prior": "MODERATE", "count": n_maj, "outcome": n_maj >= 6},
        "2_PAIR_MAJORITY_ge_3_of_4": {"prior": "LOW-MODERATE", "count": n_pair, "outcome": n_pair >= 3},
        "3_SCC_all_8": {"prior": "MODERATE-HIGH", "count": n_scc, "outcome": n_scc == 8},
        "4_STRONG_all_8": {"prior": "LOW", "count": n_strong, "outcome": n_strong == 8},
        "5_B_IMPROVES_ge_5_of_8": {"prior": "MODERATE", "count": n_impB, "outcome": n_impB >= 5},
        "6_arm_heterogeneity_ge_2_of_4": {"prior": "MODERATE-HIGH", "count": n_hetero, "outcome": n_hetero >= 2},
        "7_identical_B_label_all_8": {"prior": "LOW", "labels": sorted(labels_B), "outcome": len(labels_B) == 1}}
    agg = {"prereg": rec["prereg"], "prereg_commit": rec["prereg_commit"],
           "scores_sha256": rec["scores_sha256"], "replay": rec["replay"],
           "per_checkpoint": per, "seed_pair": seedpair,
           "primary": {"PAIR_MAJORITY_seeds_of_4": n_pair, "label": LABELS[n_pair],
                       "R488_MAJORITY_checkpoints_of_8": n_maj,
                       "uninterpretable_cells": [ck for ck in per if not per[ck]["interpretable"]]},
           "secondary": {"SCC_of_8": n_scc, "STRONG_of_8": n_strong, "NEAR_CEILING_of_8": n_nc,
                         "RAW_MAJORITY_of_8": sum(per[ck]["RAW_MAJORITY"] for ck in per)},
           "priors": priors, "semantic_beyond_all_surface_identifiable": False}
    (OUTDIR / "aggregate.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(agg["primary"], indent=1))
    for ck in per:
        print(ck, "RAW", per[ck]["RAW"]["T"], per[ck]["RAW"]["B"], "| R488", per[ck]["R488"]["T"], per[ck]["R488"]["B"],
              per[ck]["comparator_B"], "maj", per[ck]["R488_MAJORITY"], "mask0", per[ck]["mask0"]["pass"])
    print(json.dumps(seedpair, indent=0)[:1500])
    print(json.dumps(priors, indent=1))


if __name__ == "__main__":
    main()
