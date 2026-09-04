"""Stage-B aggregation for RENDER-ATLAS-FRESH-SEED-PREVALENCE-0
(adopt-not-fork of scratch/mathworld1_prband2atlasagg.py: same
metric, maximin, tier, optimum, near-set, regret, Pareto, Spearman,
basin-width and anatomy code paths, applied to the EIGHT fresh
checkpoints; plus the prevalence readouts, R488's position in each
fresh atlas, and the twelve-checkpoint descriptive SET intersections
computed against the discovery policy table without pooling scores).
Refuses unless all eight chunks are CHUNK COMPLETE with passing
replay / MASK0 / anchor gates.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scratch.mathworld1_prband2atlasagg import (FEATS, TIERS, dist,  # noqa: E402
                                                spearman, top1)
from scratch.mathworld1_prband2score import A0, B0, NAMES, fsha  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

OUTDIR = Path("logs/mathworld1/prband2atlasfresh")
POLICIES = "logs/mathworld1/prband2atlas/atlas_policies.jsonl"
DISC_TABLE = "logs/mathworld1/prband2atlasscore/policy_table.jsonl"
FREEZE = "logs/mathworld1/prband2fresh_train/fresh_checkpoint_freeze.json"
DISC = ["19001|CANONICAL", "19001|PARAM_FIRST", "20001|CANONICAL", "20001|PARAM_FIRST"]
EPS_D = 2e-05
SLACK_B, SLACK_T = 2, 4
RAW_IDX, R488 = 12, 488


def pareto(vecs):
    """Nondominated set over vectors of ANY common length (the
    four-checkpoint routine in prband2atlasagg hard-codes range(4);
    the Stage-B verifier caught that on the eight-vectors)."""
    front = []
    for i, v in vecs.items():
        n = len(v)
        dom = any(all(w[k] >= v[k] for k in range(n)) and any(w[k] > v[k] for k in range(n))
                  for j, w in vecs.items() if j != i)
        if not dom:
            front.append(i)
    return sorted(front)


def main():
    receipt = json.loads((OUTDIR / "prband2atlasfresh_receipt.json").read_text())
    gate(receipt["verdict"] == "ATLAS SCORED" and not receipt["smoke"], "SCORER VERDICT")
    fz = json.load(open(FREEZE))
    CKS = [f"{c['seed']}|{c['representation']}" for c in fz["checkpoints"]]
    gate(len(CKS) == 8, "8 CKS")
    pol = {p["atlas_index"]: p for p in map(json.loads, open(POLICIES))}
    rid = {i: pol[i]["render_id"] for i in pol}
    disc = {}
    for l in open(DISC_TABLE):
        r = json.loads(l)
        disc[r["atlas_index"]] = r
    gate(len(disc) == 720, "DISC TABLE")
    M = {ck: {} for ck in CKS}
    shas = {}
    for ck in CKS:
        seed, rep = ck.split("|")
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        cr = json.loads((cdir / "chunk_receipt.json").read_text())
        gate(cr["verdict"] == "CHUNK COMPLETE" and cr["stage_a_replay"]["pass"]
             and cr["mask0"]["pass"] and cr["anchor_reproduction_pass"]
             and cr["sequence_replay_pass"], f"CHUNK GATES {ck}")
        gate(fsha(str(cdir / "scores.jsonl")) == cr["scores_sha256"], f"CHUNK SHA {ck}")
        shas[ck] = cr["scores_sha256"]
        sc = defaultdict(lambda: defaultdict(dict))
        gold, theta, pid = {}, {}, {}
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            if r["arm"] != "FULL":
                continue
            sc[r["atlas_index"]][r["state"]][tuple(r["candidate"])] = r["sum"]
            gold[r["state"]], theta[r["state"]], pid[r["state"]] = tuple(r["gold"]), r["theta"], r["pair_id"]
        gate(sorted(sc) == list(range(720)) and all(len(sc[i]) == 96 for i in sc), f"COMPLETE {ck}")
        pairs = defaultdict(dict)
        for s in range(96):
            pairs[pid[s]][theta[s]] = s
        gate(len(pairs) == 48 and all(sorted(v) == ["COS_LOW", "SIN_LOW"] for v in pairs.values()), "PAIRS")
        for i in range(720):
            corr, ties, tops, mm = {}, 0, [], []
            for s in range(96):
                w, tie = top1(sc[i][s])
                corr[s] = w == gold[s]
                ties += tie
                tops.append(NAMES.get(w) if w else None)
                g = 1 if theta[s] == "SIN_LOW" else -1
                mm.append(g * (sc[i][s][A0] - sc[i][s][B0]))
            M[ck][i] = {"T": sum(corr.values()),
                        "B": sum(corr[p["SIN_LOW"]] and corr[p["COS_LOW"]] for p in pairs.values()),
                        "A0_correct": sum(corr[s] for s in range(96) if gold[s] == A0),
                        "B0_correct": sum(corr[s] for s in range(96) if gold[s] == B0),
                        "ties": ties, "top_census": dict(Counter(tops)),
                        "margin_census": {"gold_directed": sum(v >= EPS_D for v in mm),
                                          "opposite_gold": sum(v <= -EPS_D for v in mm),
                                          "subnoise": sum(abs(v) < EPS_D for v in mm)}}
    with open(OUTDIR / "policy_table.jsonl", "w") as f:
        for i in range(720):
            row = {"atlas_index": i, "render_id": rid[i], "roles": pol[i]["roles"]}
            for ck in CKS:
                row[ck] = M[ck][i]
            row["B_min"] = min(M[ck][i]["B"] for ck in CKS)
            row["T_min"] = min(M[ck][i]["T"] for ck in CKS)
            f.write(json.dumps(row) + "\n")
    vec = lambda i, key: tuple(M[ck][i][key] for ck in CKS)  # noqa: E731
    desc = lambda i: {"atlas_index": i, "render_id": rid[i], "roles": pol[i]["roles"],  # noqa: E731
                      "T": vec(i, "T"), "B": vec(i, "B")}
    # fresh-cohort maximin
    keyf = [lambda i: -min(vec(i, "B")), lambda i: -min(vec(i, "T")),
            lambda i: -sum(vec(i, "B")), lambda i: -sum(vec(i, "T"))]
    pool = list(range(720))
    stages = []
    for k, kf in enumerate(keyf):
        best = min(kf(i) for i in pool)
        pool = [i for i in pool if kf(i) == best]
        stages.append({"stage": k + 1, "value": -best, "tie_set_size": len(pool),
                       "tie_set": pool[:64]})
    winner = min(pool, key=lambda i: rid[i])
    maximin = {"stages": stages, "final_tie_set": pool, "winner": desc(winner),
               "B_min": min(vec(winner, "B")), "T_min": min(vec(winner, "T")),
               "R488_desc": desc(R488), "R488_B_min": min(vec(R488, "B")),
               "R488_T_min": min(vec(R488, "T"))}
    tiers = {name: {"count": len(sat := [i for i in range(720)
                                        if all(fn(M[ck][i]["T"], M[ck][i]["B"]) for ck in CKS)]),
                    "policies": [desc(i) for i in sat[:64]]} for name, fn in TIERS.items()}
    opt, r488pos = {}, {}
    for ck in CKS:
        Bs = max(M[ck][i]["B"] for i in range(720))
        Bset = [i for i in range(720) if M[ck][i]["B"] == Bs]
        Tb = max(M[ck][i]["T"] for i in range(720))
        Tset = [i for i in range(720) if M[ck][i]["T"] == Tb]
        near = [i for i in range(720) if M[ck][i]["B"] >= Bs - SLACK_B and M[ck][i]["T"] >= Tb - SLACK_T]
        opt[ck] = {"B_star": Bs, "Tbest": Tb, "ceiling": Bs == 48 and Tb == 96,
                   "B_argmax": Bset, "T_argmax": Tset, "near_set_size": len(near), "near_set": near,
                   "RAW_in_near": RAW_IDX in near, "R488_in_near": R488 in near,
                   "vacuous": len(near) == 0}
        order = sorted(range(720), key=lambda i: (-M[ck][i]["B"], -M[ck][i]["T"], rid[i]))
        r488pos[ck] = {"rank_1_based": order.index(R488) + 1, "T": M[ck][R488]["T"], "B": M[ck][R488]["B"],
                       "regret_B": Bs - M[ck][R488]["B"], "regret_T": Tb - M[ck][R488]["T"],
                       "in_near_set": R488 in near,
                       "RAW_rank_1_based": order.index(RAW_IDX) + 1}
    common8 = sorted(set.intersection(*(set(opt[ck]["near_set"]) for ck in CKS)))
    # regret
    reg = {i: (max(opt[ck]["B_star"] - M[ck][i]["B"] for ck in CKS),
               max(opt[ck]["Tbest"] - M[ck][i]["T"] for ck in CKS),
               sum(opt[ck]["B_star"] - M[ck][i]["B"] for ck in CKS),
               sum(opt[ck]["Tbest"] - M[ck][i]["T"] for ck in CKS), rid[i]) for i in range(720)}
    rw = min(range(720), key=lambda i: reg[i])
    fronts = {key: {"size": len(fr := pareto({i: vec(i, key) for i in range(720)})),
                    "policies": [desc(i) for i in fr[:64]], "R488_on_front": R488 in fr}
              for key in ("B", "T")}
    pairwise = {}
    for a in range(8):
        for b in range(a + 1, 8):
            c1, c2 = CKS[a], CKS[b]
            T1 = [M[c1][i]["T"] for i in range(720)]
            T2 = [M[c2][i]["T"] for i in range(720)]
            B1 = [M[c1][i]["B"] for i in range(720)]
            B2 = [M[c2][i]["B"] for i in range(720)]
            pairwise[f"{c1} v {c2}"] = {
                "spearman_T": spearman(T1, T2), "spearman_B": spearman(B1, B2),
                "c1_up_c2_down_B": sum(B1[i] > B1[RAW_IDX] and B2[i] < B2[RAW_IDX] for i in range(720)),
                "c2_up_c1_down_B": sum(B2[i] > B2[RAW_IDX] and B1[i] < B1[RAW_IDX] for i in range(720)),
                "near_overlap_size": len(set(opt[c1]["near_set"]) & set(opt[c2]["near_set"])),
                "same_seed": c1.split("|")[0] == c2.split("|")[0],
                "same_rep": c1.split("|")[1] == c2.split("|")[1]}
    basin = {}
    for ck in CKS:
        Ts = [M[ck][i]["T"] for i in range(720)]
        Bs_ = [M[ck][i]["B"] for i in range(720)]
        basin[ck] = {"T": dist(Ts), "B": dist(Bs_),
                     "T_gt_48": sum(t > 48 for t in Ts), "B_gt_0": sum(b > 0 for b in Bs_),
                     "majority": sum(t >= 72 and b >= 24 for t, b in zip(Ts, Bs_)),
                     "strong": sum(t >= 84 and b >= 36 for t, b in zip(Ts, Bs_)),
                     "near_ceiling": sum(t >= 90 and b >= 42 for t, b in zip(Ts, Bs_))}
    wfirst = [i for i in range(720) if pol[i]["roles"][0] == "W"]
    wcells = sum(M[ck][i]["T"] == 48 and M[ck][i]["B"] == 0 for ck in CKS for i in wfirst)
    anatomy = {}
    for feat in FEATS:
        by = defaultdict(list)
        for i in range(720):
            p = pol[i]
            key = p["roles"][0] if feat == "first_role" else "|".join(
                f"{th}:{sorted(p['surface_by_theta'][th][feat])}" for th in ("SIN_LOW", "COS_LOW"))
            by[key].append(i)
        anatomy[feat] = {k: {"n": len(v), **{ck: {"T_median": median([M[ck][i]["T"] for i in v]),
                                                  "T_max": max(M[ck][i]["T"] for i in v),
                                                  "B_max": max(M[ck][i]["B"] for i in v)} for ck in CKS}}
                         for k, v in sorted(by.items())}
    # twelve-checkpoint SET intersections (discovery table + fresh), no pooled objective
    ALL = CKS + DISC
    def cell(ck, i, key):
        return M[ck][i][key] if ck in M else disc[i][ck][key]
    tiers12 = {name: {"count": len(sat := [i for i in range(720)
                                          if all(fn(cell(ck, i, "T"), cell(ck, i, "B")) for ck in ALL)]),
                      "policies": [i for i in sat[:64]]} for name, fn in TIERS.items()}
    disc_near = {}
    for ck in DISC:
        Bs = max(disc[i][ck]["B"] for i in range(720))
        Tb = max(disc[i][ck]["T"] for i in range(720))
        disc_near[ck] = {i for i in range(720) if disc[i][ck]["B"] >= Bs - SLACK_B and disc[i][ck]["T"] >= Tb - SLACK_T}
    common12 = sorted(set(common8) & set.intersection(*disc_near.values())) if common8 else []
    r488_tiers12 = {name: all(fn(cell(ck, R488, "T"), cell(ck, R488, "B")) for ck in ALL) for name, fn in TIERS.items()}
    win_tiers12 = {name: all(fn(cell(ck, winner, "T"), cell(ck, winner, "B")) for ck in ALL) for name, fn in TIERS.items()}
    prevalence = {"fresh_ceiling_count": sum(opt[ck]["ceiling"] for ck in CKS),
                  "fresh_near_sizes": {ck: opt[ck]["near_set_size"] for ck in CKS},
                  "fresh_RAW_in_near": sum(opt[ck]["RAW_in_near"] for ck in CKS),
                  "fresh_R488_in_near": sum(opt[ck]["R488_in_near"] for ck in CKS),
                  "common_near_8": common8, "incompatibility_within_fresh": len(common8) == 0,
                  "common_near_12": common12, "incompatibility_within_twelve": len(common12) == 0,
                  "tiers_all_8": {k: v["count"] for k, v in tiers.items()},
                  "tiers_all_12": {k: v["count"] for k, v in tiers12.items()},
                  "R488_tiers_all_12": r488_tiers12, "fresh_maximin_tiers_all_12": win_tiers12,
                  "W_first_48_0_cells": wcells, "W_first_cells_total": len(wfirst) * 8}
    priors = {"1_all_8_reach_ceiling": {"prior": "MODERATE-HIGH", "count": prevalence["fresh_ceiling_count"],
                                         "outcome": prevalence["fresh_ceiling_count"] == 8},
              "2_common_near_8_empty": {"prior": "HIGH", "outcome": len(common8) == 0},
              "3_common_near_12_empty": {"prior": "HIGH", "outcome": len(common12) == 0},
              "4a_no_STRONG_all_8": {"prior": "MODERATE-HIGH", "outcome": tiers["UNIVERSAL_STRONG"]["count"] == 0},
              "4b_no_STRONG_all_12": {"prior": "HIGH", "outcome": tiers12["UNIVERSAL_STRONG"]["count"] == 0},
              "5_fresh_maximin_Bmin_le_30": {"prior": "MODERATE", "value": maximin["B_min"], "outcome": maximin["B_min"] <= 30},
              "6_W_first_all_48_0": {"prior": "MODERATE-HIGH", "count": wcells, "outcome": wcells == len(wfirst) * 8},
              "7_R488_in_near_le_1": {"prior": "MODERATE-HIGH", "count": prevalence["fresh_R488_in_near"],
                                      "outcome": prevalence["fresh_R488_in_near"] <= 1},
              "8_strong_above_115": {"prior": "NONE", "max_strong": max(basin[ck]["strong"] for ck in CKS)}}
    agg = {"prereg": receipt["prereg"], "chunk_shas": shas, "checkpoints": CKS,
           "maximin_fresh": maximin, "tiers_fresh": tiers, "individual_optima": {
               ck: {k: v for k, v in o.items() if k != "near_set"} for ck, o in opt.items()},
           "R488_position": r488pos, "minimax_regret_fresh": {"winner": desc(rw), "regret": reg[rw][:4],
                                                              "equals_maximin": rw == winner},
           "pareto_fresh": fronts, "pairwise": pairwise, "basin_width": basin,
           "surface_anatomy": anatomy, "prevalence": prevalence, "priors": priors,
           "semantic_beyond_all_surface_identifiable": False,
           "policy_table_sha256": fsha(str(OUTDIR / "policy_table.jsonl"))}
    (OUTDIR / "aggregate.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps(prevalence, indent=1))
    print(json.dumps({"maximin": maximin["winner"], "B_min": maximin["B_min"], "T_min": maximin["T_min"],
                      "stages": [(s["stage"], s["value"], s["tie_set_size"]) for s in stages]}, indent=0))
    print({ck: (o["B_star"], o["Tbest"], o["near_set_size"]) for ck, o in opt.items()})
    print({ck: (b["strong"], b["majority"], b["T_gt_48"]) for ck, b in basin.items()})
    print(json.dumps(r488pos, indent=0)[:1200])
    print(json.dumps(priors, indent=1))


if __name__ == "__main__":
    main()
