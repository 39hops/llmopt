"""MATH-CYBER-1-PRIOR-RESISTANT-EVAL-V2-RENDER-ATLAS-SCORING-0
aggregation over the four complete checkpoint chunks (RESULTS
L65585). Runs ONLY after scratch/mathworld1_prband2atlasscore.py
has closed all four chunks with verdict CHUNK COMPLETE and passing
RAW replay / MASK0 / anchor gates; refuses otherwise.

Reads the four chunk streams (FULL rows: exact SUM per candidate)
and recomputes from the raw sums, never from the chunk receipts'
own T/B tables: per (checkpoint, policy) T = strict top-1 / 96,
B = both-correct matched pairs / 48, A0/B0 correct, SCORE-TIE
count, top-action census, target-margin sign census, pair switches;
then the frozen analysis: lexicographic maximin (B_min, T_min,
sum B, sum T, render_id) with tie sets at every stage; the four
universal tiers; individual optima B*_c / Tbest_c with argmax sets;
the simultaneous near-individual-optimum test (slack 2 pairs / 4
states) with the per-checkpoint vacuity readout; minimax regret;
Pareto fronts on B and T; six pairwise Spearman correlations
(average ranks) and RAW-relative conflict counts; basin-width
distributions and tier counts; the 20001 CANONICAL narrow-basin
prior (fewer than 72/720 STRONG); and the surface-anatomy join
against the pre-existing model-blind census in atlas_policies.jsonl.

Writes small receipts under logs/mathworld1/prband2atlasscore/:
policy_table.jsonl (720 rows), aggregate.json.
"""
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scratch.mathworld1_prband2score import A0, B0, NAMES, SEM, fsha  # noqa: E402
from scratch.mathworld1_svpbirth import gate  # noqa: E402

OUTDIR = Path("logs/mathworld1/prband2atlasscore")
POLICIES = "logs/mathworld1/prband2atlas/atlas_policies.jsonl"
CKS = ["19001|CANONICAL", "19001|PARAM_FIRST", "20001|CANONICAL",
       "20001|PARAM_FIRST"]
EPS_D = 2e-05
SLACK_B, SLACK_T = 2, 4
TIERS = {"UNIVERSAL_STATE_CONDITIONED_CORRECT": lambda T, B: T > 48 and B > 0,
         "UNIVERSAL_MAJORITY": lambda T, B: T >= 72 and B >= 24,
         "UNIVERSAL_STRONG": lambda T, B: T >= 84 and B >= 36,
         "UNIVERSAL_NEAR_CEILING": lambda T, B: T >= 90 and B >= 42}
RAW_IDX = 12
FEATS = ("first_role", "lead_sign", "prompt_tokens", "minus_pos", "first_trig",
         "first_trig_degree", "first_poly_degree", "neg_role_pos")


def sgn(x):
    return 1 if x > 0 else -1 if x < 0 else 0


def top1(scores):
    ranked = sorted(scores.items(), key=lambda kv: -kv[1])
    best = ranked[0][1]
    ties = [k for k, v in scores.items() if v == best]
    return (ranked[0][0] if len(ties) == 1 else None), len(ties) > 1


def avg_rank(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    ranks = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(xs, ys):
    rx, ry = avg_rank(xs), avg_rank(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx > 0 and dy > 0 else None


def iqr(v):
    s = sorted(v)
    q = lambda p: s[min(len(s) - 1, int(p * len(s)))]  # noqa: E731
    return q(0.75) - q(0.25)


def dist(v):
    return {"min": min(v), "max": max(v), "median": median(v), "IQR": iqr(v),
            "range": max(v) - min(v)}


def pareto(vecs):
    front = []
    for i, v in vecs.items():
        dom = any(all(w[k] >= v[k] for k in range(4)) and any(w[k] > v[k] for k in range(4))
                  for j, w in vecs.items() if j != i)
        if not dom:
            front.append(i)
    return sorted(front)


def main():
    receipt = json.loads((OUTDIR / "prband2atlasscore_receipt.json").read_text())
    gate(receipt["verdict"] == "ATLAS SCORED" and not receipt["smoke"], "SCORER VERDICT")
    pol = {}
    for l in open(POLICIES):
        p = json.loads(l)
        pol[p["atlas_index"]] = p
    gate(len(pol) == 720, "POLICIES")
    rid = {i: pol[i]["render_id"] for i in pol}
    chunks = {}
    for ck in CKS:
        seed, rep = ck.split("|")
        cdir = OUTDIR / f"chunk_{seed}_{rep}"
        cr = json.loads((cdir / "chunk_receipt.json").read_text())
        gate(cr["verdict"] == "CHUNK COMPLETE" and cr["raw_replay"]["pass"]
             and cr["mask0"]["pass"] and cr["anchor_reproduction_pass"]
             and cr["sequence_replay_pass"], f"CHUNK GATES {ck}")
        gate(fsha(str(cdir / "scores.jsonl")) == cr["scores_sha256"], f"CHUNK SHA {ck}")
        sc = defaultdict(lambda: defaultdict(dict))  # idx -> state -> sem -> sum
        gold, theta = {}, {}
        for l in open(cdir / "scores.jsonl"):
            r = json.loads(l)
            if r["arm"] != "FULL":
                continue
            sc[r["atlas_index"]][r["state"]][tuple(r["candidate"])] = r["sum"]
            gold[r["state"]] = tuple(r["gold"])
            theta[r["state"]] = r["theta"]
        gate(sorted(sc) == list(range(720)), f"720 POLICIES {ck}")
        gate(all(len(sc[i]) == 96 and all(len(v) == 4 for v in sc[i].values())
                 for i in sc), f"COMPLETE {ck}")
        chunks[ck] = (sc, gold, theta, cr["scores_sha256"])
    # per (ck, policy) metrics
    M = {ck: {} for ck in CKS}
    for ck in CKS:
        sc, gold, theta, _ = chunks[ck]
        for i in range(720):
            corr, ties, tops, mm = [], 0, [], []
            for s in range(96):
                w, tie = top1(sc[i][s])
                corr.append(w == gold[s])
                ties += tie
                tops.append(NAMES.get(w) if w else None)
                g = 1 if theta[s] == "SIN_LOW" else -1
                mm.append(g * (sc[i][s][A0] - sc[i][s][B0]))
            M[ck][i] = {
                "T": sum(corr), "B": sum(corr[k] and corr[k + 1] for k in range(0, 96, 2)),
                "A0_correct": sum(c for c, s in zip(corr, range(96)) if gold[s] == A0),
                "B0_correct": sum(c for c, s in zip(corr, range(96)) if gold[s] == B0),
                "ties": ties, "top_census": dict(Counter(tops)),
                "margin_census": {"gold_directed": sum(v >= EPS_D for v in mm),
                                  "opposite_gold": sum(v <= -EPS_D for v in mm),
                                  "subnoise": sum(abs(v) < EPS_D for v in mm)},
                "pair_top_switches": sum(tops[k] != tops[k + 1] for k in range(0, 96, 2))}
    # policy table
    with open(OUTDIR / "policy_table.jsonl", "w") as f:
        for i in range(720):
            row = {"atlas_index": i, "render_id": rid[i], "roles": pol[i]["roles"]}
            for ck in CKS:
                row[ck] = M[ck][i]
            row["B_min"] = min(M[ck][i]["B"] for ck in CKS)
            row["T_min"] = min(M[ck][i]["T"] for ck in CKS)
            row["sum_B"] = sum(M[ck][i]["B"] for ck in CKS)
            row["sum_T"] = sum(M[ck][i]["T"] for ck in CKS)
            f.write(json.dumps(row) + "\n")
    vec = lambda i, key: tuple(M[ck][i][key] for ck in CKS)  # noqa: E731
    desc = lambda i: {"atlas_index": i, "render_id": rid[i],  # noqa: E731
                      "roles": pol[i]["roles"], "T": vec(i, "T"), "B": vec(i, "B")}
    # maximin with tie sets per stage
    keyf = [lambda i: -min(vec(i, "B")), lambda i: -min(vec(i, "T")),
            lambda i: -sum(vec(i, "B")), lambda i: -sum(vec(i, "T"))]
    pool = list(range(720))
    stages = []
    for k, kf in enumerate(keyf):
        best = min(kf(i) for i in pool)
        pool = [i for i in pool if kf(i) == best]
        stages.append({"stage": k + 1, "value": -best, "tie_set_size": len(pool),
                       "tie_set": pool if len(pool) <= 64 else pool[:64]})
    winner = min(pool, key=lambda i: rid[i])
    maximin = {"stages": stages, "final_tie_set": pool, "winner": desc(winner),
               "B_min": min(vec(winner, "B")), "T_min": min(vec(winner, "T"))}
    # tiers
    tiers = {}
    for name, fn in TIERS.items():
        sat = [i for i in range(720) if all(fn(M[ck][i]["T"], M[ck][i]["B"]) for ck in CKS)]
        tiers[name] = {"count": len(sat), "policies": [desc(i) for i in sat]}
    # individual optima
    opt = {}
    for ck in CKS:
        Bs = max(M[ck][i]["B"] for i in range(720))
        Bset = [i for i in range(720) if M[ck][i]["B"] == Bs]
        Ts_amongB = max(M[ck][i]["T"] for i in Bset)
        Tb = max(M[ck][i]["T"] for i in range(720))
        Tset = [i for i in range(720) if M[ck][i]["T"] == Tb]
        near = [i for i in range(720) if M[ck][i]["B"] >= Bs - SLACK_B
                and M[ck][i]["T"] >= Tb - SLACK_T]
        opt[ck] = {"B_star": Bs, "T_star_among_B": Ts_amongB, "B_argmax": Bset,
                   "Tbest": Tb, "T_argmax": Tset,
                   "own_B_argmax_pass_own_fence": [i for i in Bset if i in near],
                   "own_T_argmax_pass_own_fence": [i for i in Tset if i in near],
                   "near_set_size": len(near), "near_set": near,
                   "B_argmax_desc": [desc(i) for i in Bset[:16]],
                   "T_argmax_desc": [desc(i) for i in Tset[:16]]}
    common = sorted(set.intersection(*(set(opt[ck]["near_set"]) for ck in CKS)))
    common_basin = {"count": len(common), "policies": [desc(i) for i in common],
                    "vacuous_checkpoints": [ck for ck in CKS if opt[ck]["near_set_size"] == 0],
                    "slack": {"B": SLACK_B, "T": SLACK_T}}
    # regret
    reg = {}
    for i in range(720):
        rB = [opt[ck]["B_star"] - M[ck][i]["B"] for ck in CKS]
        rT = [opt[ck]["Tbest"] - M[ck][i]["T"] for ck in CKS]
        reg[i] = (max(rB), max(rT), sum(rB), sum(rT), rid[i])
    rw = min(range(720), key=lambda i: reg[i])
    regret = {"winner": desc(rw), "max_regret_B": reg[rw][0], "max_regret_T": reg[rw][1],
              "sum_regret_B": reg[rw][2], "sum_regret_T": reg[rw][3],
              "equals_maximin": rw == winner}
    # pareto
    fronts = {}
    for key in ("B", "T"):
        fr = pareto({i: vec(i, key) for i in range(720)})
        fronts[key] = {"size": len(fr), "policies": [desc(i) for i in fr]}
    # pairwise
    pairwise = {}
    for a in range(4):
        for b in range(a + 1, 4):
            c1, c2 = CKS[a], CKS[b]
            T1 = [M[c1][i]["T"] for i in range(720)]
            T2 = [M[c2][i]["T"] for i in range(720)]
            B1 = [M[c1][i]["B"] for i in range(720)]
            B2 = [M[c2][i]["B"] for i in range(720)]
            r1T, r2T = M[c1][RAW_IDX]["T"], M[c2][RAW_IDX]["T"]
            r1B, r2B = M[c1][RAW_IDX]["B"], M[c2][RAW_IDX]["B"]
            pairwise[f"{c1} v {c2}"] = {
                "spearman_T": spearman(T1, T2), "spearman_B": spearman(B1, B2),
                "c1_up_c2_down_T": sum(T1[i] > r1T and T2[i] < r2T for i in range(720)),
                "c2_up_c1_down_T": sum(T2[i] > r2T and T1[i] < r1T for i in range(720)),
                "c1_up_c2_down_B": sum(B1[i] > r1B and B2[i] < r2B for i in range(720)),
                "c2_up_c1_down_B": sum(B2[i] > r2B and B1[i] < r1B for i in range(720)),
                "B_argmax_overlap": sorted(set(opt[c1]["B_argmax"]) & set(opt[c2]["B_argmax"])),
                "T_argmax_overlap": sorted(set(opt[c1]["T_argmax"]) & set(opt[c2]["T_argmax"])),
                "near_overlap_size": len(set(opt[c1]["near_set"]) & set(opt[c2]["near_set"]))}
    # basin width
    basin = {}
    for ck in CKS:
        Ts = [M[ck][i]["T"] for i in range(720)]
        Bs = [M[ck][i]["B"] for i in range(720)]
        basin[ck] = {"T": dist(Ts), "B": dist(Bs),
                     "T_gt_48": sum(t > 48 for t in Ts), "B_gt_0": sum(b > 0 for b in Bs),
                     "majority": sum(t >= 72 and b >= 24 for t, b in zip(Ts, Bs)),
                     "strong": sum(t >= 84 and b >= 36 for t, b in zip(Ts, Bs)),
                     "near_ceiling": sum(t >= 90 and b >= 42 for t, b in zip(Ts, Bs))}
    prior3 = {"strong_count_20001C": basin["20001|CANONICAL"]["strong"],
              "threshold": 72,
              "verdict": "CONFIRMED" if basin["20001|CANONICAL"]["strong"] < 72 else "REFUTED"}
    # surface anatomy join (pre-existing census only)
    anatomy = {}
    for feat in FEATS:
        by = defaultdict(list)
        for i in range(720):
            p = pol[i]
            if feat == "first_role":
                key = p["roles"][0]
            else:
                key = "|".join(f"{th}:{sorted(p['surface_by_theta'][th][feat])}"
                               for th in ("SIN_LOW", "COS_LOW"))
            by[key].append(i)
        anatomy[feat] = {k: {"n": len(v),
                             **{ck: {"T_median": median([M[ck][i]["T"] for i in v]),
                                     "T_max": max(M[ck][i]["T"] for i in v),
                                     "B_median": median([M[ck][i]["B"] for i in v]),
                                     "B_max": max(M[ck][i]["B"] for i in v)}
                                for ck in CKS}}
                         for k, v in sorted(by.items())}
    anchors = {name: {ck: {"T": M[ck][i]["T"], "B": M[ck][i]["B"]} for ck in CKS}
               for name, i in (("RAW", 12), ("K_FIRST", 480), ("LOW_PAIR_FIRST", 300))}
    agg = {"prereg": receipt["prereg"], "prereg_commit": receipt["prereg_commit"],
           "chunk_shas": {ck: chunks[ck][3] for ck in CKS},
           "anchors": anchors, "maximin": maximin, "tiers": tiers,
           "individual_optima": {ck: {k: v for k, v in o.items() if k != "near_set"}
                                 for ck, o in opt.items()},
           "common_basin": common_basin, "minimax_regret": regret,
           "pareto": fronts, "pairwise": pairwise, "basin_width": basin,
           "prior3_narrow_basin_20001C": prior3, "surface_anatomy": anatomy,
           "semantic_beyond_all_surface_identifiable": False,
           "policy_table_sha256": fsha(str(OUTDIR / "policy_table.jsonl"))}
    (OUTDIR / "aggregate.json").write_text(json.dumps(agg, indent=1))
    print(json.dumps({k: agg[k] for k in ("anchors", "maximin", "common_basin",
                                          "minimax_regret", "prior3_narrow_basin_20001C")},
                     indent=1))
    print({k: v["count"] for k, v in tiers.items()})
    print({ck: {"B*": o["B_star"], "Tbest": o["Tbest"], "near": o["near_set_size"]}
           for ck, o in opt.items()})
    print({k: (v["spearman_T"], v["spearman_B"]) for k, v in pairwise.items()})
    print({ck: (b["T"], b["B"], b["strong"]) for ck, b in basin.items()})
    print({k: v["size"] for k, v in fronts.items()})


if __name__ == "__main__":
    main()
