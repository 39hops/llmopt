"""MATH-CYBER-1 SVP-LENGTH-CONTROL-DESK-0 — desk-only triage of
the length/factorization mechanism from FROZEN score artifacts.
No model inference, no training, no checkpoint load, no task
generation. Bands stay separate surfaces throughout.

Q1: score_alpha = sum_lp / T**alpha on the fixed grid
{0, .25, .5, .75, 1} (0 = booked summed-lp rider, 1 = booked
mean-lp primary), pessimistic top-1/rank per band x birth x arm
x alpha. Q2: per-decision length/margin anatomy, characterized
by the persisted inspectable sets. Q3: descriptive strata by
STATE candidate T max/min ratio (bins frozen pre-compute:
<=2, >2-<=4, >4). Q4: labeled rule family x length anatomy.

Outputs under logs/mathworld1/svpldesk/ (refuse-if-exists):
anatomy.jsonl (per decision x band), desk_receipt.json.

    .venv/bin/python scratch/mathworld1_svpldesk.py           (Mac)
"""
import hashlib
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PINS = {
    "logs/mathworld1/svpgen/scores.jsonl":
        "174cdc0eceb3599ac8808a68f1cab9d45efde4ba96361b74c8"
        "caf474d91abbfd",
    "logs/mathworld1/svpgen/inspectable_sets.json":
        "200f49fa471758e070499c8f9af6d38e33c2ee52f6ccaf0385"
        "cedca86833b9e7",
    "logs/mathworld1/svpadj/scores.jsonl":
        "66b7ceade9d0764818f7d95f16fb0e1f8fa38835e66e2c2a52"
        "61a86e581a6b07",
    "logs/mathworld1/svpadj_s10001/scores.jsonl":
        "dc0d7c24ba210bec268c776c92f19eb1db193a83d8f8efd6d2"
        "50d0cc53f32212",
    "logs/mathworld1/svpadj_s11001/scores.jsonl":
        "1ac585a822349cf37e3f264accecc6041193b19fa083f6f702"
        "3d369b8c0eaf3f",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241"
        "406a86253858c6",
}
ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
RATIO_BINS = [("r<=2", lambda r: r <= 2.0),
              ("2<r<=4", lambda r: 2.0 < r <= 4.0),
              ("r>4", lambda r: r > 4.0)]
SEEDS = [9001, 10001, 11001]
OUTDIR = Path("logs/mathworld1/svpldesk")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def rank_metrics(scores, li):
    """The booked pessimistic law, restated for desk use on
    arbitrary alpha (verified against booked top1/rank at the
    grid endpoints)."""
    lab = scores[li]
    better = sum(1 for i, s in enumerate(scores)
                 if i != li and s > lab)
    ties = sum(1 for i, s in enumerate(scores)
               if i != li and s == lab)
    return (better + ties) == 0, 1 + better + ties


def load_rows():
    """Returns {band: {seed: [row,...]}} with rows in band file
    order, plus per-band labeled-rule maps."""
    new = [json.loads(l) for l in
           open("logs/mathworld1/svpgen/scores.jsonl")]
    gate(len(new) == 237, "NEW ROWS")
    bands = {"new": {s: [r for r in new if r["birth_seed"] == s]
                     for s in SEEDS}}
    old_paths = {9001: "logs/mathworld1/svpadj/scores.jsonl",
                 10001: "logs/mathworld1/svpadj_s10001/scores.jsonl",
                 11001: "logs/mathworld1/svpadj_s11001/scores.jsonl"}
    bands["old"] = {s: [json.loads(l) for l in open(p)]
                    for s, p in old_paths.items()}
    for s in SEEDS:
        gate(len(bands["new"][s]) == 79, f"NEW N {s}")
        gate(len(bands["old"][s]) == 72, f"OLD N {s}")
    # old-band labeled rule/param_kind joined from the frozen
    # first-band decisions (text fields only)
    old_meta = {}
    for l in open("logs/mathworld1/svpeval/decisions.jsonl"):
        r = json.loads(l)
        if not r.get("primary_eligible"):
            continue
        li = [i for i, c in enumerate(r["candidates"])
              if c["is_label"]][0]
        old_meta[(r["episode_id"], r["decision_index"])] = {
            "labeled_rule": r["candidates"][li]["rule"],
            "labeled_param_kind": r["candidates"][li]["param_kind"]}
    gate(len(old_meta) == 72, "OLD META N")
    return bands, old_meta


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpldesk.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    bands, old_meta = load_rows()
    insp = json.loads(Path(
        "logs/mathworld1/svpgen/inspectable_sets.json").read_text())
    insp_ids = {k: {(d["episode_id"], d["decision_index"])
                    for d in v} for k, v in insp.items()}

    # verify desk ranker reproduces booked endpoints exactly
    for band in bands:
        for s in SEEDS:
            for r in bands[band][s]:
                for v in ("STATE", "PROGRAM"):
                    li = r["label_index"]
                    means = [su / t for su, t in
                             zip(r[v]["sum_lp"], r[v]["T"])]
                    t1, rk = rank_metrics(means, li)
                    gate(t1 == r[v]["top1"] and rk == r[v]["rank"],
                         "ALPHA1 ENDPOINT MISMATCH")
                    t1s, rks = rank_metrics(r[v]["sum_lp"], li)
                    gate(t1s == r[v]["top1_sum_rider"]
                         and rks == r[v]["rank_sum_rider"],
                         "ALPHA0 ENDPOINT MISMATCH")

    # Q1: alpha grid tables
    q1 = {}
    for band in ("old", "new"):
        n_dec = 72 if band == "old" else 79
        q1[band] = {}
        for s in SEEDS:
            q1[band][str(s)] = {}
            for v in ("STATE", "PROGRAM"):
                cell = {}
                ranks_by_alpha = {}
                for a in ALPHAS:
                    t1s, rr = 0, 0.0
                    ranks = []
                    for r in bands[band][s]:
                        li = r["label_index"]
                        sc = [su / (t ** a) for su, t in
                              zip(r[v]["sum_lp"], r[v]["T"])]
                        t1, rk = rank_metrics(sc, li)
                        t1s += t1
                        rr += 1.0 / rk
                        ranks.append(rk)
                    ranks_by_alpha[a] = ranks
                    cell[str(a)] = {"top1": t1s,
                                    "MRR": round(rr / n_dec, 4)}
                # flips v alpha=1 and any-rank-change across grid
                for a in ALPHAS:
                    flips = sum(
                        1 for i in range(n_dec)
                        if (ranks_by_alpha[a][i] == 1)
                        != (ranks_by_alpha[1.0][i] == 1))
                    cell[str(a)]["top1_flips_v_alpha1"] = flips
                anych = sum(
                    1 for i in range(n_dec)
                    if len({ranks_by_alpha[a][i]
                            for a in ALPHAS}) > 1)
                cell["decisions_rank_changes_across_grid"] = anych
                best = max(ALPHAS,
                           key=lambda a: cell[str(a)]["top1"])
                cell["posthoc_best_grid_alpha"] = {
                    "alpha": best,
                    "top1": cell[str(best)]["top1"],
                    "note": "post-hoc descriptive upper bound "
                            "only, never an adjudicating scorer"}
                q1[band][str(s)][v] = cell

    # Q2 anatomy per decision x band (birth-independent fields
    # from seed-9001 rows; correctness vectors across births)
    anatomy = []
    for band in ("old", "new"):
        n_dec = 72 if band == "old" else 79
        rows0 = bands[band][SEEDS[0]]
        for i in range(n_dec):
            r0 = rows0[i]
            key = (r0["episode_id"], r0["decision_index"])
            li = r0["label_index"]
            # T vectors identical across births (frozen bytes)
            for s in SEEDS[1:]:
                gate(bands[band][s][i]["STATE"]["T"]
                     == r0["STATE"]["T"], "T DRIFT")
            stT = r0["STATE"]["T"]
            pgT = r0["PROGRAM"]["T"]
            rival = [t for j, t in enumerate(stT) if j != li]
            gate(min(stT) > 0, "ZERO-LENGTH TARGET")
            ratio = max(stT) / min(stT)
            if band == "new":
                rule = r0["labeled_rule"]
                pk = r0["labeled_param_kind"]
            else:
                rule = old_meta[key]["labeled_rule"]
                pk = old_meta[key]["labeled_param_kind"]
            rec = {"band": band, "episode_id": key[0],
                   "decision_index": key[1],
                   "labeled_rule": rule,
                   "labeled_param_kind": pk,
                   "n_candidates": r0["n_candidates"],
                   "labeled_T": {"STATE": stT[li],
                                 "PROGRAM": pgT[li]},
                   "state_T_min": min(stT),
                   "state_T_max": max(stT),
                   "state_T_range": max(stT) - min(stT),
                   "state_T_ratio": round(ratio, 4),
                   "labeled_minus_best_rival_T":
                       (stT[li] - min(rival)) if rival else None}
            for s in SEEDS:
                r = bands[band][s][i]
                li_s = r["label_index"]
                gate(li_s == li, "LABEL DRIFT")
                for v in ("STATE", "PROGRAM"):
                    ml = r[v]["mean_lp"]
                    sl = r[v]["sum_lp"]
                    rivals_m = [x for j, x in enumerate(ml)
                                if j != li]
                    rivals_s = [x for j, x in enumerate(sl)
                                if j != li]
                    rec[f"{v}_{s}"] = {
                        "top1": r[v]["top1"],
                        "mean_margin": round(
                            ml[li] - max(rivals_m), 6)
                        if rivals_m else None,
                        "sum_margin": round(
                            sl[li] - max(rivals_s), 6)
                        if rivals_s else None,
                        "top1_by_alpha": [
                            rank_metrics(
                                [su / (t ** a) for su, t in
                                 zip(sl, r[v]["T"])], li)[0]
                            for a in ALPHAS]}
            if band == "new":
                rec["inspectable_set"] = next(
                    (k for k, ids in insp_ids.items()
                     if key in ids), "remaining")
            anatomy.append(rec)

    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "anatomy.jsonl", "w") as f:
        for r in anatomy:
            f.write(json.dumps(r) + "\n")

    def dist(xs):
        if not xs:
            return None
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1,
                             int(p * (len(xs) - 1)))]
        return {"p50": q(.5), "p90": q(.9),
                "min": xs[0], "max": xs[-1]}

    # Q2 set characterization (new band, persisted sets)
    q2 = {}
    newan = [r for r in anatomy if r["band"] == "new"]
    for setname in ("program3_state0", "program_wrong3",
                    "state_correct3", "remaining"):
        sub = [r for r in newan
               if r["inspectable_set"] == setname]
        q2[setname] = {
            "n": len(sub),
            "labeled_STATE_T": dist(
                [r["labeled_T"]["STATE"] for r in sub]),
            "labeled_PROGRAM_T": dist(
                [r["labeled_T"]["PROGRAM"] for r in sub]),
            "state_T_ratio": dist(
                [r["state_T_ratio"] for r in sub]),
            "state_T_range": dist(
                [r["state_T_range"] for r in sub]),
            "labeled_minus_best_rival_T": dist(
                [r["labeled_minus_best_rival_T"] for r in sub
                 if r["labeled_minus_best_rival_T"] is not None]),
            "labeled_rules": dict(Counter(
                r["labeled_rule"] for r in sub))}

    # Q3 strata by STATE T max/min ratio (frozen bins)
    q3 = {}
    for band in ("old", "new"):
        q3[band] = {}
        ban = [r for r in anatomy if r["band"] == band]
        for name, pred in RATIO_BINS:
            sub = [r for r in ban if pred(r["state_T_ratio"])]
            cell = {"n": len(sub)}
            for s in SEEDS:
                cell[str(s)] = {
                    v: sum(1 for r in sub if r[f"{v}_{s}"]["top1"])
                    for v in ("STATE", "PROGRAM")}
            q3[band][name] = cell

    # Q4 rule-family x length
    q4 = {}
    for band in ("old", "new"):
        q4[band] = {}
        ban = [r for r in anatomy if r["band"] == band]
        for rule in sorted({r["labeled_rule"] for r in ban}):
            sub = [r for r in ban if r["labeled_rule"] == rule]
            cell = {"n": len(sub),
                    "labeled_STATE_T": dist(
                        [r["labeled_T"]["STATE"] for r in sub]),
                    "state_T_ratio": dist(
                        [r["state_T_ratio"] for r in sub])}
            for s in SEEDS:
                cell[str(s)] = {
                    v: sum(1 for r in sub if r[f"{v}_{s}"]["top1"])
                    for v in ("STATE", "PROGRAM")}
            q4[band][rule] = cell

    # DESK RESOLUTION (frozen criterion: an alpha "materially
    # closes" if median over births of (PROGRAM top1 - STATE
    # top1) at that alpha is <= 50% of the median gap at
    # alpha=1, in BOTH bands, same alpha)
    gaps = {}
    for band in ("old", "new"):
        gaps[band] = {}
        for a in ALPHAS:
            g = [q1[band][str(s)]["PROGRAM"][str(a)]["top1"]
                 - q1[band][str(s)]["STATE"][str(a)]["top1"]
                 for s in SEEDS]
            gaps[band][str(a)] = {
                "per_seed": g,
                "median": statistics.median(g)}
    closes = [a for a in ALPHAS if all(
        gaps[b][str(a)]["median"]
        <= 0.5 * gaps[b]["1.0"]["median"]
        for b in ("old", "new"))]
    ahead_everywhere = all(
        q1[b][str(s)]["PROGRAM"][str(a)]["top1"]
        > q1[b][str(s)]["STATE"][str(a)]["top1"]
        for b in ("old", "new") for s in SEEDS for a in ALPHAS)
    if closes:
        resolution = "SCORING-LENGTH-PLAUSIBLE"
    elif ahead_everywhere:
        resolution = "SCORING-LENGTH-INSUFFICIENT"
    else:
        resolution = "MIXED"

    receipt = {
        "alpha_grid": ALPHAS,
        "q1_tables": q1,
        "q1_gap_medians": gaps,
        "q2_sets": q2,
        "q3_ratio_strata": q3,
        "q4_rule_family": q4,
        "resolution": {
            "vocabulary": {
                "SCORING-LENGTH-PLAUSIBLE":
                    "a fixed grid alpha halves the median "
                    "PROGRAM-STATE top1 gap in BOTH bands",
                "SCORING-LENGTH-INSUFFICIENT":
                    "PROGRAM directionally ahead in all births, "
                    "both bands, at EVERY grid alpha",
                "MIXED": "otherwise"},
            "alphas_that_materially_close": closes,
            "program_ahead_everywhere": ahead_everywhere,
            "verdict": resolution,
            "scope": "evaluation-normalization nuisance ONLY; "
                     "training-target length/entropy remains "
                     "confounded regardless of outcome"},
        "files": {"anatomy.jsonl": fsha(OUTDIR / "anatomy.jsonl")},
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "desk_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({"resolution": receipt["resolution"],
                      "q1_gap_medians": gaps,
                      "q3": q3}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
