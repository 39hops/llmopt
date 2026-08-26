"""MATH-CYBER-1 SVP-GENERALIZATION-SCORE-0 — ONE joint scorer:
all three paired births (9001, 10001, 11001 in that fixed order)
teacher-force scored on the frozen SECOND band (79 primary
decisions, seeds 9700-9719) under the byte-frozen first-band
scoring law. Zero training, zero generation, zero sympy/world.

Scoring primitives score_decision / rank_metrics / mcnemar_exact
are IMPORTED from the frozen first-band scorer
scratch/mathworld1_svpadj.py — one law, one source. All 237 raw
rows (3 births x 79 decisions) persist to
logs/mathworld1/svpgen/scores.jsonl BEFORE any aggregate is
computed. GENERALIZES-DIRECTION is applied mechanically per the
law frozen at SVP-GENERALIZATION-BAND-0.

    .venv/bin/python scratch/mathworld1_svpgenadj.py          (Mac)
"""
import hashlib
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from llmopt.train.mathnative import build_model  # noqa: E402
from scratch.mathworld1_actiontok import ActionGCTok  # noqa: E402
from scratch.mathworld1_svpadj import (mcnemar_exact,  # noqa: E402
                                       rank_metrics, score_decision)
from scratch.mathworld1_svpbirth import gate  # noqa: E402

BAND = {
    "logs/mathworld1/svpeval2/episodes.jsonl":
        "584c7acc2c779e01eab9293a94ab91b5930e5a19d93d318801"
        "cd93d808ade50d",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c211"
        "2cfd5e92f79315",
}
CKPTS = {
    9001: {"STATE": ("checkpoints/svp_state.pt",
                     "8e0a22f29074ee819a3936748f27939022ac9b9749"
                     "89c988fa1d3f6f0694c060"),
           "PROGRAM": ("checkpoints/svp_program.pt",
                       "d9db0049b135f326eb8fa2d9f74e7c067516e49a"
                       "e597ecaac11ecae1dfc57853")},
    10001: {"STATE": ("checkpoints/svp_state_s10001.pt",
                      "118551181a3f8904da0b6e6da9ef123a2038b83a9"
                      "0d585cb3d535739a3a3f686"),
            "PROGRAM": ("checkpoints/svp_program_s10001.pt",
                        "395dfd535d4c446c3372baa8ea33ed2eef6f703"
                        "c3c36db971559019ab4e199a9")},
    11001: {"STATE": ("checkpoints/svp_state_s11001.pt",
                      "041b3b047df80ff825ddbcff28c879de04586b961"
                      "6600eb1196ff7cf69ad4973"),
            "PROGRAM": ("checkpoints/svp_program_s11001.pt",
                        "3305782548b8e1f2b7a0b63d44ecf79a44c9430"
                        "1f692ba9b93fbdc255c3a13b7")},
}
BIRTH_ORDER = [9001, 10001, 11001]
OUTDIR = Path("logs/mathworld1/svpgen")
TOK = ActionGCTok()
N_PRIMARY = 79


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pin_all():
    for p, h in BAND.items():
        gate(fsha(p) == h, f"BAND PIN MISMATCH {p}")
    for seed, arms in CKPTS.items():
        for view, (p, h) in arms.items():
            gate(fsha(p) == h, f"CKPT PIN MISMATCH {seed} {view}")


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    pin_all()
    START = start_provenance(
        ["scratch/mathworld1_svpgenadj.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpeval2/decisions.jsonl")]
    prim = [r for r in rows if r.get("primary_eligible")]
    gate(len(prim) == N_PRIMARY, f"PRIMARY N {len(prim)}")
    ids = [(r["episode_id"], r["decision_index"]) for r in prim]
    gate(len(set(ids)) == N_PRIMARY, "PRIMARY IDS NOT UNIQUE")
    gate(torch.backends.mps.is_available(), "MPS UNAVAILABLE")
    dev = torch.device("mps")

    OUTDIR.mkdir(parents=True)
    t0 = time.time()
    per_birth = {}
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for seed in BIRTH_ORDER:
            arms = {}
            for view in ("STATE", "PROGRAM"):
                ck, h = CKPTS[seed][view]
                gate(fsha(ck) == h, f"PRE-LOAD PIN {seed} {view}")
                m = build_model(TOK.vocab_size, ctx=4096)
                m.load_state_dict(
                    torch.load(ck, weights_only=True))
                m.eval()
                arms[view] = m.to(dev)
            recs = []
            for r in prim:
                cands = r["candidates"]
                gate(len(cands) == r["n_candidates"],
                     "CAND COUNT")
                labs = [i for i, c in enumerate(cands)
                        if c["is_label"]]
                gate(len(labs) == 1, "LABEL COUNT")
                li = labs[0]
                conts = {
                    "STATE": [TOK.encode(c["child_sstr"] + "\n")
                              + [TOK.eos_id] for c in cands],
                    "PROGRAM": [TOK.encode(c["program_text"])
                                + [TOK.eos_id] for c in cands]}
                rec = {"birth_seed": seed,
                       "episode_id": r["episode_id"],
                       "decision_index": r["decision_index"],
                       "cur": r["cur"],
                       "n_candidates": len(cands),
                       "label_index": li,
                       "labeled_rule": cands[li]["rule"],
                       "labeled_param_kind":
                           cands[li]["param_kind"],
                       "labeled_child_sstr":
                           cands[li]["child_sstr"],
                       "labeled_program_text":
                           cands[li]["program_text"],
                       "exec_order": ["STATE", "PROGRAM"]}
                for view in ("STATE", "PROGRAM"):
                    triples = score_decision(
                        arms[view], dev, r["cur"], conts[view])
                    means = [t[0] for t in triples]
                    sums = [t[1] for t in triples]
                    top1, rank = rank_metrics(means, li)
                    t1s, rks = rank_metrics(sums, li)
                    rec[view] = {
                        "mean_lp": means, "sum_lp": sums,
                        "T": [t[2] for t in triples],
                        "top1": top1, "rank": rank,
                        "top1_sum_rider": t1s,
                        "rank_sum_rider": rks}
                fo.write(json.dumps(rec) + "\n")
                recs.append(rec)
            per_birth[seed] = recs
            del arms
            torch.mps.empty_cache()
            print(f"[svpgen] birth {seed}: {len(recs)} decisions "
                  f"scored", flush=True)

    # all raw rows are on disk; hard gates, THEN aggregates
    n_rows = sum(len(v) for v in per_birth.values())
    gate(n_rows == 3 * N_PRIMARY, f"ROW COUNT {n_rows}")
    for seed in BIRTH_ORDER:
        sids = [(x["episode_id"], x["decision_index"])
                for x in per_birth[seed]]
        gate(sids == ids, f"IDENTITY ORDER DRIFT birth {seed}")
    pin_all()

    def agg(recs, key):
        s = sum(1 for d in recs if d["STATE"][key])
        p = sum(1 for d in recs if d["PROGRAM"][key])
        b = sum(1 for d in recs
                if d["STATE"][key] and not d["PROGRAM"][key])
        c = sum(1 for d in recs
                if d["PROGRAM"][key] and not d["STATE"][key])
        return s, p, b, c

    primary = {}
    deltas = {}
    for seed in BIRTH_ORDER:
        recs = per_birth[seed]
        s1, p1, b, c = agg(recs, "top1")
        ss, ps, bs, cs = agg(recs, "top1_sum_rider")
        flips = sum(1 for d in recs for v in ("STATE", "PROGRAM")
                    if d[v]["top1"] != d[v]["top1_sum_rider"])
        deltas[seed] = 100.0 * (p1 - s1) / N_PRIMARY
        primary[seed] = {
            "STATE_top1": s1, "PROGRAM_top1": p1,
            "delta_pp": round(deltas[seed], 4),
            "discordant_state_only": b,
            "discordant_program_only": c,
            "mcnemar_exact_two_sided_p_rider": mcnemar_exact(b, c),
            "MRR": {v: sum(1.0 / d[v]["rank"] for d in recs)
                    / N_PRIMARY for v in ("STATE", "PROGRAM")},
            "summed_lp_top1": {"STATE": ss, "PROGRAM": ps,
                               "discordant": [bs, cs]},
            "mean_v_sum_top1_flips": flips}
    direction = all(primary[s]["PROGRAM_top1"]
                    > primary[s]["STATE_top1"]
                    for s in BIRTH_ORDER)
    verdict = ("GENERALIZES-DIRECTION" if direction
               else "BLOCKED-TIE-OR-REVERSAL")
    dvals = [deltas[s] for s in BIRTH_ORDER]

    # registered mechanism riders (descriptive only)
    def stratum(pred):
        out = {}
        for seed in BIRTH_ORDER:
            sub = [d for d in per_birth[seed] if pred(d)]
            out[str(seed)] = {
                "n": len(sub),
                "STATE_top1": sum(1 for d in sub
                                  if d["STATE"]["top1"]),
                "PROGRAM_top1": sum(1 for d in sub
                                    if d["PROGRAM"]["top1"])}
        return out

    prim_by_id = {(r["episode_id"], r["decision_index"]): r
                  for r in prim}

    def has_uc(d):
        r = prim_by_id[(d["episode_id"], d["decision_index"])]
        return any(c["param_kind"] == "u_choice"
                   for c in r["candidates"])

    rules = sorted({d["labeled_rule"]
                    for d in per_birth[BIRTH_ORDER[0]]})
    rider_rule = {ru: stratum(
        lambda d, ru=ru: d["labeled_rule"] == ru) for ru in rules}
    pkinds = sorted({d["labeled_param_kind"]
                     for d in per_birth[BIRTH_ORDER[0]]})
    rider_pk = {pk: stratum(
        lambda d, pk=pk: d["labeled_param_kind"] == pk)
        for pk in pkinds}

    # true continuation-target lengths from frozen bytes
    lab_T = {"STATE": [], "PROGRAM": []}
    spreads = []
    for r in prim:
        li = [i for i, c in enumerate(r["candidates"])
              if c["is_label"]][0]
        stT = [len(TOK.encode(c["child_sstr"] + "\n")) + 1
               for c in r["candidates"]]
        pgT = [len(TOK.encode(c["program_text"])) + 1
               for c in r["candidates"]]
        lab_T["STATE"].append(stT[li])
        lab_T["PROGRAM"].append(pgT[li])
        spreads.append({"state_min": min(stT),
                        "state_max": max(stT),
                        "program_min": min(pgT),
                        "program_max": max(pgT)})

    def dist(xs):
        xs = sorted(xs)
        q = lambda p: xs[min(len(xs) - 1,
                             int(p * (len(xs) - 1)))]
        return {"p50": q(.5), "p90": q(.9), "max": xs[-1]}

    # cross-birth correctness vectors + inspectable sets
    vec = {}
    for i, key in enumerate(ids):
        vec[key] = {
            v: [per_birth[s][i][v]["top1"] for s in BIRTH_ORDER]
            for v in ("STATE", "PROGRAM")}
    def keyrec(key):
        r = prim_by_id[key]
        li = [i for i, c in enumerate(r["candidates"])
              if c["is_label"]][0]
        return {"episode_id": key[0], "decision_index": key[1],
                "cur": r["cur"],
                "labeled_rule": r["candidates"][li]["rule"],
                "labeled_param_kind":
                    r["candidates"][li]["param_kind"],
                "labeled_child_sstr":
                    r["candidates"][li]["child_sstr"],
                "labeled_program_text":
                    r["candidates"][li]["program_text"]}
    insp = {
        "program3_state0": [keyrec(k) for k in ids
                            if all(vec[k]["PROGRAM"])
                            and not any(vec[k]["STATE"])],
        "program_wrong3": [keyrec(k) for k in ids
                           if not any(vec[k]["PROGRAM"])],
        "state_correct3": [keyrec(k) for k in ids
                           if all(vec[k]["STATE"])]}
    (OUTDIR / "inspectable_sets.json").write_text(
        json.dumps(insp, indent=1))
    xbirth = {v: {} for v in ("STATE", "PROGRAM")}
    for v in ("STATE", "PROGRAM"):
        for k in ids:
            n = sum(vec[k][v])
            xbirth[v][str(n)] = xbirth[v].get(str(n), 0) + 1

    receipt = {
        "n_primary": N_PRIMARY, "n_rows": n_rows,
        "birth_order": BIRTH_ORDER,
        "device": str(dev),
        "primary": primary,
        "resolution": {
            "law": "GENERALIZES-DIRECTION iff PROGRAM top1 > "
                   "STATE top1 in all three births; tie or "
                   "reversal blocks; per-pair p<.05 not "
                   "required (rider only)",
            "verdict": verdict,
            "deltas_pp": {str(s): round(deltas[s], 4)
                          for s in BIRTH_ORDER},
            "median_delta_pp": round(statistics.median(dvals), 4),
            "range_delta_pp": [round(min(dvals), 4),
                               round(max(dvals), 4)]},
        "riders": {
            "labeled_rule": rider_rule,
            "labeled_param_kind": rider_pk,
            "u_choice_candidate_present": stratum(has_uc),
            "u_choice_candidate_absent":
                stratum(lambda d: not has_uc(d)),
            "true_target_T": {
                "labeled_STATE": dist(lab_T["STATE"]),
                "labeled_PROGRAM": dist(lab_T["PROGRAM"])},
            "candidate_T_spread_note":
                "per-decision min/max continuation-target "
                "lengths persisted in receipt field "
                "candidate_T_spread",
            "candidate_T_spread": spreads,
            "cross_birth_correct_counts": xbirth,
            "inspectable_sets_file": "inspectable_sets.json",
            "inspectable_set_sizes": {
                k: len(v) for k, v in insp.items()}},
        "wall_s": round(time.time() - t0, 1),
        "files": {"scores.jsonl": fsha(OUTDIR / "scores.jsonl"),
                  "inspectable_sets.json":
                      fsha(OUTDIR / "inspectable_sets.json")},
        "pins": {p: fsha(p) for p in BAND} | {
            p: fsha(p) for arms in CKPTS.values()
            for p, _ in arms.values()},
        "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "svpgen_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins", "riders")},
                     indent=1), flush=True)
    print(json.dumps({"resolution": receipt["resolution"],
                      "inspectable_set_sizes":
                          receipt["riders"][
                              "inspectable_set_sizes"]},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
