"""MATH-CYBER-1 SVP-FACTOR-HASH-SCORE-0 — ONE joint scoring run:
the three sealed seed-12001 checkpoints (CANONICAL-340 / FACTOR /
HASH) teacher-force scored on the frozen third band (69 primary
decisions) under the preregistered law. Zero training, zero
generation, zero sympy, zero re-enumeration; frozen candidate
bytes only.

Mechanical anti-peeking order (frozen): verify hashes -> load
models -> score ALL 69 -> write raw scores -> hash raw scores ->
hard gates -> primary F/H aggregate -> mechanically assign the
preregistered verdict -> write receipt -> ONLY THEN rider/anatomy
summaries (written to a separate riders file). No per-decision
console output during scoring.

Scoring primitives score_decision / rank_metrics / mcnemar_exact
IMPORTED from the frozen first-band scorer; FACTOR/HASH decode
from the qualified svpcode module (roundtrip gates on the stored
codes).

Outputs under logs/mathworld1/svpfhadj/ (refuse-if-exists):
scores.jsonl, svpfhadj_receipt.json, riders.json,
inspectable_sets.json.

    .venv/bin/python scratch/mathworld1_svpfhadj.py           (Mac)
"""
import hashlib
import json
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
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        hash_decode)

BAND = {
    "logs/mathworld1/svpeval3/episodes.jsonl":
        "cb57dd356dad46abbc69dde8c33a6c187010bf5c26cddc06b661"
        "18b0a869fde7",
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f4"
        "6f1afffb7efa",
}
CKPTS = {
    "CANONICAL": ("checkpoints/svp_fh_canonical_s12001.pt",
                  "1913b53c50ed938b1430628c3e14435c80abbf74eb5a"
                  "a2d945b9f74339c08a3f"),
    "FACTOR": ("checkpoints/svp_fh_factor_s12001.pt",
               "82f4f0d76fce2dc887ec09df2757e4213bed05d9d595d1"
               "49ed48ca1798bc03dd"),
    "HASH": ("checkpoints/svp_fh_hash_s12001.pt",
             "e2b7479549f2cc2fa9c156e253fc054f43f57631bac65c75"
             "d2ff01f1d237fae3"),
}
INIT_SHA = ("e21be542c998ccb63021f1241faecd46c322448a4ee25750"
            "dbbe8608af7aabe0")
VOCAB = 340
CODE_BASE = 332
N_PRIMARY = 69
THIN_ATOM = 7  # <c:7>, FACTOR training exposure 57 (booked)
ARMS = ["CANONICAL", "FACTOR", "HASH"]
OUTDIR = Path("logs/mathworld1/svpfhadj")
TOK = ActionGCTok()
TRAIN_TUP_SRC = "data/matsub_paired.jsonl"
TRAIN_TUP_SHA = ("a943ba7fc581db743b07192e5d951fadd"
                 "dd2ba19bca3225b75d8402351d468e8")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # 1. verify all frozen hashes/checkpoints
    for p, h in BAND.items():
        gate(fsha(p) == h, f"BAND PIN {p}")
    gate(fsha(TRAIN_TUP_SRC) == TRAIN_TUP_SHA, "TRAIN PIN")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha("checkpoints/svp_fh_init_s12001.pt") == INIT_SHA,
         "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpfhadj.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py", "llmopt/lab/provenance.py"])
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpeval3/decisions.jsonl")]
    prim = [r for r in rows if r.get("primary_eligible")]
    gate(len(prim) == N_PRIMARY, f"PRIMARY N {len(prim)}")
    ids = [(r["episode_id"], r["decision_index"]) for r in prim]
    gate(len(set(ids)) == N_PRIMARY, "IDS NOT UNIQUE")
    for r in prim:
        labs = [i for i, c in enumerate(r["candidates"])
                if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        for c in r["candidates"]:
            t = ctup(c)
            gate(factor_decode(c["factor_code"]) == t, "F RT")
            gate(hash_decode(c["hash_code"]) == t, "H RT")
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")

    # 2. load all three models
    arms = {}
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(CKPTS[a][0],
                                     weights_only=True))
        gate(sum(p.numel() for p in m.parameters()) == 19142016,
             f"PARAM COUNT {a}")
        m.eval()
        arms[a] = m.to(dev)

    # 3. score ALL 69 decisions (no per-decision output)
    OUTDIR.mkdir(parents=True)
    t0 = time.time()
    recs = []
    for r in prim:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        conts = {
            "CANONICAL": [TOK.encode(c["program_text"])
                          + [TOK.eos_id] for c in cands],
            "FACTOR": [[CODE_BASE + s for s in c["factor_code"]]
                       + [TOK.eos_id] for c in cands],
            "HASH": [[CODE_BASE + s for s in c["hash_code"]]
                     + [TOK.eos_id] for c in cands]}
        rec = {"episode_id": r["episode_id"],
               "decision_index": r["decision_index"],
               "cur": r["cur"],
               "n_candidates": len(cands),
               "label_index": li,
               "labeled_tuple": list(ctup(cands[li])),
               "labeled_rule": cands[li]["rule"],
               "labeled_param_kind": cands[li]["param_kind"],
               "candidate_tuples": [list(ctup(c))
                                    for c in cands],
               "exec_order": ARMS}
        for a in ARMS:
            triples = score_decision(
                arms[a], dev, r["cur"], conts[a])
            means = [t[0] for t in triples]
            sums = [t[1] for t in triples]
            top1, rank = rank_metrics(means, li)
            t1s, rks = rank_metrics(sums, li)
            rec[a] = {"mean_lp": means, "sum_lp": sums,
                      "T": [t[2] for t in triples],
                      "top1": top1, "rank": rank,
                      "top1_sum_rider": t1s,
                      "rank_sum_rider": rks}
        recs.append(rec)
    # 4-5. write + hash raw scores
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "scores.jsonl")
    n_disk = sum(1 for _ in open(OUTDIR / "scores.jsonl"))
    gate(n_disk == N_PRIMARY, "DISK ROWS")

    # 6. hard gates
    for rec in recs:
        for a in ("FACTOR", "HASH"):
            gate(all(t == 9 for t in rec[a]["T"]), f"T!=9 {a}")
            li = rec["label_index"]
            m_rank = rank_metrics(rec[a]["mean_lp"], li)
            s_rank = rank_metrics(rec[a]["sum_lp"], li)
            gate(m_rank == s_rank, f"MEAN!=SUM RANK {a}")
            # full-ordering identity incl. pessimistic ties
            mo = sorted(range(len(rec[a]["mean_lp"])),
                        key=lambda i: (-rec[a]["mean_lp"][i], i))
            so = sorted(range(len(rec[a]["sum_lp"])),
                        key=lambda i: (-rec[a]["sum_lp"][i], i))
            gate(mo == so, f"ORDER MISMATCH {a}")
        for a in ARMS:
            gate(all(isinstance(x, float) for x in
                     rec[a]["mean_lp"]), "SCORE TYPE")

    # 7-8. primary F/H aggregate + mechanical verdict
    def agg(key, x, y):
        fx = sum(1 for d in recs if d[x][key])
        fy = sum(1 for d in recs if d[y][key])
        b = sum(1 for d in recs
                if d[x][key] and not d[y][key])
        c = sum(1 for d in recs
                if d[y][key] and not d[x][key])
        return fx, fy, b, c

    f1, h1, fb, fc = agg("top1", "FACTOR", "HASH")
    pval = mcnemar_exact(fb, fc)
    if f1 > h1 and pval < 0.05:
        verdict = "FACTOR-WIN"
    elif h1 > f1 and pval < 0.05:
        verdict = "HASH-WIN"
    else:
        verdict = "INCONCLUSIVE"
    mrr = {a: sum(1.0 / d[a]["rank"] for d in recs) / N_PRIMARY
           for a in ARMS}
    c1 = sum(1 for d in recs if d["CANONICAL"]["top1"])
    cs = sum(1 for d in recs
             if d["CANONICAL"]["top1_sum_rider"])
    cflip = sum(1 for d in recs
                if d["CANONICAL"]["top1"]
                != d["CANONICAL"]["top1_sum_rider"])

    # 9. write receipt (before any rider/anatomy inspection)
    receipt = {
        "n_primary": N_PRIMARY,
        "device": str(dev),
        "primary": {
            "FACTOR_top1": f1, "HASH_top1": h1,
            "discordant_factor_only": fb,
            "discordant_hash_only": fc,
            "mcnemar_exact_two_sided_p": pval,
            "alpha": 0.05, "verdict": verdict,
            "MRR": {"FACTOR": mrr["FACTOR"],
                    "HASH": mrr["HASH"]}},
        "canonical_bridge": {
            "top1": c1, "MRR": mrr["CANONICAL"],
            "summed_lp_top1_rider": cs,
            "mean_v_sum_flips": cflip,
            "note": "descriptive bridge only; length/"
                    "token-channel law differs from F/H"},
        "gates": {
            "T9_ALL": True, "MEAN_SUM_RANK_IDENTITY": True,
            "FULL_ORDER_IDENTITY": True,
            "note": "gate-enforced hard exits; receipt "
                    "existence = pass"},
        "wall_s": round(time.time() - t0, 1),
        "files": {"scores.jsonl": raw_sha},
        "pins": {p: fsha(p) for p in BAND} | {
            p: fsha(p) for p, _ in CKPTS.values()},
        "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "svpfhadj_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({"primary": receipt["primary"],
                      "canonical_bridge":
                          receipt["canonical_bridge"]},
                     indent=1), flush=True)

    # 10. riders/anatomy AFTER the receipt is frozen
    train_tup = set()
    for l in open(TRAIN_TUP_SRC):
        r = json.loads(l)
        train_tup.add((r["rule"], r["site_kind"],
                       r["site_ordinal"], r["param_kind"],
                       r["param_index"]))

    def cov_of(rec):
        tups = [tuple(t) for t in rec["candidate_tuples"]]
        oov = sum(1 for t in tups if t not in train_tup)
        return oov, len(tups)

    def stratum(pred):
        sub = [d for d in recs if pred(d)]
        return {"n": len(sub), **{
            f"{a}_top1": sum(1 for d in sub if d[a]["top1"])
            for a in ARMS}}

    prim_by = {(r["episode_id"], r["decision_index"]): r
               for r in prim}

    def thin_touch(rec):
        r = prim_by[(rec["episode_id"], rec["decision_index"])]
        return any(THIN_ATOM in c["factor_code"]
                   for c in r["candidates"])

    riders = {
        "coverage": {
            "all_covered": stratum(
                lambda d: cov_of(d)[0] == 0),
            "ge1_oov_rival": stratum(
                lambda d: cov_of(d)[0] > 0),
            "by_oov_count": {
                str(k): stratum(lambda d, k=k:
                                cov_of(d)[0] == k)
                for k in sorted({cov_of(d)[0] for d in recs})}},
        "thin_atom": {
            "atom": f"<c:{THIN_ATOM}>",
            "touched": stratum(thin_touch),
            "untouched": stratum(lambda d: not thin_touch(d))},
        "labeled_rule": {ru: stratum(
            lambda d, ru=ru: d["labeled_rule"] == ru)
            for ru in sorted({d["labeled_rule"]
                              for d in recs})},
        "labeled_param_kind": {pk: stratum(
            lambda d, pk=pk: d["labeled_param_kind"] == pk)
            for pk in sorted({d["labeled_param_kind"]
                              for d in recs})},
        "legal_set_size": {str(n): stratum(
            lambda d, n=n: d["n_candidates"] == n)
            for n in sorted({d["n_candidates"]
                             for d in recs})},
    }

    def keyrec(d):
        li = d["label_index"]
        oov, tot = cov_of(d)
        out = {"episode_id": d["episode_id"],
               "decision_index": d["decision_index"],
               "cur": d["cur"],
               "labeled_tuple": d["labeled_tuple"],
               "candidate_tuples": d["candidate_tuples"],
               "oov_rivals": oov, "n_candidates": tot,
               "thin_atom": thin_touch(d)}
        for a in ("FACTOR", "HASH"):
            ml = d[a]["mean_lp"]
            riv = [x for j, x in enumerate(ml) if j != li]
            out[f"{a}_label_margin"] = round(
                ml[li] - max(riv), 6) if riv else None
        return out

    insp = {
        "factor_correct_hash_wrong": [
            keyrec(d) for d in recs
            if d["FACTOR"]["top1"] and not d["HASH"]["top1"]],
        "hash_correct_factor_wrong": [
            keyrec(d) for d in recs
            if d["HASH"]["top1"] and not d["FACTOR"]["top1"]],
        "both_wrong": [keyrec(d) for d in recs
                       if not d["FACTOR"]["top1"]
                       and not d["HASH"]["top1"]],
        "both_correct_n": sum(
            1 for d in recs
            if d["FACTOR"]["top1"] and d["HASH"]["top1"]),
        "cfh_vectors": {
            f"{d['episode_id']}:{d['decision_index']}":
                [int(d[a]["top1"]) for a in ARMS]
            for d in recs}}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    (OUTDIR / "inspectable_sets.json").write_text(
        json.dumps(insp, indent=1))
    print(json.dumps({"riders": {
        "coverage": riders["coverage"]["all_covered"],
        "ge1_oov": riders["coverage"]["ge1_oov_rival"],
        "thin": riders["thin_atom"]},
        "inspectable_sizes": {
            k: (len(v) if isinstance(v, list) else v)
            for k, v in insp.items() if k != "cfh_vectors"}},
        indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
