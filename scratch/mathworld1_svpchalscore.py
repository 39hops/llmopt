"""MATH-CYBER-1 SVP-STRICT-GRID-CHALLENGE-SCORE-0 — ONE joint
scoring run: the three sealed seed-12001 checkpoints scored on
the frozen DESIGN-1 challenge bytes under the preregistered law
(274310c2). Zero training, zero checkpoint mutation, zero
challenge regeneration, zero sympy, zero legal-set
re-enumeration, zero HCE recomputation; frozen candidate bytes
only.

Mechanical anti-peeking order (frozen): verify pins -> load
models -> score ALL 216 states blind -> write raw scores -> hash
-> hard gates (T=9, finiteness, mean/sum rank + full-order
identity, exact challenge structure re-derived from row
metadata) -> ABSOLUTE HELD-OUT primary (48 heldout-I1, McNemar)
-> MATCHED TRANSFER primary (48 strict blocks, exact sign test
on q) -> mechanically assign the frozen joint verdict
(STRONG-FACTOR / STRONG-HASH / MIXED/INCONCLUSIVE) -> write
primary receipt -> ONLY THEN calibration/family/term/robustness/
bridge/inspectable riders (separate files). No per-state console
output during scoring.

LABEL SEPARATION (hard law): is_label is consumed ONLY after
candidate scores exist (rank bookkeeping). Continuations are
built solely from representation fields — CANONICAL
program_text+EOS, FACTOR factor_code+EOS, HASH hash_code+EOS —
in the stored candidate order, identical across arms.

Sign test = mcnemar_exact(q_pos, q_neg): the frozen svpadj
primitive IS the registered exact binomial two-sided test
(n = nonzero q, success = q > 0, p0 = 1/2).

Outputs under logs/mathworld1/svpchalscore/ (refuse-if-exists):
scores.jsonl, svpchalscore_receipt.json, riders.json,
inspectable_sets.json.

    .venv/bin/python scratch/mathworld1_svpchalscore.py       (Mac)
"""
import hashlib
import json
import sys
import time
from collections import Counter
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

CHAL = {
    "logs/mathworld1/svpchal2/decisions.jsonl":
        "1e3a5ef0483a7ee4970aa84c1f3d1dcc9171b8376cc6281e81d2"
        "b301a8f80d69",
    "logs/mathworld1/svpchal2/blocks.jsonl":
        "a50da492b2447b3bd10fd0409dd17d3d77ca57b4b4accbc6a703"
        "880c34a57fea",
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
ARMS = ["CANONICAL", "FACTOR", "HASH"]
OUTDIR = Path("logs/mathworld1/svpchalscore")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # 1. verify all frozen pins
    for p, h in CHAL.items():
        gate(fsha(p) == h, f"CHALLENGE PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha("checkpoints/svp_fh_init_s12001.pt") == INIT_SHA,
         "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpchalscore.py",
         "scratch/mathworld1_svpchal2.py",
         "scratch/mathworld1_svpchal.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    rows = [json.loads(l) for l in
            open("logs/mathworld1/svpchal2/decisions.jsonl")]
    # frozen structure re-derived from row metadata
    gate(len(rows) == 216, f"ROWS {len(rows)}")
    roles = Counter(r["site_role"] for r in rows)
    gate(roles == Counter({"heldout-I1": 48, "covered-I0": 48,
                           "control-I1": 24, "control-I0": 24,
                           "robustness-I1": 72}),
         f"ROLE CENSUS {dict(roles)}")
    gate(sum(r["primary"] for r in rows) == 144, "PRIMARY N")
    blocks = {}
    for r in rows:
        blocks.setdefault(r["block_id"], []).append(r)
    gate(len(blocks) == 72, f"BLOCKS {len(blocks)}")
    gate(all(len(v) == 3 for v in blocks.values()), "BLOCK SIZE")
    for bid, v in blocks.items():
        gate(len({x["term_cell"] for x in v}) == 1,
             f"BLOCK TERM HETEROGENEOUS {bid}")
        gate(len({x["family"] for x in v}) == 1,
             f"BLOCK FAMILY HETEROGENEOUS {bid}")
        gate(len({x["site_role"] for x in v}) == 3,
             f"BLOCK ROLE DUP {bid}")
    strict = {bid: v for bid, v in blocks.items()
              if v[0]["term_cell"] in (2, 3)}
    gate(len(strict) == 48, "STRICT BLOCKS")
    gate(Counter(v[0]["family"] for v in strict.values())
         == Counter({"CH-F3": 24, "CH-F4": 24}), "STRICT FAM")
    gate(Counter(v[0]["term_cell"] for v in strict.values())
         == Counter({2: 24, 3: 24}), "STRICT TERM")
    gate(len({r["cur"] for r in rows}) == 216, "CUR DUP")
    for r in rows:
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

    # 3. score ALL 216 states blind (label used only after
    # scores exist, for rank bookkeeping)
    t0 = time.time()
    recs = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        conts = {
            "CANONICAL": [TOK.encode(c["program_text"])
                          + [TOK.eos_id] for c in cands],
            "FACTOR": [[CODE_BASE + s for s in c["factor_code"]]
                       + [TOK.eos_id] for c in cands],
            "HASH": [[CODE_BASE + s for s in c["hash_code"]]
                     + [TOK.eos_id] for c in cands]}
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        rec = {"block_id": r["block_id"],
               "base_signature": r["base_signature"],
               "family": r["family"], "term": r["term_cell"],
               "site_role": r["site_role"],
               "primary": r["primary"],
               "distractor": r["distractor"],
               "cur": r["cur"],
               "n_candidates": len(cands),
               "label_index": li,
               "labeled_tuple": list(ctup(cands[li])),
               "candidate_tuples": [list(ctup(c))
                                    for c in cands],
               "exec_order": ARMS}
        for a in ARMS:
            triples = score_decision(
                arms[a], dev, r["cur"], conts[a])
            means = [t[0] for t in triples]
            sums = [t[1] for t in triples]
            top1, rank = rank_metrics(means, li)
            rec[a] = {"mean_lp": means, "sum_lp": sums,
                      "T": [t[2] for t in triples],
                      "top1": top1, "rank": rank}
        recs.append(rec)
    # 4-5. write + hash raw scores
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "scores.jsonl")) == 216,
         "DISK ROWS")

    # 6. hard gates
    n_t9 = n_rank_id = n_order_id = 0
    for rec in recs:
        for a in ("FACTOR", "HASH"):
            gate(all(t == 9 for t in rec[a]["T"]), f"T!=9 {a}")
            n_t9 += len(rec[a]["T"])
            li = rec["label_index"]
            gate(rank_metrics(rec[a]["mean_lp"], li)
                 == rank_metrics(rec[a]["sum_lp"], li),
                 f"MEAN!=SUM RANK {a}")
            n_rank_id += 1
            mo = sorted(range(len(rec[a]["mean_lp"])),
                        key=lambda i: (-rec[a]["mean_lp"][i], i))
            so = sorted(range(len(rec[a]["sum_lp"])),
                        key=lambda i: (-rec[a]["sum_lp"][i], i))
            gate(mo == so, f"ORDER MISMATCH {a}")
            n_order_id += 1
        for a in ARMS:
            gate(all(isinstance(x, float) and x == x
                     for x in rec[a]["mean_lp"]), "SCORE TYPE")

    # 7. ABSOLUTE HELD-OUT primary (48 heldout-I1)
    held = [d for d in recs if d["site_role"] == "heldout-I1"]
    gate(len(held) == 48, "HELD N")
    f1 = sum(1 for d in held if d["FACTOR"]["top1"])
    h1 = sum(1 for d in held if d["HASH"]["top1"])
    fb = sum(1 for d in held
             if d["FACTOR"]["top1"] and not d["HASH"]["top1"])
    fc = sum(1 for d in held
             if d["HASH"]["top1"] and not d["FACTOR"]["top1"])
    p_mcnemar = mcnemar_exact(fb, fc)
    if f1 > h1 and p_mcnemar < 0.05:
        held_verdict = "HELDOUT-FACTOR-WIN"
    elif h1 > f1 and p_mcnemar < 0.05:
        held_verdict = "HELDOUT-HASH-WIN"
    else:
        held_verdict = "HELDOUT-INCONCLUSIVE"
    mrr = {a: sum(1.0 / d[a]["rank"] for d in held) / 48
           for a in ARMS}

    # 8. MATCHED TRANSFER primary (48 strict blocks)
    by_role = {}
    for d in recs:
        by_role[(d["block_id"], d["site_role"])] = d
    gate(len(by_role) == 216, "BY_ROLE COLLISION")
    qrows = []
    for bid, v in sorted(strict.items()):
        c0 = by_role[(bid, "covered-I0")]
        c1 = by_role[(bid, "heldout-I1")]
        row = {"block_id": bid,
               "base_signature": v[0]["base_signature"],
               "family": v[0]["family"],
               "term": v[0]["term_cell"]}
        for a in ("FACTOR", "HASH"):
            row[f"c0_{a[0]}"] = int(c0[a]["top1"])
            row[f"c1_{a[0]}"] = int(c1[a]["top1"])
            row[f"drop_{a[0]}"] = (row[f"c0_{a[0]}"]
                                   - row[f"c1_{a[0]}"])
        row["q"] = row["drop_H"] - row["drop_F"]
        qrows.append(row)
    q_pos = sum(1 for r in qrows if r["q"] > 0)
    q_neg = sum(1 for r in qrows if r["q"] < 0)
    q_zero = sum(1 for r in qrows if r["q"] == 0)
    p_sign = mcnemar_exact(q_pos, q_neg)

    # 9. frozen joint verdict (mechanical)
    if (held_verdict == "HELDOUT-FACTOR-WIN"
            and p_sign < 0.05 and q_pos > q_neg):
        joint = "STRONG-FACTOR"
    elif (held_verdict == "HELDOUT-HASH-WIN"
            and p_sign < 0.05 and q_neg > q_pos):
        joint = "STRONG-HASH"
    else:
        joint = "MIXED/INCONCLUSIVE"
    trans = {}
    for a in ("FACTOR", "HASH"):
        trans[a] = {
            "cc": sum(1 for r in qrows
                      if r[f"c0_{a[0]}"] and r[f"c1_{a[0]}"]),
            "cw": sum(1 for r in qrows
                      if r[f"c0_{a[0]}"]
                      and not r[f"c1_{a[0]}"]),
            "wc": sum(1 for r in qrows
                      if not r[f"c0_{a[0]}"]
                      and r[f"c1_{a[0]}"]),
            "ww": sum(1 for r in qrows
                      if not r[f"c0_{a[0]}"]
                      and not r[f"c1_{a[0]}"])}

    # post-run pin re-gate (mid-run mutation halts, not records)
    for pth, h in CHAL.items():
        gate(fsha(pth) == h, f"POST-RUN PIN {pth}")
    for a, (pth, h) in CKPTS.items():
        gate(fsha(pth) == h, f"POST-RUN CKPT PIN {a}")
    gate(fsha("checkpoints/svp_fh_init_s12001.pt") == INIT_SHA,
         "POST-RUN INIT PIN")
    # 10. write primary receipt BEFORE any rider inspection
    receipt = {
        "device": str(dev),
        "absolute_heldout": {
            "n": 48, "FACTOR_top1": f1, "HASH_top1": h1,
            "discordant_factor_only": fb,
            "discordant_hash_only": fc,
            "mcnemar_exact_two_sided_p": p_mcnemar,
            "alpha": 0.05, "verdict": held_verdict,
            "MRR": {a: mrr[a] for a in ("FACTOR", "HASH")}},
        "matched_transfer": {
            "n_blocks": 48, "q_pos": q_pos, "q_neg": q_neg,
            "q_zero": q_zero,
            "sign_exact_two_sided_p": p_sign,
            "alpha": 0.05,
            "transition_anatomy": trans,
            "note": "q>0 favors FACTOR (HASH drops more)"},
        "joint_verdict": joint,
        "gates": {"T9_candidates_checked": n_t9,
                  "rank_identity_checks": n_rank_id,
                  "full_order_identity_checks": n_order_id,
                  "rederived_structure": {
                      "rows": len(rows),
                      "primary": sum(r["primary"]
                                     for r in rows),
                      "role_census": dict(roles),
                      "blocks": len(blocks),
                      "strict_blocks": len(strict),
                      "strict_by_family": dict(Counter(
                          v[0]["family"]
                          for v in strict.values())),
                      "strict_by_term": dict(Counter(
                          v[0]["term_cell"]
                          for v in strict.values()))}},
        "wall_s": round(time.time() - t0, 1),
        "files": {"scores.jsonl": raw_sha},
        "pins": {p: fsha(p) for p in CHAL} | {
            p: fsha(p) for p, _ in CKPTS.values()} | {
            "checkpoints/svp_fh_init_s12001.pt":
                fsha("checkpoints/svp_fh_init_s12001.pt")},
        "start": START, "completion_commit": completion_commit()}
    (OUTDIR / "svpchalscore_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("absolute_heldout", "matched_transfer",
                       "joint_verdict")}, indent=1), flush=True)

    # 11. riders AFTER the primary receipt is frozen
    def top1s(sub):
        return {"n": len(sub), **{
            f"{a}_top1": sum(1 for d in sub if d[a]["top1"])
            for a in ARMS}}

    def calib(fam=None):
        sub = [v for v in blocks.values()
               if v[0]["term_cell"] == 1
               and (fam is None or v[0]["family"] == fam)]
        out = {"n_blocks": len(sub)}
        for a in ARMS:
            cc = cw = wc = ww = 0
            for v in sub:
                bid = v[0]["block_id"]
                a0 = by_role[(bid, "control-I0")][a]["top1"]
                a1 = by_role[(bid, "control-I1")][a]["top1"]
                cc += a0 and a1
                cw += a0 and not a1
                wc += (not a0) and a1
                ww += (not a0) and (not a1)
            out[a] = {"cc": cc, "cw": cw, "wc": wc, "ww": ww,
                      "drop": cw - wc}
        return out

    robust = [d for d in recs
              if d["site_role"] == "robustness-I1"
              and d["term"] in (2, 3)]
    riders = {
        "t1_calibration": {"overall": calib(),
                           "CH-F3": calib("CH-F3"),
                           "CH-F4": calib("CH-F4")},
        "heldout_family_term": {
            f"{f} t{t}": top1s([d for d in held
                                if d["family"] == f
                                and d["term"] == t])
            for f in ("CH-F3", "CH-F4") for t in (2, 3)},
        "heldout_pooled": {
            "t2": top1s([d for d in held if d["term"] == 2]),
            "t3": top1s([d for d in held if d["term"] == 3]),
            "CH-F3": top1s([d for d in held
                            if d["family"] == "CH-F3"]),
            "CH-F4": top1s([d for d in held
                            if d["family"] == "CH-F4"])},
        "covered_I0_strict": top1s(
            [d for d in recs
             if d["site_role"] == "covered-I0"]),
        "robustness_strict_I1": top1s(robust),
        "robustness_by_distractor": {
            ds: top1s([d for d in robust
                       if d["distractor"] == ds])
            for ds in sorted({d["distractor"]
                              for d in robust})},
        "heldout_primary_by_distractor": {
            ds: top1s([d for d in held
                       if d["distractor"] == ds])
            for ds in sorted({d["distractor"] for d in held})},
        "canonical_bridge": {
            "heldout": top1s(held)["CANONICAL_top1"],
            "covered_I0_strict": top1s(
                [d for d in recs if d["site_role"]
                 == "covered-I0"])["CANONICAL_top1"],
            "MRR_heldout": mrr["CANONICAL"],
            "transfer_drop": sum(
                (by_role[(b, "covered-I0")]["CANONICAL"]["top1"]
                 - by_role[(b, "heldout-I1")]["CANONICAL"][
                     "top1"]) for b in strict)},
    }

    def keyrec(d):
        li = d["label_index"]
        out = {"block_id": d["block_id"],
               "base_signature": d["base_signature"],
               "family": d["family"], "term": d["term"],
               "cur": d["cur"],
               "labeled_tuple": d["labeled_tuple"],
               "candidate_tuples": d["candidate_tuples"],
               "n_candidates": d["n_candidates"]}
        for a in ("FACTOR", "HASH"):
            ml = d[a]["mean_lp"]
            riv = [x for j, x in enumerate(ml) if j != li]
            out[f"{a}_label_margin"] = round(
                ml[li] - max(riv), 6) if riv else None
        return out

    def block_pair(r):
        bid = r["block_id"]
        return {**r,
                "covered_cur": by_role[(bid,
                                        "covered-I0")]["cur"],
                "heldout_cur": by_role[(bid,
                                        "heldout-I1")]["cur"]}

    insp = {
        "heldout_F_correct_H_wrong": [
            keyrec(d) for d in held
            if d["FACTOR"]["top1"] and not d["HASH"]["top1"]],
        "heldout_H_correct_F_wrong": [
            keyrec(d) for d in held
            if d["HASH"]["top1"] and not d["FACTOR"]["top1"]],
        "heldout_both_wrong": [
            keyrec(d) for d in held
            if not d["FACTOR"]["top1"]
            and not d["HASH"]["top1"]],
        "heldout_both_correct_n": sum(
            1 for d in held
            if d["FACTOR"]["top1"] and d["HASH"]["top1"]),
        "q_positive_blocks": [block_pair(r) for r in qrows
                              if r["q"] > 0],
        "q_negative_blocks": [block_pair(r) for r in qrows
                              if r["q"] < 0],
        "q_rows_all": qrows}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    (OUTDIR / "inspectable_sets.json").write_text(
        json.dumps(insp, indent=1))
    print(json.dumps({"riders": riders}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
