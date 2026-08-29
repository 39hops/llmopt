"""MATH-CYBER-1 SVP-GRID-HELDOUT-SCORE-17001 — ONE joint strict
heldout scoring run of the two sealed seed-17001 checkpoints on
the frozen heldout_test16 artifact, under CALIBRATION-FIRED
authority verified mechanically from the frozen calibration
receipt before the heldout bytes are opened. The secondary P-OUT
robustness artifact is neither named by path nor read anywhere in
this file. Zero training, zero checkpoint mutation, zero sympy,
zero candidate regeneration; frozen bytes only.

Frozen anti-peeking order: pins + FIRE authority -> heldout bytes
-> structure/count/roundtrip gates -> load models -> blind-score
the 96 heldout-I1 PRIMARY states -> persist raw scores -> hash ->
hard gates -> PRIMARY endpoint 1 (absolute exact McNemar, n=96)
-> PRIMARY endpoint 2 (matched transfer: join each block to its
FROZEN calibration score row by block_id; calibration is NOT
rescored; q = drop_H - drop_F, exact two-sided sign test over
nonzero q; positive q favors FACTOR) -> mechanical joint verdict:
  STRONG-FACTOR iff HELDOUT-FACTOR-WIN (F>H and McNemar p<.05)
    AND sign p<.05 AND q_positive>q_negative
  STRONG-HASH symmetric; otherwise MIXED/INCONCLUSIVE
-> frozen replication-status mapping (calibration FIRED is
gated at entry): STRONG-FACTOR => REPLICATES-STRONG-FACTOR;
else F top1 > H top1 => REPLICATES-DIRECTION-ONLY; else
FAILS-REPLICATION. No pooling with seed 16001 anywhere
-> primary receipt written -> ONLY THEN the 96 robustness-I1
rider rows are scored (never in any primary denominator) and the
descriptive anatomy/riders emitted. The primary denominator is
all 96 frozen blocks, including the three FACTOR calibration-miss
blocks. No per-state top1 is printed before the raw score file is
closed and hashed.

LABEL SEPARATION (hard law): is_label is consumed for RANKING
only after both arms' candidate scores exist; a pre-load
structure gate also reads it for a cardinality/semantics check
whose result never reaches prompts, continuations, candidate
order, or scoring. Continuations are built solely from
stored factor_code / hash_code + EOS in stored candidate order,
identical across arms. Standing prompt
"Current: {cur}\\nHints: none\\nStep: ".

Outputs under logs/mathworld1/svpheldout17/ (refuse-if-exists):
primary_scores.jsonl, matched_transfer.jsonl,
svpheldout17_receipt.json, robustness_scores.jsonl, riders.json.

    .venv/bin/python scratch/mathworld1_svpheldout17.py      (Mac)
"""
import hashlib
import json
import math
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
from scratch.mathworld1_svpadj import (rank_metrics,  # noqa: E402
                                       score_decision)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        hash_decode)

HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
CAL_SCORES = "logs/mathworld1/svpcalscore17/scores.jsonl"
CAL_RECEIPT = ("logs/mathworld1/svpcalscore17/"
               "svpcalscore17_receipt.json")
PINS = {
    HELD:
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
    CAL_SCORES:
        "74b2bad6dc5c020a3687b12e96942655bc14d4346d165a090cce"
        "cede66291348",
    CAL_RECEIPT:
        "ec0b3285a9fbb737b476461f8bad762677f8f6c9d23477d6aebb"
        "5631acd93b8d",
    "logs/mathworld1/svpdiet3/covered_calibration.jsonl":
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b0850979"
        "4d0fb7978db3",
    "logs/mathworld1/svpdiet3/svpdiet3_receipt.json":
        "26cb6d0119f56e24b4025d43976ddf323a5540e0177c19583bfe"
        "2f5c984fb365",
    "logs/mathworld1/svpgbirth_s17001_receipt.json":
        "fb607dd5328973eca7dec79697449bd00ab1099f5351880ae400"
        "25b9704f4833",
}
CKPTS = {
    "FACTOR": ("checkpoints/svp_grid_factor_s17001.pt",
               "12e19fae6fe74b0b5d10fd41ce95192fe324a1f7d466da"
               "c8d713d29eb0db882c"),
    "HASH": ("checkpoints/svp_grid_hash_s17001.pt",
             "e24237b86314a3ba468f8f213d18f5d9300f4964eec530ea"
             "bfd51fb625f163d6"),
}
INIT_SHA = ("4384ed9800962f1af24a87e5901a1e348ff982532ac8250c"
            "1aafab983abbbc1a")
INIT_CK = "checkpoints/svp_grid_init_s17001.pt"
VOCAB = 340
CODE_BASE = 332
ARMS = ["FACTOR", "HASH"]
ALPHA = 0.05
OUTDIR = Path("logs/mathworld1/svpheldout17")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def binom_minlik_p(k_obs, n):
    """Exact two-sided p under Binomial(n, 0.5): sum pmf(k) over
    all k whose pmf <= pmf(k_obs) (min-likelihood convention)."""
    if n == 0:
        return 1.0
    pmf = [math.comb(n, k) / 2.0 ** n for k in range(n + 1)]
    thresh = pmf[k_obs] * (1 + 1e-12)
    return min(1.0, sum(p for p in pmf if p <= thresh))


def score_rows(rows, arms, dev):
    """Blind-score rows for both arms; label consumed only after
    both arms' scores exist for a state."""
    recs = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        conts = {
            "FACTOR": [[CODE_BASE + s for s in c["factor_code"]]
                       + [TOK.eos_id] for c in cands],
            "HASH": [[CODE_BASE + s for s in c["hash_code"]]
                     + [TOK.eos_id] for c in cands]}
        rec = {"block_id": r["block_id"],
               "base_signature": r["base_signature"],
               "term": r["term_cell"], "regime": r["regime"],
               "P": r["P"], "c": r["c"],
               "site_role": r["site_role"],
               "block_d_before": r["block_d_before"],
               "cur": r["cur"],
               "n_candidates": len(cands),
               "exec_order": ARMS}
        raw = {}
        for a in ARMS:
            raw[a] = score_decision(arms[a], dev, r["cur"],
                                    conts[a])
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        rec["label_index"] = li
        rec["labeled_tuple"] = list(ctup(cands[li]))
        for a in ARMS:
            means = [t[0] for t in raw[a]]
            sums = [t[1] for t in raw[a]]
            top1, rank = rank_metrics(means, li)
            rec[a] = {"mean_lp": means, "sum_lp": sums,
                      "T": [t[2] for t in raw[a]],
                      "top1": top1, "rank": rank}
        recs.append(rec)
    return recs


def hard_gates(recs):
    n_t9 = n_rank_id = n_order_id = 0
    for rec in recs:
        for a in ARMS:
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
            gate(all(isinstance(x, float) and math.isfinite(x)
                     for x in rec[a]["mean_lp"]
                     + rec[a]["sum_lp"]), "SCORE FINITE")
    return n_t9, n_rank_id, n_order_id


def structure_gates(rows, role, n_expect):
    gate(len(rows) == n_expect, f"{role} ROWS {len(rows)}")
    gate(all(r["site_role"] == role for r in rows), "ROLE")
    gate(Counter((r["term_cell"], r["regime"]) for r in rows)
         == Counter({(2, "IN"): 24, (2, "OUT"): 24,
                     (3, "IN"): 24, (3, "OUT"): 24}),
         f"{role} STRATA")
    gate(len({r["cur"] for r in rows}) == n_expect,
         f"{role} CUR DUP")
    for r in rows:
        labs = [c for c in r["candidates"] if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        lt = ctup(labs[0])
        gate(lt[0] == "i_unprod" and lt[1] == "I"
             and lt[2] == 1, f"LABEL SEMANTICS {lt}")
        gate(r["term_cell"] in (2, 3), "TERM CELL")
        for c in r["candidates"]:
            t = ctup(c)
            gate(factor_decode(c["factor_code"]) == t, "F RT")
            gate(hash_decode(c["hash_code"]) == t, "H RT")


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # 1. pins + calibration-FIRED authority BEFORE heldout opens
    cal_receipt = json.loads(Path(CAL_RECEIPT).read_text())
    gate(cal_receipt["verdict"] == "CALIBRATION FIRED",
         "NO HELDOUT AUTHORITY: calibration did not fire")
    gate(cal_receipt["raw_scores_sha"] == PINS[CAL_SCORES],
         "CAL RECEIPT/SCORES SHA MISMATCH")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpheldout17.py",
         "scratch/mathworld1_svpcalscore17.py",
         "scratch/mathworld1_svpdiet3.py",
         "scratch/mathworld1_svpgbirth17.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "scratch/mathworld1_actionsem.py",
         "scratch/mathworld1_srepr_export.py",
         "llmopt/search/derivation.py",
         "llmopt/search/rules.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])

    # 2-3. heldout bytes + structure gates
    allrows = [json.loads(l) for l in open(HELD)]
    gate(len(allrows) == 192, f"HELDOUT ROWS {len(allrows)}")
    pri = [r for r in allrows if r["site_role"] == "heldout-I1"]
    rob = [r for r in allrows
           if r["site_role"] == "robustness-I1"]
    gate(len(pri) + len(rob) == 192, "ROLE PARTITION")
    structure_gates(pri, "heldout-I1", 96)
    structure_gates(rob, "robustness-I1", 96)
    gate(all(r["primary"] and r["confirmatory_denominator"]
             for r in pri), "PRIMARY FLAGS")
    gate(all(not r["primary"]
             and not r["confirmatory_denominator"]
             for r in rob), "RIDER FLAGS")
    # one-to-one block match v the FROZEN calibration scores
    cal = [json.loads(l) for l in open(CAL_SCORES)]
    gate(len(cal) == 96, "CAL ROWS")
    calmap = {c["block_id"]: c for c in cal}
    gate(sorted(calmap) == sorted(r["block_id"] for r in pri),
         "BLOCK MATCH")
    for r in pri:
        c = calmap[r["block_id"]]
        gate(c["base_signature"] == r["base_signature"],
             "BASE SIG")
        gate(c["term"] == r["term_cell"], "TERM MATCH")
        gate(c["site_role"] == "covered-I0", "CAL ROLE")

    # 4. models
    gate(torch.backends.mps.is_available(), "MPS")
    dev = torch.device("mps")
    arms = {}
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(CKPTS[a][0],
                                     weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, f"PARAM COUNT {a}")
        m.eval()
        arms[a] = m.to(dev)

    # 5-7. blind-score PRIMARY, persist, hash
    t0 = time.time()
    recs = score_rows(pri, arms, dev)
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "primary_scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "primary_scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "primary_scores.jsonl"))
         == 96, "DISK ROWS")

    # 8. hard scoring gates
    n_t9, n_rank_id, n_order_id = hard_gates(recs)

    # 9. PRIMARY ENDPOINT 1: absolute heldout
    f_top = sum(1 for d in recs if d["FACTOR"]["top1"])
    h_top = sum(1 for d in recs if d["HASH"]["top1"])
    f_only = sum(1 for d in recs
                 if d["FACTOR"]["top1"]
                 and not d["HASH"]["top1"])
    h_only = sum(1 for d in recs
                 if d["HASH"]["top1"]
                 and not d["FACTOR"]["top1"])
    n_disc = f_only + h_only
    mcnemar_p = binom_minlik_p(f_only, n_disc)
    if f_top > h_top and mcnemar_p < ALPHA:
        ep1 = "HELDOUT-FACTOR-WIN"
    elif h_top > f_top and mcnemar_p < ALPHA:
        ep1 = "HELDOUT-HASH-WIN"
    else:
        ep1 = "HELDOUT-INCONCLUSIVE"
    mrr = {a: round(sum(1.0 / d[a]["rank"] for d in recs)
                    / len(recs), 4) for a in ARMS}

    # 10. PRIMARY ENDPOINT 2: matched transfer over ALL 96 blocks
    mt = []
    for d in recs:
        c = calmap[d["block_id"]]
        c0f, c0h = int(c["FACTOR"]["top1"]), int(c["HASH"]["top1"])
        c1f, c1h = int(d["FACTOR"]["top1"]), int(d["HASH"]["top1"])
        drop_f, drop_h = c0f - c1f, c0h - c1h
        mt.append({"block_id": d["block_id"],
                   "term": d["term"], "regime": d["regime"],
                   "c0_F": c0f, "c1_F": c1f,
                   "c0_H": c0h, "c1_H": c1h,
                   "drop_F": drop_f, "drop_H": drop_h,
                   "q": drop_h - drop_f})
    gate(len(mt) == 96, "MT DENOM")
    q_pos = sum(1 for m in mt if m["q"] > 0)
    q_neg = sum(1 for m in mt if m["q"] < 0)
    q_zero = sum(1 for m in mt if m["q"] == 0)
    sign_p = binom_minlik_p(q_pos, q_pos + q_neg)
    with open(OUTDIR / "matched_transfer.jsonl", "w") as fo:
        for m in mt:
            fo.write(json.dumps(m) + "\n")
    mt_sha = fsha(OUTDIR / "matched_transfer.jsonl")

    # 11. FROZEN JOINT VERDICT (mechanical)
    if (ep1 == "HELDOUT-FACTOR-WIN" and sign_p < ALPHA
            and q_pos > q_neg):
        joint = "STRONG-FACTOR"
    elif (ep1 == "HELDOUT-HASH-WIN" and sign_p < ALPHA
            and q_neg > q_pos):
        joint = "STRONG-HASH"
    else:
        joint = "MIXED/INCONCLUSIVE"
    # frozen replication-status mapping (REPLICATION-17001-
    # PREREG-0, cd6db7e3): calibration FIRED (gated at entry), so
    # REPLICATION-NOT-REACHED is unreachable here
    if joint == "STRONG-FACTOR":
        replication = "REPLICATES-STRONG-FACTOR"
    elif f_top > h_top:
        replication = "REPLICATES-DIRECTION-ONLY"
    else:
        replication = "FAILS-REPLICATION"

    # transition anatomy (required, descriptive)
    trans = {}
    for a in ARMS:
        t2 = Counter()
        for m in mt:
            key = (m[f"c0_{a[0]}"], m[f"c1_{a[0]}"])
            t2[{(1, 1): "correct->correct",
                (1, 0): "correct->wrong",
                (0, 1): "wrong->correct",
                (0, 0): "wrong->wrong"}[key]] += 1
        trans[a] = dict(t2)
    both_cc = [m for m in mt if m["c0_F"] == 1 and m["c0_H"] == 1]
    bcc = {"n_blocks": len(both_cc),
           "F_heldout_correct": sum(m["c1_F"] for m in both_cc),
           "H_heldout_correct": sum(m["c1_H"] for m in both_cc),
           "q_positive": sum(1 for m in both_cc if m["q"] > 0),
           "q_negative": sum(1 for m in both_cc if m["q"] < 0)}
    # mechanical baseline-asymmetry statement (the SCORE-0 floor
    # lesson): compares the full-denominator sign direction with
    # the both-covered-correct subset's direction
    c0f_miss = sum(1 for m in mt if m["c0_F"] == 0)
    c0h_miss = sum(1 for m in mt if m["c0_H"] == 0)
    full_dir = ("positive" if q_pos > q_neg else
                "negative" if q_neg > q_pos else "tied")
    sub_dir = ("positive" if bcc["q_positive"]
               > bcc["q_negative"] else
               "negative" if bcc["q_negative"]
               > bcc["q_positive"] else "tied")
    baseline_asym = {
        "c0_F_misses": c0f_miss, "c0_H_misses": c0h_miss,
        "q_direction_all_96": full_dir,
        "q_direction_both_covered_correct": sub_dir,
        "statement": (
            "sign direction UNCHANGED after removing covered-"
            "side baseline-miss blocks: not driven by baseline "
            "asymmetry" if full_dir == sub_dir else
            "sign direction CHANGES after removing covered-side "
            "baseline-miss blocks: baseline asymmetry is "
            "load-bearing")}

    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-HELDOUT-SCORE-17001",
        "joint_verdict": joint,
        "replication_status": replication,
        "endpoint1_absolute": {
            "n": len(recs), "FACTOR_top1": f_top,
            "HASH_top1": h_top,
            "F_only_discordant": f_only,
            "H_only_discordant": h_only,
            "n_discordant": n_disc,
            "mcnemar_p_two_sided": mcnemar_p,
            "label": ep1, "alpha": ALPHA,
            "mrr_descriptive": mrr},
        "endpoint2_matched_transfer": {
            "n_blocks": len(mt), "q_positive": q_pos,
            "q_negative": q_neg, "q_zero": q_zero,
            "sign_p_two_sided_nonzero_only": sign_p,
            "convention": "positive q favors FACTOR",
            "matched_transfer_sha": mt_sha},
        "transition_anatomy_descriptive": trans,
        "both_covered_correct_rider": bcc,
        "baseline_asymmetry_statement": baseline_asym,
        "raw_scores_sha": raw_sha,
        "calibration_authority": {
            "receipt_verdict": cal_receipt["verdict"],
            "raw_scores_sha_pinned": PINS[CAL_SCORES]},
        "gates": {"T9_candidates_checked": n_t9,
                  "rank_identity_checks": n_rank_id,
                  "full_order_identity_checks": n_order_id,
                  "rederived_structure": {
                      "heldout_rows": len(allrows),
                      "primary": {
                          f"t{t}|{rg}": n for (t, rg), n in
                          Counter((r["term_cell"], r["regime"])
                                  for r in pri).items()},
                      "rider_rows": len(rob),
                      "unique_curs": len(
                          {r["cur"] for r in allrows})}},
        "device": str(dev),
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_pins": {a: fsha(CKPTS[a][0]) for a in ARMS},
        "init_pin": fsha(INIT_CK),
        "start": START,
        "completion_commit": completion_commit()}
    # 12. post-run pin re-gates, freeze primary receipt
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    (OUTDIR / "svpheldout17_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("joint_verdict", "replication_status",
                       "endpoint1_absolute",
                       "endpoint2_matched_transfer",
                       "transition_anatomy_descriptive",
                       "both_covered_correct_rider",
                       "baseline_asymmetry_statement",
                       "wall_s")},
                     indent=1), flush=True)

    # 13. ONLY NOW: robustness rider + descriptive riders
    t1 = time.time()
    rob_recs = score_rows(rob, arms, dev)
    hard_gates(rob_recs)
    with open(OUTDIR / "robustness_scores.jsonl", "w") as fo:
        for rec in rob_recs:
            fo.write(json.dumps(rec) + "\n")

    def bucket(sel):
        return {a: sum(1 for d in sel if d[a]["top1"])
                for a in ARMS} | {"n": len(sel)}
    riders = {
        "primary_subgroups": {
            **{f"t{t}-{rg}": bucket(
                [d for d in recs
                 if d["term"] == t and d["regime"] == rg])
               for t in (2, 3) for rg in ("IN", "OUT")},
            "pooled_t2": bucket([d for d in recs
                                 if d["term"] == 2]),
            "pooled_t3": bucket([d for d in recs
                                 if d["term"] == 3]),
            "pooled_c-IN": bucket([d for d in recs
                                   if d["regime"] == "IN"]),
            "pooled_c-OUT": bucket([d for d in recs
                                    if d["regime"] == "OUT"])},
        "primary_by_P": {
            P: bucket([d for d in recs if d["P"] == P])
            for P in sorted({d["P"] for d in recs})},
        "primary_by_c": {
            str(c): bucket([d for d in recs if d["c"] == c])
            for c in sorted({d["c"] for d in recs})},
        "cand0": {
            "is_label": sum(1 for d in recs
                            if d["label_index"] == 0),
            "strict_top": {a: sum(
                1 for d in recs
                if len(d[a]["mean_lp"]) > 1
                and d[a]["mean_lp"][0]
                > max(d[a]["mean_lp"][1:])) for a in ARMS}},
        "legal_K": dict(Counter(
            d["n_candidates"] for d in recs)),
        "robustness_rider": {
            "n": len(rob_recs),
            "FACTOR_top1": sum(1 for d in rob_recs
                               if d["FACTOR"]["top1"]),
            "HASH_top1": sum(1 for d in rob_recs
                             if d["HASH"]["top1"]),
            "subgroups": {f"t{t}-{rg}": bucket(
                [d for d in rob_recs
                 if d["term"] == t and d["regime"] == rg])
                for t in (2, 3) for rg in ("IN", "OUT")},
            "mrr": {a: round(sum(
                1.0 / d[a]["rank"] for d in rob_recs)
                / len(rob_recs), 4) for a in ARMS},
            "wall_s": round(time.time() - t1, 1)}}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svpheldout17] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
