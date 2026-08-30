"""MATH-CYBER-1 SVP-GRID-KNOWN-SET-PERMUTATION-HELDOUT-SCORE-
18001 — ONE joint three-arm heldout scoring run of the sealed
seed-18001 checkpoints (FACTOR / HASH-P1 / HASH-P2) on the
frozen heldout_test16 artifact, under CALIBRATION-FIRED
authority verified mechanically from the frozen three-arm
calibration receipt before the heldout bytes open. The
secondary P-OUT artifact is neither named by path nor read.
Zero training, zero checkpoint mutation, zero sympy, zero
candidate regeneration; frozen bytes + the immutable P2 law
only.

Frozen anti-peeking order: FIRE authority -> pins -> P2
realization re-derivation (gated == the qualified pin) ->
heldout bytes -> structure/count/roundtrip gates -> load three
checkpoints -> blind-score the 96 heldout-I1 PRIMARY states for
ALL THREE arms -> persist raw scores -> hash -> hard gates ->
PAIRWISE PRIMARY 1 (F v H1: 96-state exact McNemar + 96-block
matched transfer v the FROZEN seed-18001 calibration raw scores
+ the frozen directional conjunction) -> PAIRWISE PRIMARY 2
(F v H2, identical law, independently adjudicated; H2's four
calibration-miss blocks stay in) -> mechanical IUT class label:
  both STRONG-FACTOR => KNOWN-SET-PERMUTATION-ROBUST-STRONG-FACTOR
  F-v-H1 only        => KNOWN-SET-P1-ONLY-STRONG-FACTOR
  F-v-H2 only        => KNOWN-SET-P2-ONLY-STRONG-FACTOR
  neither            => KNOWN-SET-NO-ROBUST-STRONG-FACTOR
-> PRIMARY receipt written -> ONLY THEN the H1-v-H2 sensitivity
(PERMUTATION-SENSITIVITY-MEASURED; cannot alter the class
verdict), the 96 robustness-I1 rider rows, and the descriptive
anatomy riders (per-cell counts, pooled splits, candidate-0
census, per-P, 4-cell gap vectors + preregistered n=4 Spearman
v the booked 16001/17001 F-H1 vectors). H1 and H2 rows are
never pooled; no 192-state HASH comparison exists. No per-state
top1 is printed before the raw score file is closed and hashed.

LABEL SEPARATION (hard law): is_label is consumed for RANKING
only after all three arms' candidate scores exist; a pre-load
structure gate also reads it for a cardinality/semantics check
whose result never reaches prompts, continuations, candidate
order, or scoring. Continuations: FACTOR from stored
factor_code, HASH-P1 from stored hash_code, HASH-P2 DERIVED per
candidate from its semantic tuple via the immutable qualified
P2 law (hash2_decode roundtrip gated) — + EOS, stored candidate
order, identical across arms. Standing prompt
"Current: {cur}\\nHints: none\\nStep: ".

Outputs under logs/mathworld1/svpheldout18/ (refuse-if-exists):
primary_scores.jsonl, matched_transfer.jsonl,
svpheldout18_receipt.json, robustness_scores.jsonl, riders.json.

    .venv/bin/python scratch/mathworld1_svpheldout18.py      (Mac)
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
from scratch.mathworld1_actiontok import (ActionGCTok,  # noqa: E402
                                          OPCODE_ORDER)
from scratch.mathworld1_svpadj import (rank_metrics,  # noqa: E402
                                       score_decision)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (ORD_MAX,  # noqa: E402
                                        factor_decode,
                                        hash_decode)
from scratch.mathworld1_svpp2qual import (hash2_decode,  # noqa: E402
                                          hash2_symbols)

HELD = "logs/mathworld1/svpdiet3/heldout_test16.jsonl"
CAL_SCORES = "logs/mathworld1/svpcalscore18/scores.jsonl"
CAL_RECEIPT = ("logs/mathworld1/svpcalscore18/"
               "svpcalscore18_receipt.json")
PINS = {
    HELD:
        "a3f6103b3733d909281849dcb3fd6ba9fba3891f2014bec13881"
        "b4509df46ddb",
    CAL_SCORES:
        "c5af7e8f8aa6b788d471944022751a5ed6422edd1e89f000a522"
        "e8179b78a87a",
    CAL_RECEIPT:
        "3a56bd51b194a0b4ddefff777df1923572c5fded630fe8162851"
        "e403d6d968bf",
    "logs/mathworld1/svpdiet3/covered_calibration.jsonl":
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b0850979"
        "4d0fb7978db3",
    "logs/mathworld1/svpdiet3/svpdiet3_receipt.json":
        "26cb6d0119f56e24b4025d43976ddf323a5540e0177c19583bfe"
        "2f5c984fb365",
    "logs/mathworld1/svpgbirth_s18001_receipt.json":
        "d6e4ee997780c16df970cd58ee0b2901000875d5c78ab278ee3c"
        "019b45fee659",
    "logs/mathworld1/svpp2qual/svpp2qual_receipt.json":
        "47309f22e2be3fba57a28ea0937c985d9ae616121b4cfff608c1"
        "9371246c2337",
}
CKPTS = {
    "FACTOR": ("checkpoints/svp_grid_factor_s18001.pt",
               "ecf5be31f5f7a09d16f0f9a00217983ef2c34a76cce470"
               "6332ca64a3339fcf5a"),
    "HASH_P1": ("checkpoints/svp_grid_hashp1_s18001.pt",
                "1ec2ea437737f75e57d754c992a75156f982614fb956"
                "9e50385d4186751cbaac"),
    "HASH_P2": ("checkpoints/svp_grid_hashp2_s18001.pt",
                "26e4afc35870df25a6381f68e65d90280e891399ce4c"
                "639fd5c977644b8ee844"),
}
INIT_SHA = ("a7bb5b8839e78560b6648f7471c03827796309c990514fd4"
            "bdce949b00299fc4")
INIT_CK = "checkpoints/svp_grid_init_s18001.pt"
P2_REALIZATION_SHA = (
    "952f332da4e25961b2dd52c786902e74ba4b33bbf8413f88496a"
    "0df952450ba9")
VOCAB = 340
CODE_BASE = 332
ARMS = ["FACTOR", "HASH_P1", "HASH_P2"]
HASHES = ["HASH_P1", "HASH_P2"]
ALPHA = 0.05
OUTDIR = Path("logs/mathworld1/svpheldout18")
TOK = ActionGCTok()
# booked prior-realization 4-cell F-H1 gap vectors over
# (t2-IN, t2-OUT, t3-IN, t3-OUT), from the frozen riders of
# HELDOUT-SCORE-16001 and -17001 (literals quoted from booked
# receipts; used ONLY in the preregistered descriptive Spearman
# rider, never in any gate):
PRIOR_GAPS = {"16001_F_H1": [4, 4, 19, 16],
              "17001_F_H1": [-7, 6, 10, 3]}


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


def spearman(a, b):
    """Spearman rho with average ranks (n=4 descriptive)."""
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(s):
            j = i
            while j + 1 < len(s) and v[s[j + 1]] == v[s[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[s[k]] = avg
            i = j + 1
        return r
    ra, rb = ranks(a), ranks(b)
    ma = sum(ra) / len(ra)
    mb = sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    da = math.sqrt(sum((x - ma) ** 2 for x in ra))
    db = math.sqrt(sum((y - mb) ** 2 for y in rb))
    if da == 0 or db == 0:
        return None
    return round(num / (da * db), 4)


def rederive_p2_realization_sha():
    stream = hashlib.sha256()
    for r in OPCODE_ORDER:
        for sk in ("W", "I"):
            for so in range(-1, ORD_MAX + 1):
                for pk in ("none", "u_choice", "term_index"):
                    for pi in range(-1, ORD_MAX + 1):
                        stream.update(bytes(
                            hash2_symbols(r, sk, so, pk, pi)))
    return stream.hexdigest()


def score_rows(rows, arms, dev):
    """Blind-score rows for all three arms; label consumed only
    after every arm's scores exist for a state."""
    recs = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        conts = {
            "FACTOR": [[CODE_BASE + s for s in c["factor_code"]]
                       + [TOK.eos_id] for c in cands],
            "HASH_P1": [[CODE_BASE + s for s in c["hash_code"]]
                        + [TOK.eos_id] for c in cands],
            "HASH_P2": [[CODE_BASE + s for s in c["_h2"]]
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
            gate(hash_decode(c["hash_code"]) == t, "P1 RT")
            h2 = hash2_symbols(*t)
            gate(hash2_decode(h2) == t, "P2 RT")
            c["_h2"] = h2


def pairwise(recs, calmap, hx):
    """The frozen joint law for FACTOR v one HASH arm."""
    f_top = sum(1 for d in recs if d["FACTOR"]["top1"])
    h_top = sum(1 for d in recs if d[hx]["top1"])
    f_only = sum(1 for d in recs
                 if d["FACTOR"]["top1"] and not d[hx]["top1"])
    h_only = sum(1 for d in recs
                 if d[hx]["top1"] and not d["FACTOR"]["top1"])
    n_disc = f_only + h_only
    mcnemar_p = binom_minlik_p(f_only, n_disc)
    if f_top > h_top and mcnemar_p < ALPHA:
        ep1 = "HELDOUT-FACTOR-WIN"
    elif h_top > f_top and mcnemar_p < ALPHA:
        ep1 = "HELDOUT-HASH-WIN"
    else:
        ep1 = "HELDOUT-INCONCLUSIVE"
    mt = []
    for d in recs:
        c = calmap[d["block_id"]]
        c0f, c1f = int(c["FACTOR"]["top1"]), int(d["FACTOR"]["top1"])
        c0h, c1h = int(c[hx]["top1"]), int(d[hx]["top1"])
        q = (c0h - c1h) - (c0f - c1f)
        mt.append({"block_id": d["block_id"], "hx": hx,
                   "term": d["term"], "regime": d["regime"],
                   "c0_F": c0f, "c1_F": c1f,
                   "c0_H": c0h, "c1_H": c1h, "q": q})
    gate(len(mt) == 96, "MT DENOM")
    q_pos = sum(1 for m in mt if m["q"] > 0)
    q_neg = sum(1 for m in mt if m["q"] < 0)
    q_zero = sum(1 for m in mt if m["q"] == 0)
    sign_p = binom_minlik_p(q_pos, q_pos + q_neg)
    if (ep1 == "HELDOUT-FACTOR-WIN" and sign_p < ALPHA
            and q_pos > q_neg):
        verdict = "STRONG-FACTOR"
    elif (ep1 == "HELDOUT-HASH-WIN" and sign_p < ALPHA
            and q_neg > q_pos):
        verdict = "STRONG-HASH"
    else:
        verdict = "MIXED/INCONCLUSIVE"
    trans = {}
    for arm, tag in (("FACTOR", "F"), (hx, "H")):
        t2 = Counter()
        for m in mt:
            key = (m[f"c0_{tag}"], m[f"c1_{tag}"])
            t2[{(1, 1): "correct->correct",
                (1, 0): "correct->wrong",
                (0, 1): "wrong->correct",
                (0, 0): "wrong->wrong"}[key]] += 1
        trans[arm] = dict(t2)
    both_cc = [m for m in mt if m["c0_F"] == 1
               and m["c0_H"] == 1]
    bcc = {"n_blocks": len(both_cc),
           "F_heldout_correct": sum(m["c1_F"] for m in both_cc),
           "Hx_heldout_correct": sum(m["c1_H"]
                                     for m in both_cc),
           "q_positive": sum(1 for m in both_cc if m["q"] > 0),
           "q_negative": sum(1 for m in both_cc if m["q"] < 0)}
    return {
        "hx": hx,
        "endpoint1_absolute": {
            "n": len(recs), "FACTOR_top1": f_top,
            "Hx_top1": h_top,
            "F_only_discordant": f_only,
            "Hx_only_discordant": h_only,
            "n_discordant": n_disc,
            "mcnemar_p_two_sided": mcnemar_p,
            "label": ep1, "alpha": ALPHA},
        "endpoint2_matched_transfer": {
            "n_blocks": len(mt), "q_positive": q_pos,
            "q_negative": q_neg, "q_zero": q_zero,
            "sign_p_two_sided_nonzero_only": sign_p,
            "convention": "positive q favors FACTOR"},
        "pairwise_verdict": verdict,
        "transition_anatomy_descriptive": trans,
        "both_covered_correct_rider": bcc,
        "covered_misses": {
            "F": sum(1 for m in mt if m["c0_F"] == 0),
            "Hx": sum(1 for m in mt if m["c0_H"] == 0)},
    }, mt


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    # 1. FIRE authority + pins BEFORE heldout opens
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
    # 3. P2 realization independently re-derived
    gate(rederive_p2_realization_sha() == P2_REALIZATION_SHA,
         "P2 REALIZATION DRIFT")
    START = start_provenance(
        ["scratch/mathworld1_svpheldout18.py",
         "scratch/mathworld1_svpcalscore18.py",
         "scratch/mathworld1_svpheldout17.py",
         "scratch/mathworld1_svpgbirth18.py",
         "scratch/mathworld1_svpp2qual.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])

    # 4-5. heldout bytes + structure gates
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

    # 6. models
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

    # 7-9. blind-score PRIMARY (three arms), persist, hash
    t0 = time.time()
    recs = score_rows(pri, arms, dev)
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "primary_scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "primary_scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "primary_scores.jsonl"))
         == 96, "DISK ROWS")

    # 10. scoring gates
    n_t9, n_rank_id, n_order_id = hard_gates(recs)

    # 11-12. pairwise primaries, independently adjudicated
    pw = {}
    mts = []
    for hx in HASHES:
        pw[hx], mt = pairwise(recs, calmap, hx)
        mts.extend(mt)
    with open(OUTDIR / "matched_transfer.jsonl", "w") as fo:
        for m in mts:
            fo.write(json.dumps(m) + "\n")
    mt_sha = fsha(OUTDIR / "matched_transfer.jsonl")

    # 13. mechanical IUT class label
    s1 = pw["HASH_P1"]["pairwise_verdict"] == "STRONG-FACTOR"
    s2 = pw["HASH_P2"]["pairwise_verdict"] == "STRONG-FACTOR"
    if s1 and s2:
        klass = "KNOWN-SET-PERMUTATION-ROBUST-STRONG-FACTOR"
    elif s1:
        klass = "KNOWN-SET-P1-ONLY-STRONG-FACTOR"
    elif s2:
        klass = "KNOWN-SET-P2-ONLY-STRONG-FACTOR"
    else:
        klass = "KNOWN-SET-NO-ROBUST-STRONG-FACTOR"

    mrr = {a: round(sum(1.0 / d[a]["rank"] for d in recs)
                    / len(recs), 4) for a in ARMS}
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-KNOWN-SET-"
                  "PERMUTATION-HELDOUT-SCORE-18001",
        "iut_class_verdict": klass,
        "pairwise_F_v_H1": pw["HASH_P1"],
        "pairwise_F_v_H2": pw["HASH_P2"],
        "mrr_descriptive": mrr,
        "raw_scores_sha": raw_sha,
        "matched_transfer_sha": mt_sha,
        "calibration_authority": {
            "receipt_verdict": cal_receipt["verdict"],
            "raw_scores_sha_pinned": PINS[CAL_SCORES]},
        "p2_realization_sha_rederived_pre": True,
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
    # 14. post-run re-gates, freeze primary receipt
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    p2_post = rederive_p2_realization_sha()
    gate(p2_post == P2_REALIZATION_SHA,
         "POST P2 REALIZATION DRIFT")
    receipt["p2_realization_sha_rederived_post"] = p2_post
    (OUTDIR / "svpheldout18_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("iut_class_verdict", "pairwise_F_v_H1",
                       "pairwise_F_v_H2", "mrr_descriptive",
                       "wall_s")}, indent=1), flush=True)

    # 15. ONLY NOW: sensitivity, robustness rider, anatomy
    t1 = time.time()
    h1_top = sum(1 for d in recs if d["HASH_P1"]["top1"])
    h2_top = sum(1 for d in recs if d["HASH_P2"]["top1"])
    h1_only = sum(1 for d in recs if d["HASH_P1"]["top1"]
                  and not d["HASH_P2"]["top1"])
    h2_only = sum(1 for d in recs if d["HASH_P2"]["top1"]
                  and not d["HASH_P1"]["top1"])
    sens = {
        "label": "PERMUTATION-SENSITIVITY-MEASURED",
        "H1_top1": h1_top, "H2_top1": h2_top,
        "H1_only_discordant": h1_only,
        "H2_only_discordant": h2_only,
        "mcnemar_p_two_sided":
            binom_minlik_p(h1_only, h1_only + h2_only),
        "mrr": {a: mrr[a] for a in HASHES},
        "calibration_side_by_side": {
            a: f"{sum(1 for c in cal if c[a]['top1'])}/96"
            for a in HASHES}}

    rob_recs = score_rows(rob, arms, dev)
    hard_gates(rob_recs)
    with open(OUTDIR / "robustness_scores.jsonl", "w") as fo:
        for rec in rob_recs:
            fo.write(json.dumps(rec) + "\n")

    def bucket(sel):
        return {a: sum(1 for d in sel if d[a]["top1"])
                for a in ARMS} | {"n": len(sel)}
    cellkeys = [(2, "IN"), (2, "OUT"), (3, "IN"), (3, "OUT")]
    cells = {f"t{t}-{rg}": bucket(
        [d for d in recs
         if d["term"] == t and d["regime"] == rg])
        for t, rg in cellkeys}
    gaps = {}
    for hx, tag in (("HASH_P1", "F_H1"), ("HASH_P2", "F_H2")):
        gaps[tag] = [cells[f"t{t}-{rg}"]["FACTOR"]
                     - cells[f"t{t}-{rg}"][hx]
                     for t, rg in cellkeys]
    spear = {}
    for tag in ("F_H1", "F_H2"):
        for prior, vec in PRIOR_GAPS.items():
            spear[f"{tag}_v_{prior}"] = spearman(
                gaps[tag], vec)
    riders = {
        "permutation_sensitivity_H1_v_H2": sens,
        "primary_cells": cells,
        "pooled": {
            "t2": bucket([d for d in recs if d["term"] == 2]),
            "t3": bucket([d for d in recs if d["term"] == 3]),
            "c-IN": bucket([d for d in recs
                            if d["regime"] == "IN"]),
            "c-OUT": bucket([d for d in recs
                             if d["regime"] == "OUT"])},
        "primary_by_P": {
            P: bucket([d for d in recs if d["P"] == P])
            for P in sorted({d["P"] for d in recs})},
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
        "gap_vectors_t2IN_t2OUT_t3IN_t3OUT": gaps,
        "prior_gap_vectors": PRIOR_GAPS,
        "spearman_n4_descriptive": spear,
        "robustness_rider": {
            "n": len(rob_recs),
            "top1": {a: sum(1 for d in rob_recs
                            if d[a]["top1"]) for a in ARMS},
            "cells": {f"t{t}-{rg}": {a: sum(
                1 for d in rob_recs
                if d["term"] == t and d["regime"] == rg
                and d[a]["top1"]) for a in ARMS}
                for t, rg in cellkeys},
            "mrr": {a: round(sum(
                1.0 / d[a]["rank"] for d in rob_recs)
                / len(rob_recs), 4) for a in ARMS},
            "wall_s": round(time.time() - t1, 1)}}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svpheldout18] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
