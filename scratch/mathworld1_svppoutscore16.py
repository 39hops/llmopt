"""MATH-CYBER-1 SVP-GRID-POUT-ROBUSTNESS-SCORE-16001 — ONE
scoring run of the two sealed seed-16001 checkpoints on the
frozen SECONDARY P-OUT robustness artifact pout_robustness.jsonl
(72 fresh covered-I0 states on the degree-11 two-term P class:
48 t2 PRIMARY + 24 t3 CONTROL, separately adjudicated, never
pooled). Applies the secondary law frozen at IN-SUPPORT-
CONFIRMATORY-DESIGN-0 verbatim: hypothesis FACTOR > HASH on the
48 t2 states, exact two-sided McNemar alpha .05;
SECONDARY-FACTOR-WIN iff p < .05 AND F top1 > H top1, otherwise
SECONDARY-INCONCLUSIVE (no symmetric HASH-win label is
registered; a significant reverse split books INCONCLUSIVE with
the direction disclosed). This scorer can never modify the
strict heldout verdict or any seed-17001 law. Zero training,
zero checkpoint mutation, zero sympy; frozen bytes only.

Mechanical anti-peeking order: pins -> structure gates -> load
models -> blind-score all 72 states -> raw scores persisted and
hashed -> hard gates -> primary t2 adjudication -> control t3
report -> receipt -> only then descriptive riders. No per-state
top1 printed before the raw score file is closed and hashed.

LABEL SEPARATION (hard law): is_label is consumed for RANKING
only after both arms' candidate scores exist; a pre-load
structure gate also reads it for a cardinality/semantics check
whose result never reaches prompts, continuations, candidate
order, or scoring. Standing prompt
"Current: {cur}\\nHints: none\\nStep: ".

Outputs under logs/mathworld1/svppoutscore16/ (refuse-if-exists):
scores.jsonl, svppoutscore16_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svppoutscore16.py    (Mac)
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

POUT = "logs/mathworld1/svpdiet3/pout_robustness.jsonl"
PINS = {
    POUT:
        "5c85fc1f336791522db78e681bfdadad8c4efdaafed640a7aa50"
        "3d72a82c6137",
    "logs/mathworld1/svpdiet3/svpdiet3_receipt.json":
        "26cb6d0119f56e24b4025d43976ddf323a5540e0177c19583bfe"
        "2f5c984fb365",
    "logs/mathworld1/svpgbirth_s16001_receipt.json":
        "24f1f1506575594ae494aa84fe9ba98d6c81320f508fd12c726c"
        "f2afc231b015",
}
CKPTS = {
    "FACTOR": ("checkpoints/svp_grid_factor_s16001.pt",
               "c3f7a3e974d92862478ac7a0fd48d57153f2d221db66f8"
               "0859b59dc28f63949a"),
    "HASH": ("checkpoints/svp_grid_hash_s16001.pt",
             "645fc24f6a829bd50d6e92ec02cc18d4f56f459b8df2c874"
             "f1d7351f66a474e0"),
}
INIT_SHA = ("2a580568e37ec91d976dbde0d4654a371f8a31f66dd514a7"
            "8bdfd1afed3dbbaf")
INIT_CK = "checkpoints/svp_grid_init_s16001.pt"
VOCAB = 340
CODE_BASE = 332
ARMS = ["FACTOR", "HASH"]
ALPHA = 0.05
P_FORMS = {f"x**11 + x**{n}" if n > 1 else "x**11 + x"
           for n in range(1, 11)}
OUTDIR = Path("logs/mathworld1/svppoutscore16")
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


def arm_stats(recs):
    f = sum(1 for d in recs if d["FACTOR"]["top1"])
    h = sum(1 for d in recs if d["HASH"]["top1"])
    f_only = sum(1 for d in recs if d["FACTOR"]["top1"]
                 and not d["HASH"]["top1"])
    h_only = sum(1 for d in recs if d["HASH"]["top1"]
                 and not d["FACTOR"]["top1"])
    return {"n": len(recs), "FACTOR_top1": f, "HASH_top1": h,
            "F_only_discordant": f_only,
            "H_only_discordant": h_only,
            "n_discordant": f_only + h_only,
            "mcnemar_p_two_sided":
                binom_minlik_p(f_only, f_only + h_only),
            "mrr_descriptive": {a: round(sum(
                1.0 / d[a]["rank"] for d in recs)
                / len(recs), 4) for a in ARMS}}


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svppoutscore16.py",
         "scratch/mathworld1_svpheldout16.py",
         "scratch/mathworld1_svpcalscore16.py",
         "scratch/mathworld1_svpdiet3.py",
         "scratch/mathworld1_svpgbirth16.py",
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
    rows = [json.loads(l) for l in open(POUT)]
    # structure gates re-derived from row metadata
    gate(len(rows) == 72, f"ROWS {len(rows)}")
    t2 = [r for r in rows if r["term_cell"] == 2]
    t3 = [r for r in rows if r["term_cell"] == 3]
    gate(len(t2) == 48 and len(t3) == 24, "STRATA SPLIT")
    gate(Counter(r["regime"] for r in t2)
         == Counter({"IN": 26, "OUT": 22}), "T2 CENSUS")
    gate(len({r["cur"] for r in rows}) == 72, "CUR DUP")
    gate({r["P"] for r in rows} == P_FORMS, "P CLASS")
    gate(all(r["chosen_ordinal"] == 0 for r in rows), "ORD")
    gate(all(r["stratum"] == ("primary-t2" if r["term_cell"]
                              == 2 else "control-t3")
             for r in rows), "STRATUM FIELD")
    for r in rows:
        labs = [c for c in r["candidates"] if c["is_label"]]
        gate(len(labs) == 1, "LABEL COUNT")
        lt = ctup(labs[0])
        gate(lt[0] == "i_unprod" and lt[1] == "I"
             and lt[2] == 0, f"LABEL SEMANTICS {lt}")
        for c in r["candidates"]:
            t = ctup(c)
            gate(factor_decode(c["factor_code"]) == t, "F RT")
            gate(hash_decode(c["hash_code"]) == t, "H RT")
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

    # blind-score ALL 72 states
    t0 = time.time()
    recs = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        conts = {
            "FACTOR": [[CODE_BASE + s for s in c["factor_code"]]
                       + [TOK.eos_id] for c in cands],
            "HASH": [[CODE_BASE + s for s in c["hash_code"]]
                     + [TOK.eos_id] for c in cands]}
        rec = {"base_signature": r["base_signature"],
               "term": r["term_cell"], "regime": r["regime"],
               "P": r["P"], "c": r["c"],
               "stratum": r["stratum"],
               "cur": r["cur"],
               "n_candidates": len(cands),
               "exec_order": ARMS}
        raw = {}
        for a in ARMS:
            raw[a] = score_decision(arms[a], dev, r["cur"],
                                    conts[a])
        # label consumed for ranking only AFTER both arms scored
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
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "scores.jsonl")) == 72,
         "DISK ROWS")

    # hard gates
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

    # FROZEN ADJUDICATION — primary t2 only
    p2 = [d for d in recs if d["term"] == 2]
    p3 = [d for d in recs if d["term"] == 3]
    gate(len(p2) == 48 and len(p3) == 24, "ADJ SPLIT")
    primary = arm_stats(p2)
    control = arm_stats(p3)
    if (primary["mcnemar_p_two_sided"] < ALPHA
            and primary["FACTOR_top1"] > primary["HASH_top1"]):
        verdict = "SECONDARY-FACTOR-WIN"
    else:
        verdict = "SECONDARY-INCONCLUSIVE"
    direction = ("FACTOR" if primary["FACTOR_top1"]
                 > primary["HASH_top1"] else
                 "HASH" if primary["HASH_top1"]
                 > primary["FACTOR_top1"] else "TIED")
    reverse_significant = (
        direction == "HASH"
        and primary["mcnemar_p_two_sided"] < ALPHA)
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-POUT-ROBUSTNESS-"
                  "SCORE-16001",
        "verdict": verdict,
        "primary_direction": direction,
        "reverse_significant": reverse_significant,
        "alpha": ALPHA,
        "primary_t2": primary,
        "control_t3_separate_never_pooled": control,
        "n_states_scored": len(recs),
        "raw_scores_sha": raw_sha,
        "gates": {"T9_candidates_checked": n_t9,
                  "rank_identity_checks": n_rank_id,
                  "full_order_identity_checks": n_order_id,
                  "rederived_structure": {
                      "rows": len(rows),
                      "t2_regime_census": dict(Counter(
                          r["regime"] for r in t2)),
                      "t3_regime_census": dict(Counter(
                          r["regime"] for r in t3)),
                      "unique_curs": len(
                          {r["cur"] for r in rows}),
                      "P_forms": sorted(
                          {r["P"] for r in rows})}},
        "device": str(dev),
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_pins": {a: fsha(CKPTS[a][0]) for a in ARMS},
        "init_pin": fsha(INIT_CK),
        "start": START,
        "completion_commit": completion_commit()}
    # post-run pin re-gates
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    (OUTDIR / "svppoutscore16_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "primary_t2",
                       "control_t3_separate_never_pooled",
                       "wall_s")}, indent=1), flush=True)

    # descriptive riders (post-verdict; never alter the verdict)
    def bucket(sel):
        return {a: sum(1 for d in sel if d[a]["top1"])
                for a in ARMS} | {"n": len(sel)}
    riders = {
        "primary_by_regime": {
            rg: bucket([d for d in p2 if d["regime"] == rg])
            for rg in ("IN", "OUT")},
        "control_by_regime": {
            rg: bucket([d for d in p3 if d["regime"] == rg])
            for rg in ("IN", "OUT")},
        "by_P": {P: bucket([d for d in recs if d["P"] == P])
                 for P in sorted({d["P"] for d in recs})},
        "primary_by_P": {
            P: bucket([d for d in p2 if d["P"] == P])
            for P in sorted({d["P"] for d in p2})},
        "cand0": {
            "is_label": sum(1 for d in recs
                            if d["label_index"] == 0),
            "strict_top": {a: sum(
                1 for d in recs
                if len(d[a]["mean_lp"]) > 1
                and d[a]["mean_lp"][0]
                > max(d[a]["mean_lp"][1:])) for a in ARMS}},
        "legal_K": dict(Counter(
            d["n_candidates"] for d in recs))}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svppoutscore16] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
