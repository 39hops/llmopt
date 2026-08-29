"""MATH-CYBER-1 SVP-GRID-CALIBRATION-SCORE-16001 — ONE scoring
run of the two sealed seed-16001 checkpoints on the frozen
IN-SUPPORT COVERED-CALIBRATION artifact ONLY (96 covered-I0
strict states over {t2,t3} x {c-IN,c-OUT}). This file has ZERO
path or string dependency on the sealed test artifact or the
secondary robustness artifact: it opens exactly
covered_calibration.jsonl (sha-gated pre- and post-run) and the
sealed siblings are neither named nor read anywhere here. Zero
training, zero checkpoint mutation, zero sympy; frozen
candidate bytes only.

Mechanical anti-peeking order (frozen): verify pins -> structure
gates re-derived from row metadata -> load models -> score ALL
96 states blind -> write raw scores -> hash -> hard gates (T=9,
finiteness, mean/sum rank + full-order identity) -> apply the
FROZEN FREQUENCY-STRATIFIED COMPETENCE GATE (registered at
IN-SUPPORT-CONFIRMATORY-DESIGN-0):
  per arm: c-IN covered top-1 >= 36/48 AND c-OUT covered
  top-1 >= 36/48 AND none of the four {term x regime} strata
  (24 each) below 12
  -> BOTH arms pass  => CALIBRATION FIRED (heldout authority
     granted to a LATER GO; strict heldout runs BEFORE the
     secondary scorer)
  -> ANY leg fails   => SUPPORT-NOT-LEARNED (STOP; sealed
     siblings stay unopened)
-> receipt (reporting ALL EIGHT arm x term x regime cells
separately, plus the two per-arm frequency totals the gate
consumes) -> only then descriptive riders.

LABEL SEPARATION (hard law): is_label is consumed ONLY after
candidate scores exist. Continuations are built solely from the
stored representation fields (factor_code / hash_code + EOS) in
the stored candidate order, identical across arms. Standing
prompt "Current: {cur}\\nHints: none\\nStep: ".

Outputs under logs/mathworld1/svpcalscore16/ (refuse-if-exists):
scores.jsonl, svpcalscore16_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svpcalscore16.py      (Mac)
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
from scratch.mathworld1_svpadj import (rank_metrics,  # noqa: E402
                                       score_decision)
from scratch.mathworld1_svpbirth import gate  # noqa: E402
from scratch.mathworld1_svpcode import (factor_decode,  # noqa: E402
                                        hash_decode)

CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
PINS = {
    CAL:
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b0850979"
        "4d0fb7978db3",
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
BAR_REGIME = 36   # of 48, per arm, each regime
BAR_STRATUM = 12  # of 24
OUTDIR = Path("logs/mathworld1/svpcalscore16")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpcalscore16.py",
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
    rows = [json.loads(l) for l in open(CAL)]
    # structure gates re-derived from row metadata
    gate(len(rows) == 96, f"ROWS {len(rows)}")
    gate(all(r["site_role"] == "covered-I0" for r in rows),
         "ROLE")
    gate(Counter((r["term_cell"], r["regime"]) for r in rows)
         == Counter({(2, "IN"): 24, (2, "OUT"): 24,
                     (3, "IN"): 24, (3, "OUT"): 24}),
         "STRATA")
    gate(all(r["chosen_ordinal"] == 0 for r in rows), "ORD")
    gate(len({r["cur"] for r in rows}) == 96, "CUR DUP")
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

    arms = {}
    for a in ARMS:
        m = build_model(VOCAB, ctx=4096)
        m.load_state_dict(torch.load(CKPTS[a][0],
                                     weights_only=True))
        gate(sum(p.numel() for p in m.parameters())
             == 19142016, f"PARAM COUNT {a}")
        m.eval()
        arms[a] = m.to(dev)

    # score ALL 96 states blind
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
        # label consumed only AFTER all candidate scores exist
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
    gate(sum(1 for _ in open(OUTDIR / "scores.jsonl")) == 96,
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
            gate(all(isinstance(x, float) and x == x
                     for x in rec[a]["mean_lp"]), "SCORE TYPE")

    # FROZEN FREQUENCY-STRATIFIED COMPETENCE GATE
    def cnt(arm, term=None, regime=None):
        sel = [d for d in recs
               if (term is None or d["term"] == term)
               and (regime is None or d["regime"] == regime)]
        return sum(1 for d in sel if d[arm]["top1"]), len(sel)

    cells = {}
    for a in ARMS:
        for t in (2, 3):
            for rg in ("IN", "OUT"):
                k, n = cnt(a, t, rg)
                cells[f"{a}|t{t}|{rg}"] = {"correct": k,
                                           "n": n}
    regime_totals = {}
    for a in ARMS:
        for rg in ("IN", "OUT"):
            k, n = cnt(a, regime=rg)
            regime_totals[f"{a}|{rg}"] = {"correct": k, "n": n}
    for a in ARMS:
        for rg in ("IN", "OUT"):
            gate(cells[f"{a}|t2|{rg}"]["correct"]
                 + cells[f"{a}|t3|{rg}"]["correct"]
                 == regime_totals[f"{a}|{rg}"]["correct"]
                 and regime_totals[f"{a}|{rg}"]["n"] == 48,
                 f"CELL SUM {a} {rg}")
    arm_pass = {}
    for a in ARMS:
        reg_ok = all(
            regime_totals[f"{a}|{rg}"]["correct"] >= BAR_REGIME
            for rg in ("IN", "OUT"))
        strat_ok = all(
            cells[f"{a}|t{t}|{rg}"]["correct"] >= BAR_STRATUM
            for t in (2, 3) for rg in ("IN", "OUT"))
        arm_pass[a] = {"regimes_pass": reg_ok,
                       "strata_pass": strat_ok,
                       "pass": reg_ok and strat_ok}
    fired = all(arm_pass[a]["pass"] for a in ARMS)
    verdict = ("CALIBRATION FIRED" if fired
               else "SUPPORT-NOT-LEARNED")
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-CALIBRATION-SCORE-"
                  "16001",
        "verdict": verdict,
        "gate": {"bar_regime": f">={BAR_REGIME}/48 per arm",
                 "bar_stratum": f">={BAR_STRATUM}/24",
                 "cells_arm_term_regime": cells,
                 "regime_totals": regime_totals,
                 "arm_pass": arm_pass},
        "n_states_scored": len(recs),
        "raw_scores_sha": raw_sha,
        "gates": {"T9_candidates_checked": n_t9,
                  "rank_identity_checks": n_rank_id,
                  "full_order_identity_checks": n_order_id,
                  "rederived_structure": {
                      "rows": len(rows),
                      "role_census": dict(Counter(
                          r["site_role"] for r in rows)),
                      "strata": {f"t{t}|{rg}": c2 for
                                 (t, rg), c2 in Counter(
                                     (r["term_cell"],
                                      r["regime"])
                                     for r in rows).items()},
                      "unique_curs": len(
                          {r["cur"] for r in rows})}},
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
    (OUTDIR / "svpcalscore16_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "gate", "wall_s")},
                     indent=1), flush=True)

    # descriptive riders (post-verdict; never alter the gate)
    def bucket(pred):
        sel = [d for d in recs if pred(d)]
        return {a: sum(1 for d in sel if d[a]["top1"])
                for a in ARMS} | {"n": len(sel)}
    riders = {
        "covered_mrr": {a: round(sum(
            1.0 / d[a]["rank"] for d in recs) / 96, 4)
            for a in ARMS},
        "by_block_d_before": {
            ds: bucket(lambda d, ds=ds:
                       d["block_d_before"] == ds)
            for ds in sorted({d["block_d_before"]
                              for d in recs})},
        "by_P": {P: bucket(lambda d, P=P: d["P"] == P)
                 for P in sorted({d["P"] for d in recs})},
        "by_c": {str(c): bucket(lambda d, c=c: d["c"] == c)
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
            d["n_candidates"] for d in recs))}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svpcalscore16] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
