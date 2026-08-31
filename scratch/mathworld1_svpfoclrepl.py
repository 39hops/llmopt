"""MATH-CYBER-1 SVP-FIELD-ORDER-REPLICATION-CALIBRATION-
SCORE-20001 — ONE scoring run of the two sealed seed-20001
field-order REPLICATION checkpoints (CANONICAL / PARAM-FIRST)
on the frozen covered-calibration artifact ONLY (af1a4aa1...,
the same 96 covered-I0 strict states). Stage 2 of the frozen
seed-20001 replication protocol (replication prereg 732a5312;
scoring law byte-adopted from the booked seed-19001 calibration
instrument scratch/mathworld1_svpfocal.py — deltas are
checkpoint paths/pins, fresh output namespace, and replication
provenance only). No heldout scoring, no token-onset values, no
accuracy(k)/k_step, no completion or relocation endpoint.

SCORING IDENTITY: semantic states and candidate actions are
identical across arms; only candidate SERIALIZATION differs —
CANONICAL uses the stored factor_code (roundtrip-gated);
PARAM-FIRST derives each candidate's payload via the qualified
pf_encode from the svpforder instrument (single source of the
PERM law, byte-shared with the seed-20001 replication birth), gated per candidate by pf_decode roundtrip AND the
exact permutation identity v the canonical payload. Candidates
are never regenerated.

FROZEN GATE (verbatim): per arm c-IN covered >= 36/48 AND c-OUT
covered >= 36/48 AND no {term x regime} cell < 12/24 — BOTH
arms pass => FIELD-ORDER CALIBRATION FIRED (stage-3 authority
to a later GO); EITHER arm fails ANY bar =>
FIELD-ORDER-NOT-REACHED, STOP, no heldout, no single-arm
scoring, no gate change, no onset inference. Eight
arm x term x regime cells reported. Arm differences are NEVER
field-order-superiority evidence.

RAW-FIRST LAW: blind-score all 96 states for both arms; persist
and hash raw candidate scores BEFORE any competence summary;
is_label consumed for ranking only after both arms' scores
exist (pre-load structure gate reads it for cardinality/
semantics only). Standing prompt, T=9, pessimistic top-1 by
mean logprob, mean/sum rank + full-order identity gates.

Outputs under logs/mathworld1/svpfocal20/ (refuse-if-exists):
scores.jsonl, svpfocal20_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svpfoclrepl.py       (Mac)
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
from scratch.mathworld1_svpcode import factor_decode  # noqa: E402
from scratch.mathworld1_svpforder import (PERM,  # noqa: E402
                                          pf_decode, pf_encode)

CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
PINS = {
    CAL:
        "af1a4aa1df7bf3224745e91a90e1a77c36e5c54f7ff9b0850979"
        "4d0fb7978db3",
    "logs/mathworld1/svpdiet3/svpdiet3_receipt.json":
        "26cb6d0119f56e24b4025d43976ddf323a5540e0177c19583bfe"
        "2f5c984fb365",
    "logs/mathworld1/svpforepl_s20001_receipt.json":
        "70ff3248a9b2f6e584c2d7bb8e9fc7b59853ebb7055f58193783"
        "fbd61eab78b3",
    "logs/mathworld1/svpforepl_census/"
    "svpforepl_census_receipt.json":
        "49bf9823c93b5bd91e63694b26bef9bed86d04dde298c668c434"
        "e85e80fea302",
}
CKPTS = {
    "CANONICAL": ("checkpoints/svp_forder_canonical_s20001.pt",
                  "0a841a5f2a43b6f64b0dac8259c26fd79961e6ab91"
                  "359a54be9c2582815b3e34"),
    "PARAM_FIRST": ("checkpoints/svp_forder_paramfirst_s20001"
                    ".pt",
                    "b7198ff2e7b903ab5ed075fe947cb29142c5790e"
                    "c84831434c53a598e466c322"),
}
INIT_SHA = ("7c95e77f8d7ccea5f4dd71c989e4d3225e347a178032d435"
            "39a1ae6ef62c9452")
INIT_CK = "checkpoints/svp_forder_init_s20001.pt"
VOCAB = 340
CODE_BASE = 332
ARMS = ["CANONICAL", "PARAM_FIRST"]
BAR_REGIME = 36
BAR_STRATUM = 12
OUTDIR = Path("logs/mathworld1/svpfocal20")
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
        ["scratch/mathworld1_svpfoclrepl.py",
         "scratch/mathworld1_svpfocal.py",
         "scratch/mathworld1_svpforder.py",
         "scratch/mathworld1_svpforepl.py",
         "scratch/mathworld1_svpcalscore18.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "llmopt/train/mathnative.py",
         "llmopt/lab/provenance.py"])
    rows = [json.loads(l) for l in open(CAL)]
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
            cz = c["factor_code"]
            gate(factor_decode(cz) == t, "C RT")
            pz = pf_encode(t)
            gate(pf_decode(pz) == t, "PF RT")
            gate(pz == [cz[PERM[i]] for i in range(8)],
                 "PERM IDENTITY")
            c["_pf"] = pz
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

    # blind-score all 96 states, both arms
    t0 = time.time()
    recs = []
    for r in rows:
        cands = r["candidates"]
        gate(len(cands) == r["n_candidates"], "CAND COUNT")
        conts = {
            "CANONICAL": [[CODE_BASE + s
                           for s in c["factor_code"]]
                          + [TOK.eos_id] for c in cands],
            "PARAM_FIRST": [[CODE_BASE + s for s in c["_pf"]]
                            + [TOK.eos_id] for c in cands]}
        rec = {"block_id": r["block_id"],
               "base_signature": r["base_signature"],
               "term": r["term_cell"], "regime": r["regime"],
               "P": r["P"], "c": r["c"],
               "site_role": r["site_role"],
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

    # FROZEN COMPETENCE GATE, both arms independently
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
    overall = {}
    for a in ARMS:
        for rg in ("IN", "OUT"):
            k, n = cnt(a, regime=rg)
            regime_totals[f"{a}|{rg}"] = {"correct": k, "n": n}
        k, n = cnt(a)
        overall[a] = {"correct": k, "n": n}
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
    verdict = ("FIELD-ORDER-REPLICATION CALIBRATION FIRED"
               if fired
               else "FIELD-ORDER-REPLICATION-NOT-REACHED")
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-FIELD-ORDER-REPLICATION-"
                  "CALIBRATION-SCORE-20001",
        "replication_of": "seed 19001 calibration (prereg "
                          "3ac5a70e; replication prereg "
                          "732a5312)",
        "seed": 20001,
        "verdict": verdict,
        "gate": {"bar_regime": f">={BAR_REGIME}/48 per arm",
                 "bar_stratum": f">={BAR_STRATUM}/24 per cell",
                 "cells_arm_term_regime": cells,
                 "regime_totals": regime_totals,
                 "overall_top1": overall,
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
                      "unique_curs": len(
                          {r["cur"] for r in rows})}},
        "device": str(dev),
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_pins": {a: fsha(CKPTS[a][0]) for a in ARMS},
        "init_pin": fsha(INIT_CK),
        "start": START,
        "completion_commit": completion_commit()}
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    (OUTDIR / "svpfocal20_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "gate", "wall_s")},
                     indent=1), flush=True)

    riders = {
        "covered_mrr": {a: round(sum(
            1.0 / d[a]["rank"] for d in recs) / len(recs), 4)
            for a in ARMS},
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
    print("[svpfocal20] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
