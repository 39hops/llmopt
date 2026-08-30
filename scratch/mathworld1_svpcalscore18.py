"""MATH-CYBER-1 SVP-GRID-THREE-ARM-CALIBRATION-SCORE-18001 —
ONE scoring run of the three sealed seed-18001 checkpoints
(FACTOR / HASH-P1 / HASH-P2) on the frozen covered-calibration
artifact ONLY (af1a4aa1..., the same 96 covered-I0 strict states
scored by the 16001/17001 chain). Applies the frozen ALL-THREE
competence gate of PERMUTATION-REPLICATION-PREREG-0 (7976613b):
per arm c-IN covered >= 36/48 AND c-OUT covered >= 36/48 AND no
{term x regime} cell < 12/24 — ALL THREE arms pass =>
CALIBRATION FIRED (a later explicit GO gains the KNOWN-SET
permutation heldout authority); ANY arm fails ANY bar =>
PERMUTATION-REPLICATION-NOT-REACHED, STOP, no heldout for any
subset, no rescue, no threshold change. Twelve arm x term x
regime cells reported. Calibration differences across arms are
DESCRIPTIVE ONLY — a gate, not a representation contest.

Scoring law (verbatim svpadj lineage, unchanged): standing
prompt "Current: {cur}\\nHints: none\\nStep: "; candidate sets
and stored order unchanged; FACTOR continuations from stored
factor_code + EOS; HASH-P1 from stored hash_code + EOS; HASH-P2
continuations DERIVED per candidate from its semantic tuple via
the immutable qualified P2 law (hash2_symbols, realization
re-derived and gated == 952f332d... before load and post-run),
each gated by exact hash2_decode roundtrip — candidates are
never regenerated or filtered; only the H2 arm's code rendering
comes from the qualified law, exactly as its training targets
did. T=9 every candidate; pessimistic top-1 by mean logprob;
mean/sum rank + full-order identity gates; finite-score gates;
is_label consumed for RANKING only after all three arms' scores
exist (a pre-load structure gate also reads it for a
cardinality/semantics check whose result never reaches prompts,
continuations, candidate order, or scoring); raw scores
persisted and hashed BEFORE gate adjudication. No heldout or
P-OUT artifact is named or opened.

Outputs under logs/mathworld1/svpcalscore18/ (refuse-if-exists):
scores.jsonl, svpcalscore18_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svpcalscore18.py     (Mac)
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

CAL = "logs/mathworld1/svpdiet3/covered_calibration.jsonl"
PINS = {
    CAL:
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
BAR_REGIME = 36   # of 48, per arm, each regime
BAR_STRATUM = 12  # of 24, per {term x regime} cell
OUTDIR = Path("logs/mathworld1/svpcalscore18")
TOK = ActionGCTok()


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ctup(c):
    return (c["rule"], c["site_kind"], c["site_ordinal"],
            c["param_kind"], c["param_index"])


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


def main():
    if OUTDIR.exists():
        raise SystemExit(f"REFUSING: {OUTDIR} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "INIT PIN")
    qr = json.loads(Path(
        "logs/mathworld1/svpp2qual/svpp2qual_receipt.json"
        ).read_text())
    gate(qr["p2_realization_sha"] == P2_REALIZATION_SHA,
         "P2 RECEIPT REALIZATION MISMATCH")
    gate(rederive_p2_realization_sha() == P2_REALIZATION_SHA,
         "P2 REALIZATION DRIFT")
    START = start_provenance(
        ["scratch/mathworld1_svpcalscore18.py",
         "scratch/mathworld1_svpgbirth18.py",
         "scratch/mathworld1_svpp2qual.py",
         "scratch/mathworld1_svpcalscore17.py",
         "scratch/mathworld1_svpdiet3.py",
         "scratch/mathworld1_svpadj.py",
         "scratch/mathworld1_svpbirth.py",
         "scratch/mathworld1_svpcode.py",
         "scratch/mathworld1_actiontok.py",
         "scratch/mathworld1_birth.py",
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
            gate(hash_decode(c["hash_code"]) == t, "P1 RT")
            h2 = hash2_symbols(*t)
            gate(hash2_decode(h2) == t, "P2 RT")
            c["_h2"] = h2
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

    # score ALL 96 states blind, all three arms
    t0 = time.time()
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
        # label consumed only AFTER all three arms' scores exist
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

    # FROZEN ALL-THREE COMPETENCE GATE
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
               else "PERMUTATION-REPLICATION-NOT-REACHED")
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-THREE-ARM-"
                  "CALIBRATION-SCORE-18001",
        "verdict": verdict,
        "gate": {"bar_regime": f">={BAR_REGIME}/48 per arm",
                 "bar_stratum": f">={BAR_STRATUM}/24 per cell",
                 "cells_arm_term_regime": cells,
                 "regime_totals": regime_totals,
                 "arm_pass": arm_pass},
        "n_states_scored": len(recs),
        "raw_scores_sha": raw_sha,
        "p2": {"realization_sha_pin": P2_REALIZATION_SHA,
               "rederived_pre_run": True},
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
    # post-run pin re-gates (incl. P2 realization)
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha(INIT_CK) == INIT_SHA, "POST INIT PIN")
    gate(rederive_p2_realization_sha() == P2_REALIZATION_SHA,
         "POST P2 REALIZATION DRIFT")
    (OUTDIR / "svpcalscore18_receipt.json").write_text(
        json.dumps(receipt, indent=1))
    print(json.dumps({k: receipt[k] for k in
                      ("verdict", "gate", "wall_s")},
                     indent=1), flush=True)

    # descriptive riders (post-verdict; never alter the gate)
    riders = {
        "covered_mrr": {a: round(sum(
            1.0 / d[a]["rank"] for d in recs) / 96, 4)
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
    print("[svpcalscore18] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
