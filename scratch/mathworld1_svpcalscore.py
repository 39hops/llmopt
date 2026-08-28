"""MATH-CYBER-1 SVP-GRID-CALIBRATION-SCORE-15001 — ONE scoring
run of the two sealed seed-15001 checkpoints on the frozen
COVERED-CALIBRATION artifact ONLY. This file has ZERO path or
string dependency on the sealed test artifact: it opens exactly
covered_calibration.jsonl (sha-gated), and the sealed sibling is
neither named nor read anywhere here. Zero training, zero
checkpoint mutation, zero sympy, zero legal-set/HCE
recomputation; frozen candidate bytes only.

Mechanical anti-peeking order (frozen): verify pins -> structure
gates re-derived from row metadata -> load models -> score ALL
240 states blind -> write raw scores -> hash -> hard gates
(T=9, finiteness, mean/sum rank + full-order identity) -> apply
the FROZEN COMPETENCE GATE on the 96 covered-I0 strict states:
  FACTOR top-1 >= 72/96 AND HASH top-1 >= 72/96 AND no
  {family x term} stratum (4 strata of 24) below 12/24 for
  either arm
-> verdict CALIBRATION FIRED (heldout authority granted to a
LATER GO) or SUPPORT-NOT-LEARNED (STOP; sealed sibling stays
unopened) -> write receipt -> only then descriptive riders
(t1 controls, per-stratum counts, MRR, distractor splits).

LABEL SEPARATION (hard law): is_label is consumed ONLY after
candidate scores exist. Continuations are built solely from the
stored representation fields — FACTOR factor_code+EOS, HASH
hash_code+EOS — in the stored candidate order, identical across
arms. Standing prompt "Current: {cur}\\nHints: none\\nStep: ".

Outputs under logs/mathworld1/svpcalscore/ (refuse-if-exists):
scores.jsonl, svpcalscore_receipt.json, riders.json.

    .venv/bin/python scratch/mathworld1_svpcalscore.py        (Mac)
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

CAL = "logs/mathworld1/svpdiet2/covered_calibration.jsonl"
PINS = {
    CAL:
        "90421e8b9bcab38648a20e7cd24f48e2d54cdcd20a437b18b8f5"
        "4e9e3f4d9977",
    "logs/mathworld1/svpdiet2/svpdiet2_receipt.json":
        "6313f3a54497607340700537dab1458b27c809a624e54d8de82d"
        "e0b0c3ba639e",
    "logs/mathworld1/svpgbirth_s15001_receipt.json":
        "cacee4efed9200fef1f4cf1a02b4c2197fde5d6afa12c70e5927"
        "1bfecf0bb36c",
}
CKPTS = {
    "FACTOR": ("checkpoints/svp_grid_factor_s15001.pt",
               "6ef7b85ce9215c1ba299d64fdf61e2b32573b2aaac1e0e"
               "a4bb1dedf4cf790cd3"),
    "HASH": ("checkpoints/svp_grid_hash_s15001.pt",
             "3b9da5a7ee3bd72bc40a21afcbc8b114ebd19040079ce1f7"
             "194941946052501a"),
}
INIT_SHA = ("4b085795f9e8b0be874cabdc6d58899a2a4554f8b42cb711"
            "f5154614e41797bc")
VOCAB = 340
CODE_BASE = 332
ARMS = ["FACTOR", "HASH"]
BAR_TOTAL = 72   # of 96
BAR_STRATUM = 12  # of 24
OUTDIR = Path("logs/mathworld1/svpcalscore")
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
    gate(fsha("checkpoints/svp_grid_init_s15001.pt")
         == INIT_SHA, "INIT PIN")
    START = start_provenance(
        ["scratch/mathworld1_svpcalscore.py",
         "scratch/mathworld1_svpdiet2.py",
         "scratch/mathworld1_svpgbirth.py",
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
    gate(len(rows) == 240, f"ROWS {len(rows)}")
    roles = Counter(r["site_role"] for r in rows)
    gate(roles == Counter({"covered-I0": 96, "control-I1": 48,
                           "control-I0": 48,
                           "control-robust-I1": 48}),
         f"ROLE CENSUS {dict(roles)}")
    cov = [r for r in rows if r["site_role"] == "covered-I0"]
    gate(Counter((r["family"], r["term_cell"]) for r in cov)
         == Counter({("CH-F3", 2): 24, ("CH-F3", 3): 24,
                     ("CH-F4", 2): 24, ("CH-F4", 3): 24}),
         "STRICT STRATA")
    gate(all(r["chosen_ordinal"] == 0 for r in cov), "COV ORD")
    t1 = [r for r in rows if r["term_cell"] == 1]
    gate(len(t1) == 144 and all(
        r["site_role"].startswith("control") for r in t1),
        "T1 CONTROLS")
    gate(len({r["cur"] for r in rows}) == 240, "CUR DUP")
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

    # score ALL 240 states blind
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
               "family": r["family"], "term": r["term_cell"],
               "site_role": r["site_role"],
               "distractor": r["distractor"],
               "block_d_before": r["block_d_before"],
               "cur": r["cur"],
               "n_candidates": len(cands),
               "exec_order": ARMS}
        raw = {}
        for a in ARMS:
            triples = score_decision(
                arms[a], dev, r["cur"], conts[a])
            raw[a] = triples
        # label consumed only AFTER all candidate scores exist
        li = [i for i, c in enumerate(cands)
              if c["is_label"]][0]
        rec["label_index"] = li
        rec["labeled_tuple"] = list(ctup(cands[li]))
        for a in ARMS:
            triples = raw[a]
            means = [t[0] for t in triples]
            sums = [t[1] for t in triples]
            top1, rank = rank_metrics(means, li)
            rec[a] = {"mean_lp": means, "sum_lp": sums,
                      "T": [t[2] for t in triples],
                      "top1": top1, "rank": rank}
        recs.append(rec)
    OUTDIR.mkdir(parents=True)
    with open(OUTDIR / "scores.jsonl", "w") as fo:
        for rec in recs:
            fo.write(json.dumps(rec) + "\n")
    raw_sha = fsha(OUTDIR / "scores.jsonl")
    gate(sum(1 for _ in open(OUTDIR / "scores.jsonl")) == 240,
         "DISK ROWS")

    # hard gates
    n_t9 = 0
    n_rank_id = 0
    n_order_id = 0
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

    # FROZEN COMPETENCE GATE on the 96 covered-I0 states
    covr = [d for d in recs if d["site_role"] == "covered-I0"]
    gate(len(covr) == 96, "COV N")
    totals = {a: sum(1 for d in covr if d[a]["top1"])
              for a in ARMS}
    strata = {}
    for a in ARMS:
        for f in ("CH-F3", "CH-F4"):
            for t in (2, 3):
                strata[f"{a}|{f}|t{t}"] = sum(
                    1 for d in covr
                    if d["family"] == f and d["term"] == t
                    and d[a]["top1"])
    total_pass = all(totals[a] >= BAR_TOTAL for a in ARMS)
    strata_pass = all(v >= BAR_STRATUM
                      for v in strata.values())
    fired = total_pass and strata_pass
    verdict = ("CALIBRATION FIRED" if fired
               else "SUPPORT-NOT-LEARNED")
    receipt = {
        "prereg": "MATH-CYBER-1-SVP-GRID-CALIBRATION-SCORE-"
                  "15001",
        "verdict": verdict,
        "gate": {"bar_total": f">={BAR_TOTAL}/96",
                 "bar_stratum": f">={BAR_STRATUM}/24",
                 "totals": totals,
                 "strata": strata,
                 "total_pass": total_pass,
                 "strata_pass": strata_pass},
        "n_states_scored": len(recs),
        "n_covered": len(covr),
        "raw_scores_sha": raw_sha,
        "gates": {"T9_candidates_checked": n_t9,
                  "rank_identity_checks": n_rank_id,
                  "full_order_identity_checks": n_order_id,
                  "rederived_structure": {
                      "rows": len(rows),
                      "role_census": dict(roles),
                      "covered_strata": {
                          f"{f}|t{t}": 24
                          for f in ("CH-F3", "CH-F4")
                          for t in (2, 3)},
                      "unique_curs": 240}},
        "device": str(dev),
        "wall_s": round(time.time() - t0, 1),
        "pins": {p: fsha(p) for p in PINS},
        "ckpt_pins": {a: fsha(CKPTS[a][0]) for a in ARMS},
        "init_pin": fsha("checkpoints/svp_grid_init_s15001.pt"),
        "start": START,
        "completion_commit": completion_commit()}
    # post-run pin re-gate
    for p, h in PINS.items():
        gate(fsha(p) == h, f"POST PIN {p}")
    for a, (p, h) in CKPTS.items():
        gate(fsha(p) == h, f"POST CKPT PIN {a}")
    gate(fsha("checkpoints/svp_grid_init_s15001.pt")
         == INIT_SHA, "POST INIT PIN")
    (OUTDIR / "svpcalscore_receipt.json").write_text(
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
        "t1_control_I1": bucket(
            lambda d: d["site_role"] == "control-I1"),
        "t1_control_I0": bucket(
            lambda d: d["site_role"] == "control-I0"),
        "t1_control_robust_I1": bucket(
            lambda d: d["site_role"] == "control-robust-I1"),
        "covered_mrr": {a: round(sum(
            1.0 / d[a]["rank"] for d in covr) / 96, 4)
            for a in ARMS},
        "covered_by_block_d_before": {
            ds: bucket(lambda d, ds=ds:
                       d["site_role"] == "covered-I0"
                       and d["block_d_before"] == ds)
            for ds in sorted({d["block_d_before"]
                              for d in covr})},
        "covered_by_family_term": {
            f"{f}|t{t}": bucket(
                lambda d, f=f, t=t:
                d["site_role"] == "covered-I0"
                and d["family"] == f and d["term"] == t)
            for f in ("CH-F3", "CH-F4") for t in (2, 3)}}
    (OUTDIR / "riders.json").write_text(
        json.dumps(riders, indent=1))
    print("[svpcalscore] DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
