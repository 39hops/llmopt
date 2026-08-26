"""MATH-CYBER-1 SVP-ACTION-COVERAGE-DESK-0 — outcome-independent
census of exact semantic whole-action tuple coverage: training
target actions (73,324 rows) onto the labeled and all-candidate
actions of BOTH frozen eval bands. No model scores consumed, no
model loaded, no training. Prices whether an OPAQUE whole-action
categorical code would confound the causal control with OOV
exposure.

Whole-action tuple = (rule, site_kind, site_ordinal, param_kind,
param_index) — exactly the canonical ActionProgram coordinates;
program_text is its serialization (also censused).

Outputs logs/mathworld1/svpcovdesk_receipt.json (refuse-if-
exists).

    .venv/bin/python scratch/mathworld1_svpcovdesk.py         (Mac)
"""
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)
from scratch.mathworld1_svpbirth import gate  # noqa: E402

PINS = {
    "data/matsub_paired.jsonl":
        "a943ba7fc581db743b07192e5d951fadddd2ba19bca3225b75"
        "d8402351d468e8",
    "logs/mathworld1/svpeval/decisions.jsonl":
        "f63100a62f3091d544750d679483009a473261c587f3165241"
        "406a86253858c6",
    "logs/mathworld1/svpeval2/decisions.jsonl":
        "89efbe0ea447ee937c0c130d5419112921a2dd6c2159c6c211"
        "2cfd5e92f79315",
}
RECEIPT = Path("logs/mathworld1/svpcovdesk_receipt.json")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tup(d):
    return (d["rule"], d["site_kind"], d["site_ordinal"],
            d["param_kind"], d["param_index"])


def band_census(path, n_expected, train_tup, train_txt):
    lab_cov = Counter()
    cand_cov = Counter()
    oov_lab = []
    oov_cand_rules = Counter()
    n_prim = 0
    all_tups = set()
    lab_tups = set()
    for l in open(path):
        r = json.loads(l)
        if not r.get("primary_eligible"):
            continue
        n_prim += 1
        li = [i for i, c in enumerate(r["candidates"])
              if c["is_label"]][0]
        for i, c in enumerate(r["candidates"]):
            t = tup(c)
            all_tups.add(t)
            in_t = t in train_tup
            gate((c["program_text"] in train_txt) == in_t,
                 "TUPLE/TEXT CENSUS DIVERGES")
            cand_cov[in_t] += 1
            if not in_t:
                oov_cand_rules[c["rule"]] += 1
            if i == li:
                lab_tups.add(t)
                lab_cov[in_t] += 1
                if not in_t:
                    oov_lab.append({
                        "episode_id": r["episode_id"],
                        "decision_index": r["decision_index"],
                        "rule": c["rule"],
                        "program_text": c["program_text"]})
    gate(n_prim == n_expected, f"PRIMARY N {n_prim}")
    return {
        "n_primary": n_prim,
        "labeled_covered": lab_cov[True],
        "labeled_oov": lab_cov[False],
        "candidates_covered": cand_cov[True],
        "candidates_oov": cand_cov[False],
        "distinct_candidate_tuples": len(all_tups),
        "distinct_candidate_tuples_covered": len(
            all_tups & train_tup),
        "distinct_labeled_tuples": len(lab_tups),
        "distinct_labeled_tuples_covered": len(
            lab_tups & train_tup),
        "oov_candidate_rules": dict(oov_cand_rules),
        "oov_labeled_records": oov_lab}


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpcovdesk.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    train_tup = set()
    train_txt = set()
    rule_counts = Counter()
    for l in open("data/matsub_paired.jsonl"):
        r = json.loads(l)
        train_tup.add(tup(r))
        train_txt.add(r["program_text"])
        rule_counts[r["rule"]] += 1
    receipt = {
        "training": {
            "rows": 73324,
            "distinct_whole_action_tuples": len(train_tup),
            "distinct_program_texts": len(train_txt),
            "labeled_rule_counts": dict(
                rule_counts.most_common())},
        "old_band": band_census(
            "logs/mathworld1/svpeval/decisions.jsonl", 72,
            train_tup, train_txt),
        "new_band": band_census(
            "logs/mathworld1/svpeval2/decisions.jsonl", 79,
            train_tup, train_txt),
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    gate(receipt["training"]["rows"] == sum(rule_counts.values()),
         "TRAIN ROWS")
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    print(json.dumps({k: v for k, v in receipt.items()
                      if k not in ("start", "pins")}, indent=1),
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
