"""MATH-CYBER-1 SVP-SUPPORT-OPPORTUNITY-DESK-0 — outcome-independent
census of the frozen training/eval support behind the FACTOR/HASH
null: exact semantic tuple frequency structure of the 73,324
training targets (entropy, effective support, HHI, per-coordinate
and joint supports) plus a five-class OOV hierarchy over every
labeled action, rival action, and decision in all three frozen
eval bands. No model loaded, no scores consumed, no training, no
new task generation.

Frozen exclusive precedence (field support = MARGINAL per-field
value support among training targets):
  WHOLE-COVERED > NEW-RULE > NEW-FACTOR-VALUE > RECOMBINATION-OOV
Nonexclusive flag RULE-COVERED-OOV = NEW-FACTOR-VALUE union
RECOMBINATION-OOV.

Outputs logs/mathworld1/svpsuppdesk_receipt.json (refuse-if-
exists).

    .venv/bin/python scratch/mathworld1_svpsuppdesk.py        (Mac)
"""
import hashlib
import json
import math
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
    "logs/mathworld1/svpeval3/decisions.jsonl":
        "2ff5433249622df9d421cf8014131b3907092a943040bb7b20f4"
        "6f1afffb7efa",
}
BANDS = {
    "band1": ("logs/mathworld1/svpeval/decisions.jsonl", 72),
    "band2": ("logs/mathworld1/svpeval2/decisions.jsonl", 79),
    "band3": ("logs/mathworld1/svpeval3/decisions.jsonl", 69),
}
FIELDS = ("rule", "site_kind", "site_ordinal", "param_kind",
          "param_index")
RECEIPT = Path("logs/mathworld1/svpsuppdesk_receipt.json")


def fsha(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def tup(d):
    return tuple(d[f] for f in FIELDS)


def quantile(sorted_vals, q):
    # nearest-rank on the sorted list (frozen convention, stated
    # in the receipt as "nearest-rank")
    i = min(len(sorted_vals) - 1,
            max(0, math.ceil(q * len(sorted_vals)) - 1))
    return sorted_vals[i]


def classify(t, train_tup, train_rules, field_support):
    if t in train_tup:
        return "WHOLE-COVERED"
    if t[0] not in train_rules:
        return "NEW-RULE"
    for f, v in zip(FIELDS[1:], t[1:]):
        if v not in field_support[f]:
            return "NEW-FACTOR-VALUE"
    return "RECOMBINATION-OOV"


def band_census(path, n_expected, train_tup, train_rules,
                field_support):
    lab_cls = Counter()
    rival_cls = Counter()
    dec_flags = Counter()
    lab_recomb = []
    n_prim = 0
    for l in open(path):
        r = json.loads(l)
        if not r.get("primary_eligible"):
            continue
        n_prim += 1
        rc = Counter()
        for c in r["candidates"]:
            k = classify(tup(c), train_tup, train_rules,
                         field_support)
            if c["is_label"]:
                lab_cls[k] += 1
                if k == "RECOMBINATION-OOV":
                    lab_recomb.append({
                        "episode_id": r["episode_id"],
                        "decision_index": r["decision_index"],
                        "program_text": c["program_text"]})
            else:
                rival_cls[k] += 1
            rc[k] += 1
        n_recomb_riv = sum(
            1 for c in r["candidates"]
            if not c["is_label"] and classify(
                tup(c), train_tup, train_rules,
                field_support) == "RECOMBINATION-OOV")
        n_newrule_riv = sum(
            1 for c in r["candidates"]
            if not c["is_label"] and classify(
                tup(c), train_tup, train_rules,
                field_support) == "NEW-RULE")
        if n_recomb_riv:
            dec_flags["ge1_recomb_oov_rival"] += 1
        if n_newrule_riv:
            dec_flags["ge1_new_rule_rival"] += 1
        if rc["WHOLE-COVERED"] == sum(rc.values()):
            dec_flags["only_whole_covered_candidates"] += 1
        if rc["NEW-FACTOR-VALUE"]:
            dec_flags["ge1_new_factor_value_candidate"] += 1
    gate(n_prim == n_expected, f"PRIMARY N {n_prim}")
    ruleovv = lab_cls["NEW-FACTOR-VALUE"] + lab_cls[
        "RECOMBINATION-OOV"]
    return {
        "n_primary": n_prim,
        "labels": dict(lab_cls),
        "labels_rule_covered_oov_flag": ruleovv,
        "rivals": dict(rival_cls),
        "rivals_rule_covered_oov_flag":
            rival_cls["NEW-FACTOR-VALUE"]
            + rival_cls["RECOMBINATION-OOV"],
        "decisions": dict(dec_flags),
        "labels_recombination_oov_records": lab_recomb}


def main():
    if RECEIPT.exists():
        raise SystemExit(f"REFUSING: {RECEIPT} exists")
    for p, h in PINS.items():
        gate(fsha(p) == h, f"PIN MISMATCH {p}")
    START = start_provenance(
        ["scratch/mathworld1_svpsuppdesk.py",
         "scratch/mathworld1_svpbirth.py",
         "llmopt/lab/provenance.py"])
    tup_counts = Counter()
    coord = {f: Counter() for f in FIELDS}
    joint = {"rule_x_site": Counter(),
             "rule_x_param_kind": Counter(),
             "rule_x_site_x_param_kind": Counter()}
    n_rows = 0
    for l in open("data/matsub_paired.jsonl"):
        r = json.loads(l)
        n_rows += 1
        t = tup(r)
        tup_counts[t] += 1
        for f in FIELDS:
            coord[f][r[f]] += 1
        joint["rule_x_site"][(t[0], t[1], t[2])] += 1
        joint["rule_x_param_kind"][(t[0], t[3])] += 1
        joint["rule_x_site_x_param_kind"][
            (t[0], t[1], t[2], t[3])] += 1
    gate(n_rows == 73324, f"TRAIN ROWS {n_rows}")
    gate(sum(tup_counts.values()) == n_rows, "TUPLE MASS")
    freqs = sorted(tup_counts.values(), reverse=True)
    total = float(n_rows)
    probs = [c / total for c in freqs]
    H = -sum(p * math.log(p) for p in probs)
    hhi = sum(p * p for p in probs)
    asc = sorted(freqs)
    train_tup = set(tup_counts)
    train_rules = set(coord["rule"])
    field_support = {f: set(coord[f]) for f in FIELDS}
    receipt = {
        "frozen_precedence":
            "WHOLE-COVERED > NEW-RULE > NEW-FACTOR-VALUE > "
            "RECOMBINATION-OOV; field support = marginal "
            "per-field value support among training targets; "
            "RULE-COVERED-OOV reported as nonexclusive flag "
            "(= NEW-FACTOR-VALUE + RECOMBINATION-OOV)",
        "training": {
            "rows": n_rows,
            "distinct_whole_action_tuples": len(tup_counts),
            "tuple_frequencies": {
                " ".join(map(str, t)): c
                for t, c in tup_counts.most_common()},
            "top1_mass": sum(freqs[:1]) / total,
            "top5_mass": sum(freqs[:5]) / total,
            "top10_mass": sum(freqs[:10]) / total,
            "freq_p10": quantile(asc, 0.10),
            "freq_median": quantile(asc, 0.50),
            "freq_p90": quantile(asc, 0.90),
            "freq_max": freqs[0],
            "shannon_entropy_nats": H,
            "effective_support_expH": math.exp(H),
            "hhi": hhi,
            "per_coordinate_support": {
                f: {"distinct": len(coord[f]),
                    "counts": {str(k): v for k, v in
                               coord[f].most_common()}}
                for f in FIELDS},
            "joint_support": {
                k: len(v) for k, v in joint.items()},
        },
        "bands": {
            name: band_census(path, n, train_tup, train_rules,
                              field_support)
            for name, (path, n) in BANDS.items()},
        "pins": {p: fsha(p) for p in PINS},
        "start": START, "completion_commit": completion_commit()}
    RECEIPT.write_text(json.dumps(receipt, indent=1))
    out = json.loads(json.dumps(receipt))
    out["training"].pop("tuple_frequencies")
    for f in FIELDS:
        oc = out["training"]["per_coordinate_support"][f]
        if len(oc["counts"]) > 12:
            oc["counts"] = dict(list(oc["counts"].items())[:12])
            oc["counts"]["..."] = "truncated (full in receipt)"
    out.pop("start"); out.pop("pins")
    print(json.dumps(out, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
