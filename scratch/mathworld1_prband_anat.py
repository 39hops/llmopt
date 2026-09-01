"""MATH-CYBER-1 PRIOR-RESISTANT-EVAL-MATERIALIZATION-0 — descriptive
census anatomy of the NO-FIRE horizon (read-only over
logs/mathworld1/prband/horizon_census.jsonl; no law, no selection,
no model). Writes logs/mathworld1/prband_verify/anatomy.json
(refuse-if-exists): per-signature gold census, lowest-index rule
counts and exception shapes, within-k co-occurrence, parity-v-k
alignment, gold-class-v-k table, the a-states of the two mixed
signatures, and the companion (after) side of the same quantities.

    .venv/bin/python scratch/mathworld1_prband_anat.py
"""
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path("logs/mathworld1/prband/horizon_census.jsonl")
OUT = Path("logs/mathworld1/prband_verify/anatomy.json")


def side(rows, pick):
    sig = defaultdict(Counter)
    sigjs = {}
    low = Counter()
    exc = Counter()
    for r in rows:
        v = r["variants"][pick(r)]
        g = v["gold_tuple"][4]
        sig[v["cand_sig_id"]][g] += 1
        sigjs[v["cand_sig_id"]] = v["cand_sig"]
        tl = sorted(t[4] for t in json.loads(v["cand_sig"])
                    if t[0] == "i_unprod")
        low[g == tl[0]] += 1
        if g != tl[0]:
            exc[f"{r['base_signature'][-1]}|{tl}->{g}"] += 1
    multi = {s: dict(c) for s, c in sig.items() if len(c) > 1}
    both23 = {s: dict(c) for s, c in sig.items() if 2 in c and 3 in c}
    within_k = {}
    for k in ("1", "2"):
        sk = defaultdict(Counter)
        for r in rows:
            if r["base_signature"].endswith("|k=" + k):
                v = r["variants"][pick(r)]
                sk[v["cand_sig_id"]][v["gold_tuple"][4]] += 1
        within_k[k] = {"n_signatures": len(sk),
                       "with_t2_and_t3": sum(1 for c in sk.values()
                                             if 2 in c and 3 in c)}
    return {"n_signatures": len(sig),
            "single_gold": sum(1 for c in sig.values() if len(c) == 1),
            "multi_gold": multi, "with_t2_and_t3": both23,
            "gold_is_lowest_unprod_index": {str(k): v
                                           for k, v in low.items()},
            "exception_shapes": dict(exc), "within_k": within_k,
            "signature_gold_census": {s: dict(c) for s, c in sig.items()},
            "signature_unprod_terms": {
                s: sorted(t[4] for t in json.loads(js)
                          if t[0] == "i_unprod") for s, js in sigjs.items()},
            "signature_other_rules": {
                s: sorted(t[0] for t in json.loads(js)
                          if t[0] != "i_unprod") for s, js in sigjs.items()}}


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    rows = [json.loads(l) for l in open(SRC)]
    rec = {"source": str(SRC),
           "source_sha256": hashlib.sha256(SRC.read_bytes()).hexdigest(),
           "n_rows": len(rows),
           "primary": side(rows, lambda r: r["primary_variant"]),
           "companion_after": side(rows, lambda r: "after"),
           "variant_x_k": dict(Counter(
               f"{r['primary_variant']}|k={r['base_signature'][-1]}"
               for r in rows)),
           "gold_class_x_k": dict(Counter(
               f"{r['variants'][r['primary_variant']]['gold_class']}"
               f"|k={r['base_signature'][-1]}" for r in rows)),
           "mixed_signature_a_states": sorted(
               r["base_signature"] for r in rows
               if r["variants"][r["primary_variant"]]["gold_class"] == "a"
               and r["variants"][r["primary_variant"]]["cand_sig_id"]
               in {s for s, c in side(rows, lambda r: r["primary_variant"])
                   ["with_t2_and_t3"].items()})}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rec, indent=1))
    p = rec["primary"]
    print(json.dumps({k: p[k] for k in ("n_signatures", "single_gold",
                                        "with_t2_and_t3",
                                        "gold_is_lowest_unprod_index",
                                        "exception_shapes", "within_k")},
                     indent=1))
    print("companion", rec["companion_after"]["n_signatures"],
          rec["companion_after"]["single_gold"],
          rec["companion_after"]["gold_is_lowest_unprod_index"])
    print("variant_x_k", rec["variant_x_k"])
    print("gold_class_x_k", rec["gold_class_x_k"])
    print("mixed a-states", len(rec["mixed_signature_a_states"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
