"""Observations adapter for PRE-REG QWEN-PREFILL-WALL-0: derives the
registered measurements from the held-out receipt of
scratch/qwen_prefill_wall.py (STAGE=heldout) in committed code.

Per candidate (CAND512: chunk 512; CAND1024: chunk 1,024), against the
production chunk 256 paired in ABBA order on the held-out prompts:
  1/6   ttft_ratio_1024_<cand>     median TTFT ratio at the 1,024-token prompt
  2/7   ttft_ratio_2048_<cand>     the same at 2,048
  3/8   ttft_ratio_512_<cand>      the same at 512 (no material regression)
  4/9   parity_mismatches_<cand>   runs whose final argmax or 16 greedy decode
                                   tokens differ from production's
  5/10  peak_ratio_<cand>          the largest per-run peak-memory ratio over the
                                   three lengths (the footprint is flattened by the
                                   2 GiB pool cap and is reported, not gated), with
                                   the conjuncts <id>:e2e the end-to-end (prefill
                                   plus 16 decode steps) ratio at 2,048 and
                                   <id>:decode the largest decode-time ratio
Diagnostics: production spread (ABBA max minus min over median) per length,
top-16 logit differences, peak ratios.
Validity: a clean-tree non-smoke receipt on the registered artifact chain,
the three held-out prompts present with the registered sha256s, three ABBA
blocks per candidate per length (12 runs), production self-consistent (its
own runs agree on argmax and decode).
Usage: qwen_prefill_obs.py <heldout.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

LENGTHS = {"512": 512, "1024": 1024, "2048": 2048}


def main(receipt_path, prereg_path):
    r = json.load(open(receipt_path))
    prereg = load(prereg_path)
    reasons = []
    if r.get("smoke") or r.get("tree_dirty"):
        reasons.append("smoke run or dirty tree")
    if r.get("stage") != "heldout":
        reasons.append("not the held-out stage")
    prompts = r.get("prompts", {})
    frozen = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evalsets", "qwen_prefill_heldout_prompts.json")))["prompts"]
    if set(prompts) != set(LENGTHS):
        reasons.append(f"prompts {sorted(prompts)} are not the three held-out lengths")
    for t in LENGTHS:
        if t in prompts and prompts[t].get("sha256") != frozen[t]["sha256"]:
            reasons.append(f"held-out prompt {t} differs from the frozen scratch/evalsets/qwen_prefill_heldout_prompts.json")
    runs = r.get("runs", [])
    for cand in ("CAND512", "CAND1024"):
        for t in LENGTHS:
            n = sum(1 for x in runs if x["arm"] == cand and str(x["tokens"]) == str(prompts.get(t, {}).get("tokens")))
            if n != 6:
                reasons.append(f"{cand} at {t}: {n} runs, not 6 (three ABBA blocks)")
    s = r.get("summary", {})
    if any(v.get("prod_self_mismatches", 1) for v in s.values()):
        reasons.append("production disagrees with itself on argmax or decode")
    if not r.get("engine", {}).get("chain_sha256"):
        reasons.append("no artifact chain recorded")
    if r.get("engine", {}).get("cache_limit_bytes") != 2 ** 31 or r.get("engine", {}).get("matmul") != "gemm":
        reasons.append("the engine is not the qualified serving configuration (2 GiB cache limit, GEMM prefill)")
    for x in runs:
        want = {"PROD": 256, "CAND512": 512, "CAND1024": 1024}.get(x.get("arm"))
        if want is not None and x.get("chunk") != want:
            reasons.append(f"a {x.get('arm')} run recorded chunk {x.get('chunk')}")
            break
    if any(v.get("prod_spread_pct", 0) > 10.0 for v in s.values()):
        reasons.append("production's ABBA spread exceeds 10 percent at some length: the fences cannot be read against this noise")
    valid = not reasons
    meas = {}
    if valid:
        for cand, base in (("CAND512", 1), ("CAND1024", 6)):
            c = cand.lower()
            rows = {t: s[f"{cand}:{LENGTHS[t]}"] for t in LENGTHS}
            prov = {t: {"ttft": round(rows[t]["ttft_median_s"], 3), "prod": round(rows[t]["prod_ttft_median_s"], 3), "spread_pct": round(rows[t]["prod_spread_pct"], 1), "top16_diff": rows[t]["top16_max_abs_diff"]} for t in LENGTHS}
            meas[str(base)] = {"value": rows["1024"]["ttft_ratio"], "metric": f"ttft_ratio_1024_{c}", "population": "heldout-1024:6v6", "aggregation": "median_ratio", "unit": "ratio", "provenance": json.dumps(prov)}
            meas[str(base + 1)] = {"value": rows["2048"]["ttft_ratio"], "metric": f"ttft_ratio_2048_{c}", "population": "heldout-2048:6v6", "aggregation": "median_ratio", "unit": "ratio"}
            meas[str(base + 2)] = {"value": rows["512"]["ttft_ratio"], "metric": f"ttft_ratio_512_{c}", "population": "heldout-512:6v6", "aggregation": "median_ratio", "unit": "ratio"}
            meas[str(base + 3)] = {"value": sum(rows[t]["parity_mismatches"] for t in LENGTHS), "metric": f"parity_mismatches_{c}", "population": "heldout-runs:18", "aggregation": "count", "unit": "runs",
                                   "provenance": f"top-16 max abs diff by length {[rows[t]['top16_max_abs_diff'] for t in LENGTHS]}"}
            meas[str(base + 4)] = {"value": max(rows[t]["peak_ratio"] for t in LENGTHS), "metric": f"peak_ratio_{c}", "population": "heldout-lengths:3", "aggregation": "max_ratio", "unit": "ratio",
                                   "provenance": f"footprint ratios {[round(rows[t]['footprint_ratio'], 3) for t in LENGTHS]} (flattened by the 2 GiB pool cap); transient ratios {[round(rows[t]['transient_ratio'], 3) for t in LENGTHS]}"}
            meas[f"{base + 4}:e2e"] = {"value": rows["2048"]["e2e_ratio"], "metric": f"e2e_ratio_2048_{c}", "population": "heldout-2048:6v6", "aggregation": "median_ratio", "unit": "ratio"}
            meas[f"{base + 4}:decode"] = {"value": max(rows[t]["decode_s_ratio"] for t in LENGTHS), "metric": f"decode_s_ratio_{c}", "population": "heldout-lengths:3", "aggregation": "max_median_ratio", "unit": "ratio",
                                          "provenance": "the 16 greedy decode steps after the prefill, candidate over production: a slower decode after a faster prefill would be a hidden cost"}
        meas["ttft_ratio_1024_cand512"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {receipt_path} (commit {r.get('commit')}); adapter scratch/qwen_prefill_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"PROD": {"admissible": True, "reason": "the qualified serving path: chunk 256, GEMM prefill, final head, 2 GiB cache limit"},
                    "CAND512": {"admissible": valid, "reason": "the same path with chunk 512, a diagnostic in this process only"},
                    "CAND1024": {"admissible": valid, "reason": "the same path with chunk 1,024, a diagnostic in this process only"}},
           "measurements": meas, "diagnostics": s}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
