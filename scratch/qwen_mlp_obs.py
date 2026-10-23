"""Observations adapter for PRE-REG QWEN-MLP-COUPLED-0: derives the
registered measurements from the eval receipt (scratch/qwen_mlp_coupled.py
MODE=eval) and the dev receipt (the candidate's identity check) in
committed code.

  bar 1  mlp_out_ratio: pooled complete-MLP output MSE of the candidate
         over the control, eval split, all blocks
  1:ci   mlp_out_ci_upper: the paired bootstrap 95 percent upper bound
  1:cat  mlp_out_max_category_ratio: the largest per-category ratio
  1:bits gate_bits_excess: candidate gate bits per weight minus control's
  bar 2  gate_ratio: the gate output MSE ratio (does the local gain
         reproduce on the new collection)
  bar 3  freerun_ratio: the complete-output ratio on the free-running
         distribution alone
Validity: both receipts clean-tree non-smoke runs on the same capture;
the eval receipt on the eval split with the candidate fit on dev prompts
disjoint from the eval prompts and its codebook sha256 equal to the dev
receipt's; every eval block present for both distributions.
Usage: qwen_mlp_obs.py <eval.json> <dev.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


def main(eval_path, dev_path, prereg_path):
    ev = json.load(open(eval_path))
    dv = json.load(open(dev_path))
    prereg = load(prereg_path)
    reasons = []
    for r, nm in ((ev, "eval"), (dv, "dev")):
        if r.get("smoke") or r.get("tree_dirty"):
            reasons.append(f"{nm} receipt is a smoke run or dirty tree")
    if ev.get("mode") != "eval" or ev.get("split") != "eval":
        reasons.append("the fidelity receipt is not the eval split")
    if ev.get("capture_collection_sha256") != dv.get("capture_collection_sha256") or ev.get("capture_commit") != dv.get("capture_commit"):
        reasons.append("eval and dev receipts are not on the same capture")
    if set(ev["prompts"]) & set(ev["fit_prompts"]):
        reasons.append("fit prompts overlap the eval prompts")
    if ev["candidate"]["codebook_sha256"] != dv["candidate"]["codebook_sha256"]:
        reasons.append("the candidate's codebook differs between dev and eval runs")
    if not ev.get("pin_check", {}).get("ok"):
        reasons.append("shard pin failed")
    dists = {(b["id"], b["dist"]) for b in ev["blocks"]}
    if any((p, d) not in dists for p in ev["prompts"] for d in ("prefill", "freerun")):
        reasons.append("a block is missing")
    valid = not reasons
    meas = {}
    if valid:
        s = ev["summary"]
        pop = f"eval-blocks:{s['primary_out']['blocks']}"
        meas["1"] = {"value": s["primary_out"]["ratio"], "metric": "mlp_out_ratio", "population": pop,
                     "aggregation": "pooled_sse_candidate_over_control",
                     "provenance": f"control rel MSE {s['primary_out']['ctrl_rel_mse']:.5e}, candidate {s['primary_out']['cand_rel_mse']:.5e}; by category "
                                   f"{json.dumps({c: round(v['ratio'], 4) for c, v in s['by_category'].items()})}; by distribution "
                                   f"{json.dumps({d: round(v['ratio'], 4) for d, v in s['by_dist'].items()})}"}
        meas["1:ci"] = {"value": s["primary_out"]["ci95"][1], "metric": "mlp_out_ci_upper", "population": pop, "aggregation": "paired_boot95_upper_over_blocks"}
        meas["1:cat"] = {"value": s["max_category_ratio"], "metric": "mlp_out_max_category_ratio", "population": pop, "aggregation": "max_over_categories_of_pooled_ratio"}
        meas["1:bits"] = {"value": s["storage_bits_per_weight"]["candidate_gate"] - s["storage_bits_per_weight"]["control_gate"], "metric": "gate_bits_excess",
                          "population": "tensor:22.mlp.gate_proj", "aggregation": "bits_per_weight_minus_control"}
        meas["2"] = {"value": s["gate"]["ratio"], "metric": "gate_ratio", "population": pop, "aggregation": "pooled_sse_candidate_over_control"}
        meas["3"] = {"value": s["by_dist"]["freerun"]["ratio"], "metric": "freerun_ratio", "population": f"eval-blocks:{s['by_dist']['freerun']['blocks']}-freerun",
                     "aggregation": "pooled_sse_candidate_over_control"}
        meas["mlp_out_ratio"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {eval_path} (commit {ev.get('commit')}) with {dev_path} as the candidate identity check; adapter scratch/qwen_mlp_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"CONTROL": {"admissible": True, "reason": "arm C gate, up and down through the canonical decoder"},
                    "CANDIDATE": {"admissible": valid, "reason": "E_awq gate fit on the dev split, codebook identical in dev and eval runs, up and down unchanged" if valid else "; ".join(reasons)}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
