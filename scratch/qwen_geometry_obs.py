"""Observations adapter for PRE-REG QWEN-LOSSY-GEOMETRY-0: derives the
registered measurements from the eval receipt
(scratch/qwen_geometry_quant.py MODE=eval) and the transform receipt
(scratch/qwen_geometry_transform.py) in committed code.

  bar 1  best_w4_ratio: min over the candidates B, C, D, E of
         max(ratio on gate_proj, ratio on down_proj), where ratio is the
         candidate's pooled relative output MSE over the control's on
         the EVAL prompts; the candidate attaining it is "the best"
  1:ci   best_w4_ci_upper: the larger of that candidate's bootstrap 95
         percent upper bounds on the two W4 tensors
  1:bits best_w4_bits_excess: that candidate's bits per weight minus the
         control's, the larger of the two tensors
  bar 2  best_s16_ratio: the same min over candidates on in_proj_qkv
  bar 3  transform_step_ms: the projected per-step cost of the rotation
         as unfused MLX ops (always reported; it conditions the GO/HOLD
         reading when the best candidate is rotated)
Validity: both receipts clean-tree non-smoke, the eval receipt on the
eval split, every candidate evaluated on every tensor, the awq candidate
fit on dev prompts disjoint from the eval prompts.
Usage: qwen_geometry_obs.py <eval.json> <transform.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402

CANDS = ["B_refit", "C_rht", "D_higgs", "E_awq"]


def main(eval_path, xform_path, prereg_path):
    ev = json.load(open(eval_path))
    xf = json.load(open(xform_path))
    prereg = load(prereg_path)
    reasons = []
    if ev.get("smoke") or ev.get("tree_dirty") or xf.get("tree_dirty"):
        reasons.append("a receipt is a smoke run or was written on a dirty tree")
    if ev.get("mode") != "eval" or ev.get("split") != "eval":
        reasons.append("the fidelity receipt is not the eval split")
    if set(ev.get("prompts", [])) & set(ev.get("fit_prompts_for_awq", [])):
        reasons.append("awq fit prompts overlap the eval prompts")
    by_short = {t["short"]: t for t in ev["tensors"].values()}
    need = {"22.mlp.gate_proj", "22.mlp.down_proj", "22.linear_attn.in_proj_qkv"}
    if not need.issubset(by_short):
        reasons.append(f"tensors missing: {need - set(by_short)}")
    for sh, t in by_short.items():
        for c in CANDS:
            if c not in t["candidates"] or c not in t["bootstrap_vs_control"]:
                reasons.append(f"{sh} lacks {c}")
    valid = not reasons
    meas = {}
    if valid:
        g, d, q = by_short["22.mlp.gate_proj"], by_short["22.mlp.down_proj"], by_short["22.linear_attn.in_proj_qkv"]
        score = {c: max(g["bootstrap_vs_control"][c]["ratio_candidate_over_control"], d["bootstrap_vs_control"][c]["ratio_candidate_over_control"]) for c in CANDS}
        best = min(CANDS, key=lambda c: (score[c], c))
        ci = max(g["bootstrap_vs_control"][best]["boot_ci95"][1], d["bootstrap_vs_control"][best]["boot_ci95"][1])
        bits = max(g["candidates"][best]["bits_per_weight"] - g["candidates"]["A_control"]["bits_per_weight"],
                   d["candidates"][best]["bits_per_weight"] - d["candidates"]["A_control"]["bits_per_weight"])
        s16 = {c: q["bootstrap_vs_control"][c]["ratio_candidate_over_control"] for c in CANDS}
        best16 = min(CANDS, key=lambda c: (s16[c], c))
        pop = f"eval-prompts:{len(ev['prompts'])}"
        meas["1"] = {"value": score[best], "metric": "best_w4_ratio", "population": pop, "aggregation": "min_over_candidates_of_max_over_w4_tensors",
                     "provenance": f"best {best}; scores {json.dumps({c: round(v, 4) for c, v in score.items()})}; gate ratios "
                                   f"{json.dumps({c: round(g['bootstrap_vs_control'][c]['ratio_candidate_over_control'], 4) for c in CANDS})}; down ratios "
                                   f"{json.dumps({c: round(d['bootstrap_vs_control'][c]['ratio_candidate_over_control'], 4) for c in CANDS})}"}
        meas["1:ci"] = {"value": ci, "metric": "best_w4_ci_upper", "population": pop, "aggregation": "max_over_w4_tensors_of_boot95_upper"}
        meas["1:bits"] = {"value": bits, "metric": "best_w4_bits_excess", "population": pop, "aggregation": "max_over_w4_tensors_of_bits_minus_control"}
        meas["2"] = {"value": s16[best16], "metric": "best_s16_ratio", "population": pop, "aggregation": "min_over_candidates",
                     "provenance": f"best {best16}; ratios {json.dumps({c: round(v, 4) for c, v in s16.items()})}"}
        meas["3"] = {"value": xf["projection"]["per_step_ms_unfused_mlx_ops"], "metric": "transform_step_ms", "population": "decode-step:64-layers",
                     "aggregation": "64_times_best_impl_per_layer_sum", "provenance": json.dumps(xf["projection"]["best_impl"])}
        meas["best_w4_ratio"] = dict(meas["1"])
        meas["best_candidate"] = {"value": CANDS.index(best), "metric": "best_candidate_index", "population": pop, "aggregation": "argmin"}
    obs = {"note": f"Mechanical projection of {eval_path} (commit {ev.get('commit')}) and {xform_path} (commit {xf.get('commit')}); adapter scratch/qwen_geometry_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"CONTROL": {"admissible": True, "reason": "arm C payload through the canonical decoder"},
                    "CANDIDATES": {"admissible": valid, "reason": "all four candidates on all three tensors, eval split, awq fit on dev only" if valid else "; ".join(reasons)}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
