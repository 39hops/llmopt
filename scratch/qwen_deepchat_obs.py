"""Observations adapter for PRE-REG QWEN-DEEPCHAT-0 (study B of the GO
QWEN-EFFORT-AND-DEEPCHAT-0): derives the registered diagnostic
measurements from the receipt of scratch/qwen_deepchat_diag.py in
committed code.

  bar 1  travel_empty_turns: empty replay completions in the frozen travel
         conversation (reproduction of the observed behaviour)
  bar 2  reconstruction_mismatch_turns: turns whose independently rendered
         prompt differs from the session's (template rendering)
  bar 3  incremental_first_argmax_mismatch_turns: turns whose incremental
         first-token argmax differs from replay's, with the conjunct 3:gen
         (generated ids differ) and 3:offset (measured offset mismatches)
         (token and cache accounting)
  bar 4  gemm_vs_fused_max_abs_diff: the largest absolute difference of a
         shared top-16 first-token logit between the GEMM and fused prefill
         paths on the stop-chosen turns, with the conjunct 4:argmax (argmax
         mismatches between the paths) (numerical consistency)
  bar 5  min_stop_margin: the smallest margin by which a chosen stop token
         beat the best non-stop token on the stop-chosen turns (a near tie
         would read as a numerical coin flip rather than a model decision)
Diagnostics (not adjudicated): the stop token chosen, the margins per
turn, the counterfactual single-turn and empties-removed outcomes.
Validity: a clean-tree non-smoke receipt with the three frozen
conversations complete.
Usage: qwen_deepchat_obs.py <deepchat.json> <prereg.json> > observations.json
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


def main(receipt_path, prereg_path):
    r = json.load(open(receipt_path))
    prereg = load(prereg_path)
    reasons = []
    if r.get("smoke") or r.get("tree_dirty"):
        reasons.append("smoke run or dirty tree")
    names = [c["name"] for c in r.get("conversations", [])]
    if names != ["travel", "debug", "recipe"]:
        reasons.append(f"conversations {names} are not the registered three")
    if any(len(c["turns"]) != 8 for c in r.get("conversations", [])):
        reasons.append("a conversation is incomplete")
    valid = not reasons
    meas = {}
    if valid:
        s = r["summary"]
        n_turns = sum(len(c["turns"]) for c in r["conversations"])
        n_stop = len(s["stop_margins"])
        meas["1"] = {"value": s["travel_empty_turns"], "metric": "travel_empty_turns", "population": "travel-turns:8", "aggregation": "count", "unit": "turns",
                     "provenance": f"empty replay turns {s['empty_replay_turns']}; stop tokens chosen {s['stop_tokens_chosen']}"}
        meas["2"] = {"value": s["reconstruction_mismatch_turns"], "metric": "reconstruction_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns"}
        meas["3"] = {"value": s["incremental_first_argmax_mismatch_turns"], "metric": "incremental_first_argmax_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns",
                     "provenance": f"replay vs incremental max abs top-16 logit diff {s['replay_vs_incremental_max_abs_diff']:.4g}"}
        meas["3:gen"] = {"value": s["incremental_generated_mismatch_turns"], "metric": "incremental_generated_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns"}
        meas["3:offset"] = {"value": s["offset_mismatch_turns"], "metric": "offset_mismatch_turns", "population": f"turns:{2 * n_turns}", "aggregation": "count", "unit": "turns"}
        if n_stop:
            meas["4"] = {"value": s["gemm_vs_fused_max_abs_diff"], "metric": "gemm_vs_fused_max_abs_diff", "population": "stop-chosen-turns", "aggregation": "max", "unit": "logit"}
            meas["4:argmax"] = {"value": s["gemm_vs_fused_argmax_mismatch_turns"], "metric": "gemm_vs_fused_argmax_mismatch_turns", "population": "stop-chosen-turns", "aggregation": "count", "unit": "turns"}
            meas["5"] = {"value": s["min_stop_margin"], "metric": "min_stop_margin", "population": "stop-chosen-turns", "aggregation": "min", "unit": "logit",
                         "provenance": f"margins {s['stop_margins']}; counterfactual single-turn non-empty {s['counterfactual_single_turn_nonempty']} of {n_stop}; "
                                       f"empties-removed non-empty {s['counterfactual_empties_removed_nonempty']} of {s['counterfactual_empties_removed_rows']}"}
        meas["travel_empty_turns"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {receipt_path} (commit {r.get('commit')}); adapter scratch/qwen_deepchat_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"REPLAY": {"admissible": True, "reason": "full-transcript replay sessions on the frozen conversations"},
                    "INCREMENTAL": {"admissible": valid, "reason": "incremental sessions on the same conversations"},
                    "RECONSTRUCTION": {"admissible": valid, "reason": "the transcript re-rendered outside Session; the fused prefill path on the stop-chosen turns"}},
           "measurements": meas, "diagnostics": r.get("summary", {})}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
