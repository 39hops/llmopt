"""Observations adapter for PRE-REG QWEN-CHAT-DEFAULT-0: derives the
registered measurements from the incremental-against-replay receipt
(scratch/qwen_incr_perf.py) in committed code.

  bar 1  token_mismatch_turns: turns whose generated ids or stop reason
         differ between the incremental and the replay session
  bar 2  ttft_ratio_later_median: the median over the later turns of the
         thinking-off conversations of incremental first-token latency
         over replay's
  bar 3  active_growth_mb_excess_max: the largest first-to-last-turn growth
         of active memory in an incremental conversation in excess of the
         same conversation's replay growth, MB (retention beyond the cache)
  bar 4  offset_mismatch_turns: turns whose measured KV offset differs
         from the accounting
  4:reuse later_thinking_off_reuse_fraction: the fraction of later
         thinking-off turns taken through the incremental path
Validity: a clean-tree non-smoke receipt with all five conversations
present, replay and incremental the same length, no budget stop missing.
Usage: qwen_incr_obs.py <incr.json> <prereg.json> > observations.json
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
    if names != ["travel", "debug", "recipe", "thinking", "budget-cut"]:
        reasons.append(f"conversations {names} are not the registered five")
    if any(len(c["replay"]) != len(c["incremental"]) for c in r.get("conversations", [])):
        reasons.append("a conversation has unequal turn counts between modes")
    convs = r.get("conversations", [])
    if not (len(convs) == 5 and convs[4]["replay"][1]["stop_reason"] == "max_tokens" and convs[4]["incremental"][1]["stop_reason"] == "max_tokens"):
        reasons.append("the registered budget-cut turn (conversation 5, turn 2) did not stop on the budget in both modes")
    valid = not reasons
    meas = {}
    if valid:
        s = r["summary"]
        pop = f"turns:{s['turns_compared']}"
        meas["1"] = {"value": s["token_mismatch_turns"], "metric": "token_mismatch_turns", "population": pop, "aggregation": "count",
                     "provenance": f"per conversation {[c['mismatch_turns'] for c in r['conversations']]}"}
        meas["2"] = {"value": s["ttft_ratio_later_median_thinking_off"], "metric": "ttft_ratio_later_median", "population": f"later-thinking-off-turns:{s['later_thinking_off_turns']}",
                     "aggregation": "median_of_per_turn_ratio",
                     "provenance": f"replay later TTFT s {s['replay_later_ttft_s']}; incremental {s['incremental_later_ttft_s']}"}
        meas["3"] = {"value": s["active_growth_bytes_excess_max"] / 1e6, "metric": "active_growth_mb_excess_max", "population": "conversations:5",
                     "aggregation": "max_over_conversations_of_incremental_growth_minus_replay_growth",
                     "provenance": f"incremental max growth {s['active_growth_bytes_incremental_max'] / 1e6:.1f} MB, replay max growth {s['active_growth_bytes_replay_max'] / 1e6:.1f} MB; "
                                   f"per conversation {[round((c['active_growth_bytes_incremental'] - c['active_growth_bytes_replay']) / 1e6, 1) for c in r['conversations']]}"}
        meas["4"] = {"value": s["offset_mismatch_turns"], "metric": "offset_mismatch_turns", "population": f"turns:{2 * s['turns_compared']}", "aggregation": "count"}
        meas["4:reuse"] = {"value": s["later_thinking_off_incremental_taken"] / s["later_thinking_off_turns"], "metric": "later_thinking_off_reuse_fraction",
                           "population": f"later-thinking-off-turns:{s['later_thinking_off_turns']}", "aggregation": "fraction",
                           "provenance": f"thinking conversation later modes {s['thinking_conversation_later_modes']}"}
        meas["token_mismatch_turns"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {receipt_path} (commit {r.get('commit')}); adapter scratch/qwen_incr_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"REPLAY": {"admissible": True, "reason": "full-transcript replay into a fresh cache every turn (the reference)"},
                    "INCREMENTAL": {"admissible": valid, "reason": "cache reuse under the exact rendered-token prefix check with replay fallback"}},
           "measurements": meas}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
