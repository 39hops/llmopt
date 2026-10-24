"""Observations adapter for PRE-REG QWEN-LOW-DEFAULT-1: derives the registered
measurements from the integration-smoke receipt (scratch/qwen_default1_smoke.py)
in committed code.

  1  unsafe_reuse_turns         turns taken incrementally whose rendered prompt did not
                                continue the fed ids exactly (the contract's core)
  2  replay_token_mismatch      smoke turns whose incremental ids differ from replay's,
                                plus the fixture turn against replay and the frozen receipt
  3  offset_mismatch_turns      turns whose measured KV offset differs from the accounting
  4  fixture_outcome            1 when the fixture's rendering did not extend the fed ids
                                AND the session replayed AND the tokens equal the replay
                                reference AND the offsets held, else 0
  5  incompatible_change_reused mid-session effort or thinking changes that reused the cache
  6  tests_failed               failed or errored tests across the release suites
Diagnostics: fallback count and first-token latency against incremental and
replay, tokens, time, memory, the defaults the receipt records.
Validity: a clean-tree receipt on the registered artifact chain, the defaults
recorded as thinking True / low / 768 / incremental, two smoke conversations
of four turns in both modes, the fixture and contract blocks present, the
three test suites run (returncode recorded; a non-zero code counts as failed
tests through the summary).
Usage: qwen_default1_obs.py <default1_smoke.json> <prereg.json> > observations.json
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
    if r.get("tree_dirty"):
        reasons.append("dirty tree")
    if r.get("defaults") != {"enable_thinking": True, "reasoning_effort": "low", "max_tokens": 768, "mode": "incremental"}:
        reasons.append(f"defaults recorded as {r.get('defaults')}")
    if [c["name"] for c in r.get("conversations", [])] != ["programming", "factual"] or any(len(c["incremental"]) != 4 or len(c["replay"]) != 4 for c in r.get("conversations", [])):
        reasons.append("the smoke conversations are not the registered two of four turns")
    if not r.get("fixture") or not r.get("contract") or set(r.get("tests", {})) != {"defaults_stub", "template", "release"}:
        reasons.append("fixture, contract or test blocks missing")
    if not r.get("engine", {}).get("chain_sha256"):
        reasons.append("no artifact chain recorded")
    valid = not reasons
    meas = {}
    if valid:
        s, f = r["summary"], r["fixture"]
        n_turns = s["smoke_turns"] * 2 + len([x for x in r["contract"]["rows"] if "mode" in x]) + 2
        meas["1"] = {"value": s["unsafe_reuse_turns"], "metric": "unsafe_reuse_turns", "population": "turns", "aggregation": "count", "unit": "turns", "provenance": f"{n_turns} turns in smoke, contract and fixture"}
        meas["2"] = {"value": s["smoke_token_mismatch_turns"] + int(not s["fixture_tokens_equal_replay_and_receipt"]), "metric": "replay_token_mismatch", "population": "turns", "aggregation": "count", "unit": "turns",
                     "provenance": f"smoke mismatches {s['smoke_token_mismatch_turns']} of {s['smoke_turns']}; fixture equal to replay {f['generated_equal_replay']} and to the frozen receipt {f['generated_equal_frozen_receipt']}"}
        meas["3"] = {"value": s["offset_mismatch_turns"] + int(not f["offsets_ok"]), "metric": "offset_mismatch_turns", "population": "turns", "aggregation": "count", "unit": "turns"}
        meas["4"] = {"value": int(s["fixture_detected_and_replayed"] and s["fixture_tokens_equal_replay_and_receipt"] and f["offsets_ok"]), "metric": "fixture_outcome", "population": "fixture:1", "aggregation": "indicator", "unit": "fixtures",
                     "provenance": f"extends {f['rendering_extends_fed']} mode {f['mode']} break offset {f['break_offset_in_generation']} n_generated {f['n_generated']} first token fallback {f['first_token_s_fallback']:.2f} s replay {f['first_token_s_replay']:.2f} s"}
        meas["5"] = {"value": s["incompatible_change_reused"], "metric": "incompatible_change_reused", "population": "changes:2", "aggregation": "count", "unit": "turns",
                     "provenance": f"effort change {r['contract']['effort_change_mode']}, thinking change {r['contract']['thinking_change_mode']}; budget stop {r['contract']['budget_stop']}, cap refused {r['contract']['cap_refused']}, cache dropped {r['contract']['cap_dropped_cache']}, recovery {r['contract']['recovery_mode']}, reset {r['contract']['reset_ok']}"}
        meas["6"] = {"value": s["tests_failed"] + sum(1 for v in s["tests_returncodes"].values() if v != 0 and s["tests_failed"] == 0), "metric": "tests_failed", "population": "suites:3", "aggregation": "count", "unit": "tests",
                     "provenance": f"passed {s['tests_passed']}; returncodes {s['tests_returncodes']}; summaries {[t['summary'] for t in r['tests'].values()]}"}
        meas["unsafe_reuse_turns"] = dict(meas["1"])
    obs = {"note": f"Mechanical projection of {receipt_path} (commit {r.get('commit')}); adapter scratch/qwen_default1_obs.py",
           "registration_sha256": registration_digest(prereg),
           "measurement_valid": valid, **({} if valid else {"measurement_reason": "; ".join(reasons)}),
           "arms": {"DEFAULT": {"admissible": valid, "reason": "Session and CLI with no controls given: thinking True, low, 768, incremental, greedy, generation 5"},
                    "REPLAY": {"admissible": valid, "reason": "the same transcripts through full replay (the reference)"}},
           "measurements": meas, "diagnostics": r.get("summary", {})}
    json.dump(obs, sys.stdout, indent=1)
    sys.stdout.write("\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
