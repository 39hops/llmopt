"""Observations adapter for PRE-REG QWEN-DEEPCHAT-0 as amended by
AMENDMENT QWEN-DEEPCHAT-0-ADAPTER (study B of the GO
QWEN-EFFORT-AND-DEEPCHAT-0): derives the registered diagnostic
measurements from the receipt of scratch/qwen_deepchat_diag.py in
committed code. Bar 2 re-renders every replay turn's transcript as a
ChatML string built BY HAND from the user texts and the decoded replies
(the thinking-off form: an empty think block opens each assistant turn),
tokenises it with the pinned tokenizer WITHOUT the chat template, and
compares the sha256 of the ids with the session's recorded prompt
sha256; the conjunct counts later incremental turns that fell back.
Bar 3's offset conjunct compares the incremental session's measured
offset with replay's on each turn.

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

import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from llmopt.lab.prereg import load, registration_digest  # noqa: E402


VENDOR = os.path.expanduser("~/qwen_vendor")
GEN_PROMPT_OFF = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def chatml_sha(tok, turns, k):
    """sha256 of the ids of a hand-built thinking-off ChatML transcript up to the user turn k (0-based)."""
    s = ""
    for j in range(k):
        s += f"<|im_start|>user\n{turns[j]['user'].strip()}<|im_end|>\n"
        s += GEN_PROMPT_OFF + turns[j]["replay"]["text"].strip() + "<|im_end|>\n"
    s += f"<|im_start|>user\n{turns[k]['user'].strip()}<|im_end|>\n" + GEN_PROMPT_OFF
    ids = [int(t) for t in tok(s, add_special_tokens=False)["input_ids"]]
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest(), len(ids)


def main(receipt_path, prereg_path):
    r = json.load(open(receipt_path))
    prereg = load(prereg_path)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(VENDOR)
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
        chatml_mism, lens = [], []
        for c in r["conversations"]:
            if c["thinking"]:
                continue
            for k, t in enumerate(c["turns"]):
                h, n = chatml_sha(tok, c["turns"], k)
                lens.append((n, t["replay"]["prompt_len"]))
                if h != t["replay"]["prompt_sha256"]:
                    chatml_mism.append((c["name"], t["turn"], n, t["replay"]["prompt_len"]))
        later_fallbacks = [(c["name"], t["turn"]) for c in r["conversations"] for t in c["turns"][1:] if t["incremental"]["mode"] != "incremental"]
        offset_mism = [(c["name"], t["turn"]) for c in r["conversations"] for t in c["turns"]
                       if t["incremental"]["cache_offset_measured"] != t["replay"]["cache_offset_measured"] or t["incremental"]["cache_tokens"] != t["replay"]["cache_tokens"]]
        meas["2"] = {"value": len(chatml_mism), "metric": "chatml_reconstruction_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns",
                     "provenance": f"hand-built ChatML tokenised without the template; mismatches {chatml_mism}; in-run (template) reconstruction mismatches {s['reconstruction_mismatch_turns']}"}
        meas["2:reuse"] = {"value": len(later_fallbacks), "metric": "incremental_later_fallback_turns", "population": f"later-turns:{n_turns - len(r['conversations'])}", "aggregation": "count", "unit": "turns",
                           "provenance": f"fallbacks {later_fallbacks}"}
        meas["3"] = {"value": s["incremental_first_argmax_mismatch_turns"], "metric": "incremental_first_argmax_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns",
                     "provenance": f"replay vs incremental max abs top-16 logit diff {s['replay_vs_incremental_max_abs_diff']:.4g}"}
        meas["3:gen"] = {"value": s["incremental_generated_mismatch_turns"], "metric": "incremental_generated_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns"}
        meas["3:offset"] = {"value": len(offset_mism), "metric": "replay_vs_incremental_offset_mismatch_turns", "population": f"turns:{n_turns}", "aggregation": "count", "unit": "turns",
                            "provenance": f"mismatches {offset_mism}; within-session mismatches abort the run (counted {s['offset_mismatch_turns']})"}
        if n_stop:
            diffs = [t["gemm_vs_fused_first_logits"]["max_abs_diff_common"] for c in r["conversations"] for t in c["turns"] if "gemm_vs_fused_first_logits" in t]
            meas["4"] = {"value": 99.0 if any(x is None for x in diffs) else max(diffs), "metric": "gemm_vs_fused_max_abs_diff", "population": "stop-chosen-turns", "aggregation": "max", "unit": "logit",
                         "provenance": f"{n_stop} stop-chosen turns; per-turn max abs diff over the shared top-16 {[None if x is None else round(x, 6) for x in diffs]}; an empty intersection reads 99"}
            meas["4:argmax"] = {"value": s["gemm_vs_fused_argmax_mismatch_turns"], "metric": "gemm_vs_fused_argmax_mismatch_turns", "population": "stop-chosen-turns", "aggregation": "count", "unit": "turns"}
            meas["5"] = {"value": s["min_stop_margin"], "metric": "min_stop_margin", "population": "stop-chosen-turns", "aggregation": "min", "unit": "logit",
                         "provenance": f"margins {s['stop_margins']}; counterfactual single-turn non-empty {s['counterfactual_single_turn_nonempty']} of {n_stop}; "
                                       f"empties-removed non-empty {s['counterfactual_empties_removed_nonempty']} of {s['counterfactual_empties_removed_rows']}"}
        meas["chatml_reconstruction_mismatch_turns"] = dict(meas["2"])
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
