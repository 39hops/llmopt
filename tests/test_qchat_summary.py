"""The CLI's operational accounting (QWEN-PREFILL-WALL-0): derived from the
recorded turn results only, never from cache logic."""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
import qwen_chat  # noqa: E402


def _turn(mode, note, n_prompt, ttft, total, active, footprint, peak, cache_tokens):
    return {"mode": mode, "note": note, "prompt_ids": list(range(n_prompt)), "generated_ids": [1, 2], "text": "x", "stop_reason": "stop_token",
            "stop_token_id": 248046, "cache_tokens": cache_tokens, "timings": {"first_token_s": ttft, "total_s": total, "decode_tok_per_s": 9.0},
            "memory": {"active_bytes": active, "footprint_bytes": footprint, "peak_bytes": peak, "cache_bytes": 0}}


def test_summary_counts_modes_fallbacks_latency_and_memory_from_the_turns():
    turns = [_turn("fresh", "", 40, 2.0, 5.0, 9.0e9, 11.0e9, 11.5e9, 60),
             _turn("incremental", "", 20, 2.1, 4.0, 9.0e9, 11.1e9, 11.6e9, 100),
             _turn("replay-fallback", "the rendered transcript does not extend the fed token sequence; replayed into a fresh cache", 130, 8.0, 10.0, 9.1e9, 11.2e9, 12.0e9, 150),
             _turn("incremental", "", 25, 2.3, 4.5, 9.0e9, 11.1e9, 11.7e9, 190)]
    s = qwen_chat.summarize_turns(turns)
    assert s["turns"] == 4 and s["incremental_reuses"] == 2 and s["replay_fallbacks"] == 1 and s["fresh_turns"] == 1
    assert s["fallback_reasons"] == {"the rendered transcript does not extend the fed token sequence; replayed into a fresh cache": 1}
    assert s["first_token_s_by_mode"] == {"fresh": [2.0], "incremental": [2.1, 2.3], "replay-fallback": [8.0]}
    assert s["first_token_s_median_by_mode"]["incremental"] == 2.2 and s["first_token_s_median_by_mode"]["replay-fallback"] == 8.0
    assert s["replayed_prompt_tokens"] == 130 and s["replay_cost_s"] == 8.0        # the fallback's whole prompt was re-prefilled at that latency
    assert s["prompt_tokens_total"] == 215 and s["generated_tokens_total"] == 8
    assert s["peak_bytes_max"] == 12.0e9 and s["active_bytes_last"] == 9.0e9 and s["footprint_bytes_max"] == 11.2e9 and s["cache_tokens_last"] == 190


def test_summary_of_no_turns_is_empty_and_well_formed():
    s = qwen_chat.summarize_turns([])
    assert s["turns"] == 0 and s["incremental_reuses"] == 0 and s["replay_fallbacks"] == 0 and s["replayed_prompt_tokens"] == 0
    assert s["first_token_s_median_by_mode"] == {} and s["peak_bytes_max"] is None
