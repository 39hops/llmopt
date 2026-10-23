"""QWEN-CHAT-DEFAULT-0, part A: the registered comparison of incremental
cache reuse against full-transcript replay on realistic multi-turn
conversations, both modes from a fresh engine state per conversation,
one process, the serving defaults (kernel generation 5, the 2 GiB cache
limit, chunk-256 GEMM prefill, fused cached decode, greedy).

Conversations (fixed text, fixed per-turn budgets): three eight-turn
thinking-off conversations (a travel plan, a short debugging session, a
recipe), one four-turn conversation with enable_thinking True (the
pinned template's own control), and one thinking-off conversation whose
second turn is cut by a six-token budget before the follow-up. Replay runs
first, then incremental, each from a fresh session; the session is
deleted and the allocator cache cleared between the two.

Per turn: the mode taken (fresh, incremental, replay-fallback) and its
note, prompt and generated ids, stop reason, first-token latency, total
time, active, footprint and peak memory after the turn, the accounted
cache tokens and the KV cache's measured offset. Summary: the number of
turns whose generated ids differ between modes; the median over the
later turns (2 and on) of the thinking-off conversations of the
incremental first-token latency divided by replay's; the largest growth
of active memory from the first to the last turn of an incremental
conversation in excess of the same conversation's replay growth (the
cache's own growth with the transcript appears in both); the number of later thinking-off turns taken through the
incremental path; the measured-offset mismatches (zero by construction:
the Engine raises on any). Receipt OUT (default
logs/qwen_serve/incr_v1.json; refuses to overwrite). SMOKE=1 runs the
first two turns of the first and the thinking conversation into
logs/qwen_serve/smoke/incr.json.
"""
from __future__ import annotations

import json
import os
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/qwen_serve/smoke/incr.json" if SMOKE else os.environ.get("OUT", "logs/qwen_serve/incr_v1.json")

CONVERSATIONS = [
    {"name": "travel", "thinking": False, "turns": [
        ("I am planning three days in Lisbon in October. Suggest a rough outline, one line per day.", 96),
        ("Swap day two for a day trip to Sintra. Keep the same format.", 96),
        ("What should I pack for that weather? Five items, one per line.", 64),
        ("Which of those days needs an early start, and why? Two sentences.", 64),
        ("Name one restaurant style I should try and one dish. One sentence.", 48),
        ("Give me a two-line summary of the whole plan.", 64),
        ("Translate that summary into French.", 96),
        ("Thanks. Reply with just: safe travels.", 16)]},
    {"name": "debug", "thinking": False, "turns": [
        ("My Python script prints 'KeyError: user_id' when reading a JSON file. What are the three most likely causes? One line each.", 96),
        ("The key exists but is spelled 'userId'. What is the smallest fix? Show one line of code.", 64),
        ("How would I make the code tolerate both spellings? Short snippet.", 96),
        ("Should I log a warning when the old spelling is used? One sentence.", 48),
        ("Write a one-line pytest assertion for the new behaviour.", 48),
        ("Summarise the changes as a commit message title, under 60 characters.", 32),
        ("What is one risk of accepting both spellings? One sentence.", 48),
        ("Reply with just: done.", 16)]},
    {"name": "recipe", "thinking": False, "turns": [
        ("Give me a simple lentil soup recipe: ingredients only, one per line, at most eight.", 96),
        ("Now the steps, numbered, at most six.", 128),
        ("I have no cumin. What is a substitute? One line.", 32),
        ("How long does it keep in the fridge? One sentence.", 32),
        ("Double the recipe: list only the ingredient amounts that change.", 96),
        ("Suggest one side dish. One line.", 32),
        ("Write the shopping list for the doubled recipe, one item per line.", 96),
        ("Reply with just: enjoy.", 16)]},
    {"name": "thinking", "thinking": True, "turns": [
        ("A train leaves at 9:40 and the trip takes 2 hours 35 minutes. When does it arrive? Answer briefly.", 256),
        ("If it is delayed by 50 minutes, when does it arrive? Answer briefly.", 256),
        ("How many minutes is that after noon? Answer briefly.", 256),
        ("Reply with just: ok.", 128)]},
    {"name": "budget-cut", "thinking": False, "turns": [
        ("Describe the colour of the sea at dawn in one sentence.", 48),
        ("Now describe it at dusk in one long sentence.", 6),
        ("Which of the two descriptions was longer? One word.", 16),
        ("Reply with just: end.", 16)]},
]


def run_conversation(eng, spec, mode):
    from llmopt.lab import qserve
    import mlx.core as mx
    sess = qserve.Session(eng, mode=mode, enable_thinking=spec["thinking"])
    rows = []
    for text, budget in spec["turns"]:
        r = sess.ask(text, max_tokens=budget)
        rows.append({"user": text, "budget": budget, "mode": r.mode, "note": r.note, "prompt_ids": r.prompt_ids, "generated_ids": r.generated_ids,
                     "text": r.text, "stop_reason": r.stop_reason, "n_prompt": r.n_prompt, "n_generated": r.n_generated,
                     "cache_tokens": r.cache_tokens, "cache_offset_measured": r.cache_offset_measured, "first_token_s": r.timings["first_token_s"],
                     "total_s": r.timings["total_s"], "active_bytes": r.memory["active_bytes"], "footprint_bytes": r.memory["footprint_bytes"],
                     "peak_bytes": r.memory["peak_bytes"]})
    del sess, r
    mx.clear_cache()
    return rows


def main() -> None:
    from llmopt.lab import qserve
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    specs = CONVERSATIONS
    if SMOKE:
        specs = [dict(CONVERSATIONS[0], turns=CONVERSATIONS[0]["turns"][:2]), dict(CONVERSATIONS[3], turns=CONVERSATIONS[3]["turns"][:2])]
    eng = qserve.Engine()
    convs = []
    t0 = time.perf_counter()
    for spec in specs:
        rep = run_conversation(eng, spec, "replay")
        inc = run_conversation(eng, spec, "incremental")
        mism = [i for i, (a, b) in enumerate(zip(rep, inc)) if a["generated_ids"] != b["generated_ids"] or a["stop_reason"] != b["stop_reason"]]
        later = list(range(1, len(spec["turns"])))
        convs.append({"name": spec["name"], "thinking": spec["thinking"], "replay": rep, "incremental": inc, "mismatch_turns": mism,
                      "incremental_later_modes": [inc[i]["mode"] for i in later],
                      "ttft_ratio_later": [inc[i]["first_token_s"] / rep[i]["first_token_s"] for i in later],
                      "active_growth_bytes_incremental": inc[-1]["active_bytes"] - inc[0]["active_bytes"],
                      "active_growth_bytes_replay": rep[-1]["active_bytes"] - rep[0]["active_bytes"]})
        print(f"[incr] {spec['name']:11s} mismatches {mism} later modes {[m[:4] for m in convs[-1]['incremental_later_modes']]} "
              f"ttft ratios {[round(x, 2) for x in convs[-1]['ttft_ratio_later']]} growth inc {convs[-1]['active_growth_bytes_incremental'] / 1e6:.1f} MB", flush=True)
    off_rows = [(c["name"], m, i) for c in convs for m in ("replay", "incremental") for i, r in enumerate(c[m]) if r["cache_offset_measured"] != r["cache_tokens"]]
    off_ratios = [x for c in convs if not c["thinking"] for x in c["ttft_ratio_later"]]
    later_off = [m for c in convs if not c["thinking"] for m in c["incremental_later_modes"]]
    summary = {"token_mismatch_turns": sum(len(c["mismatch_turns"]) for c in convs),
               "turns_compared": sum(len(c["replay"]) for c in convs),
               "ttft_ratio_later_median_thinking_off": statistics.median(off_ratios) if off_ratios else None,
               "ttft_ratio_later_all": [round(x, 3) for c in convs for x in c["ttft_ratio_later"]],
               "active_growth_bytes_incremental_max": max(c["active_growth_bytes_incremental"] for c in convs),
               "active_growth_bytes_replay_max": max(c["active_growth_bytes_replay"] for c in convs),
               # the cache itself grows with the transcript in both modes (replay's last turn holds a fresh cache of the
               # whole transcript), so retention shows as incremental growth IN EXCESS of replay's
               "active_growth_bytes_excess_max": max(c["active_growth_bytes_incremental"] - c["active_growth_bytes_replay"] for c in convs),
               "later_thinking_off_turns": len(later_off), "later_thinking_off_incremental_taken": sum(m == "incremental" for m in later_off),
               "thinking_conversation_later_modes": [c["incremental_later_modes"] for c in convs if c["thinking"]],
               "offset_mismatch_turns": len(off_rows), "budget_stop_turns": sum(r["stop_reason"] == "max_tokens" for c in convs for r in c["replay"]),
               "replay_later_ttft_s": [round(r["first_token_s"], 2) for c in convs if not c["thinking"] for r in c["replay"][1:]],
               "incremental_later_ttft_s": [round(r["first_token_s"], 2) for c in convs if not c["thinking"] for r in c["incremental"][1:]]}
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "engine": {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"),
               "chain_sha256": eng.info.get("chain_sha256"), "stop_ids": sorted(eng.stop_ids), "context_cap": eng.context_cap,
               "cache_limit_bytes": eng.cfg.cache_limit_bytes, "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {})},
               "conversations": convs, "summary": summary, "wall_s": time.perf_counter() - t0}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[incr] summary {json.dumps({k: v for k, v in summary.items() if not isinstance(v, list)})}; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
