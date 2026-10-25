"""Local greedy chat with the qualified Qwen3.8-27B arm C on this Mac
(QWEN-INFERENCE-BASELINE-0): the command-line face of llmopt.lab.qserve.

    .venv/bin/python scripts/qwen_chat.py --interactive                  # ordinary: thinking at low effort, 768 tokens per turn
    .venv/bin/python scripts/qwen_chat.py --interactive --no-thinking    # FAST: direct answers, no reasoning block
    .venv/bin/python scripts/qwen_chat.py --prompt "What is 17 * 23?"
    .venv/bin/python scripts/qwen_chat.py --prompt ... --json out.json

Loads the frozen artifact under its digest chain (no download), serves on
the qualified Metal generation 5 path with the 2 GiB serving cache limit,
renders the pinned tokenizer's chat template, decodes greedily to the
output-token budget or a stop token, and prints the text, the token ids
and the timings (cold load, time to first token, prefill and decode
throughput, total), peak and held memory and the stop reason. --json
writes the full receipt. Defaults (VERDICT QWEN-LOW-DEFAULT-1): thinking on at the pinned
template's reasoning_effort low with a 768-token budget; --no-thinking
restores direct answers; --reasoning-effort xhigh|medium|low and
--max-tokens override. Sessions reuse the cache across turns by default
(VERDICT QWEN-CHAT-DEFAULT-0: the mechanical prefix check on the rendered
transcript every turn, the KV offset checked against the accounting, a
replay into a fresh cache whenever the rendering does not extend the fed
tokens); --replay re-prefills the whole transcript every turn, the
reference. The DeltaNet recurrence runs mlx-lm's Metal kernel by default
(VERDICT QWEN-GDN-KERNEL-0, promoted under QWEN-GDN-PROMOTE-0), selected
by name per layer; --recurrence loop selects the Python reference loop the
earlier receipts ran. Nothing in the model, codec or kernels changes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _report(res, load_s=None):
    t = res.timings
    print(res.text)
    print(f"--- {res.n_prompt} prompt tokens, {res.n_generated} generated, stop {res.stop_reason}"
          f"{' (' + str(res.stop_token_id) + ')' if res.stop_token_id is not None else ''}; "
          f"first token {t['first_token_s']:.2f}s, prefill {t['prefill_tok_per_s'] and round(t['prefill_tok_per_s'], 1)} tok/s, "
          f"decode {t['decode_tok_per_s'] and round(t['decode_tok_per_s'], 2)} tok/s, total {t['total_s']:.2f}s"
          f"{'; load ' + format(load_s, '.1f') + 's' if load_s is not None else ''}; "
          f"peak active {res.memory['peak_bytes'] / 1e9:.2f} GB, footprint {res.memory['footprint_bytes'] / 1e9:.2f} GB; "
          f"cache holds {res.cache_tokens} tokens; mode {res.mode}", file=sys.stderr)


def summarize_turns(turns: list) -> dict:
    """Operational accounting derived from the recorded turn results only
    (QWEN-PREFILL-WALL-0): mode counts, fallback reasons, first-token latency
    by mode, the prompt tokens re-prefilled by fallbacks and their cost,
    token totals and memory. No cache logic is duplicated here."""
    import statistics
    by_mode: dict = {}
    reasons: dict = {}
    for t in turns:
        by_mode.setdefault(t["mode"], []).append(t["timings"]["first_token_s"])
        if t["mode"] == "replay-fallback":
            reasons[t.get("note") or "unspecified"] = reasons.get(t.get("note") or "unspecified", 0) + 1
    fallbacks = [t for t in turns if t["mode"] == "replay-fallback"]
    return {"turns": len(turns), "incremental_reuses": sum(1 for t in turns if t["mode"] == "incremental"),
            "replay_fallbacks": len(fallbacks), "fresh_turns": sum(1 for t in turns if t["mode"] == "fresh"), "fallback_reasons": reasons,
            "first_token_s_by_mode": by_mode, "first_token_s_median_by_mode": {m: statistics.median(v) for m, v in by_mode.items()},
            "replayed_prompt_tokens": sum(len(t["prompt_ids"]) for t in fallbacks), "replay_cost_s": sum(t["timings"]["first_token_s"] for t in fallbacks),
            "prompt_tokens_total": sum(len(t["prompt_ids"]) for t in turns), "generated_tokens_total": sum(len(t["generated_ids"]) for t in turns),
            "peak_bytes_max": max((t["memory"]["peak_bytes"] for t in turns), default=None),
            "footprint_bytes_max": max((t["memory"]["footprint_bytes"] for t in turns), default=None),
            "active_bytes_last": turns[-1]["memory"]["active_bytes"] if turns else None,
            "cache_tokens_last": turns[-1]["cache_tokens"] if turns else None}


def resolve_mode(thinking_flag: bool, no_thinking_flag: bool, effort):   # thinking_flag is kept for the record: --thinking alone is the default configuration
    """The CLI's configuration: no flags = thinking on at low effort (VERDICT
    QWEN-LOW-DEFAULT-1); --no-thinking = direct answers; --reasoning-effort
    sets the template's control explicitly (xhigh is the template's own
    default and must be asked for)."""
    if no_thinking_flag:
        return False, None
    return True, (effort or "low")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prompt", help="one user turn (otherwise --interactive)")
    ap.add_argument("--interactive", action="store_true", help="read turns from stdin until EOF or /quit; /reset starts over")
    ap.add_argument("--system", default=None, help="optional system message")
    ap.add_argument("--max-tokens", type=int, default=768, help="output budget per turn (default 768, the promoted configuration's)")
    ap.add_argument("--no-thinking", action="store_true", help="FAST mode: direct answers with no reasoning block (enable_thinking False)")
    ap.add_argument("--thinking", action="store_true", help="thinking on (the default since VERDICT QWEN-LOW-DEFAULT-1, at low effort unless --reasoning-effort says otherwise)")
    ap.add_argument("--reasoning-effort", choices=("xhigh", "medium", "low"), default=None,
                    help="the pinned template's reasoning_effort (default low; xhigh is the template's own default); refused with --no-thinking")
    ap.add_argument("--replay", action="store_true", help="re-prefill the whole transcript every turn (the reference) instead of reusing the cache")
    ap.add_argument("--incremental", action="store_true", help=argparse.SUPPRESS)   # the former opt-in; now the default
    ap.add_argument("--context-cap", type=int, default=None)
    ap.add_argument("--json", default=None, help="write the turn receipts (ids, text, timings, memory) to this file")
    ap.add_argument("--recurrence", choices=("kernel", "loop"), default=None,
                    help="the GatedDeltaNet recurrence implementation: kernel (the default, mlx-lm's Metal kernel) or loop (the Python reference loop)")
    ap.add_argument("--art-dir", default=None)
    ap.add_argument("--vendor-dir", default=None)
    a = ap.parse_args()
    if not a.prompt and not a.interactive:
        ap.error("give --prompt or --interactive")
    if a.thinking and a.no_thinking:
        ap.error("--thinking and --no-thinking exclude each other")
    if a.incremental and a.replay:
        ap.error("--replay and --incremental exclude each other")
    if a.max_tokens < 1:
        ap.error("--max-tokens must be at least 1")
    if a.context_cap is not None and a.context_cap < 1:
        ap.error("--context-cap must be at least 1")
    if a.reasoning_effort and a.no_thinking:
        ap.error("--reasoning-effort has no effect with --no-thinking (the template ignores it)")
    from llmopt.lab import qserve
    thinking, effort = resolve_mode(a.thinking, a.no_thinking, a.reasoning_effort)
    kw = {"context_cap": a.context_cap} if a.context_cap is not None else {}
    if a.recurrence is not None:
        kw["recurrence"] = a.recurrence
    eng = qserve.Engine(art_dir=a.art_dir, vendor_dir=a.vendor_dir, **kw)
    print(f"[qwen_chat] loaded {eng.info.get('fused')} fused tensors, kernel generation {eng.info.get('kernel_version')}, "
          f"chain {eng.info.get('chain_sha256', '')[:12]}, recurrence {eng.info.get('recurrence')} on {eng.info.get('recurrence_layers')} layers, "
          f"in {eng.load_s:.1f}s; stop ids {sorted(eng.stop_ids)}; "
          f"cache limit {eng.cfg.cache_limit_bytes >> 20} MiB; context cap {eng.context_cap}", file=sys.stderr)
    sess = qserve.Session(eng, system=a.system, mode="replay" if a.replay else "incremental", enable_thinking=thinking, reasoning_effort=effort)
    receipts = {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"),
                "recurrence": eng.info.get("recurrence"), "recurrence_layers": eng.info.get("recurrence_layers"), "recurrence_vendor": eng.info.get("recurrence_vendor"),
                "recurrence_of": __import__("llmopt.lab.qtower", fromlist=["x"]).recurrence_of(eng.model) if eng.info.get("recurrence") else None,
                "thinking": thinking, "reasoning_effort": effort, "max_tokens": a.max_tokens, "mode": sess.mode, "turns": []}

    def turn(text):
        try:
            res = sess.ask(text, max_tokens=a.max_tokens)
        except (qserve.InputError, qserve.ContextLimitError) as e:
            print(f"[qwen_chat] refused: {e}", file=sys.stderr)
            return
        _report(res, eng.load_s if not receipts["turns"] else None)
        receipts["turns"].append({"user": text, "prompt_ids": res.prompt_ids, "generated_ids": res.generated_ids, "text": res.text,
                                  "stop_reason": res.stop_reason, "stop_token_id": res.stop_token_id, "cache_tokens": res.cache_tokens,
                                  "mode": res.mode, "note": res.note, "timings": res.timings, "memory": res.memory})

    if a.prompt:
        turn(a.prompt)
    if a.interactive:
        print("[qwen_chat] interactive; /reset clears the session, /quit exits", file=sys.stderr)
        for line in sys.stdin:
            line = line.rstrip("\n")
            if line.strip() == "/quit":
                break
            if line.strip() == "/reset":
                sess.reset()
                print("[qwen_chat] session reset", file=sys.stderr)
                continue
            turn(line)
    summary = summarize_turns(receipts["turns"])
    receipts["summary"] = summary
    if summary["turns"]:
        med = summary["first_token_s_median_by_mode"]
        print(f"[qwen_chat] {summary['turns']} turns: {summary['incremental_reuses']} reused, {summary['replay_fallbacks']} replayed"
              f"{' (' + '; '.join(summary['fallback_reasons']) + ')' if summary['fallback_reasons'] else ''}, {summary['fresh_turns']} fresh; "
              f"first token median by mode {{{', '.join(f'{m}: {v:.2f}s' for m, v in med.items())}}}; replayed {summary['replayed_prompt_tokens']} prompt tokens "
              f"costing {summary['replay_cost_s']:.1f}s; peak {summary['peak_bytes_max'] / 1e9:.2f} GB, footprint {summary['footprint_bytes_max'] / 1e9:.2f} GB, "
              f"active {summary['active_bytes_last'] / 1e9:.2f} GB, cache holds {summary['cache_tokens_last']} tokens", file=sys.stderr)
    if a.json:
        json.dump(receipts, open(a.json, "w"), indent=1)
        print(f"[qwen_chat] wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
