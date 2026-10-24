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
reference. Nothing in the model, codec or kernels changes.
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
    eng = qserve.Engine(art_dir=a.art_dir, vendor_dir=a.vendor_dir, **kw)
    print(f"[qwen_chat] loaded {eng.info.get('fused')} fused tensors, kernel generation {eng.info.get('kernel_version')}, "
          f"chain {eng.info.get('chain_sha256', '')[:12]}, in {eng.load_s:.1f}s; stop ids {sorted(eng.stop_ids)}; "
          f"cache limit {eng.cfg.cache_limit_bytes >> 20} MiB; context cap {eng.context_cap}", file=sys.stderr)
    sess = qserve.Session(eng, system=a.system, mode="replay" if a.replay else "incremental", enable_thinking=thinking, reasoning_effort=effort)
    receipts = {"load_s": eng.load_s, "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"),
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
    if a.json:
        json.dump(receipts, open(a.json, "w"), indent=1)
        print(f"[qwen_chat] wrote {a.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
