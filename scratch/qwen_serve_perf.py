"""QWEN-INFERENCE-BASELINE-0, phase D: the user-facing performance and
reliability of the local arm C engine (llmopt.lab.qserve), measured as a
user sees it, not as isolated kernel time.

  cold load         Engine() wall from process start to ready, and the
                    first generation's time to first token (kernel
                    compilation and first-touch costs), reported apart
  prompts           real text prompts rendered through the chat template
                    and padded with real tokens (the E0 teacher records,
                    qwen_metal_prefill_head.real_tokens) to 512, 1,024 and
                    2,048 prompt tokens; for each: time to first token
                    (the prefill), prefill throughput, steady-state decode
                    throughput over 64 greedy tokens (steps 5 on), total
                    response time, stop reason, token counts, peak active
                    and held memory
  repeated turns    a 6-turn replay session on one prompt: memory after
                    every turn (growth), cache tokens, per-turn timings;
                    and the same in incremental mode with the mechanical
                    prefix check, comparing tokens turn by turn
  parity            the engine's greedy tokens on a 1,024-token prompt
                    against a direct reference loop (prefill_chunked plus
                    the plain model call, the generation 5 qualified path)
                    on the same ids: identical ids expected
  tokenizer         round trip of the three prompts (decode(encode(text))
                    == text) and the chat template's stop token at the end
                    of the rendered assistant turn
Receipt logs/qwen_serve/perf.json (refuse-if-exists; OUT overrides).
SMOKE=1 uses 64-token prompts and 8 decode steps into logs/qwen_serve/smoke/.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

SMOKE = os.environ.get("SMOKE") == "1"
OUT = "logs/qwen_serve/smoke/perf.json" if SMOKE else os.environ.get("OUT", "logs/qwen_serve/perf.json")
LENGTHS = [64] if SMOKE else [512, 1024, 2048]
STEPS = 8 if SMOKE else 64
TURNS = 3 if SMOKE else 6
TEXTS = ["Summarise the following notes in three bullet points, then suggest one next step.",
         "Here is a long passage. Answer the question at the end in one sentence.",
         "Read the material below and list every number that appears, in order."]


def main() -> None:
    import mlx.core as mx
    from llmopt.lab import qprefill, qserve
    from qwen_metal_prefill_head import real_tokens
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t_proc = time.perf_counter()
    eng = qserve.Engine()
    cold = time.perf_counter() - t_proc
    tok = eng.tokenizer
    pool = real_tokens(4096)
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "cold_load_s": cold, "engine_load_s": eng.load_s,
               "kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"), "stop_ids": sorted(eng.stop_ids),
               "context_cap": eng.context_cap, "cache_limit_bytes": eng.cfg.cache_limit_bytes, "prompts": [], "turns": {}, "parity": {}, "tokenizer": {}}
    # ---- tokenizer round trip and template stop ----
    rt = []
    for t in TEXTS:
        ids = tok(t)["input_ids"]
        rt.append({"text": t[:40], "n": len(ids), "round_trip": tok.decode(ids) == t})
    rendered = eng.render_chat([{"role": "user", "content": TEXTS[0]}], enable_thinking=False)
    receipt["tokenizer"] = {"round_trips": rt, "rendered_prompt_tokens": len(rendered), "rendered_tail": tok.decode(rendered[-8:]),
                            "eos_token": tok.eos_token, "eos_id": tok.eos_token_id}
    # ---- the cold first generation (kernel compilation and first-touch costs land here) ----
    warm = eng.generate_ids(eng.render_chat([{"role": "user", "content": "Say hi."}], enable_thinking=False), max_tokens=4)
    receipt["cold_first_generation"] = {"prompt_tokens": warm.n_prompt, "first_token_s": warm.timings["first_token_s"], "total_s": warm.timings["total_s"],
                                        "decode_tok_per_s": warm.timings["decode_tok_per_s"]}
    print(f"[perf] cold first generation: first token {warm.timings['first_token_s']:.2f}s, total {warm.timings['total_s']:.2f}s", flush=True)
    del warm
    # ---- prompts at three lengths (warm) ----
    for L, text in zip(LENGTHS, TEXTS):
        head_ids = eng.render_chat([{"role": "user", "content": text}], enable_thinking=False)
        # the user content is padded with real tokens inside the template: rebuild as text so the template stays exact
        filler = tok.decode(pool[: max(0, L - len(head_ids) - 8)])
        msgs = [{"role": "user", "content": text + "\n\n" + filler}]
        ids = eng.render_chat(msgs, enable_thinking=False)
        res = eng.generate_ids(ids, max_tokens=STEPS)
        steps = res.timings["decode_steps"]
        receipt["prompts"].append({"target": L, "prompt_tokens": res.n_prompt, "generated": res.n_generated, "stop_reason": res.stop_reason,
                                   "first_token_s": res.timings["first_token_s"], "prefill_tok_per_s": res.timings["prefill_tok_per_s"],
                                   "decode_tok_per_s": res.timings["decode_tok_per_s"], "decode_steps": steps, "total_s": res.timings["total_s"],
                                   "memory": res.memory, "text_head": res.text[:80]})
        print(f"[perf] {res.n_prompt} prompt tokens: first token {res.timings['first_token_s']:.2f}s, prefill {res.timings['prefill_tok_per_s']:.1f} tok/s, "
              f"decode {res.timings['decode_tok_per_s'] or 0:.2f} tok/s over {steps} steps, total {res.timings['total_s']:.2f}s, peak {res.memory['peak_bytes']/1e9:.2f} GB, "
              f"footprint {res.memory['footprint_bytes']/1e9:.2f} GB, stop {res.stop_reason}", flush=True)
        del res
        mx.clear_cache()
    # ---- repeated turns: replay and incremental ----
    qs = ["Give me a one-sentence fact about copper.", "And one about tin.", "Which of those two metals has the lower melting point? One word.",
          "Name an alloy of the two. One word.", "In what century was that alloy first used? One sentence.", "Thank you. Reply with exactly: bye."][:TURNS]
    for mode in ("replay", "incremental"):
        sess = qserve.Session(eng, mode=mode, enable_thinking=False)
        rows = []
        for q in qs:
            r = sess.ask(q, max_tokens=48)
            rows.append({"user": q, "mode": r.mode, "note": r.note, "n_prompt": r.n_prompt, "n_generated": r.n_generated, "cache_tokens": r.cache_tokens,
                         "stop_reason": r.stop_reason, "first_token_s": r.timings["first_token_s"], "total_s": r.timings["total_s"],
                         "active_bytes": r.memory["active_bytes"], "footprint_bytes": r.memory["footprint_bytes"], "peak_bytes": r.memory["peak_bytes"],
                         "generated_ids": r.generated_ids, "text": r.text})
            print(f"[perf] {mode:11s} turn {len(rows)}: {r.mode:15s} prompt {r.n_prompt:4d} gen {r.n_generated:3d} cache {r.cache_tokens:4d} "
                  f"active {r.memory['active_bytes']/1e9:.2f} GB footprint {r.memory['footprint_bytes']/1e9:.2f} GB  {r.text[:50]!r}", flush=True)
        receipt["turns"][mode] = rows
        del sess
        mx.clear_cache()
    a, b = receipt["turns"]["replay"], receipt["turns"]["incremental"]
    receipt["turns"]["incremental_equals_replay_tokens"] = [x["generated_ids"] == y["generated_ids"] for x, y in zip(a, b)]
    receipt["turns"]["replay_footprint_growth_bytes"] = a[-1]["footprint_bytes"] - a[0]["footprint_bytes"]
    receipt["turns"]["incremental_footprint_growth_bytes"] = b[-1]["footprint_bytes"] - b[0]["footprint_bytes"]
    # ---- parity against the direct qualified path ----
    L = LENGTHS[min(1, len(LENGTHS) - 1)]
    msgs = [{"role": "user", "content": TEXTS[1] + "\n\n" + tok.decode(pool[: max(0, L - 40)])}]
    ids = eng.render_chat(msgs, enable_thinking=False)
    res = eng.generate_ids(ids, max_tokens=STEPS)
    cache = eng.model.make_cache()
    lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=256, head="final", matmul="gemm")
    mx.eval(lg)
    cur = int(lg[0, -1].argmax())
    ref = []
    while len(ref) < STEPS and cur not in eng.stop_ids:
        ref.append(cur)
        cur = int(eng.model(mx.array([[cur]]), cache=cache)[0, -1].argmax())
    receipt["parity"] = {"prompt_tokens": len(ids), "engine_ids": res.generated_ids, "reference_ids": ref, "identical": res.generated_ids == ref,
                         "engine_stop": res.stop_reason, "reference_stopped_on_stop_token": cur in eng.stop_ids}
    print(f"[perf] parity on {len(ids)} prompt tokens: identical {res.generated_ids == ref}", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[perf] cold load {cold:.1f}s; wrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
