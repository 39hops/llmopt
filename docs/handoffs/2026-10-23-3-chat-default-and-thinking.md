# Handoff 2026-10-23-3: QWEN-CHAT-DEFAULT-0 (GO, incremental promoted) and QWEN-THINKING-BATTERY-0 (thinking recovers arithmetic and JSON, loses code)

Session: Fable 5.1 on the Mac. Reviewer on Opus 5.5 (cache correctness, thinking parsing, evaluator consistency, receipts) read while the chain was live; its scoring findings became AMENDMENT QWEN-THINKING-BATTERY-0-RULES before any thinking score was read.
Previous handoff: 2026-10-23-2 (the inference baseline). Read BOARD first.

## What happened

1. Artin accepted QWEN-INFERENCE-BASELINE-0 and issued the GO
   QWEN-CHAT-DEFAULT-AND-THINKING-0: part A promote incremental
   conversation if it passes a registered comparison; part B the frozen
   v2 battery under the pinned template's thinking control at 384 and
   768 tokens; part C a mechanism classification of the baseline's
   failures; part D review, gates, CI, 3080. Model and kernels frozen.
2. PRE-REG QWEN-CHAT-DEFAULT-0 and PRE-REG QWEN-THINKING-BATTERY-0 at
   5dc6adf3 (JSON in docs/preregs/). Eight stub scenario tests (stop
   token, budget cut, empty completion, thinking on and off, changed
   rendering, dropped special token, cap refusal and recovery, reset
   and isolation; the stub template mirrors the pinned template's
   structure) and two live tests added; qbattery.split_final,
   format_ok and score_thinking; the runner's THINK and BUDGET
   configuration; drivers and adapters.
3. Chain qwen-chat-default-0 at 7ae6a556 (live tests 5 passed; the
   five-conversation comparison; the two thinking batteries; 64 min).
4. The review (read during the run, applied after disarm and before
   any thinking score was read): the two-turn rule made stricter than
   registered (fixed to the registration), a reopened block, format on
   the final answer only, reasoning tokens to the last close, a one-run
   fence on the battery bars (the amended registration
   docs/preregs/qwen-thinking-battery-0a.json), every KV layer's offset
   must agree, BaseException discards the cache, tests tightened. The
   adapter recomputes thinking verdicts from the stored replies under
   the amended scorer; no row changed.
5. VERDICT QWEN-CHAT-DEFAULT-0: all four bars FIRE; incremental is the
   default of Session and scripts/qwen_chat.py (--replay the reference).
   VERDICT QWEN-THINKING-BATTERY-0: 33 / 30 / 36 of 46 (baseline, 384,
   768); arithmetic plus JSON 8 to 16 of 18 (FIRE); other kinds 25 to
   20 (NO-FIRE, code 7 to 2 through unterminated blocks); tokens 4.7x
   and 6.6x. Part C annotation in
   scratch/evalsets/qwen_battery_v2_failure_mechanisms.json.
6. Side finding to measure next: thinking-off multi-turn chat emitted
   empty completions on three consecutive turns of the eight-turn travel
   conversation in both modes.

## Next (held for Artin's GO)

- The pinned template's reasoning_effort low (and medium) at 768 tokens
  on the unchanged v2 battery, registered with the same fence; if low
  closes the code blocks while keeping the arithmetic recovery, thinking
  at low effort as the chat default.
- The empty-completion behaviour of thinking-off multi-turn chat: a
  registered multi-turn measurement before the default is called usable
  at depth.
- Prefill kernels (the single-turn wall) need their own GO. No HTTP
  API, fine-tuning, recompression, speculative decoding or tool use.

## Hygiene

Receipts force-added by file name: incr_v1.json and its adjudication,
the two thinking receipts and their adjudication, the chain log, the
liverun record, the two smoke receipts. Both checkouts at origin/main.
