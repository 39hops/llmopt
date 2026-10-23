# Handoff 2026-10-23-4: QWEN-EFFORT-0 (low effort qualifies, recommended not flipped) and QWEN-DEEPCHAT-0 (empty completions are the model's decided stops)

Session: Fable 5.1 on the Mac. Auditor on Opus 5.5 (template routing, scoring, cache accounting, diagnostic validity, receipts), read while the chain was live; its two blockers and the rest became AMENDMENT QWEN-DEEPCHAT-0-ADAPTER before any receipt was read.
Previous handoff: 2026-10-23-3 (chat default and thinking). Read BOARD first.

## What happened

1. Artin accepted QWEN-CHAT-DEFAULT-0 (GO) and QWEN-THINKING-BATTERY-0 and
   issued QWEN-EFFORT-AND-DEEPCHAT-0: study A the pinned template's
   reasoning_effort low and medium at 768 tokens on the frozen v2 battery;
   study B a read-only diagnostic of the deep-chat empty completions. No
   model, kernel or default changes; no flip without another GO.
2. Template semantics verified mechanically in fresh subprocesses
   (tests/test_qtemplate.py, ids and shas pinned): omitted equals xhigh
   (the detailed instruction), low the brief instruction, medium no
   system message at all, thinking False ignores the control (the Engine
   refuses that combination). reasoning_effort routed through
   Engine.render_chat, chat, Session and the CLI (--reasoning-effort);
   default unchanged; stub tests for routing, reuse under each effort and
   fallback on a mid-session change. The Engine records the prefill's
   top-16 first-token logits and the stop ids' logits on every result.
3. PRE-REG QWEN-EFFORT-0 and PRE-REG QWEN-DEEPCHAT-0 at ad3d1ce3; chain
   qwen-effort-0 at b638a497 (template and live tests 12 passed; low;
   medium; the diagnostic; 51 min).
4. The audit's repairs (applied after disarm, before any receipt was
   read): the deep-chat adapter's unregistered key and the conjunct unit
   (both would have refused adjudication), bar 2 re-measured against a
   hand-built ChatML rendering tokenised without the template, bar 3's
   offset conjunct across modes, validity checks in the effort adapter,
   jinja template refusals as InputError, CLI flag refusal before the
   cold load, the vacuous stub effort test fixed. Amended registration
   docs/preregs/qwen-deepchat-0a.json.
5. VERDICT QWEN-EFFORT-0: low fires all five bars (40 of 46; arithmetic
   plus JSON 15 of 18; code 7 of 8; other 18 of 20; 2 unterminated; 7,957
   tokens, 977 s); medium fires three (38; code 4 of 8; 12,421 tokens).
   Low is RECOMMENDED as the chat default; not implemented.
   VERDICT QWEN-DEEPCHAT-0: every bar FIRE; the empties are decided
   <|im_end|> stops (2.1 to 10.0 logits) at the end of a transcript whose
   replies shorten with depth; template, accounting and the two Metal
   paths excluded; a quality limitation, no serving defect.

## Next (held for Artin's GO)

- The default flip to enable_thinking True, reasoning_effort low, 768
  tokens, with one confirmation on a prompt set this battery did not
  shape (the battery has been read under four configurations).
- The multi-turn shortening: a registered measurement with thinking on
  (low) and wider per-turn budgets, the two variables the frozen
  conversation held fixed.
- Prefill kernels (the single-turn wall) need their own GO. No HTTP API,
  fine-tuning, recompression, speculative decoding or tool use.

## Side note

A security check of the Mac for the 2026-10-21 litellm supply-chain
incident and the Node Brotli CVE found no litellm anywhere and no
indicators; Homebrew node 26.10.0 should be upgraded to 26.12.1. Not a
repo matter; recorded here so the next session does not redo it.

## Hygiene

Receipts force-added by file name: the two effort receipts and their
adjudication, the deep-chat receipt and its adjudication, the chain log,
the liverun record, the two smoke receipts. Both checkouts at origin/main.
