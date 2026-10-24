# Handoff 2026-10-23-6: QWEN-LOW-DEFAULT-1 released (thinking at low effort, 768 tokens, incremental reuse is the chat default)

Session: Fable 5.1 on the Mac. Opus 5.5 reviewer (code, cache correctness, release protocol) read while the first smoke was live; its findings became AMENDMENT QWEN-LOW-DEFAULT-1-SMOKE before the receipt was read.
Previous handoff: 2026-10-23-5 (the HOLD). Read BOARD first.

## What happened

1. Artin accepted the HOLD of QWEN-LOW-RELEASE-0 as booked and issued the
   release-engineering GO QWEN-LOW-DEFAULT-1: register the operational
   cache contract as the decision, preserve the tokenizer mismatch as a
   fixture, promote the defaults if a fresh smoke passes, document cost.
2. PRE-REG at 5a3a0e27 with the defaults flipped in code: Session()
   with nothing given = thinking True, low, incremental, 768 tokens;
   explicit enable_thinking with no effort keeps the template default
   (every research driver reproduces); explicit effort implies thinking;
   CLI no flags = low 768, --no-thinking fast mode, overrides kept,
   incompatible flags refused at parse. tests/fixtures/qwen_planning_turn7.json
   (ids 1 and 2523 emitted, 20946 re-encoded, offset 47) with
   tests/test_qrelease.py and tests/test_qdefaults.py.
3. First smoke at 520a11db; the Opus review (read while it ran) found the
   cap probe too short and vacuous bars; AMENDMENT -SMOKE and the repaired
   registration docs/preregs/qwen-low-default-1a.json committed at
   1e0e09b5 BEFORE the first receipt was read; the first receipt kept,
   superseded, disclosed (its only failure was the cap probe).
4. Second smoke at 1e0e09b5: six of six bars FIRE (RESULTS L85431).
   Defaults released. README quick-start and limitations note landed.
5. DeepInfer review (previous handoff) unchanged.

## Exact default and override behaviour

- Session(engine): thinking True, reasoning_effort low, mode incremental,
  max_tokens 768 per ask. Session(engine, enable_thinking=False): fast
  mode, effort None. Session(engine, enable_thinking=True): thinking at
  the template's default effort (xhigh). Session(engine, enable_thinking=None):
  the template's own default. Session(engine, reasoning_effort="medium"):
  thinking at medium. enable_thinking False with an effort: InputError.
- CLI: no flags = thinking low 768; --no-thinking; --reasoning-effort
  xhigh|medium|low; --max-tokens N; --replay; --thinking (default effort
  low); refused: --thinking with --no-thinking, --reasoning-effort with
  --no-thinking, --replay with --incremental, --max-tokens below 1,
  --context-cap below 1.

## Next (held for Artin's GO)

- A fallback-rate and fallback-latency counter in the CLI receipts.
- The first-token wall on long prompts (prefill kernels, own GO); the
  deterministic sampler design and the multi-row fused kernel from the
  DeepInfer review as the studied options.
- No sampling, grammar, speculation, HTTP serving, adapters or new
  quantization were implemented in this release.

## Security side note (evidence, carried from 2026-10-23-5)

Mac clean of litellm 1.107.0 indicators; Node upgraded by Artin, `node
--version` prints v26.12.1; PHP CLI only; no GitLab tooling. No further
CVE work authorised.

## Hygiene

Receipts force-added by file name: both smoke receipts, the adjudication
pair, the two smoke logs, the two liverun records. Both checkouts at
origin/main.
