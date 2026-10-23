# Handoff 2026-10-23-2: QWEN-INFERENCE-BASELINE-0 (the inference program and the first usefulness baseline)

Session: Fable 5.1 on the Mac. Auditors on Opus 5.5 (code and cache-correctness audit before scoring; receipt audit before booking).
Previous handoff: 2026-10-23-1 (the MLP-coupled null). Read BOARD first.

## What happened

1. Artin's GO (QWEN-INFERENCE-BASELINE-0): turn the qualified arm C into a
   reliable local inference program, then establish an independent,
   mechanically scored functional baseline; no model, codec, kernel or
   default changes; no HTTP API.
2. Phase A and B: llmopt/lab/qtower.py adopts the qualified tower build
   verbatim from scratch/qwen_metal_qual.py (tests/test_qtower_adoption.py
   pins source identity and the mlx-lm GatedDeltaNet guard).
   llmopt/lab/qserve.py: Engine (artifact verified under its digest chain,
   kernel generation 5 only, the 2 GiB cache limit, chunk-256 GEMM prefill
   with the final head, fused cached decode, pinned tokenizer and template,
   stop tokens <|im_end|> and <|endoftext|> never continued past, a context
   cap of 8,192 that refuses rather than truncates), GenerationResult (ids,
   text, stop reason, timings, memory, the exact cache accounting and the
   KV cache's measured offset, which the Engine checks on entry and exit),
   Session (full-transcript replay as the reference; incremental reuse only
   when the template's rendering of the transcript has the fed ids as an
   exact prefix; a refused turn leaves the transcript unchanged; thinking
   text split into reasoning_content and content for the template).
   scripts/qwen_chat.py is the CLI (--prompt or --interactive, --system,
   --no-thinking/--thinking, --incremental, --json). tests/test_qserve.py:
   a deterministic stub pins every accounting branch and the incremental
   path; three live tests pass on the real tower.
3. Phase C: llmopt/lab/qbattery.py and the frozen collection
   scratch/evalsets/qwen_battery_v2.json (46 tasks: seeded arithmetic,
   JSON with schema and semantic checks, sandboxed Python with hidden
   tests, constrained instructions, two-turn retention, termination).
   The sandbox is macOS sandbox-exec (no network, no writes outside the
   task dir, no reads under /Users, no exec but the interpreter, empty
   environment, CPU/file-size/process limits inside the sandboxed process;
   memory not limitable on macOS). Freeze booked at L84146 before scoring.
4. The Opus code audit after the v1 chain and before any battery result
   was read found three mis-specified v1 tasks and real defects (replay
   cache leak, dangling refused turn, unmeasured accounting check, thinking
   history, perf filler, harness); all repaired, v2 re-frozen, AMENDMENT at
   L84244 (commit 3ed19ee6). v1 receipts kept, the v1 battery receipt never
   read.
5. Phase D and E at 3ed19ee6 under liverun qwen-inference-1: OBSERVATION
   QWEN-INFERENCE-BASELINE-0 at L84293 (receipts logs/qwen_serve/perf_v2.json,
   logs/qwen_battery/results_v2.json). Numbers in the BOARD row. Honest
   battery count 33 of 46; the code-08 error row is a budget-cut syntax
   error booked error by a harness classification defect, fixed at the
   booking commit (candidate compiled from its own file after the start
   sentinel; test added RED then GREEN) without rescoring.
6. The receipt audit's remaining findings are disclosed in the entry: the
   perf filler is a chat-transcript pastiche (timings stand, reply texts
   are not instruction-following evidence); the cold load ran on a warm
   page cache; per-call peak grows with the prompt as prefill scratch.

## Next (in order; all held for Artin's GO)

- Make incremental cache reuse the chat default behind the existing
  prefix check (zero kernel work; later-turn first token 2.2 s against
  7.8 s at a 217-token transcript).
- Run the unchanged v2 collection under enable_thinking True at a larger
  budget as its own versioned receipt (no prompt or validator edits).
- A performance filler from plain prose (no role words or traces) and a
  true cold load before any other stage, if the perf numbers are re-cut.
- Prefill kernel work (the 32 to 36 tok/s GEMM ceiling) is out of this
  round's boundaries and would need its own GO.
- Not started, per the GO: HTTP API, fine-tuning, model changes.

## Hygiene

Receipts force-added by file name: both chains' perf and battery receipts,
the smoke battery receipt, the two study logs, the two liverun records.
Both checkouts at origin/main after this commit (3080 synced if reachable).
