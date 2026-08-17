---
name: qualify
description: Software qualification before any expensive run — the ladder that keeps a 27B execution from being the first test of new code
---

# Qualifying code before an expensive run

Origin: 2026-08-17. Three bugs in one day surfaced INSIDE the runs
they invalidated (teacher RoPE zeroing, runtime io OOM, and a
2-day rollout scaling bug), each costing a full launch. /probe
prices the SCIENCE of a run; this prices the SOFTWARE.

## The law

No run expected to cost >10 minutes, >1 GiB of output, or a full
model sweep may be the FIRST test of new or copy-modified code.
`/rung` step 4 (launch) requires this ladder first.

## The ladder (cheap -> expensive, failure-localizing)

1. **Static checks, seconds.** Artifact/manifest structure:
   digest chain identity (rung 0), exact key-set conservation v a
   PINNED source (sha the index), payload-length formulas, exact
   span cover (no gaps, no trailing bytes), duplicate-key refusal
   at parse time. `scratch/qwen_qualify.py` is the model.
2. **Golden fixtures, seconds, in pytest.** Every binary format
   and execution convention gets committed byte fixtures with
   hand-computed expectations (tests/test_qwen_codec.py: nibble
   order, exponent bias, section offsets). Production bytes from
   the frozen producer, committed once, decode forever.
3. **Cross-implementation parity, a minute.** Independent encoder
   and decoder meet on tiny real tensors in the suite — never for
   the first time inside a model-scale run.
4. **Resource preflight, fail-closed.** Estimated peak residency
   in the requested dtype v ACTUAL available memory (WSL VM's
   budget, not the host's; vm_stat page size parsed, not
   assumed), refuse above 0.8x. Record estimate AND observed peak
   (ru_maxrss: KiB on Linux, BYTES on macOS) in the same receipt
   so the gate calibrates.
5. **Mechanism-complete smoke.** The smallest run that exercises
   EVERY mechanism class — for a 64-layer hybrid tower that means
   traversal census 64/48/16 + rope calls asserted, not a 2-layer
   slice (the teacher's 2-layer smoke was blind to the RoPE bug
   by construction). One full forward before any generation;
   two cached tokens before 32.
6. Only then the long run.

## Rules that travel

- Every incident becomes an executable regression BEFORE its fix
  is accepted. A bug documented only in RESULTS.md can recur; a
  fixture stays dead.
- One canonical codec/format module (llmopt/lab/qcodec.py
  pattern); a second decode implementation is a live
  contradiction, not redundancy. Tests may keep a deliberately
  simple reference implementation.
- Correctness references REFUSE on any oracle failure
  (fp16-losslessness, sha mismatch) — warn-and-continue is
  performance-experiment behavior, never reference behavior.
- Receipts record device_actual, dtype, estimated and observed
  peak bytes — "running on the 3080" must not be ambiguous about
  CPU v CUDA.
- "Green" means the FULL suite (CI's run), not the focused subset
  you just wrote. The oracle for repository state is the whole
  suite; a focused pass is a development convenience.
