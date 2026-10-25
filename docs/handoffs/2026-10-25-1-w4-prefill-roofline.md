# Handoff 2026-10-25-1: QWEN-W4-PREFILL-ROOFLINE-0 (the W4 prefill wall on the promoted path is the machine's float32 matrix rate); README on the kernel default; the mathematics study's figures restated

Session: Fable 5.1 on the Mac. Opus 5.5: a methodology review of the
registration, driver and adapter BEFORE registration (three blockers,
fourteen should-fix items, all applied, then a second smoke), a
pre-registration audit (three blockers, two unsupported numbers and a
process count, and ten should-fix wording items, all applied) and a
receipt audit (no false provenance; one instrument contamination
disclosed in the verdict; the adapter correction committed before the
observations were regenerated) before booking. HEAD at close: the
handoff commit (git log -1), on origin/main; 3080 fast-forwarded to it
(ff-only, HEAD asserted). Previous handoff: 2026-10-25-0 (the kernel
promotion). Read BOARD first.

## What happened

1. README (d516a43e): the Metal GatedDeltaNet recurrence kernel as the
   serving default, `--recurrence loop` as the reference option, the
   measured first-token latencies of VERDICT QWEN-GDN-PROMOTE-0 (2.2 s on
   short CLI prompts, 8.9 / 18.1 / 34.9 s at 512 / 1,024 / 2,048 tokens;
   decode steps 94 to 96 ms). The August 2026 mathematics study's 0.4 to
   0.65 dB figures and the CVP-hardness item restated, on Artin's
   ruling, as guidance under their assumptions, never as caps on the
   activation-weighted W4 distortion or as a theorem that useful
   codebooks must be structured lattices (precision note in the spec,
   the RIFF bank and the previous handoff). The negative result stands.
2. PRE-REG QWEN-W4-PREFILL-ROOFLINE-0 (L86620, 21632afc, code_commit
   6e1f30ec): the QWEN-PREFILL-WALL-0 marginal decomposition repeated on
   the promoted path with the W4 family split in situ (decode removal at
   a fixed float32 GEMM; the GEMM at a fixed decode), the DeltaNet
   stand-ins installed on the promoted class and counted, additive
   attribution with the W4 and S16 decodes timed exclusively, per-shape
   microbenchmarks at T = 256, 512, 1,024 against measured square-GEMM
   and streaming rates with mx.quantized_matmul as the library
   fused-dequant reference, the shape census and roofline arithmetic in
   the receipt; eight PREDICTION bars (not hurdles), a candidate rule
   with every term a named receipt field, at most one further
   microbenchmark under its own pre-reg. Driver
   scratch/qwen_w4_roofline.py, adapter scratch/qwen_w4_roofline_obs.py,
   chain scratch/qwen_w4_roofline_study.sh.
3. The run (liverun qwen-w4-prefill-roofline-0 at 6e1f30ec, 12 min):
   the 1,017-token development prompt prefills in 17.65 s; W4 11.68 s
   (66.2 percent; decode 2.54, GEMM 9.07, sum 0.994, equal to 768
   isolated decode-then-matmul calls within 0.4 percent), S16 body 4.98
   (28.2), DeltaNet core 0.57 (3.2), everything else under 0.6 percent;
   the float32 GEMM at 3.93 and 3.67 TFLOPS against a 4.06 TFLOPS
   4,096-square rate (float16 4.62); the decode 105 GB/s of 133 cleared
   and in situ; the vendor's fused affine 4-bit kernel 0.84 and 0.82 of
   production with float32 activations (1.03 to 1.08 times the float32
   matmul alone); one slab's allocation under 1 percent; GPU-side 0.997.
4. VERDICT (L86622): seven of eight bars FIRE, bar 4 NO-FIRE as
   registered: the gate_proj warm-pool decode row (10.4 ms) was timed
   first after the rates section freed a 1 GiB array without emptying
   the pool, and a plain fill into the same buffer was equally slow;
   the cleared-pool decode and every down_proj row read 3.6 ms. Observed,
   not explained; disclosed with the derived fields that inherit it.
   Ceilings as fractions of TTFT under the registered rule: B float16
   decode and GEMM 0.147 (the float16 copies of the W4 weights are
   inexact: a new served function), A1 0.145 (decode removal at a fixed
   float32 GEMM), A2 0.112 (what the vendor's fused kernel achieves;
   0.190 with float16 activations, descriptive), C 0.109 and infeasible
   (68 GB float32 or 34 GB float16 against a 30 GB working set), D
   0.055. No microbenchmark implemented (the rule requires its own
   pre-reg). Design assessment with the ranked table, falsifiers and a
   proposed QWEN-W4-F16-PREFILL-0:
   docs/superpowers/specs/2026-10-25-w4-prefill-roofline-design-assessment.md.
5. One post-run adapter correction (22f38a2c): the registered unit
   label on every measurement (the adjudicator refused the first
   document with metric_unit_mismatch) plus a cross-receipt model-pin
   and vendor equality floor; no value, bar or threshold moved.
6. FINDINGS bullet, BOARD row, RIFF MEASURED bank (with the instrument
   lesson: empty the pool between a rates section and a kernel probe),
   receipts force-added (decomp.json, micro.json, adjudicate_1, the study
   log), index, lock, codemap, scoreboard, INDEX.

## Open for Artin

- The next GO: a bounded float16 qualification (QWEN-W4-F16-PREFILL-0,
  spec section 9; ceiling 15 percent; it changes the served weights and
  needs the near ties re-read) or the S16 body split (28 percent of the
  wall, its decode 1.59 s) before any precision change. Nothing armed.
- Whether the allocator-state effect deserves a serving-path check (a
  chunk's decodes in the additive trace at over 6 ms each would mean it
  bites in situ; none did here).
- Bar 4's instrument: a future rung that registers a warm-pool decode
  bandwidth must clear the pool first or time a cleared-pool call.

## Invariants kept

QWEN-GDN-PROMOTE-0 and QWEN-GDN-KERNEL-0 as booked (bar 9 UNRESOLVED);
the qualified kernels, the codec, the serving defaults, every frozen
driver and every historical receipt unchanged; no Muon, no
distillation, no new compression algorithm; the mathematics study's
negative result stands.
