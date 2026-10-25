# QWEN-W4-PREFILL-ROOFLINE-0: the W4 prefill wall on the promoted path, a ranked design assessment

Status: measured 2026-10-25 (Mac, one session, commit 6e1f30ec for the
receipts, liverun qwen-w4-prefill-roofline-0), booked as VERDICT
QWEN-W4-PREFILL-ROOFLINE-0 against PRE-REG QWEN-W4-PREFILL-ROOFLINE-0
(docs/preregs/qwen-w4-prefill-roofline-0.json; seven of eight
prediction bars FIRE, bar 4 NO-FIRE). Assessment and bounded
measurement only, Artin's GO of 2026-10-25 after VERDICT
QWEN-GDN-PROMOTE-0: no serving change, no kernel, no census.
Receipts: logs/qwen_w4_roofline/decomp.json, logs/qwen_w4_roofline/micro.json,
logs/qwen_w4_roofline/adjudicate_1/. Driver scratch/qwen_w4_roofline.py,
adapter scratch/qwen_w4_roofline_obs.py.

## 1. The serving path as it is

prefill_chunked (llmopt/lab/qprefill.py) feeds the prompt in chunks of
256 tokens through the body with every cache array evaluated per chunk,
the head on the final row, inside qmetal.gemm_scope(True). Under that
scope every W4 and S16 resident with a tile of at least 16 (W4) or 32
(S16) rows takes _Resident._gemm_decoded (llmopt/lab/qmetal.py): the
decode kernel (_W4_DECODE_SRC: one thread per 32-bit index word, four
codebook gathers, 16 float32 weights written, the codebook value times
the block's power-of-two scale) writes the whole matrix as a float32
slab (every body matrix fits one 400 MiB slab), mx.matmul multiplies the
chunk's activations against the slab's transpose, and the slab is a
temporary. Nothing is kept across chunks: a prompt of k chunks decodes
every W4 and S16 matrix k times. The 48 DeltaNet layers run mlx-lm's
Metal recurrence kernel (VERDICT QWEN-GDN-PROMOTE-0).

Shape census of every fused resident (micro.json census):

| family | matrices | shape | payload each | float32 slab each |
|---|---|---|---|---|
| W4 gate_proj, up_proj | 128 | 17408 x 5120 | 22.98 MB (2.06 bits per weight) | 356.5 MB |
| W4 down_proj | 64 | 5120 x 17408 | 22.98 MB | 356.5 MB |
| S16 body | 208 | six shapes | 4.31 GB in all | 28.9 GB in all |
| S16 head | 1 | 248320 x 5120 | 645.6 MB | once per prompt, one row, fused |

Per 256-token chunk, the W4 family: 17.11 billion weights, 8.76 TFLOP,
4.41 GB of payload read, 68.45 GB of float32 slab WRITTEN by the decode
kernels and 68.45 GB READ by the GEMMs (one pass, a lower bound; a tiled
GEMM re-reads weight tiles once per activation tile), 1.81 GB of
activations read once, 2.62 GB of outputs written. The S16 body adds
4.34 TFLOP and 28.9 GB of slab written and read. A 1,024-token prompt is
four of these.

## 2. Roofline at the rates this process measured

| rate | measured |
|---|---|
| 4,096-square float32 matmul | 4.06 TFLOPS |
| 4,096-square float16 and bfloat16 matmul | 4.62 TFLOPS (1.14 times float32; the framework's rate in this process, not a vendor peak) |
| streaming add over 1 GiB (read plus write) | 133 GB/s |
| fill over 1 GiB (write) | 133 GB/s |

Floors for the 1,024-token prompt (four chunks) at those rates against
the in-situ marginals of section 4:

| term | floor | measured in situ | ratio |
|---|---|---|---|
| W4 GEMM, 34.8 TFLOP (the last chunk has 249 tokens) | 8.64 s | 9.07 s | 0.95 |
| W4 slab write, 274 GB | 2.05 s | 2.54 s | 0.81 |
| W4 payload read, 17.6 GB | 0.13 s | inside the decode | |
| S16 body GEMM, 14.8 TFLOP | 3.64 s | 3.96 s (additive) | 0.92 |
| S16 body slab write, 115 GB | 0.87 s | 1.59 s (additive) | 0.55 |

Theoretical against achievable. A fused dequant-GEMM removes the 274 GB
of slab write and the 274 GB of slab read per prompt: 4.1 s at the
streaming rate IF the path were bandwidth-bound. It is not. At T = 256
the float32 GEMM does 128 FLOP per slab byte (each decoded weight, 4
bytes, enters 2 x 256 = 512 FLOP) against a ridge of 4.06e12 / 1.33e11
= 31 FLOP per byte, so the slab READ hides under the FLOPs and only the
WRITE (the decode) is exposed. The exposed part is 2.54 s of 17.65.

## 3. Decomposition on the promoted path (one development prompt of 1,017 tokens, seed prefill-wall-dev-1-1024, the held-out prompts never read; chunk 256)

Marginal replacement (full minus the path with one family replaced by a
near-free stand-in; medians of 4 order-balanced runs; the full path
17.65 s, spread 3 percent; dispatch counted: 7,104 kernel executions, 0
loop, in every counted run; the recurrence stand-in counted 192 times
in each of its 4 runs):

| family removed | wall | marginal | share |
|---|---|---|---|
| W4 MLP (192 linears) | 5.97 s | 11.68 s | 66.2 % |
| W4 decode at a fixed float32 GEMM | 15.11 s | 2.54 s | 14.4 % |
| W4 GEMM at a fixed decode | 8.58 s | 9.07 s | 51.4 % |
| S16 body (208 linears) | 12.68 s | 4.98 s | 28.2 % |
| DeltaNet core (conv, recurrence, gated norm, small projections) | 17.09 s | 0.57 s | 3.2 % |
| recurrence alone (the Metal kernel) | 17.25 s | 0.40 s | 2.3 % |
| head | 17.56 s | 0.09 s | 0.5 % |
| attention core | 17.57 s | 0.08 s | 0.5 % |
| norms | 17.64 s | 0.01 s | 0.1 % |
| every linear | 1.05 s | 16.60 s | 94.1 % |

Additive synchronous attribution (every call evaluated as issued, inner
calls subtracted; the decodes timed exclusively inside their linear's
frame): W4 matmul 9.38 s, W4 decode 3.09, S16 matmul 3.96, S16 decode
1.59, DeltaNet core 0.65, attention 0.19, norms 0.18, head 0.05; sum
19.08 against a synchronised wall of 19.44 and an unsynchronised 17.52
(serialisation penalty 1.9 s). One instrumented run: GPU completion
0.997 of the wall, CPU issue 0.05 s; peak 12.08 GB, footprint 11.51 GB.

Against VERDICT QWEN-PREFILL-WALL-0 (loop path, 26.2 s): the recurrence
went from 9.0 s to 0.4; the W4 family stayed at 11.7 to 12.6 s and went
from 48 percent to 66; the S16 body from 20 to 28.

## 4. The W4 family split

Decode 0.218 of the family, GEMM 0.776, sum 0.994 (no overlap between
the two in the pipelined path). Per call over the 768 W4 calls of the
prompt: 3.3 ms decode, 11.8 ms matmul; the isolated production call in
stage micro reads 14.9 ms (gate_proj) and 15.6 (down_proj), 768 of
them 11.64 s, within 0.4 percent of the in-situ marginal. The family is
the sum of its matrix calls; there is no hidden framework or dispatch
term.

## 5. Per-shape microbenchmarks (stage micro, medians of 7, single-call timing)

gate_proj 17408 x 5120:

| T | f32 matmul | TFLOPS (of square) | production | pool cleared | f16 | bf16 | qmm f32 | qmm f16 |
|---|---|---|---|---|---|---|---|---|
| 256 | 11.6 ms | 3.93 (0.97) | 14.9 | 15.0 | 10.3 | 10.3 | 12.5 | 10.7 |
| 512 | 22.9 | 3.99 (0.98) | 26.2 | 26.2 | 20.3 | 20.3 | 24.7 | 21.2 |
| 1024 | 45.3 | 4.03 (0.99) | 48.6 | 48.6 | 40.1 | 40.1 | 49.1 | 42.1 |

down_proj 5120 x 17408:

| T | f32 matmul | TFLOPS (of square) | production | pool cleared | f16 | bf16 | qmm f32 | qmm f16 |
|---|---|---|---|---|---|---|---|---|
| 256 | 12.4 ms | 3.67 (0.90) | 15.6 | 15.7 | 10.6 | 10.6 | 12.8 | 10.9 |
| 512 | 27.9 | 3.27 (0.81) | 31.1 | 30.2 | 20.6 | 20.5 | 25.1 | 21.4 |
| 1024 | 58.6 | 3.11 (0.77) | 62.0 | 63.8 | 40.5 | 40.5 | 49.9 | 42.4 |

Decode kernel alone: down_proj 3.6 ms single (105 GB/s, 0.79 of the
stream), 3.3 pipelined, 3.6 with the pool cleared, a plain fill of the
same slab 2.75 ms (the whole decode is 1.31 of the fill).
gate_proj, timed first after the rates section with the allocator pool
holding the just-freed 1 GiB and 4,096-square arrays: 10.4 ms single
(36 GB/s), the plain fill 9.4 ms, and 3.6 ms with the pool cleared;
3.3 ms inside its production call seconds later (14.9 minus 11.6) and 3.3 ms per decode averaged over the W4 family in situ. Bar 4
(decode bandwidth at least 0.60 of the stream, worst shape, warm-pool
single call) therefore reads NO-FIRE as registered: the registered
aggregation coincides with an equally slow plain fill into the buffer
the cache handed back after the 1 GiB and 4,096-square frees under the
2 GiB limit (2.9 times the cleared-pool decode) and does not separate
the kernel's own rate from that state; observed, not explained, and not
seen in the serving path in situ. Allocation of one fresh slab costs under 1 percent of a
production call (bar 6). The production decode-then-matmul is bit-equal
to the matmul on the pre-decoded slab on both shapes at every T.

Dtype: the float16 and bfloat16 copies of the W4 weights are NOT exact
on either shape (small codebook values times 2^-16 scales fall under
float16's normal range; bfloat16 drops three mantissa bits), so a
float16 GEMM is a different function, not just a different accumulation;
float16 maximum absolute output error 3e-4 to 4e-4 of the largest
float32 output, bfloat16 3e-3, at the same speed, on a seeded random
activation (not a real one; numerics unqualified). mx.quantized_matmul (affine 4-bit, group 128, a different codec
at about 4.5 bits per weight with its scales and biases, timing only)
runs at 1.03 to 1.08 times the float32 matmul alone at T = 256: the
framework's fused kernel buys the decode, not a better GEMM schedule, at
this tile; at T = 1,024 on down_proj it reaches 0.85 of the float32
matmul.

## 6. Design comparison and Amdahl ceilings

Registered rule (pre-reg note), every term a receipt field: w4_share
0.662; decode_share 0.219 and matmul_share 0.781 (normalised by the
split sum 0.994); micro terms the T = 256 rows count-weighted 128:64 as
ratios of weighted sums (the mean of per-shape ratios moves every
ceiling by under 0.002).
Ceilings are fractions of the 17.65 s first-token wall; "realistic"
is this study's reading, not a measurement.

| rank | design | removes | mechanism shown by | ceiling | realistic | memory | complexity, risk |
|---|---|---|---|---|---|---|---|
| 1 | B float16 decode and float16 GEMM | half the slab write (proxied: a float16 decode kernel is unmeasured) and 12 percent of the GEMM | measured f16 matmul 0.89 (gate) and 0.85 (down) of f32 | 0.147 (2.6 s) | 0.10 to 0.15 | halves the 0.9 GB of slab temporaries | changes the SERVED WEIGHTS and the accumulation: a parity requalification (census argmaxes, the 15 near ties, chat turns) under a new pre-reg; a decode kernel variant to write; the float16 rate on this silicon is only 14 percent above float32 |
| 2 | A fused tiled dequant-GEMM (W4 codebook, simdgroup_matrix) | the slab write and allocation (A1); a better GEMM schedule only if the house kernel beats mx.matmul (A2 shows the vendor's fused kernel does not at T = 256) | library qmm at 0.83 of production (f32); 0.71 with f16 activations | A1 0.145, A2 0.112 (0.190 with f16 activations) | 0.05 to 0.11 | removes the slab temporaries | a new Metal GEMM competing with the vendor's steel kernels; the vendor's own tuned fused kernel reaches 0.83; a per-weight codebook gather (256 entries of four halfs, threadgroup-staged) once per activation tile; a new reduction order, full parity requalification; weeks of kernel work for a tenth |
| 3 | C decoded-weight caching across chunks | three of four decodes | none | 0.109 | 0 | 68.5 GB float32 or 34.2 GB float16 resident against a 30.2 GB recommended working set | infeasible as stated; a partial cache (the slabs of n layers) scales linearly and starves the KV and recurrent state |
| 4 | D chunk 512 | half the decode passes | measured T = 512 rows (0.92 per matrix); consistent with the 5 percent of PREFILL-WALL-0 on the loop path, no new in-situ measurement | 0.055 (0.97 s) | 0.05 | +6 percent peak (PREFILL-WALL-0) | the kernel recurrence is qualified at chunk 256 only (Engine refuses other chunks on the kernel path); a requalification for 5 percent |
| 5 | E the vendor's affine 4-bit codec as the served format | the decode and the slab | library measurement | not applicable | | | a different artifact, a different function, twice the bits per weight; out of scope |
| 6 | F a faster float32 GEMM | GEMM efficiency | square-matmul rate | at most 0.03 (gate) to 0.10 (down) of the matmul | 0 | | the vendor's GEMM is at 0.90 to 0.97 of its own square rate; nothing to take |

Combined: B and A are not additive (A's decode removal and B's decode
halving overlap; a fused float16 kernel is A with B's numerics) and the
best stacked case, a fused float16 dequant-GEMM at the vendor's float16
rate, is w4_share x (1 - 0.877 x 0.781) = 0.21 of TTFT: the first-token
wall of the promoted path is the machine's matrix rate, and every W4
design is a slice of the 22 percent of the family that is not float32
GEMM or a precision change worth 14 percent of the GEMM.

## 7. Candidate selection under the registered rule

B (0.147) and A2 (0.112) clear the 0.10 line; C (0.109) is infeasible;
D is under it. B is the feasible candidate with the largest ceiling whose
mechanism the measurements demonstrate (the float16 GEMM measured on
both shapes; the float16 decode proxied). No microbenchmark was
implemented under this GO: the rule requires its own pre-registration
before any code, and B's ceiling rests on changing the served weights,
which is Artin's call, not a kernel question.

## 8. Falsifiers

- If a fused W4-codebook simdgroup-matrix kernel at T = 256 ever runs
  under 0.80 of the production decode-then-matmul on gate_proj with
  float32 accumulation, the reading "A2 is what a fused kernel can
  reach" is wrong (the vendor's affine kernel reads 0.84).
- If a float16-output decode kernel runs slower than 0.6 of the float32
  decode, B's ceiling drops below A1 and the ranking flips.
- If the 1,024-token wall on the promoted path moves by more than 3
  percent on a second session at the same commit, the shares here carry
  a session fence they do not declare.
- If the gate_proj warm-pool decode anomaly (10.4 ms) reproduces inside
  the serving path (a chunk's decodes at over 6 ms each in the additive
  trace), the allocator-state effect is a serving cost and bar 6's
  "allocation is not limiting" reading is wrong in situ.
- If the S16 body's split reads its decode above 0.4 of the family, the
  S16 side has a larger decode lever than W4 (1.59 of 5.55 s additive
  here, 0.29).

## 9. Proposed next qualification GO (for Artin; nothing armed)

QWEN-W4-F16-PREFILL-0, bounded: (1) desk number, the count of W4
weights whose float16 copy is inexact across all 192 matrices and the
largest relative change (a census of the artifact's codebooks, no model
run); (2) a float16-output variant of the decode kernel on the two
shapes, timing and bit-exactness against the float16 cast of the
float32 decode; (3) the float16 GEMM on real activations from one
chunk, output error against the float32 product; (4) only if (1) to (3)
hold, a parity rung on the frozen census positions and the 15 near
ties, the held-out prompts ABBA with TTFT, memory and 16-step decode;
hurdle TTFT at most 0.88 at 1,024 and 2,048 with a 0.03 fence, zero
argmax flips. Ceiling 15 percent; kill at (1) if the inexact count is
material or at (2) if the float16 decode is not under 0.7 of the
float32 decode. Alternative for the same GO slot: the S16 body split
(28 percent of the wall, decode 0.29 of it) before any W4 precision
change.
