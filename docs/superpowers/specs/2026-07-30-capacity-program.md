# Spec: THE CAPACITY PROGRAM (2026-07-30; drafted 07-29 late)

The packed-crystal C-series closed with one unifying criterion:
calibration-free quantization is an AT-CAPACITY property. This
spec turns that criterion + the C-series wins into the next
program. Seven points, ordered by payoff; each cell pre-regs in
RESULTS before it fires. Mac unless noted.

## P1 — C7: sigma-law on MoE routed experts (RUNNING)
Pre-reg'd 07-29 late; OLMoE-1B-7B on Mac. The transport claim:
at-capacity weights need no calibration ANYWHERE they occur.
In-model control (dense attn arm) tests whether the meter
ORDERS damage within one checkpoint. Lands -> the paper's
scope jumps from micro-crystals to the expert layers of
frontier-class MoE (most params in modern deployments).

## P2 — THE HYBRID ALLOCATOR (desk + 3080, behind P1)
Per-tensor routing, still zero calibration: meter M(tensor) ->
sigma-grid where M small, max-anchored RTN-grid where large;
both closed-form. Cells: (a) Qwen-0.5B full table v rtn/hqq at
matched bits — bar: within 2x of hqq DeltaKL at <2s wall (the
whole C6 table, won or honestly lost); (b) threshold sweep
M* in {2.0, 2.5, 3.0} (n=1 each, same harness); (c) OLMoE
whole-model (experts sigma + attn max-anchored) if P1 lands.
FALSIFIER: hybrid no better than pure max-anchored -> the
meter is diagnosis-only, not an allocator input.

## P3 — DETERMINISTIC DECODE END-TO-END (Mac + 3080 short test)
Extend C4's exact integer GEMM to a full decode step: integer
GEMMs (fp32 carrier, partials < 2^24) + fp norms/softmax
BETWEEN layers, quantized activations at each GEMM input
(scale = power of 2 so requant is exact shifts). Cells: (a)
single-layer forward hash Mac=cuda; (b) full 40-token greedy
decode hash Mac=cuda (the product claim: bit-reproducible
decode across vendors); (c) capability price: gate the
integer-forward path v fp (bar: within sigma). Ties to axiom
FX-V1 (their exact fixed-point NN) — cross-lab cell: same
packed crystal, house integer path v axiom FX-V1, hash
equality with ZERO adjudication. (d) wall-time honest card.

## P4 — BORN-PACKED (3080 overnight, needs Artin's nightly GO)
Birth ON the 5-bit sigma lattice from step 0 (STE round to the
per-tensor grid; q_t refreshed per epoch from live sigma).
Precedent: born-rational parity (07-27/28), born-ternary
premium. Arms: born-packed d64 v fp control, same seeds/
device; gate + pack (the artifact needs NO post-hoc step).
Prediction: parity (exactness free); the win is pipeline
simplicity + the C5 lesson (no post-hoc tax risk). Rider:
TRAINING LENS on this birth (proxy gate n=8 + flips/token +
per-layer update mass every N steps — the banked 07-29 riff,
still unfired).

## P5 — THE PRE-DEPLOY CARD (desk, cheap, anytime)
Standardize the two-number check: capacity meter M (regime) +
flips/token k_c probe (fragility, rho .883). One script, one
card per checkpoint: {M, kurt, flips/tok, predicted knee,
allocator recommendation}. Validation cell: retrodict the
month's snap verdicts (d56 Q8 bite, d64h8 3-bit shrug, C5
matryoshka tax, C6 Qwen crater) from cards alone — 4/4 bar.
This is the reviewer-facing artifact for the paper.

## P6 — ENTROPY-CODED SIGMA GRIDS (desk)
The C6c killer was the fixed-width penalty (bits priced by the
worst outlier). Cell: rANS/arith-code the sigma-grid code
stream per tensor (C1 showed deflate recovers ~1.8 bits on
crystals); re-read the Qwen table at ENTROPY-matched (not
span-matched) bits. Prediction: gap v rtn narrows materially
but hqq keeps the zero-point edge on tails. Also prices P4's
artifact (born-packed + entropy-coded = smallest honest file).

## P7 — PAPER ASSEMBLY (after P1; P2/P3 strengthen)
Working title: "Quantization at the Entropy Bound:
Calibration-Free Packing of At-Capacity Networks."
Claims (each already measured, C-series + P1): (1) at-capacity
criterion + capacity meter (zero-inference predicate); (2)
closed-form sigma-allocation = calibrated methods on
at-capacity weights, 33x fence on web-dense, mechanism named;
(3) entropy within 1% of Gaussian capacity; (4) bit-packed
kernel: disk format runs live at 2.39x; (5) cross-device
integer determinism (2 seeds); (6) tiered bytes + escalation
economics. Honest-negatives section is load-bearing: C5 tax,
C6 falsifier, C6b null, micro-shape kernel losses, prediction
2's missed threshold. Venue class: MLSys / efficient-ML
workshop first pass. Needs: related-work rigor (GPTQ/AWQ/HQQ/
QLoRA/LLM.int8/TurboQuant lineage — THEORY row has the
citations), n>=2 seeds on headline cells, C7 in the table.

## Standing constraints
Lab charter fences all of this to math/physics engines and
generic inference machinery — nothing here touches
chem/bio. Pre-reg before every run; falsifiers named; fences
travel (sigma per-tensor, gates same-device, meter thresholds
n=1 until P2b sweeps them). 3080 via wsl.sh; nightly-GO rules
for P4. Query first: scripts/results_query.py.

## Order
P1 (running) -> P2a -> P5 (validation retrodiction) -> P3a/b ->
P7 draft skeleton -> P4 (next nightly GO) -> P6 -> P2c/P3c/d.
Diet-evolution + multi-ply farmer stay queued behind this
program (separate spec).
