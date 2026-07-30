# Quantization at the Entropy Bound: Calibration-Free Packing of At-Capacity Networks

PROSE DRAFT v1 (2026-07-30 afternoon). Source of truth for
every number: docs/RESULTS.md (pre-registered). Skeleton and
cite registry: docs/paper-draft-entropy-bound.md. Figures
TODO'd inline.

## Abstract

Post-training quantization methods earn their complexity on
some models and waste it on others, and current practice
offers no way to tell which regime a model is in without
running the quantizers. We give a zero-inference criterion. A
network is *at capacity* when its weight code stream, on a
per-row sigma/2 grid, carries entropy equal to the Gaussian
capacity of that grid; we show that small math-task
transformers trained to convergence sit within 1% of this
bound. At capacity there is no structure left for calibration
to exploit: GPTQ (with its true Hessian), AWQ, and HQQ all
tie a closed-form sigma allocation at matched bits, and
MSE-optimal Lloyd-Max codebooks tie naive uniform grids. Off
capacity, the premium those methods earn is *monotone* in a
disk statistic M = span_bits − code_entropy that costs
nothing to compute: measured premiums run from ~1x at M
0.8–1.6 (our born crystals) through 16x at M 2.85 (MoE
experts) to 34x at M 3.62 (web-dense LLMs). The criterion
turns quantizer selection into a file read. Around it we
ship: a closed-form packing artifact whose disk format is its
runtime format (a bit-packed 5-bit GPU kernel at 2.39x over
fp16); a lossless rANS stage that reaches 3.67x over bf16 on
a 30B MoE with zero calibration on a laptop; and a fixed-point
decode path whose logit traces are bit-identical across GPU
vendors and laboratories — including on a frontier model's
own shipped MXFP4 expert, consumed natively with zero
requantization. All claims are pre-registered, and the
falsified predictions are reported alongside the confirmed
ones.

## 1. Introduction

The quantization literature is a contest of increasingly
sophisticated calibrated methods, each demonstrated on
web-trained LLMs, each earning real margins there. This paper
starts from an observation those demonstrations skip: on some
networks, all of that machinery ties a two-line closed-form
baseline exactly.

The networks where calibration ties are not toys chosen to
make the point. They are transformers trained to convergence
on a verified task distribution — models we can gate-score
exactly, so parity claims are measured in solved problems,
not proxy perplexity. On these models we measure (Sec. 2)
that the weight entropy at a fixed quantization step equals
the capacity of a Gaussian source at matched variance to
within 1%. Nothing in the weights prefers any code over any
other; there is no structure for a Hessian or an activation
profile to find. We call such networks *at capacity*.

Web-trained LLMs are not at capacity, and the distance from
capacity turns out to be measurable from the weight file
alone. Our meter M = span_bits − code_entropy prices exactly
one thing: what a fixed-width uniform grid pays to reach the
worst outlier in each row. Across every group we measured —
born crystals, MoE experts from four production model
families, attention blocks, dense web LLMs — the premium that
calibrated methods earn over the closed form is monotone in M
(Sec. 3). The practical consequence is a decision rule you
can run before loading the model onto an accelerator: below M
~ 2, use the closed form and keep the calibration budget;
above it, per-row max-anchored grids and their calibrated
refinements earn their cost.

Three artifacts fall out of taking the entropy bound
seriously (Secs. 4–6): a packed format at ~5 bits/weight
whose gate score matches the fp parent exactly on our
headline class; an entropy-coded container that is within a
rounding error of the measured code entropy, lossless, at
30B scale on a laptop; and — because integer codes admit
exact arithmetic — a decode path that produces bit-identical
logit traces on Apple and NVIDIA silicon, replicated
cross-laboratory, and demonstrated on a frontier MoE's own
shipped 4-bit expert format.

We report our falsified predictions with the same prominence
as the confirmed ones. Three of them (Secs. 3, 7) sharpened
the criterion into its final form; one of them closes an
entire class of post-hoc MoE compression and redirects the
effort to the systems layer (Sec. 7).

TODO figure 1: the meter-premium curve (M on x, measured
DeltaKL premium on y, all measured groups labeled).

## 2. The entropy bound, measured

Fix a weight tensor W with per-row standard deviation
sigma_r. Quantize row r on a uniform grid of step
sigma_r/2 — the coarsest step that keeps our models at gate
parity (Sec. 4) — and read the resulting integer codes as a
symbol stream. Two numbers describe that stream: its
empirical entropy H, and the capacity of a Gaussian source
quantized on the same grid, H* = ½log2(2*pi*e) −
log2(step/sigma). If the weights were drawn iid Gaussian, H
would equal H*; any learned structure — correlations,
heavy tails, clustering — shows up as H < H* (compressible
structure) or as grid overhead (outliers stretching the
span).

Measured on our born crystals (d64 8-head math-task
transformers, three seeds), H sits within 1% of H*. This is
the paper's anchor measurement: these networks are
*at capacity* — maximum-entropy on their own grid, nothing
left for a smarter code to find. Convergent training appears
to use its weight budget the way a channel at capacity uses
its bandwidth.

The at-capacity state has a sharp operational consequence,
which we verify directly (Sec. 4): every calibrated
quantizer we tested collapses to the closed form. And it has
a converse: any model whose code stream falls short of
capacity, or whose span outruns its entropy, is *not* at
capacity, and the shortfall is exploitable — by exactly the
methods the literature already built. The rest of the paper
is the measurement of that gap and the toolchain that lives
on its floor.

## 3. The meter: regime detection from disk

M = span_bits − code_entropy, computed at per-row sigma/2,
no forward passes. Span_bits is what the fixed-width grid
must pay per weight; code_entropy is what the weights
actually carry. Their difference prices the worst-case
outlier against the typical weight — which is, mechanically,
the same thing a calibrated outlier-aware method gets paid
for handling well.

TODO table 1: the dial. Born crystals M 0.8–1.6 -> premium
~1x. NNUE 0.82 -> sigma-law domain. MoE experts: Qwen3-30B
(5M/expert) 2.93, OLMoE (6M) 2.85, DeepSeek-V3 (45M) 2.33,
Kimi-K2 (40M) 2.01 -> 16x at 2.85. Attention 3.11 -> 22x.
Web-dense 3.62–3.9 -> 34x. Premiums are sigma-v-calibrated
DeltaKL, same device, matched bits.

Two falsifications built this instrument, and we report them
as such. First, "strong transport" — the claim that sigma
allocation would hold on any expert-shaped tensor — failed
on OLMoE experts (16x premium to HQQ at matched bits); the
meter had flagged the miss before the arms ran, which is
what promoted it from diagnostic to dial. Second, kurtosis,
the obvious moment statistic, is demoted by measurement: a
tensor group at kurtosis 3.50 (nominally Gaussian) still
carried M 2.78's worth of grid overhead; M caught what the
fourth moment blurred.

The dial comes with a second condition, also learned from a
falsification: zero-tax deployment requires an evaluation
metric with knee slack. Under outcome scoring (does the
model still solve the task), our packs are free below M ~ 2;
under per-token KL, nothing is free — round-to-nearest wins
that comparison. We argue outcome scoring is the
deployment-realistic metric, and we say explicitly that this
is an argument, not a theorem.

A scaling regularity rides on the MoE rows of the dial:
expert capacity is monotone in *per-expert size*, not expert
count — 5M-parameter experts sit at M 2.93, 45M at 2.33,
40M at 2.01, at the sigma-law boundary. Frontier practice is
consistent with the trend: Kimi-K3 ships 896 latent experts
(33M parameters each — a figure we established by pulling a
single expert from the 2.8T release by HTTP byte-range) in a
4-bit block format whose code stream carries 3.643
bits/param, within ~9% of that format's own entropy
capacity. We fence this row honestly: the meter on a
quantized release reads the shipped grid's image, not the fp
master, so K3 supports the band, not a rank ordering.

TODO figure 2: the atlas — Qwen3-30B, all 18,673 tensors
metered in 136.6 laptop-minutes; routers incompressible at
M 4.45; expert up_proj already in the sigma domain.

## 4. The artifact: packing at the bound

The packing rule is two lines. For each tensor t with
standard deviation sigma_t, choose the integer grid
q_t = ceil(2/sigma_t) — i.e. step sigma_t/2 — round, and
store the codes bit-packed at the width the span requires
(~5 bits/weight on our models; interface tensors emb/head
take a sigma/8 step, a distinction the fragility probe
prices, Sec. 9). No calibration data, no optimization, no
inference. The pack runs in 0.9 s where HQQ takes 61.7 s at
0.5B scale, and 16.6 s where HQQ takes 675.5 s on 6.4B
expert parameters — a 41x wall-time gap that matters
precisely because, in this regime, the expensive method buys
nothing:

*Parity.* On the d64 8-head class, gate score is preserved
exactly (58 -> 58 solved of 120), replicated at three seeds
(+2/−3/0). On the L4 depth-floor class the pack pays −5 at
weak seeds; we scope the zero-tax claim to the d64h8 class
and name per-crystal fragility (Sec. 9) as the axis the
entropy argument does not see. Compression is 6.15–6.65x
over fp32.

*The tie.* At matched 5 bits on at-capacity weights, GPTQ
with its real Hessian, AWQ with real activation profiles,
and HQQ with its per-tensor optimization all tie the closed
form at the gate — as the capacity measurement predicts:
there is no structure for them to spend their budget on. A
3-bit arm shows where damage lands when it does come:
solve rates stay flat while validity and DeltaKL degrade
first.

*The disk format is the runtime format.* A Metal GEMV kernel
consumes the bit-packed 5-bit codes directly — six codes per
uint32, no unpack pass — at 2.39x over fp16 at large shapes,
*beating* the byte-aligned variant of the same kernel: the
denser format wins on bandwidth. We report the honest
losses: at micro shapes the same kernel runs 0.91x, and our
first attempts at both kernels lost to their baselines.

One negative fences the artifact class: on a
jointly-trained matryoshka (tiered) crystal, every tier of
the nested pack pays 1–2 sigma — tiered packing is real
(−15% bytes per solve under escalation economics) but its
zero-tax version requires packing EMA parents, not
joint-STE children. Fragility, again, is orthogonal to
entropy.

## 5. Lossless coding: the bound as bytes

If the code stream carries H bits/weight, an entropy coder
should store it in H bits/weight, and the meter already
measured H from disk. A range-coder (rANS) stage over the
sigma-law codes achieves exactly that: on crystals, 9.10x
and 8.25x over fp32 (the residual gap to H is the coder's
~0.1% overhead); on Qwen3-30B, the packed-plus-coded
artifact is 16.48 GB — 3.67x smaller than the bf16 release —
lossless, produced on a laptop in one pass with zero
calibration. The pipeline composes: rANS is the at-rest
container, and decompression yields the same bit-packed
codes the Sec. 4 kernel executes, so the storage format and
the execution format are two states of one object. Deep
Compression Huffman-coded pruned CNNs in 2016; this is that
idea revived at LLM scale, with the coding gain *predicted
in advance* by the same statistic that selects the
quantization regime.

## 6. Determinism: exact arithmetic as a deployment property

Integer codes admit exact arithmetic, and exact arithmetic
is order-invariant: an int64 sum gives the same bits under
any reduction schedule, which is precisely what
floating-point inference cannot promise across GPUs. We
build this out in three stages, each verified by hash
equality, not tolerance.

*Stage 1 — exact GEMMs.* Weights as integer codes, GEMMs on
an exact-fp32 carrier with hi/lo splitting keeping every
partial below 2^24 (the fp32 mantissa bound, asserted at
runtime). Forward hashes are bit-identical between Apple
(MPS) and NVIDIA (cuda) silicon at two independent seeds,
while the fp logits of the same model differ.

*Stage 2 — full fixed-point decode.* RMSNorm as int64
sums-of-squares with integer-Newton isqrt; SiLU, softmax-exp
and RoPE as integer tables generated once and shipped as
bytes (libm variation across platforms is exactly the thing
being excluded, so tables must travel, never be
regenerated). The complete per-step logit *traces* — every
number, not just the argmax — hash identically on MPS and
cuda. The capability price, measured honestly: 96.66%
teacher-forced argmax agreement against the fp parent, with
disagreements concentrated at coin-flip margins (median
0.177 vs 7.6 overall); the reference implementation runs
10–40x slower than fp, and speed is not the claim. An
independent laboratory reproduced both digests from the
sha-pinned tables file on first attempt; the replication
protocol — pinned artifact, pinned instrument, full-digest
compare — deletes the tolerance column that replication
arguments usually turn on.

*Stage 3 — a frontier model's own format.* Kimi-K3's routed
experts ship as MXFP4: e2m1 integer codes times power-of-two
scales — already exact. We pulled one expert (17.5 MB by
safetensors byte-range, out of a 2.8T release), consumed the
shipped codes natively with zero requantization, and
measured sha-identical full-expert forwards (both GEMVs,
SiLU table, gating product, down-projection) on cpu, MPS,
and cuda. Bit-reproducible execution of frontier weights
requires no cooperation from the format: the industry's
4-bit block formats are already integer formats.

TODO figure 3: the three-stage hash ladder with digests.

## 7. What routing does to weights

The MoE rows of the dial (M ~ 2–2.9) invite an obvious
post-hoc program: find the redundancy between experts and
compress it — merge similar experts, share bases, factor
deltas. We measured the preconditions and closed the
program.

In weight space, production experts are decorrelated to
zero: across OLMoE expert pairs, delta-sigma ratio 0.995 and
correlation ~0.005 — there is no shared component for a
base-plus-delta factorization to extract, and merging the
*most* similar pair costs +3.4 perplexity. But the
redundancy did not vanish; it moved. Co-routing mutual
information between adjacent layers runs 300–500x above a
token-shuffle control at every depth. Storage-side
redundancy became usage-side structure — which is a systems
lever (prefetch, placement, caching, as the offloading
literature already exploits), not a parameters lever.

A causal experiment sharpens the mechanism. Training
matched micro-MoEs (0.9M parameters, four experts, top-1
switch routing; two seeds) under (a) the standard
load-balance loss, (b) no balance loss, and (c) tied
base-plus-delta experts: removing the balance loss changed
*neither* the decorrelation (0.0080 vs 0.0085) *nor* the
routing structure (256x vs 288x shuffle), and the router did
not collapse. The split is created by sparse assignment
itself — hard-routed experts never see the same tokens and
cannot stay correlated. The balance loss shapes load, not
structure. (Our pre-registered predictions said otherwise;
they were falsified, and the mechanism above is what the
falsification taught.) Two riders: the production
signature reproduces at 1/30,000th production scale, making
the phenomenon cheap to study; and experts *tied at birth*
gate within sigma of a dense control both seeds — the
factorization that post-hoc analysis proves impossible is
approximately free if imposed before training.

## 8. Related work / 9. Fences / 10. Reproducibility

[Related work and fences: lift from skeleton secs 8–9 —
already prose-shaped. Reproducibility: every experiment
pre-registered in an append-only log with falsifiers named
before runs; amendments preserved; instruments and seeds in
the repository; replication protocol as in Sec. 6.]
