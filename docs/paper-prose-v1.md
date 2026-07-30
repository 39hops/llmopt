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

[PROSE TODO — from C0/C1/C2/C2b/C3/C5/C6 bookings: the
closed-form q_t = ceil(2/sigma_t); gate parity table at n=3;
the honest L4 floor scoping; GPTQ/AWQ/HQQ tie table; kernel
cards including the micro-shape losses; wall-time column.]

## 5. Lossless coding: the bound as bytes

[PROSE TODO — from P6/P6-v2: rANS at 9.10x/8.25x over fp32
on crystals; Qwen3-30B at 16.48 GB = 3.67x bf16, zero
calibration, laptop wall-clock; coding gain predicted by the
meter's entropy reading.]

## 6. Determinism: exact arithmetic as a deployment property

[PROSE TODO — from C4/P3/K3-D1/K3-D2/FX-V1-H: integer
carrier; shipped tables doctrine; bit-identical logit traces
across vendors; cross-lab replication protocol (sha-pinned
artifact, pinned instrument, full-digest compare); the
Kimi-K3 expert consumed natively; 96.66% argmax agreement
price at coin-flip margins.]

## 7. What routing does to weights

[PROSE TODO — from N3/B4/UMOE-1: decorrelation ~0 +
co-routing MI 300–500x shuffle on production models; merge
probe +3.4 ppl closes post-hoc params-side compression; the
causal 3-arm: sparse assignment, not the balance loss,
creates the split (n=2 seeds); tie-at-birth as the live
lever; what this means for offloading systems.]

## 8. Related work / 9. Fences / 10. Reproducibility

[Related work and fences: lift from skeleton secs 8–9 —
already prose-shaped. Reproducibility: every experiment
pre-registered in an append-only log with falsifiers named
before runs; amendments preserved; instruments and seeds in
the repository; replication protocol as in Sec. 6.]
