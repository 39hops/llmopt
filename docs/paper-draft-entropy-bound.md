# DRAFT SKELETON: Quantization at the Entropy Bound (P7/B5)

Working title: **"Quantization at the Entropy Bound:
Calibration-Free Packing of At-Capacity Networks"**
(alt, the Artin cut: "Black-Hole Experts: Router Focusing
Drives Weights Toward Maximum Entropy")

Status: skeleton, 2026-07-30 morning. Every number below is
booked in docs/RESULTS.md with a pre-registration. n counts
noted; headline C1 cell at n=3 pending today's seed births.

## 1. The criterion (the paper's spine)
A network is AT CAPACITY when its weight code stream at step
sigma/2 carries entropy = Gaussian capacity (measured <1% gap
on born crystals). At capacity there is NO structure for a
calibrated quantizer to exploit:
- GPTQ (real Hessian), AWQ, HQQ tie closed-form sigma
  allocation at matched bits (C3, crystals).
- MSE-optimal Lloyd-Max codebooks tie naive uniform (07-25).
- The premium calibrated methods DO earn elsewhere is
  MONOTONE in a zero-inference disk statistic M = span_bits
  - code_entropy (the capacity meter/dial): crystals ~1x at
  M 0.8-1.6; OLMoE experts 16x at 2.85; attn 22x at 3.11;
  web-dense 34x at 3.62 (C7 + C6b; 6 points).
- Second condition (N2): zero-tax deployment additionally
  requires a metric with knee slack (outcome scoring, not
  logit distance) — deployment-realistic, argued not assumed.

## 2. The artifact (what we ship)
- Closed-form allocation q_t = ceil(2/sigma_t); pack at ~5
  bits/wt; gate parity EXACT on d64h8 (58->58), -1 on L4d56
  (C1; n=3 pending). 6.15-6.65x v fp32.
- Entropy within 1% of capacity (C0); entropy coding worth a
  further 1.29 bits/wt at 30B scale (P6).
- Bit-packed 5-bit Metal GEMV runs the DISK format at 2.39x
  over fp16 at large shapes, beating the byte-aligned kernel
  (C2b); honest micro-shape losses booked (C2).
- Wall-time: 0.9s v HQQ 61.7s at 0.5B; 16.6s v 675.5s at
  6.4B expert params (41x) (C6, C7).

## 3. Determinism (the claim nobody else makes)
Integer-GEMM forward hashes bit-identical MPS v cuda (exact
fp32 carrier, partials < 2^24; 2 independent seeds), fp
logits differ, greedy streams match anyway (C4 + R-pass).
Cross-vendor bit-reproducible deployment for packed models.
Upgraded twice since: (P3) FULL fixed-point decode — shipped
integer tables, no libm — hashes identical logit TRACES on
MPS and cuda at 96.66% argmax agreement v fp; (K3-D1) the
same discipline consumes a frontier model's shipped MXFP4
natively: one Kimi-K3 expert (17.5 MB byte-range out of
2.8T), sha-identical GEMV traces on cpu/mps/cuda, zero
requantization.

## 4. The scaling law (Artin's law)
Expert capacity is monotone in PER-EXPERT SIZE, not count:
Qwen3-30B ~5M/exp M 2.93 | OLMoE ~6M 2.85 | V3 ~45M 2.33 |
K2 ~40M 2.01 (AT the sigma boundary). K2 flat in depth
(2.04-2.09). Industry rider (K3-D1, measured): K3 ships 896
LATENT experts (3584-dim latent, ~33M/expert — discovered
by pulling one expert via byte-range) at MXFP4; metered M
1.94-2.15 straddles K2's boundary (grid-image confound
fenced); its 4-bit codes carry 3.643 bits/param — frontier
practice ships within ~9% of its format's entropy capacity.
The atlas figure: Qwen3-30B, 18,673 tensors, 136.6-min
laptop pass, zero calibration (B0); routers incompressible
(M 4.45, keep fp); up_proj already in the sigma domain.

## 5. Structure lives in routing, not weights (the split law)
- Experts decorrelated to zero in weight space (N3: delta
  sigma ratio 0.995, corr ~0.005) — no post-hoc merging/
  sharing/replication.
- Co-routing MI 300-500x shuffle at every layer (B4) — the
  redundancy the router removed from storage reappears as
  usage correlation; merging the top pair costs +3.4 ppl.
- Consequence: params-side compression is closed post-hoc;
  the exploitable levers are (a) our packing at rest, (b)
  routing-side systems (prefetch/placement/caching).

## 6. Honest negatives (load-bearing)
C5 tier tax on joint-STE tensors (fragility k_c is an
orthogonal axis; meter blind to it); C6 33x falsifier fired
(mechanism: span-pricing); C6b granularity null; P2a clip
catastrophe (outliers load-bearing at FULL magnitude — ppl
138,890 at k=4); C7 strong transport falsified (dial saved
it); B0 prediction-2 falsified (size not fineness); B3
mid-dip not a regularity; N2 dial-pack loses to rtn on KL.
Three instrument bugs caught by controls, all amended.

## 7. Needed before submission
- [x] C1 at n=3 DONE: h8 parity replicates (+2/-3/0); L4 floor pays -5 at weak seeds (claim scoped to d64h8 class; fragility axis named)
- [x] rANS coder DONE (P6-v2): crystals 9.10x/8.25x fp32; Qwen3-30B 16.48 GB = 3.67x bf16, lossless, laptop, zero calibration
- [x] P5 card DONE at CLASS scope: flips/tok ranks architectures correctly (L4 1.3x h8) pre-pack; seed-level unresolved (matched-operator probe = named follow-up)
- [x] P3 DONE: full fixed-point decode, bit-identical logit traces MPS=cuda (2 table versions), 96.66% argmax agreement v fp at low-margin disagreements; sec.3 upgrades to end-to-end
- [x] Related work drafted (sec. 8) with honest deltas
- [ ] Venue: MLSys or efficient-ML workshop first
- [x] Fences section drafted (sec. 9)
- [ ] Verify flagged citations (TurboQuant authorship) at
      writing time; every cite must be checked against the
      actual paper before submission (THEORY discipline)

## 8. Related work (honest deltas)

**Calibrated PTQ.** GPTQ (Frantar et al., 2022; layer-wise
Hessian), AWQ (Lin et al., 2023; activation-aware scaling),
HQQ (Badri & Shaji, 2023; calibration-free but
per-tensor-optimized). Our delta is NOT "we beat them": on
web-dense LLMs they earn a 16-34x premium and we say USE
THEM. The delta is the criterion: on at-capacity networks
(M < ~2) they tie closed-form sigma allocation at matched
bits (C3, measured on all three), and the meter predicts
which regime you are in from disk alone, zero inference.
No prior work we know of offers a pre-quantization
regime test.

**Outlier methods.** LLM.int8() (Dettmers et al., 2022)
and SqueezeLLM (Kim et al., 2023) isolate outliers as
mixed-precision exceptions. Our P2a falsifier is direct
evidence FOR their premise from the opposite direction:
clipping web-dense outliers to k sigma is catastrophic
(ppl 138,890 at k=4) — outliers are load-bearing at full
magnitude. Our M statistic is exactly a measure of the
fixed-width price those outliers impose on a uniform grid;
the dial quantifies when their machinery is needed.

**Information-theoretic codebooks.** NF4/QLoRA (Dettmers
et al., 2023) argues quantile codebooks are
information-optimal for Gaussian weights; QuIP# (Tseng et
al., 2024) and AQLM (Egiazarian et al., 2024) use
incoherence processing / learned lattices to Gaussianize
then vector-quantize. We measure the endpoint of that
line: networks BORN at capacity need none of the
machinery — their sigma/2 code stream already carries
Gaussian capacity (<1% gap), and Lloyd-Max ties uniform
(07-25). Incoherence processing is a transform TOWARD the
regime our training produces natively. TurboQuant
(rotation-based, 2025 — cite to verify) belongs to the
same transform family.

**Weight entropy coding.** Deep Compression (Han et al.,
2016) Huffman-codes pruned CNNs; DeepCABAC (Wiedemann et
al., 2019) arithmetic-codes them. Our P6 is the LLM-scale
revival with a twist: rANS on sigma-law codes is lossless
ON TOP of the quantization the gate already priced, and
the coding gain (1.29 bits/wt at 30B) is PREDICTED by the
same entropy statistic the meter reads. Disk format = rANS
stream; runtime format = the bit-packed codes the C2b
kernel consumes directly.

**MoE structure.** Switch (Fedus et al., 2021) and
DeepSeekMoE (Dai et al., 2024) argue fine-grained experts
specialize; our split law is the weight-space receipt:
experts decorrelate to zero (N3) while co-routing MI stays
300-500x shuffle (B4) — the redundancy moved to the
router. Expert offloading/prefetch (e.g. Eliseev & Mazur,
2023, Mixtral offloading) exploits routing locality for
memory; our B4 measurement says co-routing MI is the
statistic such systems are implicitly betting on, and our
merge probe (+3.4 ppl top pair) closes the params-side
alternative. Expert pruning (REAP, 2025) is orthogonal:
it removes experts; we compress the survivors at rest.

**Determinism.** Integer-only inference exists for edge
CNNs (Jacob et al., 2018 quantized inference); we know of
no prior LLM decode demonstrating bit-identical logit
traces across GPU vendors with shipped transcendental
tables (P3), nor one running a frontier model's own
shipped format exactly (K3-D1).

## 9. Fences (all named, none silent)

- C1 parity claim is scoped to the d64h8 class (n=3); L4
  floor pays -5 at weak seeds. Fragility (k_c, flips/tok)
  is an ORTHOGONAL axis the meter does not see; the class
  gate (P5) ranks architectures, seed-level unresolved.
- The meter on quantized-release models (K2, K3) reads the
  shipped grid's IMAGE, not the fp master — banded claims
  only, no rank claims across format boundaries.
- CE-400 is format-BOUND; sigma never transports across
  devices/widths; all paired arms same-device same-seed.
- P3's 96.66% agreement is v the fp model at coin-flip
  margins (median 0.177 v 7.6); the determinism claim is
  about the fixed-point path's OWN reproducibility, not fp
  equivalence. Reference speed ~10-40x slower; speed is
  not the claim.
- K3-D1 is two experts, GEMV-level; full-expert chain is
  composition of tested pieces but not yet run.
- Wall-time wins (0.9s v 61.7s) are calibration-time, not
  inference-time, comparisons; C2 micro-shape kernel
  losses booked and kept.
- n=1 cells: C4 second seed only (n=2); C5 tier tax n=1;
  B3/B4 single-model (K2, OLMoE respectively); P6 30B
  single-model.
