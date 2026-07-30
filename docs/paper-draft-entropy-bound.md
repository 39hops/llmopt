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

## 4. The scaling law (Artin's law)
Expert capacity is monotone in PER-EXPERT SIZE, not count:
Qwen3-30B ~5M/exp M 2.93 | OLMoE ~6M 2.85 | V3 ~45M 2.33 |
K2 ~40M 2.01 (AT the sigma boundary). K2 flat in depth
(2.04-2.09). Industry rider: K3 ships 896 experts x ~66M at
MXFP4 (4-bit) — frontier practice consistent with the law.
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
- [ ] P3a/b end-to-end deterministic decode (strengthens S3)
- [ ] Related work with honest deltas: GPTQ/AWQ/HQQ/QLoRA/
      LLM.int8/SqueezeLLM/TurboQuant; MoE: Switch, DeepSeek
      fine-grained, expert offloading lit (co-routing
      prefetch has precedent to cite and measure against)
- [ ] Venue: MLSys or efficient-ML workshop first
- [ ] Fences section: all n=1 cells named as such
