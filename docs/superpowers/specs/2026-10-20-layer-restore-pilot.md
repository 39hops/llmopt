# Spec 2026-10-20: QWEN-LAYER-RESTORE-0, one bounded layer-restoration feasibility pilot

Status: Phase 1 plan (target, storage, memory, runtime, validation) written
before any vendor byte beyond shard 18 is retrieved. The shard download is
a separately authorized step (Artin's GO of 2026-10-20, section 6). Nothing
in this document was chosen by looking at E0 outcomes.

Follows VERDICT QWEN-HEAD-RESTORE-0 (NULL, RESULTS L81143): the vendor
BF16 lm_head on arm C's frozen hidden states recovered no E0 agreement,
so by elimination the teacher disagreement sits upstream of the head.
That reading is a hypothesis; this pilot is its first test.

## 1. Target and storage

### 1.1 Vendor layout (Qwen/Qwen3.8-27B, revision 1d4bf0f2, index read 2026-10-20)

Text stack: 64 decoder layers, hidden 5120, intermediate 17408. Layer
types alternate three gated DeltaNet layers then one full-attention layer
(full attention at 3, 7, 11, ..., 63: 16 layers; DeltaNet: 48 layers).
Tensor names carry the prefix `model.language_model.layers.N.`.

DeltaNet layer (14 tensors, 766.5 MB BF16; arm C holds 128.6 MB):
in_proj_qkv [10240, 5120] S16, in_proj_z [6144, 5120] S16, out_proj
[5120, 6144] S16, in_proj_a and in_proj_b [48, 5120] raw BF16, A_log and
dt_bias [48] raw, conv1d [10240, 1, 4] raw, norm [128] raw, the MLP
gate/up [17408, 5120] and down [5120, 17408] W4, two RMSNorm vectors raw.

Full-attention layer (11 tensors, 744.5 MB BF16; arm C holds 122.2 MB):
q_proj [12288, 5120] (queries and the output gate, interleaved per head),
k_proj and v_proj [1024, 5120], o_proj [5120, 6144], all S16; q_norm and
k_norm [256] raw; the same W4 MLP and raw norms.

Shards do not align with layers. Eight layers straddle a shard boundary
(4, 21, 29, 37, 45, 53 and 61 are DeltaNet; 15 is full attention). Complete layers per shard:

| shard | bytes | complete layers | straddling |
|---|---|---|---|
| 00001 | 3,966,730,552 | 0, 1, 2, 3 (plus 333 vision tensors) | 4 |
| 00002 | 3,043,080,328 | 5, 6, 7 | 4 |
| 00003 | 2,542,796,952 | none (embed_tokens only) | |
| 00004 | 3,988,973,152 | 10 to 14 | 15 |
| 00005 | 2,099,339,864 | 8, 9 (both DeltaNet) | 15 |
| 00006 | 3,979,553,696 | 16 to 20 | 21 |
| 00007 | 2,108,759,344 | 22 (DeltaNet), 23 (full attention) | 21 |
| 00008 to 00016 | 3.98 GB each (even) | five layers each | one |
| 00009, 11, 13, 15, 17 | 2,108,759,344 each | one DeltaNet, one full attention | one |
| 00018 | 3,392,197,344 | none (lm_head, MTP) | |

### 1.2 Recommended candidate: layers 22 and 23 from shard 00007

Chosen on architecture and cost, not on E0:

- Both layer families in one contiguous group (22 DeltaNet, 23 full
  attention), so the DeltaNet recurrence and the attention cache are
  each exercised once. Single-layer arms (22 alone, 23 alone) come from
  the same bytes and separate the families as a diagnostic.
- Every tensor of both layers is inside the shard; nothing straddles.
  The partial layer 21 in the same shard is not restored (a half-restored
  layer is not an interpretable unit).
- Cheapest download tier (2.11 GB; the 3.98 GB tier buys five layers but
  this pilot restores a pair). Shard 00001 would cost 3.97 GB and carry
  vision tensors.
- Mid-stack: 40 layers downstream for the sensitivity instrument
  (section 5) and 22 upstream, so the restored pair runs on inputs the
  compressed body produced, which is the counterfactual of interest.

Alternatives for a later candidate, by position: shard 00017 (62, 63,
the last pair before the final norm) and shard 00002 (5, 6, 7, early).
Neither is selected now; a second candidate is justified only by the
pilot's verdict and is Artin's call.

### 1.3 Download, storage, memory, runtime

Download: exactly one file, `model-00007-of-00018.safetensors`,
2,108,759,344 bytes, LFS sha256
80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892 (hub
metadata at the pinned revision, re-read at fetch and required to equal
these pins), to `~/qwen_vendor_shards/1d4bf0f2/` beside the head shard,
outside the repository; sha and size verified after the transfer, the
safetensors header parsed, its name set required to equal the index's
30 names for the shard, and the 25 target tensors checked for BF16 dtype
and the manifest shapes.
Disk: 47 GB free on the Mac volume. The head shard (3.39 GB) stays.

Memory: the Metal census of arm C peaks at 14.3 GB (footprint 11.0 GB
with the 2 GiB cache limit). The restored pair held as float32 (exact
widening of BF16, so the only change is the weights) adds 1.53 GB plus
1.49 GB; the two layers' residents (0.25 GB) are released. Projected
peak about 17.1 GB on a 36 GiB machine.

Runtime (measured basis: the census prefills 6,359 E0 tokens in 215 s at
29 tokens per second; the 454 validation tokens take about 20 s):

| stage | arms | wall |
|---|---|---|
| boundary capture for the oracle (control model, validation contexts) | 1 | 1 min |
| one-layer oracle, mlx side and torch side (CPU, float32, 454 tokens) | 2 layers | 2 min |
| score: CONTROL, DENSE_C, L22, L23, GROUP on E0 + validation (dense arms ran 1.4x slower in smoke) | 5 | 20 to 25 min |
| sensitivity from the score dumps | | 1 min |

## 2. One-layer oracle (implementation equivalence)

Before any restored-layer measurement, the mlx-lm Qwen3.5 decoder layer
(the class arm C runs on, with the l2norm override the tower applies) is
compared against the vendor PyTorch implementation
(transformers 5.15.0 `Qwen3_5DecoderLayer`, `.venv_teacher`) on identical
frozen inputs: the residual stream entering layer 22 and layer 23,
captured from the compressed model's prefill of the six prefix snippets
and the corpus document (454 positions, full sequence, fresh cache). Both
sides run float32 and the same weights. Two weight sources: the decoded
arm C tensors (before the download, the machinery check) and the vendor
BF16 tensors (after the download, the registered gate).

Semantics checked by construction, not assumed: RMSNorm zero-centred
weights (torch `1 + w`, mlx-lm `+1` at load); DeltaNet q/k l2norm with
eps inside the square root (torch `l2norm`, the tower's override);
conv1d layout [D, 1, K] against [D, K, 1] and the zero initial
convolution state; the gated RMSNorm order (norm, weight, silu gate);
query/gate split per head; partial rotary (64 of 256 dims, non
interleaved, text position ids equal across the three MRoPE axes);
attention scale, q/k norm before rotary; the DeltaNet chunked rule
(torch, chunk 64) against the Metal kernel (both from a zero state).

Registered tolerance (float32 both sides, different reduction orders):
relative L2 of the layer output difference at most 1e-4 and the maximum
absolute difference at most 1e-3 of the output row RMS, per layer, over
all 454 positions. A cache-split check on the mlx side (one shot against
chunk 256 with the cache carried) is held to the same tolerance. A
violation stops the pilot before any restoration is scored.

## 3. Counterfactual restoration (the measurement)

Arms, all through `prefill_chunked` with a fresh cache per sequence,
chunk 256, the GEMM prefill path, head full, live vocabulary 248,077:

- CONTROL: arm C as served (fused residents).
- DENSE_C: layers 22 and 23 replaced by dense float32 modules holding
  the decoded arm C tensors. Same machinery as the restoration, no
  weight change. Control for the swap: argmax must equal CONTROL at every
  position whose CONTROL margin exceeds 0.01 and the logits at the
  teacher's top-16 must agree within 1e-2 absolute.
- L22, L23: one layer restored to the vendor BF16 weights (float32
  widening), the other compressed. Diagnostic arms.
- GROUP: both restored. The registered contrast is GROUP against CONTROL.

Test surfaces: E0 (the 24 teacher-greedy rollouts, 4,722 positions,
diagnostic) and the validation set (corpus 356 in five paragraph blocks,
prefixes 98 in six snippet blocks). Reported per arm and per set: teacher
top-1 agreement, paired corrected and broken counts with the exact sign
test and a 10,000-draw block bootstrap on the block units, teacher-token
rank and margin changes, forward KL and cross-entropy, the opener z(We)
minus z(The) at the first generated position of each rollout, the
numerical parity of the unchanged operations (DENSE_C), memory footprint
and throughput. Resolution of the 454 validation positions, from the
one previous pass on them (the head restoration: 13 discordant
positions, bootstrap interval [-0.78, +3.41], half-width about 2.1
points): the +1.0 bar sits inside the set's noise band, so a fire whose
interval includes zero is inconclusive at this resolution, counts as no
fire in the refutation ratio, and selects no further layer on this set.

## 4. Registration (to be frozen in PRE-REG QWEN-LAYER-RESTORE-0)

Bars on GROUP minus CONTROL: (1) E0 agreement at least +1.0 point,
diagnostic, reported beside the verdict and never required for it; (2)
validation agreement at least +1.0 point pooled with neither subset net
negative, VALIDATED only with a bootstrap interval excluding zero; (3)
validation KL at least ten percent lower, a calibration reading. Preconditions (measurement invalid if they fail):
the oracle tolerance of section 2 on the vendor weights, the DENSE_C
parity, the shard sha. Refutation: bar 1 NO-FIRE and bar 2 not a
resolved fire (no fire, a conjunct failure, or a fire whose interval
includes zero) means restoring this pair recovers nothing measurable at
the registered resolution; the null is booked without expanding to
another layer.
Single-layer arms are reported, never adjudicated; three contrasts on one
validation set is disclosed as multiplicity.

## 5. Sensitivity instrument (observation, not a bar)

On the validation contexts and every E0 rollout, the residual stream is
captured at nine boundaries for CONTROL and GROUP: the input of layer 22,
the output of layer 23, the outputs of the full-attention layers 27, 31,
39, 47, 55 and 63, and the final norm output. Reported: the per-tensor
weight reconstruction error of the compressed pair (relative Frobenius
against BF16), the local activation difference at the output of 23
(relative L2 per position), its evolution through the downstream
boundaries (ratio to the local difference, median and quantiles), and
the final logits response (KL, argmax flips). This separates weight
error, local activation difference, propagation and prediction change for
one pair inside an otherwise compressed network; it does not identify
the source of all compression error.

## 6. Evidence rules

Frozen and unchanged: arm C, the E0 rows, the Metal and prefill receipts,
the head-restoration receipts, the teacher records. New receipts under
`logs/qwen_layer_restore/` refuse to overwrite; smoke under
`logs/qwen_layer_restore_smoke/` with a labelled stand-in (the decoded
arm C tensors as the "restored" weights, which is also the DENSE_C
control). Independent Opus review of the driver before registration and
of the receipts before booking. No fitting, no selection on outcomes, no
second shard without a separate GO.
