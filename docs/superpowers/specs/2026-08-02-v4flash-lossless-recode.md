# Spec: DeepSeek-V4-Flash lossless re-coding (2026-08-02, design pass v0)

The quantization program has only ever measured house crystals and
two external models (SmolLM2-1.7B, Qwen2.5-0.5B) plus a read of
DeepSeek-V3's routed experts. V4-Flash is the largest and newest
object available and it is a 91%-MoE, which makes it the natural
next cell. But the obvious experiment is invalid, and saying why
is what shapes this whole spec.

## The catch: the experts are already fp4

Verified from the model's own `config.json` (2026-08-02):

| field | value |
|---|---|
| params / layers / shards | 304B / 43 / 48 x ~3.5GB |
| routed experts | 256 per layer, +1 shared, top-6 |
| `moe_intermediate_size` / `hidden_size` | 2048 / 4096 |
| `expert_dtype` | **fp4** |
| `quantization_config` | fp8 e4m3, block [128,128], `scale_fmt: ue8m0` |

Arithmetic: 256 experts x 3 matrices x 4096x2048 x 43 layers is
about 277B of the 304B total, so roughly **91% of this model is
routed-expert weight**, shipped at four bits.

THEREFORE THE CAPACITY METER DOES NOT APPLY. M = span_bits -
code_entropy detects heavy tails in a master. On weights already
projected onto a 16-point lattice per 128x128 block, the format
has destroyed the tail information the meter reads. A number
computed there would look like a capacity reading and would not be
comparable to any house-crystal reading, all of which were taken on
fp32/bf16 masters. This is the K3 expert fence (reading the shipped
quantized format, not the master) in a worse form: fp4 rather than
fp8. Booking a meter number for V4-Flash without this paragraph
attached would be an instrument error, not a result.

## The hard fence, stated first because it governs every rung

**No capability claim is possible on this machine.** 152GB at fp4
against 36GB of RAM: we cannot run this model, so we cannot gate
it, and the standing rider (loss claims are teacher-forced claims;
capability claims need the oracle gate) means any lossy proposal
here would be unfalsifiable by construction.

Two claim types survive, and the spec admits only these:

- **LOSSLESS**, verified by bit-identical round-trip. Exact,
  auditable, and needs no gate we cannot run.
- **PER-EXPERT function-space**, at expert granularity only. One
  expert is ~25M params and is comfortably runnable; the precedent
  is `blackhole_b0.py`'s function-space spot check.

Anything that would require a model-level capability verdict is out
of scope for the Mac and stays out of this spec.

## Why lossless is the interesting question anyway

The weights are a **symbol stream on a known lattice**. Roughly
Gaussian weights on a 16-point lattice do not have a uniform symbol
distribution, so the empirical entropy is strictly below the 4 bits
each symbol is stored in. The measured house precedent is that code
streams land within 1% of the Gaussian-capacity entropy bound
(packed crystal, RESULTS). The only question is the size of the gap
on someone else's model, and it is computable per shard with zero
capability risk.

A dyadic detail that makes the arithmetic clean: `ue8m0` scales are
exponent-only, so every block scale is a power of two and the whole
expert set lives on a common dyadic lattice. Differences across
experts with different block scales are then **exactly**
representable after an integer shift — no rounding is introduced by
aligning them. That is what makes rung 2 exact rather than
approximate, and it is squarely the lab's exact-integer wheelhouse.
VERIFY THIS against the reference implementation in the model repo's
own `inference/` directory before relying on it.

## Rungs

Each rung is independently killable and rung 0 is the gate.

### Rung 0 — fp4 symbol entropy (the gate)

Stream a **stratified sample** of 4-6 shards (early / middle / late
layers), dequant-free: read the fp4 codes and the ue8m0 scales as
symbols. Report per-tensor and pooled empirical entropy in bits,
against the 4 bits stored.

- Pre-registered prediction: the stream is non-uniform, so entropy
  is measurably below 4 bits. Magnitude NOT predicted.
- Kill condition: near-uniform symbols. Then rungs 1-3 die for the
  cost of one afternoon of bandwidth, which is the point of
  sampling first.
- Cost: ~25GB download, one afternoon.

### Rung 1 — lossless re-code

rANS over the rung-0 symbol statistics. Acceptance bar is
**bit-identical round-trip**, not a size ratio: decode must
reproduce the original bytes exactly, asserted per shard.

- Prediction: within ~1% of the rung-0 empirical bound (the house
  packing precedent).
- Report: GB saved on the sampled shards, extrapolated with an
  explicit fence that extrapolation is not measurement.

### Rung 2 — mu + delta expert coding

Write each expert as `w_e = mu + delta_e`, store mu once and
entropy-code the residuals. Lossless by construction, in exact
integer arithmetic on the shared dyadic lattice.

- What it tests that nothing here has: the ledger says experts are
  FUNCTIONALLY distinct (strict agreement 0.0000; large merge
  damage). Functional distinctness does not imply CODE
  distinctness — two experts can compute different functions while
  sharing most of their bits. 256 experts per layer gives real
  statistical power where the battery's 4 did not.
- Prediction from our own ledger: **modest or zero win**. The merge
  tests say the experts are diverse. A null is the likely outcome
  and is publishable.
- Note this is NOT gravity: GRAV-0T/REV measured post-hoc gravity
  destructive in both directions, and merge-free is a property OF
  the pulled basin. Rung 2 never replaces an expert with mu; it
  only codes relative to mu. None of the merge-damage findings
  apply to a lossless change of basis.

### Rung 3 — the block-scale stream

The `ue8m0` scales are a second, entirely separate symbol stream
(one exponent per 128x128 block) and are usually ignored. Code them
on their own statistics.

- Prediction: none registered; this is unmeasured anywhere in the
  ledger.
- Cost: tiny, reuses rung-1 machinery.

## Reuse inventory (write as little as possible)

- `scratch/blackhole_b0.py` — one shard on disk at a time
  (download -> process -> DELETE, the C7 OOM lesson applied to
  disk), atlas rows to jsonl, zero calibration. This is the
  streaming skeleton; V4 needs a naming/arch shim, not an
  algorithm.
- `scratch/capacity_meter.py:102` — a DeepSeek-V3 **fp8**
  block-dequant cell that already reads `<name>_scale_inv` tensors
  at 128x128. V4 plausibly shares that naming convention, but it is
  fp4 with ue8m0 scales: re-verify, do not assume it carries.
- `llmopt/quantize/pack.py:108` — `rans_size(codes, verify=True)`
  entropy-codes an integer symbol array, returns bytes-with-table
  AND entropy bits/symbol, and **round-trip-verifies with an
  assert**. Rungs 0 and 1 are essentially this one call: rung 0 is
  its `ent` return, rung 1 is its byte count under `verify=True`.
  The genuinely new code is only the fp4/ue8m0 symbol reader.
  `pack_tensor`/`unpack_tensor` and `allocator.py`,
  `sensitivity.py` sit alongside it.
- Exact-integer primitives in `llmopt/intmath.py` for rung 2's mu
  computation, so the decomposition is reproducible bit-for-bit —
  the deterministic-birth doctrine applied to compression.

## Instrument fences

1. The meter fence above: no capacity reading on fp4 weights.
2. Extrapolation from sampled shards is not measurement; any
   whole-model number is labelled as an estimate with its sample.
3. Bandwidth is a real cost, not a footnote: 168GB for the full
   set. Sample, always, and delete as you go.
4. The fp4/ue8m0 dequant must be validated against the model's own
   reference implementation before any number is booked.
5. Lossless claims are verified by round-trip, never by a size
   ratio alone; a size win with a failed round-trip is a bug
   report, not a result.

## Observed but unexplained (deliberately not rungs yet)

The config carries structure this spec does not touch and should
not guess at: `q_lora_rank`/`o_lora_rank` 1024, an attention index
(`index_n_heads` 64, `index_topk` 512), `num_hash_layers` 3 with
`hc_sinkhorn_iters` 20, `dspark_*` on layers 40-42 with
`dspark_markov_rank` 256, and per-layer `compress_ratios`
alternating 4 and 128. The Sinkhorn iteration is an
optimal-transport method and the low-rank ranks are exactly the
kind of structure the lab's rank-floor work speaks to, but none of
that is measured here. Listed so a later pass can pick it up
honestly rather than as decoration.

## Status

v0, design pass. A read-only reviewer scan of the ledger and
tooling is in flight to propose further rungs; this document gets a
v1 after those are verified line-by-line. Nothing runs until the
rungs are pre-registered in RESULTS.
