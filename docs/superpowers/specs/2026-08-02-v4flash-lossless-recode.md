# Spec: DeepSeek-V4-Flash lossless re-coding (2026-08-02, v1)

v0 was a design pass. v1 folds in a read-only reviewer scan of the
ledger, every claim of which was verified line-by-line before
adoption (see "Provenance" at the end). The scan changed three
things materially: it found a **free read that gates the whole
program**, it found that one of my rungs was **already run and
nulled at production scale**, and it found an **instrument hazard
that would have silently voided my acceptance bar**.

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
projected onto a 16-point lattice per block, the format has
destroyed the tail information the meter reads. A number computed
there would look like a capacity reading and would not be
comparable to any house-crystal reading, all taken on fp32/bf16
masters. This is the K3 expert fence in a worse form: fp4 rather
than fp8.

## The hard fence, stated first because it governs every rung

**No capability claim is possible on this machine.** The shipped
footprint is ~168GB (48 shards; 277B at 4 bits plus ~27B at 8 bits
≈ 165GB — the earlier "152GB" was the all-params-at-fp4
idealization, so the case is ~10% stronger than v0 stated) against
36GB of RAM. We cannot run this model, cannot gate it, and the
standing rider — loss claims are teacher-forced claims, capability
claims need the oracle gate — makes any lossy proposal here
unfalsifiable by construction.

Two claim types survive, and this spec admits only these:

- **LOSSLESS**, verified by bit-identical round-trip.
- **PER-EXPERT function-space or exact-forward**, at expert
  granularity only (one expert is ~25M params).

Because no capability claim is possible, **no seed-count argument
applies anywhere in this program** — stated once so it covers every
rung.

## Rungs

Ordered bandwidth-optimally. Rung -1 is free and gates the rest.

### Rung -1 — the safetensors HEADER read (free, do this first)

`scratch/k3_expert_demo.py:44-79` already does exactly this:
`struct.unpack("<Q", _get(url, 0, 8))` for the header length, then
`json.loads` of the tensor index, then HTTP Range-fetches only the
spans it wants — every blob sha256'd at write and re-asserted at
load. This is the booked K3-D1 extraction cell.

What the header settles that `config.json` does not:

- **The expert scale granularity.** `weight_block_size [128,128]`
  plausibly describes the *fp8* path; the fp4 expert path may carry
  its own layout (MXFP4 is group-32 with E8M0 scales; NVFP4 is
  group-16). The scale tensor's shape decides it and is in the
  header.
- **This swings rung 3 by ~500x.** At [128,128], the scale stream
  is 277e9/16384 ≈ 17 MB, i.e. 0.01% of the artifact — no
  meaningful "GB saved" exists there. At group-32 it is
  277e9/32 ≈ 8.7 GB ≈ 5% of the artifact — a real rung. One free
  read decides which.
- Whether the fp4 path is a power-of-two-scaled integer lattice at
  all, which is what makes rung A exact rather than
  fp-with-tolerance.
- The true expert param count, the shared expert's storage, and
  what the `index_*` / `num_hash_layers` / `dspark_*` tensors are
  by name and shape.

Cost: minutes, <1 MB. No prediction to register — this is a
fact-read.

### Rung 0 — fp4 symbol entropy (with a real prior)

Stream a stratified sample of 4-6 shards; read fp4 codes and scales
as symbols; report per-tensor and pooled empirical entropy.

**Prior, and it is a good one:** VERDICT K3-D1 (RESULTS.md:11450)
measured a frontier shipped 4-bit stream at **3.643 bits/param,
rANS = Shannon to three decimals, ~9% lossless margin**. So:

- Pre-registered prediction: V4's expert stream reads **3.6-3.9
  bits/param** order-0, with rANS within ~0.5% of it (the P6-v2
  precedent).
- Directional refinement, mechanistic: if V4's scale blocks are
  *coarser* than K3's group-32, within-block magnitude spread is
  larger, the histogram flatter, and entropy should land **at or
  above** 3.643.
- **The kill condition changed.** v0 said "near-uniform symbols
  kills it"; K3 says that almost certainly will not fire. The
  honest framing is that rung 0 is a **second point on a one-point
  law**, and the payoff is the *contrast* — how much coding margin
  does block coarseness cost? That contrast is new.
- Prior status: K3-D1 is n=1-ish (two experts, GEMV-level;
  RESULTS.md:11459). The pre-reg must say "prior from n=1".

### Rung 0b — split the stream into SIGN and MAGNITUDE (free)

The e2m1 alphabet is sign x 8 magnitudes; the house LUT is pinned
at `k3_expert_demo.py:33` (`[0,1,2,3,4,6,8,12]`).

- For a symmetric weight distribution H(sign) = 1.000 bits exactly
  and is incompressible, so K3's 3.643 implies **H(magnitude) ≈
  2.643 of 3 bits — all the coding gain lives in the magnitudes.**
- Predictions: (a) H(sign) >= 0.9995 bits per expert tensor; (b)
  all rung-0 margin is in the magnitude sub-stream.
- **A falsifier worth firing:** any tensor with H(sign) < 0.999 has
  systematic sign asymmetry, contradicting the kurt≈3.0 Gaussian
  readings across four MoEs — that would be a real finding about
  V4's training.
- Cost: free, same pass.

### Rung 1 — lossless re-code

rANS over the rung-0 statistics. Bar is **bit-identical
round-trip**, asserted per tensor, never a size ratio.

Use `llmopt/quantize/pack.py:108` `rans_size(codes, verify=True)`
and pin `verify=True` unconditionally. **Do not** reuse
`scratch/pack_rans.py:18` — see hazard 1.

### Rung 2 — naive mu+delta: ALREADY NULLED, run only as a control

v0 predicted "modest or zero win" from the wrong evidence. I cited
the 208k-param 4-expert house micro-MoE's output agreement; the
right citation is **N3, which is this exact rung run at production
scale**:

- Pre-reg (RESULTS.md:11095): "decode all 128 experts' gate_proj,
  compute the mean-expert and per-expert delta; report
  sigma(delta)/sigma(weight)... If the ratio reads < 0.5 anywhere,
  base+delta compression arms."
- **N3 VERDICT (RESULTS.md:11107): "experts share NOTHING in
  weight space — dynamic replication is dead post-hoc, alive only
  at birth." sigma(delta)/sigma(W) = 0.995 / 0.993; pairwise
  correlations mean 0.0024 / 0.0054.**

Quantitative prediction now available: residual sigma ratio
0.98-1.00, residual entropy >= raw entropy, so naive mu+delta is a
**net loss** of roughly -0.0 to -0.1 bits/param once the extra
frequency table and stored mu are counted. Run it only as a cheap
confirmed-null on a new vendor and format (~1 hour), not as a
candidate.

Note this was never gravity: GRAV-0T/REV measured post-hoc gravity
destructive both ways, and rung 2 only *codes* relative to mu, it
never replaces an expert with mu.

### Rung 2b — permutation-aligned mu+delta (the gauge-legal version)

N3's instrument was entrywise pairwise correlation, which is
exactly what `CLAUDE.md` forbids for weight comparison: "the same
function lives at many weight arrangements (neuron permutations,
rescalings)". Two experts computing overlapping functions with
permuted hidden units read correlation 0. **So N3 correctly kills
coordinate-aligned mu+delta and says nothing about the aligned
version.**

- Lossless-compatible: a permutation is a bijection; one 2048-entry
  index per expert is ~2.8 KB against 25M params (0.001%).
- Prediction is honestly two-sided. The split law says hard-routed
  experts never see the same tokens, so alignment may buy nothing;
  if it buys > 0.2 bits/param, N3 needs an amendment.
- Kill: aligned residual entropy >= unaligned. Cost: one layer, 32
  experts, `scipy.linear_sum_assignment` on 2048x2048 — an
  afternoon, zero extra bandwidth.

### Rung 2c — the pooled-table question (free, predicted to WIN)

The version of "do experts share code mass" that N3 did *not* kill:
do the 256 experts share a **frequency table**?

Mechanism: the black-hole law says router focusing drives expert
weights to max-entropy Gaussian at their own scale (V3 experts read
kurt 3.07, "Gaussian to two decimals"). Gaussian-on-a-fixed-lattice
implies near-identical *marginal* histograms across experts even
when the weights are fully decorrelated. Universal-coding
decomposition: total rate = N·H(pooled) + sum_e KL(p_e || p_pooled).

- Prediction: mean KL(expert || pooled) **< 0.01 bits/param**, so
  one global 16-symbol table serves all ~33,000 expert tensors —
  collapsing per-tensor table overhead and making the artifact
  seekable. Per-tensor table overhead is already a named fence in
  the paper draft; this rung is its measured amortization.
- Kill: KL > 0.05 bits/param — which would itself be a new finding
  (experts differing in distribution, not just coordinates) and
  would revive rung 2 in a per-expert-table form.
- Cost: free — a histogram pass over the rung-0 data.

### Rung 3 — the scale stream, reframed as a STRUCTURE probe

Two corrections to v0. First, the GB payoff is probably ~17 MB (see
rung -1), so this is not a size rung unless the header says
group-32. Second, v0 said "no prediction registered" — the ledger
does predict the ordering:

P6 (RESULTS.md:11206): experts raw 5.60 v entropy 4.31; **router
raw 6.00 v entropy 2.84 — "the LEAST meter-compressible tensors are
the MOST entropy-codable."**

- Prediction, a cross-model replication of a booked regularity:
  rank per-param rANS gain as **ue8m0 scales > router/gate and
  `index_*`/hash tensors > shared expert > routed-expert fp4
  codes**. Exponent streams are the extreme concentrated-code case:
  adjacent blocks of a Gaussian tensor share scales, so predict
  **H(scale) < 2.5 bits/symbol, gain > 5.5 bits/symbol**.
- The interesting readout is spatial: is the exponent field smooth
  across (row-block, col-block) and across experts and layers? An
  order-1 conditional entropy costs one extra histogram.

### Rung A — port K3-D2: one expert, exact, hash-locked

The strongest form of the per-expert claim type, already built:
`k3_expert_demo.py:99-151` has `det_gemv` (exact int64 GEMV on
shipped codes with power-of-two shift scales and an overflow assert)
and `chain` (w1/w3 GEMV → shift requant → sha-pinned SiLU table →
gate*up → w2 GEMV → sha256 of the int64 trace). **VERDICT K3-D2
(RESULTS.md:11513): hashes identical on cpu, mps and cuda — a
frontier model's routed expert running exactly, same integers, any
backend, directly on the vendor's shipped format.**

Port delta: projection naming (`gate/up/down_proj`), shapes
4096↔2048, and a 2-D block scale broadcast instead of
group-32-along-last-dim — **the donor for which already exists** at
`capacity_meter.py:102-107` (the V3-proven 128x128
`repeat_interleave` dequant). Both halves exist; the new code is the
join.

- Prediction: sha256 identical on cpu and mps (cuda optional, 3080
  is Artin's and GO-gated).
- Kill: if the expert scale is not power-of-two (e4m3 rather than
  ue8m0), exact integer accumulation needs a different carrier and
  the rung weakens to fp-with-tolerance. Rung -1 decides.
- Fence: tables travel as bytes, sha-verified, never regenerated
  per device (P3 doctrine).
- Cost: ~35 MB byte-range fetch, an afternoon.

### Rung 13 — subspace overlap vs a random-matrix null (speculative)

The complement N3 never ran: principal angles between expert
row-spaces are permutation-invariant and ask whether experts share
a **subspace** despite zero coordinate correlation. One layer, 64
experts, SVD the `gate_proj`s, compare the principal-angle spectrum
against a matched random-Gaussian null.

- Indistinguishable from the null → N3 confirmed at the geometry
  level and the whole mu+delta family closes for good. Measurably
  more overlap → a shared basis exists and rung 2 revives as a
  change-of-basis (still exactly invertible if kept integer).
- The *rung* is speculative; the *criticism of N3's instrument* is
  grounded in the standing gauge law. The null must be generated,
  not assumed — 2048-dim subspaces in R^4096 have non-trivial
  generic overlap.

## Instrument hazards (all verified in-tree)

1. **The verify gate.** `scratch/pack_rans.py:84` reads
   `rans_bytes(c, verify=(tot_n < 2e9))` — round-trip verification
   **silently switches off** past 2 billion cumulative symbols. At
   277B expert params that threshold is crossed inside the first
   shard, so reusing that driver would leave my "asserted per
   tensor" bar unmet on ~99% of the stream. Use
   `pack.py:108 rans_size(..., verify=True)` and pin it.
   (The booked P6-v2 verdict discloses this honestly — "first 2B
   Qwen symbols verified, coder identity thereafter" — so it is a
   fence, not an error; but a V4 full-verification claim must not
   be read as equivalent to it.)
2. **Code per TENSOR, never per shard.** `rans_size` materializes
   symbols as int32: one 3.5GB fp4 shard is ~7e9 symbols = **28 GB
   as int32**, which kills the Mac. One expert tensor (8.4M
   symbols) is 34 MB. This is the C7 OOM lesson — its amendment
   records the first run being killed for cloning expert params on
   top of the model. Per-tensor loop, `blackhole_b0.py:73-93`
   shape, with delete-as-you-go on disk.
3. **Derive the function-space bar, do not guess it.** VERDICT
   B0-B2 was "FALSIFIED BY MIS-SPECIFICATION": a <=2% bar ignored
   that a sigma/2 grid's *intrinsic* relative output error is
   sqrt(1/48) ≈ 14.4%. Compute fp4's intrinsic error at V4's block
   granularity before writing any bar, and state that
   function-space numbers are comparative between arms, never a
   quality readout.
4. **Do not add V4 to the expert-size ladder** in
   `llmopt/quantize/meter.py`. A V4 expert is 25.2M params and
   lands temptingly between Qwen3 (5M) and V3 (45M) — but fp4 caps
   span even harder than K3's MXFP4, and V3's own ladder point came
   through an fp8 dequant approximation. Two different format
   images cannot be rank-ordered.
5. **Pin the coder version.** `pack.py:126` uses
   `Categorical(probs, perfect=False)`, which approximates the
   probability model internally — a stored stream plus stored
   counts is only decodable by a compatible `constriction` version.
   If "lossless artifact" is the claim, record the version beside
   the frequency tables, the way the lab already ships sha-pinned
   tables as bytes.
6. **Extrapolation is not measurement.** Any whole-model number is
   labelled an estimate with its sample named.
7. **Validate the fp4/ue8m0 dequant** against the model repo's own
   `inference/` reference before booking any number.

## Reuse inventory (verified by reading the code)

| Tool | Path:line | Interface | Change for V4 |
|---|---|---|---|
| Header + byte-range fetch | `scratch/k3_expert_demo.py:44-79` | `_get(url, lo, hi)`, `fetch_expert()` → named uint8 arrays, sha-asserted | repo/shard constants only |
| fp4 nibble unpack + LUT | `k3_expert_demo.py:33,82-96` | `dequant(packed, scale)` | scale broadcast: group-32 → 2-D block |
| 2-D block-scale broadcast | `scratch/capacity_meter.py:102-107` | `repeat_interleave` 128x128 dequant | none — this is the donor |
| Exact integer expert forward | `k3_expert_demo.py:99-151` | `det_gemv`, `chain` → sha256 | naming + shapes; re-derive the overflow assert |
| rANS, verified | `llmopt/quantize/pack.py:108` | `rans_size(codes, verify=True)` → (bytes incl. table, entropy bits/symbol) | none — call this, not `pack_rans.py` |
| Streaming shard skeleton | `scratch/blackhole_b0.py:61-102` | START/END env, per-tensor loop, jsonl atlas, `os.remove` | `group_of()` needs V4 tensor names |
| Function-space probe | `blackhole_b0.py:56-57` | 64 Gaussian probes, relative output error | bar must be derived (hazard 3) |
| Capacity meter | `llmopt/quantize/meter.py` | `meter(w)`, `meter_group(...)` | **not on fp4 experts**; legitimate on fp8 non-expert tensors |

**Dropped from v0's list:** `llmopt/quantize/allocator.py`. Verified
— `allocate_bits` requires per-layer `delta_kl` per bit-width as
input, which can only be measured by running the model. Under the
no-capability fence it has no input it can be fed, so listing it as
available would be misleading.

## Rung order (bandwidth-optimal)

`-1` header read (free; gates rung 3's value and rung A's
feasibility) → `0` + `0b` + `2c` in **one streaming pass** over the
stratified sample (entropy, sign/magnitude split, pooled-table KL —
all histograms, one read) → `1` lossless rANS with `verify=True`
per tensor → `A` K3-D2 port on one expert (separate 35 MB fetch) →
`3` scale stream, free from the rung-0 pass → `2` naive mu+delta as
a confirmed null → `2b` permutation-aligned, only if an N3
amendment is wanted → `13` subspace overlap, only if 2b is
ambiguous.

## Provenance

Rungs -1, 0b, 2b, 2c, 13, the rung-0 and rung-3 priors, all seven
hazards, and the reuse corrections came from a read-only Opus 5
reviewer scan (2026-08-02). Every load-bearing citation was
verified in-tree before adoption: the verify gate at
`pack_rans.py:84`, N3's pre-reg and verdict at RESULTS.md:11095 and
:11107, the K3-D1 entropy anchor at :11450, the P6 group ordering
at :11206, K3-D2's cross-backend hashes at :11513, the byte-range
fetch and `det_gemv` in `k3_expert_demo.py`, and the allocator's
input requirement. The scan also corrected v0's citation for rung 2
(I had cited the 208k micro-MoE's output agreement instead of N3's
production-scale weight-space null) and v0's footprint arithmetic.

## Status

v1. Nothing runs until the rungs are pre-registered in RESULTS.
