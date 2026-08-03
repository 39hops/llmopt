# Spec: DeepSeek-V4-Flash — exact re-coding and the streaming question (2026-08-02, v3)

v3 exists because a read-only reviewer scan found **four real errors in
v2**, each verified in-tree before adoption. They are listed first,
because they change what this program is allowed to claim:

1. **The merged lattice is weight-exact but BYTE-LOSSY**, so it cannot
   meet v2's own "bit-identical round-trip" fence. The claim type had to
   be renamed, not weakened by degrees.
2. **The "shared component was factored out at training time" reading is
   over-stated**, and the house already booked a falsification-tested
   mechanism that explains the same observation without it.
3. **"Double the bit width for the part every token uses" is wrong** —
   my own receipt says *all* non-expert tensors are fp8. The asymmetry is
   routed-vs-not-routed.
4. **"One global table makes the artifact seekable" misattributes the
   cause**, and the in-tree coder cannot express either property.

## What is measured (booked; the program's actual assets)

| fact | value | entry |
|---|---|---|
| expert format | group-32 MXFP4, byte-identical to K3 | RECEIPT V4-RUNG-MINUS-1 |
| expert code entropy | 3.8646 bits/param (stored 4) | VERDICT V4-RUNG-0/1 |
| scale stream | 0.964 bits (stored 8); 5.9% of bytes, **62% of headroom** | same |
| sign bit | exactly 1.00000 bits, incompressible | same |
| pooled table (nibble alphabet) | mean KL 0.00075 bits/param | same |
| merged code+scale lattice | 3.5903 v 3.8975 separate, **one expert** | OBSERVATION V4-MERGED-LATTICE |
| exact integer expert forward | one hash on cpu / mps / cuda | VERDICT V4-RUNG-A + rider |
| experts share weight structure? | no — coordinate, permutation, and router-neighbour | N3; V4-RUNG-2B; V4-RUNG-R + 2B-ROUTER |
| router keys | all 32,640 pairs positive; +0.385 shared direction — but that direction is a **level**: deleting it leaves 97.4-98.0% of top-6 routing | V4-RUNG-R + 2B-ROUTER; **V4-RUNG-D** |
| byte-lossless expert | 13.37 MB → **12.26 MB (8.3%)**, decoding at **38 MB/s** | V4-RUNG-D + S0 |
| hash routing table | `tid2eid` [129280, 6] confirmed, layer 0 | same |
| non-expert tensors | **F8_E4M3 with F8_E8M0 scales** | RECEIPT V4-RUNG-MINUS-1 |

## Claim types admitted (corrected)

- **WEIGHT-EXACT** — the dequantised tensor is bit-identical after a
  round-trip. The *bytes* are not, and the artifact cannot carry the
  vendor's file sha. This is the honest name for what the merged lattice
  achieves, because the merge is many-to-one **by construction**: `0x0`
  and `0x8` both mean zero (12.70% of codes), and `2·2¹` aliases `4·2⁰`.
- **BYTE-LOSSLESS** — bit-identical shard bytes. Achievable only by
  coding the nibble and scale streams *separately* (rung 1 did this).
- **PER-EXPERT** function-space or exact-forward.

No capability claim about V4 is possible on this machine. **That
exemption is scoped to the V4 rungs and does NOT extend to Q1**, which
gates a runnable model at `N_EVAL = 120` (`scripts/eval_pruned_moe.py:26`)
and therefore inherits the resolution law's sigma ≈ 5 — deltas under ~5
solves are unresolved at n=1.

## The arithmetic (corrected, and the conclusion is stronger)

277e9 × 3.5903 / 8 = **124.3 GB** against ~30 GB usable: **4.1× short.**

But "provably" was wrong. 3.5903 is an **order-0 empirical rate on one
expert**, not a lower bound — any model with memory can beat it, and the
ledger already contains proof that it can, since the merge itself found
0.307 bits/param that per-stream order-0 coding could not see. The
correct wording is *best measured rate*. The conclusion survives anyway:
even another 2× leaves 62 GB, still 2× short.

**The stronger argument v2 failed to make**: the ~27B **non-routed**
params are fp8, i.e. ~27 GB — which alone nearly fills the machine before
a single routed expert is considered. The dense every-token path is the
binding constraint, not the experts.

**Owed, free**: the whole-artifact arithmetic does not close. 43 expert
shards × 3.42 GB = 147.1 GB leaves 5 shards for ~27B non-expert params,
i.e. ~5.4 GB/shard, contradicting the "~3.5 GB shard" figure elsewhere.
One header read of a non-expert shard reconciles it.

## Streaming, corrected

**v3 priced this wrong and S0 caught it.** 11.29 MB is 25.17M params at
3.5903 bits — the **merged-lattice** rate, which is weight-exact,
byte-LOSSY and n=1. A stream has to be executable, and the executable
form is packed fp4 at **13.37 MB**. So per token, experts only:
6 × 43 × 13.37 MB = **3.45 GB**, not 2.9 GB, and every bound below
tightens accordingly (the ~1.7 tok/s ceiling becomes **~1.45 tok/s**).
Three omissions in v2, the first load-bearing:

1. **The dense path is missing.** Every token also reads attention, the
   shared expert, norms and routers for all 43 layers. Resident, that
   ~27 GB consumes the entire budget and leaves no room for an expert
   cache — which undercuts the cache motivation. Non-resident, it adds
   ~20 GB/token, ~7× the expert traffic.
2. **The reads are 43 dependent batches of 6, not one 2.9 GB stream** —
   layer L's routing is unknown until L−1 runs. Floor: 43 × (68 MB /
   5 GB/s) = 585 ms **plus 43 round-trip latencies plus decode plus
   compute**. So "**at most ~1.7 tok/s**, ignoring decode and attention",
   not "roughly 1-2".
3. **Prefill has a computable crossover.** The full expert store is
   43 × 256 × 11.29 MB = 124.3 GB, and balanced routing (which DeepSeek's
   aux-loss-free bias actively enforces) reaches near-full coverage fast.
   Per-token cost equals a full-model read at **T ≈ 43 tokens**; beyond
   that, cost is 124.3 GB / T. That is the amortisation law and it is a
   desk calculation.

**Seekability, corrected.** It comes from *one stream per tensor plus a
stored offset index* — both format decisions not yet made — not from the
global table. rANS is stack-structured, so within a stream you cannot
skip; seek granularity is whatever unit you encode. The global table
buys **decode-path simplicity** (no per-expert table load/parse, one
decoder configuration, a table small enough for constant memory — which
is what makes a SIMD or GPU decoder practical), not seeking and not
space: per-tensor tables are 72 bytes × ~33,024 tensors ≈ **2.4 MB
total**.

## Why gravity is closed — and the correct mechanism

Closed three ways: GRAV-0T/REV (post-hoc destructive), N3 (sigma ratio
0.995), V4-RUNG-2B and 2B-ROUTER (nothing up to the optimal permutation,
nothing between router-nearest pairs, wrong sign).

**v2's causal story is withdrawn.** UMOE-2 VERDICT already booked the
mechanism with causal arms: expert correlation is **0.0016 at
initialisation**, training raises it only to ~0.008 in every regime, and
the rider states it outright — *"'experts decorrelated' is not an
achievement of MoE training, it is the absence of a correlating force."*
Under that law V4's nulls need no factoring story: independently
parameterised experts start orthogonal and nothing pushes them together.
v2 laid an additional, unmeasured causal claim on top of a law the house
had already tested twice.

**What survives, and it is supported by the same verdict**: UMOE-2 also
found that *"the ROUTING-side structure is where training actually
writes"* (MI 205-374× shuffle). So the router result — every key sharing
a +0.385 direction — is a *frontier-scale instance of a booked house
law*, which is a better claim than the one v2 made.

**Corrections to the same paragraph**: the shared expert is stored like
*every* non-routed tensor (fp8), so there is no deliberate singling-out;
and the DIET-COND-SEED connection is a **resonance** between a 304B
artifact and a 208k-param integer battery, not "the same statement".

## Rungs

**Done**: `-1` header · `0` entropy · `0b` sign/magnitude · `1` rANS ·
`2c` pooled table · `3` scale stream · `A` exact forward ×3 backends ·
`2b` permutation · `R` router · `2b-router` neighbours.

### M1 — merged lattice, re-scoped (next)

Register as **weight-exact**, not byte-lossless. Four additions the scan
earned:

- **Sparse alphabet, first contact.** The merged value set is
  `{0,1,2,3,4,6,8,12} × {1,2,4,8}`; 5, 7, 9, 10… are unreachable, so
  `np.bincount` hands `Categorical(probs, perfect=False)` a vector with
  exact zeros. No prior rung did this — rung 0 coded dense alphabets and
  rung 2b used `np.unique`, never a coder. **Predict**: `rans_size`
  either raises or silently floors, and the coded rate lands measurably
  above 3.5903 rather than the 0.003% overshoot rung 1 saw. Fix: map to
  the dense observed alphabet (~24-97 entries) and store it.
- **The rung-A hash is a free end-to-end verifier**, stronger than a
  symbol round-trip: decode the merged stream, re-run
  `scratch/v4flash_rungA.py`, and the trace **must** reproduce
  `a68256ce…`. Any mismatch is a merge bug, not rounding.
- **The pooled-table result does not transfer.** KL 0.00075 was measured
  on the **nibble** alphabet (`v4flash_rung0.py:141`); the merged
  alphabet is defined per-tensor relative to that tensor's own `emin`.
  **Predict** merged KL an order of magnitude higher. The format
  survives either way (2.4 MB of tables); the *prose* does not.
- **All 9 cached experts, not one** — v2's 124 GB headline extrapolates
  an n=1 number, violating its own hazard 6. Zero download.

### S0 — rANS decode throughput, pre-registered to FAIL — **RUN, fired**

**Measured 2026-08-03: 38.2 MB/s, 131× short. Conclusion adopted — the
entropy-coded form is an archive, the bit-packed fp4 form is the runtime.
C5 stays dormant. The 15.6% below was also wrong (same 11.29 MB error):
byte-lossless saves 8.3%.** Original registration kept for the record:

House doctrine already answers this in the negative: P6-v2's fences say
*"decode-side rANS throughput not benched (storage format; **the runtime
twin remains crystal5/int8**)"* and cite *"the C2b lesson that the
bit-packed form is directly executable"*. And the payoff is small: a raw
expert is 13.37 MB against 11.29 MB coded — **15.6%** — bought with a
full entropy decode. **Predict** single-threaded decode at 20-300 MB/s,
1-2 orders short of 5 GB/s. Run it anyway (minutes) because it converts
an inference into a number, but register the expected conclusion: **the
entropy-coded form is an archive format; the bit-packed fp4 form is the
streaming format.**

### R-d — is the shared router direction routing-INERT? — **RUN, fired**

**Measured 2026-08-03: 97.4-98.0% set agreement at three layers and
three input scales; deflation removes ~99% of the logit mean and <0.1%
of the across-expert spread. The direction is a LEVEL. Headline
qualified in FINDINGS and in the table above.** Original registration:

Hazard 7 says a constant bias offset is a top-k no-op. **The same
argument was never applied to the shared key direction.** Keys align at
+0.385 ± 0.045 — equal to within ±12% — so the shared component
contributes a near-identical additive term to every score and is
*mostly* inert by the same logic. Cell: draw x ~ N(0, I), compare top-6
selections from `Wx` and `(W − uuᵀW)x`, report set agreement. **Predict
≥90% agreement.** If it fires, "the confluence lives in the router" is a
statement about key *geometry*, not a routing *mechanism*, and the
headline needs qualification. Fence: isotropic x is a null model, so this
licenses "consistent with inert", never "proven inert".

### H1 — exact activation frequencies from `tid2eid` (free)

Layers 0-2 route by an exact public token→expert table, so expert load
and the **exact pairwise co-activation matrix** follow from any corpus's
unigram distribution with zero inference — the only place in this model
where the co-activation graph is known rather than estimated. Check
whether layers 1-2 carry their own tables (only layer 0 is confirmed).
Fence: hash routing is designed, not learned; it does not extrapolate to
the other 40 layers.

### W1 — shared expert, format-matched (corrected)

v2 compared the shared expert (fp8) against routed experts (fp4) —
exactly the format-image confound of hazard 4. The legal comparison is
shared expert vs **the other fp8 non-expert tensors**. Also free: one
header read of the shared-expert scale *shape* settles whether it really
uses [128,128] blocks, which is currently taken from `config.json` — the
same source that misdescribed the routed path.

### Q1 — the falsifiable arm

Qwen3-30B-A3B-4bit (16 GB) and OLMoE-1B-7B (13 GB) are cached and
MLX-runnable; `scripts/eval_pruned_moe.py` + `llmopt/moe/prune.py`
(`mask_router`, with the `len(kept) >= top_k` guard) +
`scripts/moe_router_stats.py` implement routing-masked pruning with the
disjoint-eval fence. `llmopt/moe/offload.py` `ExpertCache` (LRU +
`warm()`) is the cache vehicle H1 would feed. **Sigma ≈ 5 applies.**
Neither vehicle has a shared expert (to verify), so Q1 tests
compressibility and prunability generically — it **cannot** gate the
structural claim. Ledger hygiene owed: BOARD.md:102 cites a
"61%-keep / 50% count-quantile / ~28% cliff" result to "RESULTS MoE
pruning" that the scan could not locate in RESULTS.md; if Q1's baseline
rests on it, it may be un-booked.

### 13 — subspace overlap, with the instrument fixed

Expert `w1` row space is 2048-dim in R^4096 — exactly half rank, where
principal-angle spectra concentrate and the null and any signal overlap
almost completely. **Register the fix before running**: measure on
**top-k energy subspaces** (k = 32, 64, 128) where the Jacobi/MANOVA null
is sharp, report the full spectrum, and generate the null empirically.

## Speculative (labelled; each with its cheapest killer)

- **C1 conditional scale coding.** The E8M0 field is span-3 over a
  2048×128 grid; measure H(s | left) and H(s | left, up) on cached
  blobs. Competes with M1 rather than stacking — the merge deletes the
  scale stream. Killer: no drop below ~0.5 bits.
- **C2 do the zeros cluster?** 12.70% are exactly zero; order-0 already
  prices i.i.d. zeros optimally, so run-length gains need clustering.
  Beyond bits: identically-zero nibble columns would be a **format-free
  statement about an expert's effective input rank**. Killer: run
  lengths matching a Bernoulli(0.127) null.
- **C3 group-structured coding.** The exponent may be *derivable* from
  its group's magnitude profile rather than stored. Measure
  H(exponent | group profile). Killer: near its 2-bit marginal.
- **C4 parametric tables.** Fit a generalised Gaussian through the
  dyadic quantiser; if it matches within sampling noise, a table is two
  numbers — and, unlike the empirical pooled table, a parametric one
  survives per-tensor `emin` rebasing (the A5 problem).
- **C5 interleaved rANS** only if S0 surprises; otherwise accept the
  archive/executable split.
- **C6 hash layers as an exact co-activation graph**, extending H1 —
  charter-clean combinatorics, exact rather than estimated. Speculative
  prediction: load skew max/mean between 1.5 and 4, dominated by the
  head of the token distribution.
- **C7 Grassmannian framing** for rung 13 — the formal reason the k =
  2048 instrument is powerless and k = 32-128 is not.

## Hazards (all verified in-tree)

1. `scratch/pack_rans.py:84` is `verify=(tot_n < 2e9)` — round-trip
   checking switches OFF past 2B symbols. Use
   `llmopt/quantize/pack.py:108` and pin `verify=True`.
2. **`rans_size` cannot take a global table and rebases per array**
   (`sym - sym.min()`), and returns a size, not a stream — there is no
   artifact writer in-tree. MENTION for Fable: a `probs=` parameter and
   a stream-returning sibling are needed before any format work.
3. Code per **tensor**, never per shard (a 3.5 GB shard is ~28 GB as
   int32 — the C7 OOM lesson).
4. Derive function-space bars from the grid's arithmetic (B0-B2 was
   falsified by mis-specification).
5. Do not add V4 to `meter.py`'s expert-size ladder (format images are
   not rank-orderable).
6. Pin the coder version — `Categorical(perfect=False)`; constriction
   0.5.0.
7. Extrapolation is not measurement; label estimates with their sample.
8. Router bias: a constant is a top-k no-op; only deviations carry
   signal.
9. Tables travel as bytes (`scratch/v4flash_ref/`, sha-pinned).
10. `df` inside WSL reports the sparse vhdx ceiling, not host free space.

## Reuse

`scratch/v4flash_rungA.py` is the de-facto V4 library — `header()`,
`cached()` (sha-pinned byte-range), `decode()` (with a vendor-semantics
assertion), `det_gemv`, `rdiv`; `v4flash_rung2b.py` and
`v4flash_router.py` already import it and M1/S0/W1/R-d should too. The
merged lattice is already built at `v4flash_rung2b.py load_expert`.
Streaming skeleton: `scratch/blackhole_b0.py:61-102`. fp8 dequant donor
for W1: `scratch/capacity_meter.py:102-107` — **but** it assumes a float
`weight_scale_inv` while V4's non-expert scales are `F8_E8M0`, so the
port needs an exponent decode, not a multiply. Not usable:
`llmopt/quantize/allocator.py` (needs delta-KL from a model run).

## Status

v3, amended 2026-08-03: **R-d and S0 are RUN** (VERDICT V4-RUNG-D + S0)
and both fired as registered — the router direction is a level, the
coded form is an archive. Remaining order: **M1** (re-scoped, zero
download) → **W1** format-matched → **Q1** with sigma ≈ 5 → **13** with
the fixed instrument. Nothing runs until its rung is pre-registered in
RESULTS.

**Standing note earned by R-d**: the mean projection of the router keys
onto the shared direction is stable to 0.0008 across four defensible
definitions of that direction, while the MINIMUM spans 0.2397-0.2643.
Book extrema with their definition attached, or book the mean. This is
the third wrong-extreme finding on this branch.
