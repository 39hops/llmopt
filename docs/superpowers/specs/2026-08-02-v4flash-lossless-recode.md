# Spec: DeepSeek-V4-Flash — lossless re-coding and the streaming question (2026-08-02, v2)

v0 was a design pass; v1 folded in a reviewer scan; **v2 is written
after ten rungs actually ran**. Three things changed the shape of the
program:

1. **Lossless compression provably cannot make this model fit.** The
   measured floor is 3.59 bits/param, so the arithmetic ends the
   "compress it into RAM" framing. Streaming is the only honest path,
   and the compression work turns out to be what makes streaming
   *viable* rather than a competing idea.
2. **The gravity / mu+delta family is CLOSED** — on three independent
   tests, the last of which looked at exactly the pairs most likely to
   work.
3. **The router is free**, and reading it produced the most interesting
   structural result of the program.

## What is now measured (all booked, all in RESULTS)

| fact | value | entry |
|---|---|---|
| expert format | group-32 MXFP4, byte-identical to K3 | RECEIPT V4-RUNG-MINUS-1 |
| expert code entropy | 3.8646 bits/param (stored 4) | VERDICT V4-RUNG-0/1 |
| scale stream entropy | 0.964 bits (stored 8) | same |
| scales: share of bytes / of headroom | **5.9% / 62%** | same |
| sign bit | exactly 1.00000 bits, incompressible | same |
| one global table serves all experts | mean KL 0.00075 bits/param | same |
| merged code+scale lattice | 3.5903 v 3.8975 separate | OBSERVATION V4-MERGED-LATTICE |
| exact integer expert forward | one hash on cpu / mps / cuda | VERDICT V4-RUNG-A + rider |
| experts share weight structure? | no, even up to permutation | VERDICT V4-RUNG-2B |
| experts share it with routing neighbours? | no — wrong sign | VERDICT V4-RUNG-R + 2B-ROUTER |
| router key geometry | all 32,640 pairs positive; +0.385 shared direction | same |
| hash routing table | `tid2eid` [129280, 6] confirmed | same |

## The arithmetic that governs

277B expert params at the lossless floor of 3.59 bits is **~124 GB**
against ~30 GB of usable Mac memory. **Lossless is 4x short, as a
measurement, not an engineering shortfall.** Fitting would need ~0.9
bits/param, which is a lossy regime — and lossy is *unfalsifiable here*,
because a 304B model cannot be gated on this machine. That combination
closes the "shrink it to fit" path completely.

What is not closed: **top-6 of 256 means 2.3% of expert parameters touch
any given token.** The model does not need to be resident, only
reachable.

  per token: 6 experts x 43 layers x ~11.3 MB coded = **~2.9 GB read**

At ~5 GB/s that is ~0.6 s/token of pure bandwidth before latency — so
roughly 1-2 tok/s for interactive decode, and much better for
prefill, where many tokens share experts and the read amortises. That
is the honest performance envelope: **batch and prefill workloads are
plausible, interactive chat is not.**

**Why the compression rungs matter to this**: VERDICT V4-RUNG-0/1
measured mean KL(expert || pooled) = 0.00075 bits/param, so ONE global
16-symbol table serves every expert. That is precisely what makes an
entropy-coded artifact **seekable** — decode expert 47 of layer 22
without touching anything else, with no per-tensor table to load first.
Rung 2c was not a side quest; it is the format requirement.

## The hard fence, unchanged and still governing

**No capability claim is possible on this machine.** Only two claim
types are admitted anywhere in this spec:

- **LOSSLESS**, verified by bit-identical round-trip.
- **PER-EXPERT** function-space or exact-forward, at expert granularity.

Because no capability claim is possible, **no seed-count argument
applies** to any rung here.

## Why gravity is closed, and what replaced it

Three independent results, strongest last:

- **GRAV-0T/REV**: post-hoc gravity is destructive both ways; merge-free
  is a property *of the pulled basin*.
- **N3** (Qwen3-30B, 128 experts): sigma(delta)/sigma(W) = 0.995.
- **V4-RUNG-2B and 2B-ROUTER**: nothing shared up to the optimal
  permutation, and nothing shared even between the pairs closest in
  routing space (+0.026, the wrong sign against a -0.05 bar).

**But the confluence is real — it is just not in the routed weights.**
Every one of layer 22's 32,640 gate-key pairs is positively aligned, and
every key shares a common direction at +0.385. And the architecture
carries an explicit shared component: `n_shared_experts: 1`, stored at
**fp8 with [128,128] blocks while the routed experts get fp4 at
group-32** — double the bit width for the part every token uses.

So the reading that survives: **the shared part was factored out at
training time, into the router and the shared expert.** The routed
experts are the residual, which is why no post-hoc method finds anything
in them. This is the same statement as VERDICT DIET-COND-SEED's
"consensus pull works at birth", arrived at from a frontier model.

## Rungs

### Done

`-1` header read | `0` symbol entropy | `0b` sign/magnitude |
`1` lossless rANS | `2c` pooled table | `3` scale stream |
`A` exact forward, three backends | `2b` permutation alignment |
`R` router read | `2b-router` neighbour retest

### Next, in order

**M1 — confirm the merged lattice (owed, and now on the critical path).**
OBSERVATION V4-MERGED-LATTICE measured 3.5903 v 3.8975 bits/param on one
expert, entropy only. Owed: the rung-0 sample, rANS with `verify=True`
per tensor, round-trip asserted. It is the streaming format, so this
stops being a curiosity and becomes the artifact spec. If it holds,
147.2 GB -> ~124 GB, about 15.6% rather than 8.4%.

**S0 — rANS decode throughput (MB/s, Mac).** One number decides whether
streaming is bandwidth-bound or decode-bound, and therefore whether the
whole path is viable. If decode is slower than ~5 GB/s the format needs
a faster coder and the ladder stops here. No model run needed.

**H1 — exact activation frequencies for the hash-routed layers (free).**
`tid2eid` [129280, 6] is a public token -> expert table, so for layers
0-2 the expert-activation distribution over ANY corpus is a histogram
computation with zero inference. Gives a real (if partial) answer to
"how skewed is expert usage", which is the input a cache policy needs.
Fence: three layers, and hash routing may be deliberately flatter than
score routing — it cannot be extrapolated to the other 40.

**W1 — shared-expert asymmetry probe.** Measure the shared expert's
entropy and kurtosis against the routed ones. The mu/delta reading above
predicts it differs; the precision asymmetry (fp8 v fp4) says DeepSeek
treats it differently. ~25M params, already reachable.

**Q1 — the falsifiable arm, on a model we can actually run.**
Qwen3-30B-A3B-4bit (16 GB) and OLMoE-1B-7B (13 GB) are both cached
locally and MLX-runnable, and `scripts/eval_pruned_moe.py` +
`moe_router_stats.py` already implement routing-masked pruning with the
disjoint-eval fence ("otherwise the keep-sets would be fitted to the
eval set"). Everything this program claims about compression and pruning
should be **gated there**, where a capability claim is legal, and
transferred to V4 only as conjecture. This is where "how much can we cut"
gets a real number.

**Note against caching**: DeepSeek's aux-loss-free balancing actively
FLATTENS expert usage in training, which works against a hot-expert
cache. Whether inference on a narrow domain re-skews it is exactly what
Q1 and H1 measure.

**13 — subspace overlap vs a random-matrix null.** The last untested
gauge: principal angles between expert row-spaces are
permutation-invariant. 2b and 2b-router make a null likely; the
random-matrix baseline makes it decisive either way. Must be generated,
not assumed — 2048-dim subspaces in R^4096 have non-trivial generic
overlap.

### Not proposed, and why

Lossy compression to fit in RAM (unfalsifiable here); post-hoc gravity
or mu+delta in any form (closed, three ways); a capacity-meter reading on
the fp4 experts (measures the format, not the model — the K3 fence).

## Instrument hazards (all verified in-tree)

1. **The verify gate.** `scratch/pack_rans.py:84` is
   `verify=(tot_n < 2e9)` — round-trip checking switches OFF past 2B
   symbols, which at 277B params happens inside the first shard. Use
   `llmopt/quantize/pack.py:108 rans_size(..., verify=True)` and pin it.
2. **Code per TENSOR, never per shard.** One 3.5 GB shard is ~28 GB as
   int32. Per-tensor loop, delete as you go (the C7 OOM lesson).
3. **Derive function-space bars from the grid's arithmetic** — VERDICT
   B0-B2 was falsified by mis-specification when a 2% bar ignored a
   14.4% intrinsic error.
4. **Do not add V4 to the expert-size ladder** in `meter.py`: fp4 caps
   span harder than K3's MXFP4 and V3's ladder point came through an fp8
   approximation. Two format images cannot be rank-ordered.
5. **Pin the coder version** — `Categorical(perfect=False)` makes a
   stored stream version-coupled. constriction 0.5.0 recorded.
6. **Extrapolation is not measurement**: whole-model figures are
   estimates, labelled, with their sample named.
7. **Router bias**: a constant offset is a topk no-op, so ~99% of the
   bias magnitude is inert. Only deviations from its mean carry signal.
8. **Tables travel as bytes.** The SiLU table is a committed sha-pinned
   artifact (`scratch/v4flash_ref/`) because `wsl.sh` has no copy
   subcommand and regenerating on another libm would silently differ.
9. **WSL disk**: `df` inside WSL reports the sparse vhdx ceiling, not
   host free space, and the vhdx does not shrink back.

## Reuse inventory (verified by reading the code)

| Tool | Path | What it gives |
|---|---|---|
| byte-range fetch + fp4 unpack | `scratch/k3_expert_demo.py:44-96` | header read, sha-pinned blobs, MXFP4 decode |
| exact integer expert forward | `k3_expert_demo.py:99-151` | `det_gemv`, `chain` |
| V4 ports of both | `scratch/v4flash_{header,rungA}.py` | shipped, three backends |
| rANS, verified | `llmopt/quantize/pack.py:108` | entropy + round-trip in one call |
| 2-D block dequant donor | `scratch/capacity_meter.py:102-107` | for the fp8 non-expert path |
| streaming shard skeleton | `scratch/blackhole_b0.py:61-102` | download -> process -> delete |
| routing-masked pruning + eval | `scripts/eval_pruned_moe.py`, `moe_router_stats.py` | the Q1 harness, disjoint-eval fence built in |
| router read | `scratch/v4flash_router.py` | keys, bias, null comparison |

**Not usable here**: `llmopt/quantize/allocator.py` needs per-layer
delta-KL, which requires running the model.

## Status

v2. M1 and S0 are the next two; Q1 is where the program becomes
falsifiable. Nothing runs until its rung is pre-registered in RESULTS.
