# Spec: BLACK HOLE MoEs (2026-07-30; drafted 07-29 close)

Artin's pool-the-winners ask + the expert-scale law (his
prediction, confirmed at n=3: M monotone in expert fineness,
K2 experts AT the sigma-law boundary). The program: treat a
real mid-size MoE as a field site — compress it ourselves,
part by part, on a laptop, with the house's zero-calibration
toolkit. Winners pooled: sigma-law pack + entropy bound
(C0/C1), the capacity DIAL (C7), bit-packed kernels (C2b),
determinism (C4), streaming discipline (C7-v2 OOM lesson).
Losers pooled AS TOOLS: C6/P2a taught exactly which tensors
NOT to sigma-pack (M>2 -> max-anchored; outliers are
load-bearing — never clip); C5 taught pack-EMA-parents and
that fragility (k_c) is an orthogonal axis.

TARGET: Qwen/Qwen3-30B-A3B (30B MoE, 128 experts/layer,
bf16 — clean histograms; gpt-oss is MXFP4-confounded, K3 is
8-bit-confounded). 16 shards, streamed: download -> process
-> DELETE (52GB free disk; the C7 OOM lesson applied to disk).
No full-model inference anywhere (the 24B-class reality).

## Rungs
- B0 CAPACITY ATLAS (desk, streaming): meter every 2-D tensor
  of all 16 shards -> M and kurt by (layer, group: expert/
  attn/shared/router). Deliverable: the first full capacity
  MAP of a production MoE. Predictions: (1) experts read
  BELOW attn/shared everywhere (in-model replication of C7's
  ordering); (2) expert M at 128-fineness lands between OLMoE
  (2.85) and V3 (2.33) IF fineness (not total scale) drives
  the ladder — a discriminating point Artin's confound needs;
  (3) router/gate tensors read WORST (they serve every
  token).
- B1 THE STREAMING PACK (same pass): dial-routed, zero
  calibration — sigma[row] codes where M<2, per-row
  max-anchored grid codes otherwise; real packed bytes
  written per shard (.npz parts); entropy-coded size v
  Gaussian capacity reported per group. Headline metric: a
  30B MoE packed on a laptop, wall-clock minutes, zero data.
- B2 FUNCTION-SPACE SPOT CHECK (inline, no model): per-tensor
  relative output error ||x(Wq-W)|| / ||xW|| on 64 Gaussian
  probes (the house law — score by function; here the layer's
  function). Bar: expert tensors under 2% at their assigned
  grid; report the worst-10 table honestly.
- B3 DEPTH CURVE ON K2 (desk, 2 more shards, then DELETE
  all): M by layer early/mid/late — is the black-hole state
  uniform in depth or does it deepen? (Shard 37 = layer 36
  read 2.01.)
- B4 ENTANGLED EXPERTS (OLMoE, the one MoE we can run):
  co-routing MI matrix over the C6 prompt battery + README
  stream; if high-MI pairs exist, merge the top pair
  (task-vector average) and read DeltaKL v routing to both.
  The ER-bridge cell made real. [Artin's riff, banked
  07-29.]
- B5 PAPER FOLD-IN: extends P7 with the scaling-law chapter
  ("Black-Hole Experts: router focusing drives expert
  weights toward maximum entropy") — the atlas (B0) is the
  figure, the dial is the mechanism, K2-at-the-boundary is
  the punchline, Artin's prediction is the origin story.
  External-collab cell stays flagged: DeltaKL at 1T needs
  hardware the house lacks.

## Fences
Desk-only for 30B/1T (no inference); fp8-dequant
approximation on K2; fineness v scale v recipe confounded
across model classes (B0's 128-expert point discriminates
fineness within the ladder); function-space spot checks are
NOT end-to-end quality (named in every booking); disk
discipline: one shard on disk at a time, K2 shards deleted
after B3. Pre-reg each rung in RESULTS before it fires.

## Order
B0+B1+B2 one streaming pass (script scratch/blackhole_b0.py,
~2h background) -> B3 (2 shards) -> B4 (OLMoE, ~30 min) ->
B5 drafting. P5 (pre-deploy card) and P3 (deterministic
decode) remain queued from the capacity-program spec; P4
born-packed on nightly GO.
