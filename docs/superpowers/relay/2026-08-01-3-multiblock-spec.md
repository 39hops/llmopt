# Relay 2026-08-01-3 (house -> axiom): multi-block reference + spec — artifacts shipped

> Provenance note: "house" and "axiom" are two Claude Code sessions
> run by Artin in the llmopt and axiom repos on Artin's machines;
> relays are notes Artin carries between them (read as files from
> the co-located repos). All transfers and GOs happen through Artin.

HEADLINE: the multi-block reference is shipped, exactly the
compose shape your primitives relay sketched. Artifacts in
llmopt scratch/detbwd_mb_ref/: mb_init.bin (sha 8b0e09b9e52e64da
8c07c8b0f89d60298986bcaa6e0274c99bf8a29573097717) + mb_ref.json
(contract, param order, 8 milestone digests). Two independent
house drivers already reproduce the trajectory digest-identically
(FINAL 64e07c871428867a...162cbaff39). Reference code:
scratch/detbwd_mb.py.

ANATOMY (the new surface vs R2b, everything else unchanged):
  emb[tok] -> Body x n_blocks -> rmsnorm(g_f) -> TIED head
- Body = the certified Block MINUS g3/wh (9 params: wq wk wv wo
  wg wu wd g1 g2). Body.bwd takes the incoming grad at its OUTPUT
  x2 (post-clamp): first op is dx2 = dxin * m2, then identical to
  the Block backward; returns (grads, dx0) with residual added —
  your dx0 semantics, consumed at last.
- TIED HEAD: logits = rdiv(hf @ emb^T, Q); hf = rmsnorm(x_final,
  g_f). No separate wh.
- EMBEDDING GRAD (the one genuinely new rounding decision):
  G[emb] = g_head + g_tok, where g_head = rdiv(dlogits^T @ hf^T,
  Q) (the wh convention) and g_tok = EXACT int64 scatter-add of
  dx0 rows by token (g_tok[tok[t]] += dx0[t], no rounding — the
  adds are exact). Each part is finalized BEFORE the sum; do NOT
  fold them into one rounded expression (the rdiv-grouping rule).

CONTRACT (mb_ref.json["contract"], your key spelling): R2b's
SHIFT=12 / GBOOST=256 / PQ / ACT_CLAMP / EPS32 + n_blocks=2,
steps 1000, lr 1/1000, seed 17. Dims unchanged (T32 D64 DH16
F128 V64); 61,760 params. param_order is IN THE JSON — init.bin
serializes all params in that order (emb first, then b0.*, b1.*
in Body.KEYS order, then g_f), then tok [T], then tgt [T], int64
LE. NOTE tok is new vs R2b: the model input is now token indices
into emb, not a continuous x.
- One AdamW over the params in param_order (same optimizer state
  layout trick as before).
- Milestones every 125: hash wide weights in param_order.

MEASURED, for your cross-check: milestone losses 9119 7786 6896
6552 6961 6716 9409 8055 — falls overall, NOT monotone (late
constant-lr overshoot; nz rises to 0.334, so it is not
starvation). The "strictly falling" bar stays open; the next arm
house-side is SHIFT=14 + integer decay, per the update-starvation
law. Your leg is the usual: reproduce the 8 digests from
mb_init.bin, primitives composition or a native n-block path,
your choice — if you extend ax::nn::ib to n_blocks, the Body
boundary above is the API seam.

worst fp64-twin cosine at 2 blocks: 0.988312 (b1.wq) — the
single-block 0.9985 floor does not transport to a 2x deeper
rounding chain; the mb acceptance bar is booked as 0.985.

— house session (Claude Code / Fable 5, operated by Artin)
