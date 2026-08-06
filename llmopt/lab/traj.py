"""lab/traj — unified MoE router instrument (module 4; DESK TIER ONLY).

STATUS 2026-08-06: divergence enumeration ONLY. No unification code
exists yet — per spec 2026-08-06-lab-traj-session.md the authority
table below must be settled from the actual three-way diff BEFORE any
code is written, because this module fails the verbatim-adoption test
that made oracle/config/keepsets safe: it unifies three DIVERGENT
copies, which is design, not copy.

The three source copies (frozen evidence record; do not edit):

  A. scratch/moe_gt1.py       `instrument`  — free-routing recorder:
       RouterStats + first-touch + TRAJ per-token rows (certified
       instrument 4f3dc6c; D0 regression = 590,736 rows bit-identical)
  B. scratch/moe_gt1_arm2.py  `instrument`  — masked routing + closed-
       loop recall counters (every GT-1/2/3/5/6 masked verdict)
  C. scripts/moe_router_stats.py `instrument` — the minimal ancestor:
       RouterStats only (checkpoints/router_stats.json provenance)

SHARED CORE (identical in all three, adopt verbatim):
  - layer discovery: enumerate(model.model.layers), keep layer.mlp
    with hasattr gate AND top_k; layer_of = {id(block): li}
  - CLASS-level patch (obj(x) dispatches through type(obj).__call__;
    instance __call__ is never consulted)
  - routing math: softmax(..., precise=True) -> argpartition kth=-k
    top-k -> take_along_axis -> norm_topk_prob renorm ->
    switch_mlp(x, inds) -> (y * scores[..., None]).sum(axis=-2)
  - precise-softmax flag: precise=True in ALL THREE, both call sites
    (A/C unmasked, B masked AND recall probe). PINNED surface.

DIVERGENCE TABLE (surface -> what differs -> AUTHORITY):

  1. masking          A/C: softmax(self.gate(x)) directly.  B: raw
                      logits kept, additive -inf mask per block
                      (mx.array of 0/-inf), masked softmax; assert
                      len(kept) >= top_k at build.
                      AUTHORITY: B verbatim when keep is given;
                      keep=None must reduce to the A/C expression
                      byte-for-byte (softmax OF the raw logits with no
                      mask add — note B's unmasked-equivalent is
                      softmax(logits + 0-mask); the unified keep=None
                      path must NOT add a zero mask, it must take the
                      A expression, or D0 byte-identity is at risk).
  2. recall counters  B only: unmasked top-k of the SAME logits via
                      argpartition on RAW LOGITS (A tops the SOFTMAXED
                      gates — monotone-equivalent but a known fp16
                      near-tie rounding surface); hits/slots counters.
                      AUTHORITY: B verbatim, including the raw-logits
                      domain — recall numbers in booked verdicts came
                      from exactly that expression.
  3. stats recording  A/C: state["stats"].update(li, flat_i, flat_s);
                      pause semantics = state["stats"] is None.
                      B: no stats. AUTHORITY: A/C verbatim (identical
                      in both); pause semantics kept.
  4. first-touch      A only: per-layer dict expert -> POOLED routed-
                      token index; pooled `pos` counter NEVER reset
                      (certified arm-0 artifact surface).
                      AUTHORITY: A verbatim.
  5. pos vs tpos      A only: TWO counters — pooled `pos` (first-touch,
                      never reset) and per-prompt `tpos` (TRAJ row
                      position, reset by the DRIVER via
                      state["tpos"] = {} alongside state["tail_done"]
                      = {} at each prompt boundary + before the probe).
                      AUTHORITY: A verbatim, INCLUDING driver-side
                      reset responsibility — the unified API exposes a
                      begin_prompt(state, prompt_id) helper that does
                      exactly the certified resets (tpos, tail_done,
                      prompt id), nothing more.
  6. prompt_tail      A only (TRAJ v3 phase tagging): len(flat_i) > 1
                      -> "prefill" (and tail_done[li] = False);
                      first 1-token call per layer after prefill ->
                      "prompt_tail" (tail_done[li] = True); thereafter
                      "decode". Per-layer tail_done dict. (mlx_lm
                      prefill loops `while y.size > 1`, leaving the
                      last prompt token to the first 1-token step —
                      counting it as decode inflated cross-domain
                      Jaccards; reviewer bug 2026-08-04.)
                      AUTHORITY: A verbatim.
  7. write rounding   A only: scores round(s, 6); H round(float, 4);
                      entropy in nats over the FULL 128-way softmax
                      with +1e-12 epsilon, computed BEFORE top-k.
                      AUTHORITY: A verbatim.
  8. row schema       A only, insertion order IS the byte contract
                      (json.dumps preserves dict order): {"prompt",
                      "layer", "pos", "topk", "scores", "phase", "H"}
                      with "ok" appended by the DRIVER at write time.
                      AUTHORITY: A verbatim; field order pinned.
  9. restore          B only: saves original cls.__call__, restore()
                      in the driver's finally. A/C never restore.
                      AUTHORITY: NEW DESIGN (spec mandate, no source
                      authority): context manager; a raising gate must
                      not leave the class patched; on failure emit a
                      loud INSTRUMENT_NOT_RESTORED line (loud-failure
                      contract).
 10. n_experts        A: model.args.num_experts fallback 128.
                      B: block.gate.weight.shape[0]. C: caller passes
                      cfg value. Cosmetic (all equal 128 on the 30B);
                      AUTHORITY: B's per-block shape (derived from the
                      weights themselves, no fallback constant).

OPEN DESIGN QUESTIONS (flagged before unification, not decided here):

  Q1. traj + keep TOGETHER is a combination NO source copy ever ran:
      A trajs only free routing; B masks without traj. If both are
      requested, which distribution do H/scores come from — the
      masked softmax (what the model actually did) or the unmasked
      one (what the router wanted)? No certified artifact constrains
      it. Proposal: refuse the combination in v1 (loud ValueError)
      until a registered run needs it; acceptance never exercises it.
  Q2. Live-tier comparison target: spec says "TRAJ rows byte-identical
      + recall counters equal" vs the frozen arm2 path, but arm2
      writes no TRAJ rows. Reading: recall counters + per-problem
      rows compare vs frozen B on the masked arm; TRAJ byte-identity
      is the D0 regression (tier 2) plus a fresh free-routing row
      compare vs A. Confirm before the 30B block.

ACCEPTANCE (spec order): 1. this desk enumeration; 2. D0 590,736-row
bit-identity through the unified patch; 3. live gate run vs frozen B;
4. keepsets' always-on acceptance closes the loop over regenerated
rows.
"""
