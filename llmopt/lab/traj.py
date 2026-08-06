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

DESIGN DECISIONS (both Artin-nodded 2026-08-06; were Q1/Q2):

  D-1. traj + keep TOGETHER is REFUSED (loud ValueError). No source
       copy ever ran the combination (A trajs only free routing; B
       masks without traj), so no certified artifact constrains which
       distribution H/scores would come from. Revisit only under a
       registered run that needs it.
  D-2. Live tier SPLITS (spec updated in the same commit): 3a masked
       arm — recall counters + per-problem rows byte-identical vs
       frozen B; 3b free arm — fresh TRAJ rows byte-identical vs
       frozen A, on top of tier 2's D0 regression.

ACCEPTANCE (spec order): 1. this desk enumeration; 2. D0 590,736-row
bit-identity through the unified patch; 3a/3b. live gate runs vs
frozen B (masked) and A (free); 4. keepsets' always-on acceptance
closes the loop over regenerated rows.

UNIFICATION (written after the table + D-1/D-2 nods, same session):
`patch_moe_router(model, traj=False, keep=None)` — a context manager.
keep=None -> A's wrapper body verbatim (traj gated on the state slot,
not the TRAJ env — the only deliberate delta, param-for-env); keep
given -> B's wrapper body verbatim (recall counters, raw-logits want,
additive -inf mask). traj+keep raises (D-1). `begin_prompt` performs
exactly the certified driver-side resets (surface 5). mlx is imported
lazily so the module imports on non-Metal machines.
"""
from __future__ import annotations


def begin_prompt(state, prompt_id):
    """The certified per-prompt driver resets (surface 5, A verbatim:
    resets fire only when TRAJ is recording), nothing more."""
    state["prompt"] = prompt_id
    if state.get("traj") is not None:
        state["tpos"] = {}
        state["tail_done"] = {}


class patch_moe_router:
    """Context manager unifying the three router instruments.

    keep=None: free-routing recorder (A) — RouterStats via
      state["stats"] (None pauses), first-touch, optional TRAJ rows.
    keep={layer: set(experts)}: masked routing + closed-loop recall
      (B) — state["hits"]/state["slots"].
    traj=True with keep is REFUSED (D-1).

    __exit__ always restores the class __call__; if the class is found
    patched by something other than this instrument at exit, it emits
    a loud INSTRUMENT_NOT_RESTORED line and restores anyway (the
    loud-failure contract: a raising gate must not leave the class
    patched).
    """

    def __init__(self, model, *, traj=False, keep=None):
        if traj and keep is not None:
            raise ValueError(
                "traj+keep REFUSED (D-1, 2026-08-06): no source copy "
                "ever ran the combination and no certified artifact "
                "constrains which distribution H/scores come from")
        self.model = model
        self.traj = traj
        self.keep = keep

    def __enter__(self):
        import mlx.core as mx

        model = self.model
        moe_layers = [
            (i, layer.mlp)
            for i, layer in enumerate(model.model.layers)
            if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
        ]
        cls = type(moe_layers[0][1])
        self._cls = cls
        self._original = cls.__call__
        n_exp = moe_layers[0][1].gate.weight.shape[0]

        if self.keep is None:
            state = {"stats": None, "first": {}, "pos": {}, "tpos": {},
                     "prompt": None,
                     "traj": [] if self.traj else None}
            layer_of = {id(block): li for li, block in moe_layers}

            def wrapped(self, x):
                gates = mx.softmax(self.gate(x), axis=-1, precise=True)
                k = self.top_k
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if self.norm_topk_prob:
                    scores = scores / mx.sum(scores, axis=-1, keepdims=True)
                if state["stats"] is not None:
                    li = layer_of[id(self)]
                    flat_i = inds.reshape(-1, k).tolist()
                    flat_s = scores.reshape(-1, k).tolist()
                    state["stats"].update(li, flat_i, flat_s)
                    first = state["first"].setdefault(li, {})
                    pos = state["pos"].get(li, 0)
                    for t, picks in enumerate(flat_i):
                        for e in picks:
                            if e not in first:
                                first[e] = pos + t
                    state["pos"][li] = pos + len(flat_i)
                    if state["traj"] is not None:
                        # `gates` is already the full softmax; entropy
                        # per token in nats. tpos has its OWN counter
                        # (reset per prompt) so the pooled pos/first-
                        # touch path stays byte-identical to the
                        # certified arm-0 artifact (surface 5).
                        p = gates.reshape(-1, gates.shape[-1])
                        ent = (-(p * mx.log(p + 1e-12)).sum(axis=-1)
                               ).tolist()
                        # phase is RECORDED, not inferred (surface 6):
                        # prefill hits the router with the whole prompt
                        # batch, decode with 1 token; mlx_lm leaves the
                        # LAST PROMPT TOKEN to the first 1-token step —
                        # labeled prompt_tail, never decode.
                        li_tail = state.setdefault("tail_done", {})
                        if len(flat_i) > 1:
                            phase = "prefill"
                            li_tail[li] = False
                        elif not li_tail.get(li, False):
                            phase = "prompt_tail"
                            li_tail[li] = True
                        else:
                            phase = "decode"
                        flat_sc = scores.reshape(-1, k).tolist()
                        tpos = state["tpos"].get(li, 0)
                        for t, picks in enumerate(flat_i):
                            state["traj"].append({
                                "prompt": state["prompt"], "layer": li,
                                "pos": tpos + t, "topk": picks,
                                "scores": [round(s, 6)
                                           for s in flat_sc[t]],
                                "phase": phase,
                                "H": round(float(ent[t]), 4)})
                        state["tpos"][li] = tpos + len(flat_i)
                y = self.switch_mlp(x, inds)
                return (y * scores[..., None]).sum(axis=-2)

        else:
            state = {"hits": 0, "slots": 0}
            masks, keepsets = {}, {}
            for li, block in moe_layers:
                kept = self.keep[li]
                assert len(kept) >= block.top_k
                masks[id(block)] = mx.array(
                    [0.0 if e in kept else float("-inf")
                     for e in range(n_exp)])
                keepsets[id(block)] = kept

            def wrapped(self, x):
                logits = self.gate(x)
                k = self.top_k
                # closed-loop recall: what the UNMASKED router wants
                # (surface 2: raw-logits domain, B verbatim)
                want = mx.argpartition(logits, kth=-k, axis=-1)[..., -k:]
                kept = keepsets[id(self)]
                for picks in want.reshape(-1, k).tolist():
                    state["slots"] += k
                    state["hits"] += sum(1 for e in picks if e in kept)
                # actual routing: masked
                gates = mx.softmax(logits + masks[id(self)], axis=-1,
                                   precise=True)
                inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
                scores = mx.take_along_axis(gates, inds, axis=-1)
                if self.norm_topk_prob:
                    scores = scores / mx.sum(scores, axis=-1,
                                             keepdims=True)
                y = self.switch_mlp(x, inds)
                return (y * scores[..., None]).sum(axis=-2)

        cls.__call__ = wrapped
        self._wrapped = wrapped
        state["n_experts"] = n_exp
        print(f"[lab.traj] instrumented {len(moe_layers)} MoE layers "
              f"({cls.__name__}), {n_exp} experts, "
              f"mode={'masked' if self.keep is not None else 'free'}"
              f"{'+traj' if self.traj else ''}", flush=True)
        self.state = state
        return state

    def __exit__(self, exc_type, exc, tb):
        if self._cls.__call__ is not self._wrapped:
            print("[lab.traj] INSTRUMENT_NOT_RESTORED — class __call__ "
                  "was re-patched by something else while this "
                  "instrument was live; restoring the saved original "
                  "anyway", flush=True)
        self._cls.__call__ = self._original
        return False

