"""EX6-B43 decomposition census (the NEXT INSTRUMENT of
OBSERVATION EX6-B43-CONTROL-DESK-0): the full local counterfactual
of the B43 mask, computed inside ONE outcome-blind native
execution per problem.

Because the B43 arm has no upstream intervention, native and
masked worlds enter block 43 with the identical hidden state h, so
at the (block-position 43, z1) call the driver computes, without
altering the native forward it returns:

  d_del     = -p_71 * E_71(h)          (deleted term)
  d_entrant =  p'_r * E_r(h)           (entrant term)
  d_renorm  =  sum_C (p'_i - p_i) E_i(h)   (shared-seven renorm)

with p = native softmax-top8-renormalized gates and p' = the
frozen masked-gate math (softmax over logits + -inf mask, then
argpartition top8, then renorm — verbatim ex6depth1 semantics).
Logged per problem: term norms, pairwise cosines, |delta| both as
the sum and directly as |masked_out - native_out| (residual =
agreement check; the single-expert switch_mlp calls group the
quantized matmuls differently from the batched top-8 call, so the
residual is finite-precision-scale, never zero); native p_71 raw
and renormalized mass; top-16
logit values (post-hoc margins); BOTH selectors on the SAME
logits — argsort top-8 and argpartition top-8 — plus the actual
masked top-8 and entrant (pays the tie-adjudication precondition
and the dose census of AMENDMENT EX6-B43-IDENTITY-0-SCOPE in one
run).

OUTCOME-BLIND: native generation only (the wrapped call always
returns the NATIVE output), max_tokens=2, no oracle call anywhere.

Receipts: logs/ex6b43decomp/{PRE}decomp.jsonl (refuse-if-exists;
SMOKE=1 -> smoke_ paths, 4 problems, seed 7001).

    .venv/bin/python scratch/ex6b43_decomp.py                 (Mac)
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

SMOKE = os.environ.get("SMOKE") == "1"
PRE = "smoke_" if SMOKE else ""
DIR = Path("logs/ex6b43decomp")
OUT = DIR / f"{PRE}decomp.jsonl"

import scratch.moe_gt1_arm2 as m  # noqa: E402
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

SEEDS = [7001] if SMOKE else [7001, 8002, 9003]
N_EVAL = 4 if SMOKE else 120
KEEPSET = "checkpoints/ex3_del_invp.json"
TARGET_BP = 43


def instrument(model, keep):
    import mlx.core as mx
    moe_layers = [
        (i, layer.mlp)
        for i, layer in enumerate(model.model.layers)
        if hasattr(layer.mlp, "gate") and hasattr(layer.mlp, "top_k")
    ]
    masks, zeros, keepsets, t1_count, blk_band = {}, {}, {}, {}, {}
    for bp, (li, block) in enumerate(moe_layers):
        kept = keep[li]
        n_exp = block.gate.weight.shape[0]
        # float32 mask/zeros added on BOTH paths — verbatim frozen
        # ex6depth1 gate math, so native and masked softmax run at
        # the same (upcast) precision the booked intervention used
        masks[id(block)] = mx.array(
            [0.0 if e in kept else float("-inf") for e in range(n_exp)])
        zeros[id(block)] = mx.array([0.0] * n_exp)
        keepsets[id(block)] = kept
        blk_band[id(block)] = bp
    cls = type(moe_layers[0][1])
    original = cls.__call__
    rows = []

    def phase_of(self, n_tokens):
        if n_tokens > 1:
            t1_count[id(self)] = 0
            return "prefill"
        c = t1_count.get(id(self), 0) + 1
        t1_count[id(self)] = c
        return f"z{c}" if c <= 3 else "decode"

    def expert_out(self, x, e):
        inds = mx.full(x.shape[:-1] + (1,), e, dtype=mx.uint32)
        return self.switch_mlp(x, inds)[..., 0, :].astype(mx.float32)

    def norm(v):
        return float(mx.sqrt(mx.sum(v * v)))

    def cos(a, b):
        na, nb = norm(a), norm(b)
        if na == 0.0 or nb == 0.0:
            return None
        return float(mx.sum(a * b)) / (na * nb)

    def wrapped(self, x):
        logits = self.gate(x)
        k = self.top_k
        n_tokens = 1
        for d in logits.shape[:-1]:
            n_tokens *= d
        phase = phase_of(self, n_tokens)
        gates = mx.softmax(logits + zeros[id(self)],
                           axis=-1, precise=True)
        inds = mx.argpartition(gates, kth=-k, axis=-1)[..., -k:]
        scores = mx.take_along_axis(gates, inds, axis=-1)
        if self.norm_topk_prob:
            scores = scores / mx.sum(scores, axis=-1, keepdims=True)
        y = self.switch_mlp(x, inds)
        native = (y * scores[..., None]).sum(axis=-2)
        if phase == "z1" and blk_band[id(self)] == TARGET_BP:
            kept = keepsets[id(self)]
            flat_logits = logits.reshape(-1)
            flat_gates = gates.reshape(-1)
            order = mx.argsort(flat_logits)[::-1][:16].tolist()
            ap_top8 = sorted(inds.reshape(-1).tolist())
            as_top8 = sorted(order[:8])
            # native renormalized coefficients over the argpartition
            # top-8 (the set the native forward actually used)
            nat_p = {e: float(v) for e, v in zip(
                inds.reshape(-1).tolist(),
                scores.reshape(-1).tolist())}
            # frozen masked-gate math on the SAME logits
            mgates = mx.softmax(logits + masks[id(self)],
                                axis=-1, precise=True)
            minds = mx.argpartition(mgates, kth=-k, axis=-1)[..., -k:]
            mscores = mx.take_along_axis(mgates, minds, axis=-1)
            if self.norm_topk_prob:
                mscores = mscores / mx.sum(mscores, axis=-1,
                                           keepdims=True)
            my = self.switch_mlp(x, minds)
            masked_out = (my * mscores[..., None]).sum(axis=-2)
            msk_p = {e: float(v) for e, v in zip(
                minds.reshape(-1).tolist(),
                mscores.reshape(-1).tolist())}
            outside = [e for e in nat_p if e not in kept]
            entrants = [e for e in msk_p if e not in nat_p]
            common = [e for e in nat_p if e in msk_p]
            d_del = mx.zeros_like(native).astype(mx.float32)
            for e in outside:
                d_del = d_del - nat_p[e] * expert_out(self, x, e)
            d_ent = mx.zeros_like(native).astype(mx.float32)
            for e in entrants:
                d_ent = d_ent + msk_p[e] * expert_out(self, x, e)
            d_ren = mx.zeros_like(native).astype(mx.float32)
            for e in common:
                d_ren = d_ren + (msk_p[e] - nat_p[e]) * \
                    expert_out(self, x, e)
            delta_direct = (masked_out - native).astype(mx.float32)
            d_sum = d_del + d_ent + d_ren
            rows.append({
                "bp": TARGET_BP,
                "argsort_top16": order,
                "logits_top16": [float(flat_logits[e]) for e in order],
                "raw_gate_top16": [float(flat_gates[e]) for e in order],
                "argsort_top8": as_top8,
                "argpartition_top8": ap_top8,
                "selectors_agree": as_top8 == ap_top8,
                "masked_top8": sorted(msk_p),
                "outside": outside,
                "entrants": entrants,
                "native_p": nat_p,
                "masked_p": msk_p,
                "norm_native_out": norm(native.astype(mx.float32)),
                "norm_d_del": norm(d_del),
                "norm_d_entrant": norm(d_ent),
                "norm_d_renorm": norm(d_ren),
                "norm_delta_direct": norm(delta_direct),
                "norm_residual_sum_vs_direct": norm(
                    d_sum - delta_direct),
                "cos_del_entrant": cos(d_del, d_ent),
                "cos_del_renorm": cos(d_del, d_ren),
                "cos_entrant_renorm": cos(d_ent, d_ren),
            })
        return native

    cls.__call__ = wrapped

    def restore():
        cls.__call__ = original

    return rows, restore


def main():
    if OUT.exists():
        raise SystemExit(f"REFUSING: {OUT} exists")
    DIR.mkdir(parents=True, exist_ok=True)
    from huggingface_hub import snapshot_download
    art_dir = snapshot_download(m.MODEL,
                                allow_patterns=["*.json"])
    START = start_provenance(
        ["scratch/ex6b43_decomp.py", "scratch/moe_gt1_arm2.py",
         KEEPSET],
        artifacts={"model": art_dir})
    from mlx_lm import generate, load

    from llmopt.mathgen.problems import make_dataset
    global mx
    import mlx.core as mx
    model, tok = load(m.MODEL)
    keep = {int(li): set(v) for li, v in
            json.loads(Path(KEEPSET).read_text()).items()}
    f = OUT.open("a")
    t0 = time.time()
    for seed in SEEDS:
        problems = make_dataset(N_EVAL, seed=seed)
        for i, p in enumerate(problems):
            msgs = [{"role": "system", "content": m.SYSTEM},
                    {"role": "user", "content": p.prompt}]
            text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False,
                enable_thinking=False)
            rows, restore = instrument(model, keep)
            try:
                generate(model, tok, prompt=text, max_tokens=2)
            finally:
                restore()
            f.write(json.dumps({"seed": seed, "idx": i,
                                "level": getattr(p, "level", None),
                                "z1": rows}) + "\n")
            f.flush()
        print(f"[dc] seed {seed} done {time.time()-t0:.0f}s",
              flush=True)
    f.write(json.dumps({"meta": {
        "note": "EX6-B43 decomposition census, outcome-blind "
                "(native output returned; no oracle; max_tokens=2)",
        "start": START,
        "completion_commit": completion_commit()}}) + "\n")
    f.close()
    print("[dc] done", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
