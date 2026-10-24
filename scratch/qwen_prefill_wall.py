"""QWEN-PREFILL-WALL-0: the user-facing first-token latency of the released
arm C serving path (generation 5, chunk-256 GEMM prefill, final-position
head, 2 GiB allocator-cache limit, greedy), measured, decomposed and probed
on a small registered design space. The qualified kernels and the serving
defaults are not changed; every variant here is a diagnostic in this
process only.

Stages (STAGE env; each stage writes its own receipt and refuses to overwrite):
  baseline  the development prompts at 512, 1,024 and 2,048 tokens (and 4,096
            after a memory preflight), fixed token ids rendered through the
            pinned template with thinking off, order-balanced repeats
            (A B C C B A, twice); per run the end-to-end wall of the
            qualified prefill_chunked call, and an instrumented replica of
            the same loop that separates CPU issue time from GPU completion
            per chunk; cold load (Engine() in this process) and the first
            prefill of the process (kernel compilation and first touch)
            recorded apart from the warm runs; peak, active and cache memory.
  decomp    the 1,024-token development prompt at chunk 256 under the GEMM
            scope: (a) MARGINAL replacement, the full path against the path
            with one family replaced by a near-free stand-in that keeps a data
            dependence (W4 MLP projections; S16 body projections; the head;
            the DeltaNet core, convolution plus recurrence plus its gated norm;
            the DeltaNet recurrence alone; the attention core; the norms),
            order-balanced, so the marginal cost of a family is full
            minus without, NOT additive across families; (b) ADDITIVE
            synchronous attribution, every family's calls evaluated as they
            are issued and timed exclusively (inner calls subtracted), which
            serialises the GPU and inflates the total, so the sum is reported
            against the unsynchronised wall and the difference is overlap plus
            dispatch.
  design    the development prompts under diagnostic variants of the qualified
            path, each paired with the production configuration in ABBA
            order: chunk 128, 512 and 1,024; the GEMM slab limit raised to
            1,600 MB; per run TTFT, memory and parity against production (the
            final argmax, the top-16 logits, 16 greedy decode steps). A
            throwaway microbenchmark on one W4 MLP matrix: fp32 decode and
            matmul against an fp16 decode and matmul, time and max abs error
            against the fp32 product, under an explicit memory check.
  heldout   the registered comparison on the HELD-OUT prompts, read once:
            production chunk 256 against the two registered candidates
            (chunk 512, chunk 1,024), ABBA three times per length, medians,
            parity, footprint, and the end-to-end wall of prefill plus 16
            decode steps.
Receipts logs/qwen_prefill/<stage>.json (OUT overrides). SMOKE=1 shortens
every stage (one small prompt, one repeat) into logs/qwen_prefill/smoke/.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE = os.environ.get("STAGE", "baseline")
SMOKE = os.environ.get("SMOKE") == "1"
OUT = os.environ.get("OUT", f"logs/qwen_prefill/{'smoke/' if SMOKE else ''}{STAGE}.json")
TARGETS = [512, 1024, 2048]
DECODE_STEPS = 16
WORDS = ("river valley stone bridge market lantern harbour copper meadow granite willow autumn signal ledger quiet "
         "morning timber kettle orchard compass thread window saddle anchor meadow garden pencil engine basket "
         "cloud summer evening parcel lattice canvas silver marble ribbon falcon ember tulip cobalt harvest").split()


def prose(seed: str, n_words: int) -> str:
    r = random.Random(seed)
    out, k = [], 0
    while k < n_words:
        n = min(r.randint(7, 14), n_words - k)
        sent = " ".join(r.choice(WORDS) for _ in range(n))
        out.append(sent[0].upper() + sent[1:] + ".")
        k += n
    return " ".join(out)


def fixed_prompt(eng, seed: str, target: int) -> list:
    """Chat-rendered ids (thinking off) of a plain-prose request whose length lands within 8 of target."""
    lo, hi = 10, target * 3
    best = None
    for _ in range(40):
        mid = (lo + hi) // 2
        text = "Summarise the following notes in two sentences.\n\n" + prose(seed, mid)
        ids = eng.render_chat([{"role": "user", "content": text}], enable_thinking=False)
        best = ids
        if abs(len(ids) - target) <= 8:
            break
        if len(ids) < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()


def mem():
    from llmopt.lab import qprefill
    return qprefill.memory_report()


def prefill_instrumented(model, ids, cache, chunk, matmul="gemm"):
    """The prefill_chunked loop with per-chunk CPU-issue and GPU-completion times."""
    import mlx.core as mx
    from llmopt.lab import qmetal, qprefill
    body, lm = qprefill.split_body_head(model)
    rows = []
    last = None
    for s in range(0, len(ids), chunk):
        t0 = time.perf_counter()
        with qmetal.gemm_scope(matmul == "gemm"):
            h = body(mx.array([ids[s:s + chunk]]), cache)
            lg = lm(h[:, -1:, :])
            t_issue = time.perf_counter() - t0
            mx.eval(lg, *qprefill.cache_arrays(cache))
        t_done = time.perf_counter() - t0
        rows.append({"start": s, "tokens": min(chunk, len(ids) - s), "issue_s": t_issue, "complete_s": t_done})
        last = lg
        del h
    return last, rows


def topk(lg, k=16):
    import mlx.core as mx
    row = lg[0, -1].astype(mx.float32)
    top = mx.argpartition(-row, k)[:k]
    order = mx.argsort(-row[top])
    return [int(i) for i in top[order].tolist()], [float(v) for v in row[top][order].tolist()]


def top16_diff(x, ref):
    """largest absolute logit difference over the ids shared by the two top-16 lists; 99 if they share none"""
    a = dict(zip(x["top16_ids"], x["top16_logits"]))
    b = dict(zip(ref["top16_ids"], ref["top16_logits"]))
    common = set(a) & set(b)
    return max(abs(a[i] - b[i]) for i in common) if common else 99.0


def decode_steps(model, cache, first, n):
    import mlx.core as mx
    out, cur = [], first
    for _ in range(n):
        out.append(cur)
        lz = model(mx.array([[cur]]), cache=cache)[0, -1]
        mx.eval(lz)
        cur = int(lz.argmax())
    return out


def run_prefill(eng, ids, chunk, matmul="gemm", decode=0, instrument=False, clear_pool=True):
    """One measured prefill through the qualified call (and optionally the instrumented replica and a decode).
    clear_pool=True empties the allocator pool first (every run pays a fresh refill; serving never clears);
    clear_pool=False measures with whatever pool the previous run left (the serving condition)."""
    import mlx.core as mx
    from llmopt.lab import qprefill
    if clear_pool:
        mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    cache = eng.model.make_cache()
    t0 = time.perf_counter()
    lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=chunk, head="final", matmul=matmul)
    mx.eval(lg)
    wall = time.perf_counter() - t0
    ids_top, vals_top = topk(lg)
    m = mem()
    m["active_before_bytes"] = active_before
    m["transient_bytes"] = m["peak_bytes"] - active_before
    row = {"tokens": len(ids), "chunk": chunk, "matmul": matmul, "clear_pool": clear_pool, "ttft_s": wall, "prefill_tok_per_s": len(ids) / wall, "argmax": ids_top[0],
           "top16_ids": ids_top, "top16_logits": vals_top, "memory": m}
    if decode:
        t1 = time.perf_counter()
        row["decode_ids"] = decode_steps(eng.model, cache, ids_top[0], decode)
        row["decode_s"] = time.perf_counter() - t1
        row["end_to_end_s"] = wall + row["decode_s"]
    if instrument:
        del cache
        mx.clear_cache()
        cache2 = eng.model.make_cache()
        mx.reset_peak_memory()
        t2 = time.perf_counter()
        lg2, chunks = prefill_instrumented(eng.model, ids, cache2, chunk, matmul)
        mx.eval(lg2)
        row["instrumented"] = {"wall_s": time.perf_counter() - t2, "chunks": chunks, "issue_s_total": sum(c["issue_s"] for c in chunks),
                               "complete_s_total": sum(c["complete_s"] for c in chunks), "gpu_side_s_total": sum(c["complete_s"] - c["issue_s"] for c in chunks),
                               "argmax_equal": int(lg2[0, -1].argmax()) == ids_top[0],
                               "note": "issue_s is graph construction plus the host-side embedding decode; complete_s is measured from the same start and includes it; the GPU-side time is complete minus issue"}
        del cache2
    else:
        del cache
    mx.clear_cache()
    return row


def preflight_ok(extra_gb: float) -> dict:
    import mlx.core as mx
    active = mx.get_active_memory() / 1e9
    limit = 30.0
    return {"active_gb": active, "needed_gb": active + extra_gb, "limit_gb": limit, "ok": active + extra_gb < limit}


# ---------------------------------------------------------------- stages

def stage_baseline(eng, receipt):
    import mlx.core as mx
    seed = "prefill-wall-dev-1"
    prompts = {t: fixed_prompt(eng, f"{seed}-{t}", t) for t in (TARGETS if not SMOKE else [256])}
    receipt["prompts"] = {str(t): {"tokens": len(p), "sha256": sha(p)} for t, p in prompts.items()}
    first = min(prompts)
    receipt["first_prefill_of_process"] = run_prefill(eng, prompts[first], 256)        # kernel compilation and first touch land here
    receipt["warmup"] = run_prefill(eng, prompts[first], 256)
    order = (list(prompts) + list(prompts)[::-1]) * (1 if SMOKE else 2)
    receipt["runs"] = []
    for t in order:
        r = run_prefill(eng, prompts[t], 256, decode=DECODE_STEPS, instrument=True)
        receipt["runs"].append(r)
        print(f"[prefill] baseline {t} ttft {r['ttft_s']:.2f}s ({r['prefill_tok_per_s']:.1f} tok/s) issue {r['instrumented']['issue_s_total']:.2f}s complete {r['instrumented']['complete_s_total']:.2f}s "
              f"peak {r['memory']['peak_bytes'] / 1e9:.2f} GB decode16 {r['decode_s']:.2f}s", flush=True)
    if not SMOKE:
        pf = preflight_ok(4.0)
        receipt["preflight_4096"] = pf
        if pf["ok"]:
            p4 = fixed_prompt(eng, f"{seed}-4096", 4096)
            receipt["prompts"]["4096"] = {"tokens": len(p4), "sha256": sha(p4)}
            receipt["runs_4096"] = [run_prefill(eng, p4, 256, decode=DECODE_STEPS, instrument=True) for _ in range(2)]
            for r in receipt["runs_4096"]:
                print(f"[prefill] baseline 4096 ttft {r['ttft_s']:.2f}s ({r['prefill_tok_per_s']:.1f} tok/s) peak {r['memory']['peak_bytes'] / 1e9:.2f} GB", flush=True)
    receipt["warm_pool_runs"] = [run_prefill(eng, prompts[max(prompts)], 256, clear_pool=False) for _ in range(2)]   # the serving condition: no pool clear between runs
    by = {}
    for r in receipt["runs"]:
        by.setdefault(str(r["tokens"]), []).append(r["ttft_s"])
    receipt["summary"] = {t: {"n": len(v), "median_ttft_s": statistics.median(v), "min_s": min(v), "max_s": max(v), "spread_pct": 100 * (max(v) - min(v)) / statistics.median(v),
                              "tok_per_s_median": int(t) / statistics.median(v) if t.isdigit() else None} for t, v in by.items()}
    receipt["summary"]["issue_vs_complete"] = {str(r["tokens"]): {"issue_s": r["instrumented"]["issue_s_total"], "complete_s": r["instrumented"]["complete_s_total"]} for r in receipt["runs"][: len(by)]}


def stage_decomp(eng, receipt):
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.models.qwen3_5 as q35
    from llmopt.lab import qmetal, qprefill
    model = eng.model
    tm = model.language_model.model
    QLinearT = type(tm.layers[0].mlp.gate_proj)
    assert QLinearT.__name__ == "QLinear"
    target = 256 if SMOKE else 1024
    prompt = fixed_prompt(eng, f"prefill-wall-dev-1-{target}", target)
    chunk = 256
    receipt["prompt"] = {"tokens": len(prompt), "sha256": sha(prompt), "chunk": chunk}

    class ConstLinear(nn.Module):       # keeps a cheap data dependence on x so the upstream graph is not pruned
        def __init__(self, out_features):
            super().__init__()
            self.out_features = out_features
            self.c = mx.full((out_features,), 1e-3, dtype=mx.float32)

        def __call__(self, x):
            return x[..., :1] * 0.0 + self.c

    class Identity(nn.Module):
        def __call__(self, x, *a, **k):
            return x

    def linears():
        for ly in tm.layers:
            blk = ly.linear_attn if hasattr(ly, "linear_attn") else ly.self_attn
            for nm in ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a", "out_proj", "q_proj", "k_proj", "v_proj", "o_proj"):
                if hasattr(blk, nm) and isinstance(getattr(blk, nm), QLinearT):
                    yield (blk, nm, getattr(blk, nm))
            for nm in ("gate_proj", "up_proj", "down_proj"):
                if isinstance(getattr(ly.mlp, nm), QLinearT):
                    yield (ly.mlp, nm, getattr(ly.mlp, nm))
    fam = {"w4_mlp": [], "s16_body": []}
    for parent, nm, mod in linears():
        fam["w4_mlp" if mod.res.codec == "w4" else "s16_body"].append((parent, nm, mod))
    head_mod = model.language_model.lm_head
    assert isinstance(head_mod, QLinearT)
    fam["head"] = [(model.language_model, "lm_head", head_mod)]
    receipt["families"] = {k: len(v) for k, v in fam.items()}
    receipt["families"]["head_codec"] = head_mod.res.codec
    receipt["family_members"] = {k: sorted({n for _, n, _ in v}) for k, v in fam.items()}
    receipt["family_definitions"] = {"w4_mlp": "every W4-codec linear (all are MLP gate, up and down projections)", "s16_body": "every S16-codec body linear (attention q, k, v, o; DeltaNet in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj)",
                                     "head": "the vocabulary head (final position only)", "gdn_core": "the DeltaNet block minus in_proj_qkv, in_proj_z and out_proj: in_proj_b, in_proj_a, the convolution, the recurrence, the gated norm and the cache writes",
                                     "gdn_recurrence": "gated_delta_update alone (the state carried unchanged)", "attn_core": "the attention block minus q, k, v, o: q_norm, k_norm, rope, the score matmul and softmax, the output gate",
                                     "norms": "every RMSNorm (input and post-attention layer norms, the attention q and k norms, the final norm)"}
    attn_cls = next(getattr(q35, c) for c in ("Qwen3NextAttention", "Attention", "Qwen3_5Attention") if hasattr(q35, c))
    gdn_call, attn_call = q35.GatedDeltaNet.__call__, attn_cls.__call__

    def gdn_passthrough(self, x, mask=None, cache=None):
        qkv = self.in_proj_qkv(x)
        z = self.in_proj_z(x)
        v = qkv[..., -z.shape[-1]:]
        return self.out_proj(v * z)

    def attn_passthrough(self, x, mask=None, cache=None):
        # projections kept (S16 body), the KV cache still fed so the chunk loop's cache arrays exist;
        # norms, rope, the score matmul and the softmax removed
        B, L, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x).reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        v = self.v_proj(x)
        vh = v.reshape(B, L, self.num_key_value_heads, -1).transpose(0, 2, 1, 3)
        if cache is not None:
            cache.update_and_fetch(k, vh)
        half = q.shape[-1] // 2
        o = q[..., :half] * 0.0 + mx.broadcast_to(v[..., :1], q[..., :half].shape)
        return self.o_proj(o)

    def norms():
        for ly in tm.layers:
            for nm in ("input_layernorm", "post_attention_layernorm"):
                yield (ly, nm, getattr(ly, nm))
            blk = getattr(ly, "self_attn", None)
            if blk is not None:
                for nm in ("q_norm", "k_norm"):
                    if hasattr(blk, nm):
                        yield (blk, nm, getattr(blk, nm))
        yield (tm, "norm", tm.norm)
    norm_list = list(norms())

    def timed_full():
        mx.clear_cache()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(model, prompt, cache, chunk=chunk, head="final", matmul="gemm")
        mx.eval(lg)
        w = time.perf_counter() - t0
        del cache
        return w

    gdu_real = q35.gated_delta_update

    def gdu_standin(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel=True):
        # the recurrence alone replaced: the output keeps v's shape and a cheap dependence on q, k, a and b;
        # the state is carried unchanged (or zeros on the first chunk) so the cache update still happens
        dep = q.sum(axis=(-1, -2), keepdims=True) + k.sum(axis=(-1, -2), keepdims=True) + a.sum(-1)[..., None, None] + b.sum(-1)[..., None, None]   # (B, S, 1, 1): q and k have fewer heads than v
        out = v + 0.0 * dep
        if state is None:
            state = mx.zeros((v.shape[0], v.shape[2], k.shape[-1], v.shape[-1]), dtype=v.dtype)
        return out, state

    def with_swaps(swaps=(), gdn=False, attn=False, nrm=False, rec=False):
        saved = [(p, n, m) for p, n, m in swaps]
        for p, n, m in swaps:
            setattr(p, n, ConstLinear(m.res.shape[0]))
        saved_norm = []
        if nrm:
            for p, n, m in norm_list:
                saved_norm.append((p, n, m))
                setattr(p, n, Identity())
        if gdn:
            q35.GatedDeltaNet.__call__ = gdn_passthrough
        if attn:
            attn_cls.__call__ = attn_passthrough
        if rec:
            q35.gated_delta_update = gdu_standin
        try:
            return timed_full()
        finally:
            for p, n, m in saved:
                setattr(p, n, m)
            for p, n, m in saved_norm:
                setattr(p, n, m)
            q35.GatedDeltaNet.__call__ = gdn_call
            attn_cls.__call__ = attn_call
            q35.gated_delta_update = gdu_real

    configs = {"full": {}, "no_w4_mlp": {"swaps": fam["w4_mlp"]}, "no_s16_body": {"swaps": fam["s16_body"]}, "no_head": {"swaps": fam["head"]},
               "no_gdn_core": {"gdn": True}, "no_gdn_recurrence": {"rec": True}, "no_attn_core": {"attn": True}, "no_norms": {"nrm": True},
               "no_all_linears": {"swaps": fam["w4_mlp"] + fam["s16_body"] + fam["head"]}}
    reps = 1 if SMOKE else 2
    order = list(configs) + list(configs)[::-1]
    marg = {k: [] for k in configs}
    with_swaps()        # warm
    for _ in range(reps):
        for k in order:
            w = with_swaps(**configs[k])
            marg[k].append(w)
            print(f"[prefill] decomp marginal {k:14s} {w:.3f}s", flush=True)
    full = statistics.median(marg["full"])
    receipt["marginal"] = {k: {"median_s": statistics.median(v), "runs": v, "marginal_s": full - statistics.median(v), "marginal_pct_of_full": 100 * (full - statistics.median(v)) / full}
                           for k, v in marg.items()}
    receipt["marginal_note"] = "marginal cost = full minus the path with the family replaced by a near-free stand-in; GPU work overlaps and dispatch is shared, so these do NOT add to the full time"

    # additive synchronous attribution: exclusive time per family with inner calls subtracted
    acc = {"w4_mlp": 0.0, "s16_body": 0.0, "head": 0.0, "gdn_core": 0.0, "attn_core": 0.0, "norms": 0.0, "embed_other": 0.0}
    stack = []
    fam_of = {id(m): ("w4_mlp" if m.res.codec == "w4" else "s16_body") for lst in (fam["w4_mlp"], fam["s16_body"]) for _, _, m in lst}
    fam_of[id(head_mod)] = "head"
    orig_qlin = QLinearT.__call__

    def exclusive(name_fn, call):
        def wrapper(self, *a, **k):
            name = name_fn(self)
            # evaluate the inputs BEFORE the timed region: under lazy evaluation the cost of upstream work
            # otherwise lands on the first consumer that forces it (a recurrence's time would be booked to the
            # out_proj linear that first evaluates it)
            ins = [v for v in a if isinstance(v, mx.array)] + [v for v in k.values() if isinstance(v, mx.array)]
            if ins:
                mx.eval(*ins)
            frame = [name, 0.0]
            stack.append(frame)
            t0 = time.perf_counter()
            out = call(self, *a, **k)
            mx.eval(out)
            dt = time.perf_counter() - t0
            stack.pop()
            acc[name] += dt - frame[1]
            if stack:
                stack[-1][1] += dt
            return out
        return wrapper

    norm_cls = type(tm.norm)
    orig_norm = norm_cls.__call__
    QLinearT.__call__ = exclusive(lambda self: fam_of.get(id(self), "embed_other"), orig_qlin)
    q35.GatedDeltaNet.__call__ = exclusive(lambda self: "gdn_core", gdn_call)
    attn_cls.__call__ = exclusive(lambda self: "attn_core", attn_call)
    norm_cls.__call__ = exclusive(lambda self: "norms", orig_norm)
    try:
        mx.clear_cache()
        cache = model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(model, prompt, cache, chunk=chunk, head="final", matmul="gemm")
        mx.eval(lg)
        sync_wall = time.perf_counter() - t0
        del cache
    finally:
        QLinearT.__call__ = orig_qlin
        q35.GatedDeltaNet.__call__ = gdn_call
        attn_cls.__call__ = attn_call
        norm_cls.__call__ = orig_norm
    unsync = statistics.median([timed_full() for _ in range(reps)])
    receipt["additive_synchronous"] = {"exclusive_s": acc, "sum_s": sum(acc.values()), "synchronised_wall_s": sync_wall, "unsynchronised_wall_s": unsync,
                                       "untimed_residual_s": sync_wall - sum(acc.values()), "serialisation_penalty_s": sync_wall - unsync,
                                       "note": "inputs evaluated before each timed call, outputs after; the untimed residual (synchronised wall minus the sum) is the host-side embedding decode, the residual adds, the chunk-end evaluation and anything in no family; the serialisation penalty (synchronised minus unsynchronised wall) is what forcing every call costs over the pipelined path; neither is overlap alone"}
    print(f"[prefill] decomp additive {json.dumps({k: round(v, 3) for k, v in acc.items()})} sum {sum(acc.values()):.2f}s sync wall {sync_wall:.2f}s unsync {unsync:.2f}s", flush=True)


def stage_design(eng, receipt):
    import mlx.core as mx
    from llmopt.lab import qmetal
    targets = TARGETS if not SMOKE else [256]
    prompts = {t: fixed_prompt(eng, f"prefill-wall-dev-1-{t}", t) for t in targets}
    receipt["prompts"] = {str(t): {"tokens": len(p), "sha256": sha(p)} for t, p in prompts.items()}
    variants = {"prod_chunk256": {"chunk": 256}, "chunk128": {"chunk": 128}, "chunk512": {"chunk": 512}, "chunk1024": {"chunk": 1024}}
    # (a GEMM slab variant was dropped: the largest body matrix, 17408 x 5120 float32, already fits one 400 MiB slab and the head runs fused on one row)
    if SMOKE:
        variants = {"prod_chunk256": {"chunk": 256}, "chunk512": {"chunk": 512}}
    slab0 = qmetal.GEMM_SLAB_BYTES
    receipt["runs"] = []
    reps = 1 if SMOKE else 2
    for t in targets:
        pf = preflight_ok(3.0)
        receipt.setdefault("preflight", {})[str(t)] = pf
        if not pf["ok"]:
            continue
        for name, v in variants.items():       # one discarded warm-up per (variant, length): kernel compilation and first touch
            w = run_prefill(eng, prompts[t], v["chunk"])
            receipt.setdefault("warmups", []).append({"tokens": t, "variant": name, "ttft_s": w["ttft_s"]})
        for name, v in variants.items():
            if name == "prod_chunk256":
                continue
            for _ in range(reps):
                for which in ("prod_chunk256", name, name, "prod_chunk256"):   # ABBA
                    cfg = variants[which]
                    r = run_prefill(eng, prompts[t], cfg["chunk"], decode=DECODE_STEPS)
                    r["variant"], r["pair"] = which, name
                    receipt["runs"].append(r)
                    print(f"[prefill] design {t} {which:18s} ttft {r['ttft_s']:.2f}s peak {r['memory']['peak_bytes'] / 1e9:.2f} GB footprint {r['memory']['footprint_bytes'] / 1e9:.2f} GB argmax {r['argmax']}", flush=True)
    # summary per length and variant against production (paired within each ABBA block)
    summ = {}
    for t in targets:
        prod = [r for r in receipt["runs"] if r["tokens"] == len(prompts[t]) and r["variant"] == "prod_chunk256"]
        if not prod:
            continue
        ref = prod[0]
        for name in variants:
            rows = [r for r in receipt["runs"] if r["tokens"] == len(prompts[t]) and r["variant"] == name]
            if not rows:
                continue
            prod_pair = [r for r in prod if r["pair"] == name] or prod
            summ[f"{t}:{name}"] = {"median_ttft_s": statistics.median(x["ttft_s"] for x in rows), "prod_median_ttft_s": statistics.median(x["ttft_s"] for x in prod_pair),
                                    "ttft_ratio": statistics.median(x["ttft_s"] for x in rows) / statistics.median(x["ttft_s"] for x in prod_pair),
                                    "argmax_equal_prod": all(x["argmax"] == ref["argmax"] for x in rows),
                                    "decode_equal_prod": all(x.get("decode_ids") == ref.get("decode_ids") for x in rows),
                                    "top16_max_abs_diff": max(top16_diff(x, ref) for x in rows),
                                    "footprint_ratio": max(x["memory"]["footprint_bytes"] for x in rows) / max(x["memory"]["footprint_bytes"] for x in prod_pair),
                                    "peak_ratio": max(x["memory"]["peak_bytes"] for x in rows) / max(x["memory"]["peak_bytes"] for x in prod_pair),
                                    "transient_ratio": max(x["memory"]["transient_bytes"] for x in rows) / max(1, max(x["memory"]["transient_bytes"] for x in prod_pair)),
                                    "decode_s_ratio": statistics.median(x["decode_s"] for x in rows) / statistics.median(x["decode_s"] for x in prod_pair),
                                    "prod_self_consistent": all(x["argmax"] == ref["argmax"] and x["decode_ids"] == ref["decode_ids"] for x in prod),
                                    "peak_gb": max(x["memory"]["peak_bytes"] for x in rows) / 1e9}
    receipt["summary"] = summ
    # throwaway microbenchmark: one W4 MLP matrix, fp32 decode+matmul against fp16 decode+matmul (an oracle on the fp32 product)
    tm = eng.model.language_model.model
    res = tm.layers[0].mlp.gate_proj.res
    pf = preflight_ok(2.0)
    receipt["microbench_preflight"] = pf
    if pf["ok"] and res.gemm_ok:
        R, C = res.shape
        mb = {"matrix": "layers.0.mlp.gate_proj", "shape": [R, C], "rows": {}}
        w32 = res.decode_f32(0, R)
        w16 = w32.astype(mx.float16)
        mx.eval(w32, w16)
        mb["fp16_weights_finite"] = bool(mx.all(mx.isfinite(w16)).item())
        mb["fp16_weights_flushed_to_zero"] = int(mx.sum((w16 == 0) & (w32 != 0)).item())
        mb["decode_only_s"] = None
        for T in ((64,) if SMOKE else (256, 512, 1024)):
            x = mx.random.normal((T, C), dtype=mx.float32, key=mx.random.key(T)) * 0.1
            x16 = x.astype(mx.float16)
            mx.eval(x, x16)
            def mm32():
                return mx.matmul(x, w32.T)
            def mm16():
                return mx.matmul(x16, w16.T)
            def dec_only():
                return res.decode_f32(0, R)
            def tm_(fn, n=3):
                mx.eval(fn())
                ts = []
                for _ in range(n):
                    t0 = time.perf_counter(); y = fn(); mx.eval(y); ts.append(time.perf_counter() - t0)
                return statistics.median(ts), y
            t32, y32 = tm_(mm32)
            t16, y16 = tm_(mm16)
            tdec, _ = tm_(dec_only)
            mb["decode_only_s"] = tdec
            y16f = y16.astype(mx.float32)
            err = float(mx.max(mx.abs(y32 - y16f)).item()); scale = float(mx.max(mx.abs(y32)).item())
            mb["rows"][str(T)] = {"matmul_fp32_s": t32, "matmul_fp16_s": t16, "decode_only_s": tdec, "fp16_over_fp32_matmul": t16 / t32, "max_abs_err": err, "max_abs_ref": scale,
                                  "rel_err": err / scale if scale else None, "note": "matmul alone on pre-decoded, pre-cast weights and a seeded random activation (not a real activation); the fp16 output cast back for the error; whether MLX accumulates fp16 matmuls in float32 is not verified here"}
            print(f"[prefill] microbench T={T} matmul fp32 {t32 * 1e3:.1f} ms fp16 {t16 * 1e3:.1f} ms (decode alone {tdec * 1e3:.1f}) rel err {err / scale:.2e}", flush=True)
        receipt["microbench_throwaway"] = mb


def stage_heldout(eng, receipt):
    targets = TARGETS if not SMOKE else [256]
    if SMOKE:
        prompts = {256: fixed_prompt(eng, "prefill-wall-heldout-1-256", 256)}
    else:
        frozen = json.load(open("scratch/evalsets/qwen_prefill_heldout_prompts.json"))["prompts"]    # read once, asserted against the frozen sha256s
        prompts = {t: frozen[str(t)]["ids"] for t in targets}
        for t in targets:
            assert sha(prompts[t]) == frozen[str(t)]["sha256"], f"held-out prompt {t} does not match its frozen sha"
    receipt["prompts"] = {str(t): {"tokens": len(p), "sha256": sha(p)} for t, p in prompts.items()}
    cands = {"CAND512": 512, "CAND1024": 1024}
    reps = 1 if SMOKE else 3
    receipt["runs"] = []
    for t in targets:
        pf = preflight_ok(3.0)
        receipt.setdefault("preflight", {})[str(t)] = pf
        if not pf["ok"]:
            continue
        for which, ch in (("PROD", 256), *cands.items()):     # one discarded warm-up per (arm, length)
            w = run_prefill(eng, prompts[t], ch)
            receipt.setdefault("warmups", []).append({"tokens": t, "arm": which, "ttft_s": w["ttft_s"]})
        for name, chunk in cands.items():
            for _ in range(reps):
                for which, ch in (("PROD", 256), (name, chunk), (name, chunk), ("PROD", 256)):
                    r = run_prefill(eng, prompts[t], ch, decode=DECODE_STEPS)
                    r["arm"], r["pair"] = which, name
                    receipt["runs"].append(r)
                    print(f"[prefill] heldout {t} {which:9s} ttft {r['ttft_s']:.2f}s e2e {r['end_to_end_s']:.2f}s footprint {r['memory']['footprint_bytes'] / 1e9:.2f} GB", flush=True)
    summ = {}
    for t in targets:
        for name in cands:
            rows = [r for r in receipt["runs"] if r["tokens"] == len(prompts[t]) and r["arm"] == name]
            prod = [r for r in receipt["runs"] if r["tokens"] == len(prompts[t]) and r["arm"] == "PROD" and r["pair"] == name]
            if not rows or not prod:
                continue
            ref = prod[0]
            summ[f"{name}:{t}"] = {"n": len(rows), "ttft_median_s": statistics.median(x["ttft_s"] for x in rows), "prod_ttft_median_s": statistics.median(x["ttft_s"] for x in prod),
                                    "ttft_ratio": statistics.median(x["ttft_s"] for x in rows) / statistics.median(x["ttft_s"] for x in prod),
                                    "e2e_ratio": statistics.median(x["end_to_end_s"] for x in rows) / statistics.median(x["end_to_end_s"] for x in prod),
                                    "prod_spread_pct": 100 * (max(x["ttft_s"] for x in prod) - min(x["ttft_s"] for x in prod)) / statistics.median(x["ttft_s"] for x in prod),
                                    "parity_mismatches": sum(1 for x in rows if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                                    "prod_self_mismatches": sum(1 for x in prod if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                                    "top16_max_abs_diff": max(top16_diff(x, ref) for x in rows),
                                    "footprint_ratio": max(x["memory"]["footprint_bytes"] for x in rows) / max(x["memory"]["footprint_bytes"] for x in prod),
                                    "peak_ratio": max(x["memory"]["peak_bytes"] for x in rows) / max(x["memory"]["peak_bytes"] for x in prod),
                                    "transient_ratio": max(x["memory"]["transient_bytes"] for x in rows) / max(1, max(x["memory"]["transient_bytes"] for x in prod)),
                                    "decode_s_ratio": statistics.median(x["decode_s"] for x in rows) / statistics.median(x["decode_s"] for x in prod)}
    receipt["summary"] = summ


def main() -> None:
    from llmopt.lab import qserve
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t_proc = time.perf_counter()
    eng = qserve.Engine()
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "stage": STAGE, "cold_load_s": eng.load_s, "process_to_ready_s": time.perf_counter() - t_proc,
               "engine": {"kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"), "cache_limit_bytes": eng.cfg.cache_limit_bytes,
                          "chunk": eng.cfg.chunk, "matmul": eng.cfg.matmul, "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {}),
                          "gemm_min_t_by_codec": dict(__import__("llmopt.lab.qmetal", fromlist=["x"]).GEMM_MIN_T_BY_CODEC), "gemm_slab_bytes": __import__("llmopt.lab.qmetal", fromlist=["x"]).GEMM_SLAB_BYTES}}
    {"baseline": stage_baseline, "decomp": stage_decomp, "design": stage_design, "heldout": stage_heldout}[STAGE](eng, receipt)
    receipt["wall_s"] = time.perf_counter() - t_proc
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[prefill] {STAGE} wrote {OUT} in {receipt['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
