"""QWEN-GDN-KERNEL-0: qualification of mlx-lm's Metal GatedDeltaNet
recurrence kernel as a substitute for the Python reference loop the released
arm C serving path runs today (VERDICT QWEN-PREFILL-WALL-0 found the loop
selected by the vendor's training-mode flag). Nothing here changes a default:
the implementation is selected per measured call through
llmopt.lab.qgdn.recurrence_path, whose receipt counts which implementation
EXECUTED; the baseline arm LOOP runs today's code path unchanged.

Oracle. Arm C against arm C: identical weights, prompts, cache construction,
chunking (the serving path: chunk 256, GEMM prefill, 2 GiB cache limit),
loop against kernel, in one process. Teacher agreement and the frozen
generation-5 census rows (logs/qwen_metal_qual/C_v5, arm C on Metal through
the loop, one-shot fused forward) are DESCRIPTIVE cross-checks only.

Stages (STAGE env; each writes its own receipt and refuses to overwrite):
  layers    the 1,017-token development prompt through the LOOP path with the
            recurrence operands (q, k, v, a, b, state in) captured at six
            DeltaNet layers on every chunk; offline, loop and kernel re-run from
            the identical operands and state (output and new state compared
            with normalized errors), the kernel carried across the chunks
            against the loop's carried state (accumulated divergence), a
            T = 1 decode step from a carried state, and the kernel run twice
            (determinism).
  census    the 24 E0 teacher-greedy rollouts (prompt plus recorded
            continuation, 4,722 positions), full-head chunked prefill through
            both arms (arm order alternating by prompt): per position both
            argmaxes and margins, the kernel's logits at the loop's top-16 ids,
            the full-vocabulary max abs difference, agreement with the teacher
            token (descriptive), the frozen census argmaxes (descriptive); the
            15 near-tie positions (metal_margin under 0.02 logits in the frozen
            generation-5 census: arm C on Metal through the loop) listed one by one; the recurrent and convolution states of
            all 48 DeltaNet layers after the full prefill compared per layer.
  decode    two rollout prompts: free-run greedy 256 steps in both arms
            (streams compared, first divergence and its margins), the kernel
            arm twice (determinism), and the loop's stream fed teacher-forced
            through the kernel arm with the per-step full-vocabulary max abs
            logit difference and the recurrent-state errors every 32 steps
            (accumulated decode error).
  heldout   the three frozen held-out prompts (scratch/evalsets/
            qwen_prefill_heldout_prompts.json): LOOP against KERNEL at the
            serving chunk, ABBA three times per length with 16 decode steps
            (TTFT, peak and transient memory, decode time, parity), one
            instrumented run per arm per length (CPU issue against GPU
            completion), 4,096 tokens once ABBA after a memory preflight, and
            one long generation (256 greedy steps) per arm at 1,024 and 2,048
            (end-to-end wall, steady decode step).
  cachehist three fixed three-turn conversations (thinking off, 32 tokens a
            turn) in both arms, incremental and replay sessions: per turn the
            generated ids, mode and fallback note; incremental against replay
            within the kernel arm, kernel against loop across arms.
Receipts logs/qwen_gdn/<stage>.json (census also census_positions.jsonl);
SMOKE=1 shortens every stage into logs/qwen_gdn/smoke/.
"""
from __future__ import annotations

import hashlib
import json
import os
import statistics
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

STAGE = os.environ.get("STAGE", "layers")
SMOKE = os.environ.get("SMOKE") == "1"
OUTDIR = os.environ.get("OUTDIR", f"logs/qwen_gdn/{'smoke/' if SMOKE else ''}")
OUT = os.path.join(OUTDIR, f"{STAGE}.json")
if SMOKE != ("smoke" in OUTDIR):
    raise SystemExit(f"REFUSING: SMOKE={int(SMOKE)} with OUTDIR {OUTDIR!r}; smoke receipts live under a path containing 'smoke' and real ones never do")
ARMS = {"LOOP": "loop", "KERNEL": "kernel"}
TEACHER_PINS = {"logs/qwenteacher_v2": "0ca4151"}
FROZEN_CENSUS = "logs/qwen_metal_qual/C_v5/positions.jsonl"
NEAR_TIE_MARGIN = 0.02
LAYER_ORDINALS = [0, 9, 19, 29, 39, 47] if not SMOKE else [0, 47]
DECODE_STEPS = 16
LONG_STEPS = 256 if not SMOKE else 24
TARGETS = [512, 1024, 2048] if not SMOKE else [256]
CONVERSATIONS = [
    ["What is the capital of Portugal?", "And its population, roughly?", "Name one river there."],
    ["Write a haiku about granite.", "Now one about copper.", "Which did you prefer and why?"],
    ["Give me three prime numbers above 100.", "Add them together.", "Is the sum prime?"],
]
if SMOKE:
    CONVERSATIONS = CONVERSATIONS[:1]
    CONVERSATIONS[0] = CONVERSATIONS[0][:2]


def sha_ids(ids):
    return hashlib.sha256(json.dumps(list(map(int, ids))).encode()).hexdigest()


def fsha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def mem():
    from llmopt.lab import qprefill
    return qprefill.memory_report()


def selected(path):
    """The selector for one arm; the caller records sel.receipt() after the block."""
    from llmopt.lab import qgdn
    return qgdn.recurrence_path(path)


def record_dispatch(receipt, sel, label):
    sel.count.assert_exclusive()
    receipt.setdefault("dispatch", []).append({"label": label, **sel.receipt()})


def dnet_layers(model):
    tm = model.language_model.model
    return [(i, ly) for i, ly in enumerate(tm.layers) if getattr(ly, "is_linear", False)]


def state_errors(cache_a, cache_b, model):
    """Per DeltaNet layer: normalized errors of the recurrent state and the conv state of cache_a against cache_b."""
    from llmopt.lab import qgdn
    out = []
    for i, _ in dnet_layers(model):
        ca, cb = cache_a[i], cache_b[i]
        rec = qgdn.normalized_errors(np.asarray(ca.cache[1], dtype=np.float32), np.asarray(cb.cache[1], dtype=np.float32))
        conv = qgdn.normalized_errors(np.asarray(ca.cache[0], dtype=np.float32), np.asarray(cb.cache[0], dtype=np.float32))
        out.append({"layer": i, "state": rec, "conv": conv})
    return out


def worst(errs, key="state"):
    return {"max_abs_over_rms": max(e[key]["max_abs_over_rms"] for e in errs), "max_rel_on_large": max(e[key]["max_rel_on_large"] for e in errs),
            "all_finite": all(e[key]["finite"] for e in errs), "n_layers": len(errs)}


def prose(seed, n_words):
    import random
    words = ("river valley stone bridge market lantern harbour copper meadow granite willow autumn signal ledger quiet "
             "morning timber kettle orchard compass thread window saddle anchor meadow garden pencil engine basket "
             "cloud summer evening parcel lattice canvas silver marble ribbon falcon ember tulip cobalt harvest").split()
    r = random.Random(seed)
    out, k = [], 0
    while k < n_words:
        n = min(r.randint(7, 14), n_words - k)
        sent = " ".join(r.choice(words) for _ in range(n))
        out.append(sent[0].upper() + sent[1:] + ".")
        k += n
    return " ".join(out)


def fixed_prompt(eng, seed, target):
    lo, hi, best = 10, target * 3, None
    for _ in range(40):
        mid = (lo + hi) // 2
        ids = eng.render_chat([{"role": "user", "content": "Summarise the following notes in two sentences.\n\n" + prose(seed, mid)}], enable_thinking=False)
        best = ids
        if abs(len(ids) - target) <= 8:
            break
        lo, hi = (mid + 1, hi) if len(ids) < target else (lo, mid - 1)
    return best


def topk(row, k=16):
    import mlx.core as mx
    row = row.astype(mx.float32)
    top = mx.argpartition(-row, k)[:k]
    order = mx.argsort(-row[top])
    return [int(i) for i in top[order].tolist()], [float(v) for v in row[top][order].tolist()]


def top1_margin(z):
    """argmax and the gap to the runner-up of a float64 vector."""
    i = int(z.argmax())
    second = np.partition(z, -2)[-2]
    return i, float(z[i] - second)


def teacher_rollouts(eng):
    tdir = "logs/qwenteacher_v2"
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    if man.get("smoke") or man["code_commit"] != TEACHER_PINS[tdir]:
        raise SystemExit("REFUSING: teacher record pin")
    ro = man["records"]["rollouts"]
    if ro["padding_side"] != "left":
        raise SystemExit("REFUSING: unexpected padding side")
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids, mask, gl = np.asarray(ro["input_ids"]), np.asarray(ro["attention_mask"]), list(ro["gen_lengths"])
    out = []
    for i, rec in enumerate(recs):
        prompt = [int(x) for x in input_ids[i][mask[i] == 1]]
        gen = [int(x) for x in rec["tokens"][:gl[i]]]
        out.append({"id": rec["id"], "category": rec["category"], "prompt": prompt, "gen": gen})
    return out, {"dir": tdir, "code_commit": man["code_commit"], "rollout_tokens_sha256": fsha(os.path.join(tdir, "rollout_tokens.jsonl"))}


# ---------------------------------------------------------------- layers

def stage_layers(eng, receipt):
    import mlx.core as mx
    from mlx_lm.models import gated_delta, qwen3_5
    from llmopt.lab import qgdn, qprefill
    model = eng.model
    chunk = eng.cfg.chunk
    target = 1024 if not SMOKE else 64
    seed = None
    for n in range(2, 12):          # gdn-dev-1-1024 was the pre-registration calibration prompt (read); the registered run uses the first unread seed whose length fits exactly four chunks
        seed = f"gdn-dev-{n}-{target}"
        ids = fixed_prompt(eng, seed, target)
        if target - 24 <= len(ids) <= target:
            break
    else:
        raise SystemExit("REFUSING: no development seed in 2..11 landed within 24 tokens under the target")
    receipt["prompt"] = {"seed": seed, "tokens": len(ids), "sha256": sha_ids(ids), "chunk": chunk}
    n_dnet = len(dnet_layers(model))
    want = set(LAYER_ORDINALS)
    captured = {}        # (ordinal, chunk_index) -> operands as numpy
    calls = {"n": 0}
    with selected("loop") as sel:
        inner = qwen3_5.gated_delta_update       # the selector's forwarding wrapper

        def capturing(q, k, v, a, b, A_log, dt_bias, state=None, mask=None, use_kernel=True):
            idx = calls["n"]
            calls["n"] += 1
            ordinal, ci = idx % n_dnet, idx // n_dnet
            if ordinal in want:
                mx.eval(q, k, v, a, b, A_log, dt_bias, *( [state] if state is not None else []))
                captured[(ordinal, ci)] = {"q": np.asarray(q), "k": np.asarray(k), "v": np.asarray(v), "a": np.asarray(a), "b": np.asarray(b),
                                           "A_log": np.asarray(A_log), "dt_bias": np.asarray(dt_bias), "state": None if state is None else np.asarray(state),
                                           "mask": None if mask is None else np.asarray(mask), "use_kernel_requested": bool(use_kernel), "dtypes": [str(x.dtype) for x in (q, k, v, a, b, A_log, dt_bias)] + [None if state is None else str(state.dtype)]}
            return inner(q, k, v, a, b, A_log, dt_bias, state, mask, use_kernel)
        qwen3_5.gated_delta_update = capturing
        try:
            cache = model.make_cache()
            lg = qprefill.prefill_chunked(model, ids, cache, chunk=chunk, head="final", matmul=eng.cfg.matmul)
            mx.eval(lg)
        finally:
            qwen3_5.gated_delta_update = inner
    record_dispatch(receipt, sel, "capture through the loop path")
    n_chunks = (len(ids) + chunk - 1) // chunk
    receipt["capture"] = {"recurrence_calls": calls["n"], "dnet_layers": n_dnet, "chunks": n_chunks, "captured": len(captured), "ordinals": sorted(want),
                          "any_mask": any(c["mask"] is not None for c in captured.values()), "dtypes": next(iter(captured.values()))["dtypes"],
                          "use_kernel_requested_by_vendor_flag": sorted({c["use_kernel_requested"] for c in captured.values()})}
    if calls["n"] != n_dnet * n_chunks:
        raise SystemExit(f"REFUSING: {calls['n']} recurrence calls, expected {n_dnet * n_chunks}")

    offline = {"loop": {"ops": 0, "kernel": 0, "calls": 0}, "kernel": {"ops": 0, "kernel": 0, "calls": 0}}

    def run(c, state, use_kernel):
        # every offline call goes through the selector so its dispatch is counted and asserted like the capture's
        path = "kernel" if use_kernel else "loop"
        args = [mx.array(c[n]) for n in ("q", "k", "v", "a", "b", "A_log", "dt_bias")]
        st = None if state is None else mx.array(state)
        with selected(path) as sel:
            y, s = qwen3_5.gated_delta_update(*args, st, None if c["mask"] is None else mx.array(c["mask"]), use_kernel=not use_kernel)   # the vendor flag inverted on purpose: the selector must override it
            mx.eval(y, s)
        sel.count.assert_exclusive()
        for k in ("ops", "kernel"):
            offline[path][k] += sel.count.executed[k]
        offline[path]["calls"] += sel.count.update_calls
        return np.asarray(y), np.asarray(s)

    rows, carried = [], []
    for ordinal in sorted(want):
        k_state = None
        for ci in range(n_chunks):
            c = captured[(ordinal, ci)]
            t0 = time.perf_counter()
            y0, s0 = run(c, c["state"], False)
            t_loop = time.perf_counter() - t0
            t0 = time.perf_counter()
            y1, s1 = run(c, c["state"], True)
            t_kernel = time.perf_counter() - t0
            y2, s2 = run(c, c["state"], True)
            rows.append({"ordinal": ordinal, "chunk": ci, "T": int(c["q"].shape[1]), "state_in_from": "zeros" if c["state"] is None else "loop-carried",
                         "y": qgdn.normalized_errors(y1, y0), "state": qgdn.normalized_errors(s1, s0),
                         "kernel_repeat_max_abs": float(max(np.abs(y2 - y1).max(), np.abs(s2 - s1).max())),
                         "loop_s": t_loop, "kernel_s": t_kernel})
            # the kernel carried through the chunks on its own state against the loop's carried state
            yk, k_state = run(c, k_state, True)
            if ci + 1 < n_chunks:
                nxt = captured[(ordinal, ci + 1)]["state"]
                carried.append({"ordinal": ordinal, "after_chunk": ci, "state": qgdn.normalized_errors(k_state, nxt), "y": qgdn.normalized_errors(yk, y0)})
            else:
                carried.append({"ordinal": ordinal, "after_chunk": ci, "state": qgdn.normalized_errors(k_state, s0), "y": qgdn.normalized_errors(yk, y0)})
        # a T = 1 decode step from the carried state: the last position of the last chunk re-fed
        c = captured[(ordinal, n_chunks - 1)]
        one = {n: c[n][:, -1:] for n in ("q", "k", "v", "a", "b")}
        one.update({"A_log": c["A_log"], "dt_bias": c["dt_bias"], "mask": None})
        y0, s0 = run(one, s0, False)
        y1, s1 = run(one, k_state, True)
        rows.append({"ordinal": ordinal, "chunk": "decode-step", "T": 1, "state_in_from": "each arm's own carried state",
                     "y": qgdn.normalized_errors(y1, y0), "state": qgdn.normalized_errors(s1, s0), "kernel_repeat_max_abs": None, "loop_s": None, "kernel_s": None})
    receipt["rows"] = rows
    receipt["carried"] = carried
    receipt["offline_dispatch"] = offline
    for path, want in (("loop", "ops"), ("kernel", "kernel")):
        other = "kernel" if want == "ops" else "ops"
        if offline[path][other] or not offline[path][want]:
            raise SystemExit(f"REFUSING: offline dispatch proof failed for {path}: {offline[path]}")
    receipt["nonfinite_arrays"] = sum(1 for r in rows for k in ("y", "state") if not r[k]["finite"]) + sum(1 for r in carried for k in ("y", "state") if not r[k]["finite"])
    receipt["summary"] = {"same_state_in": {"y_max_abs_over_rms": max(r["y"]["max_abs_over_rms"] for r in rows), "state_max_abs_over_rms": max(r["state"]["max_abs_over_rms"] for r in rows),
                                            "y_max_rel_on_large": max(r["y"]["max_rel_on_large"] for r in rows), "state_max_rel_on_large": max(r["state"]["max_rel_on_large"] for r in rows),
                                            "all_finite": all(r["y"]["finite"] and r["state"]["finite"] for r in rows),
                                            "kernel_repeat_max_abs": max(r["kernel_repeat_max_abs"] for r in rows if r["kernel_repeat_max_abs"] is not None)},
                          "kernel_carried": {"state_max_abs_over_rms": max(r["state"]["max_abs_over_rms"] for r in carried), "y_max_abs_over_rms": max(r["y"]["max_abs_over_rms"] for r in carried),
                                             "state_max_rel_on_large": max(r["state"]["max_rel_on_large"] for r in carried), "all_finite": all(r["y"]["finite"] and r["state"]["finite"] for r in carried)},
                          "per_call_ms": {"loop_median": 1e3 * statistics.median(r["loop_s"] for r in rows if r["loop_s"]), "kernel_median": 1e3 * statistics.median(r["kernel_s"] for r in rows if r["kernel_s"])}}
    print(f"[gdn] layers: y {receipt['summary']['same_state_in']['y_max_abs_over_rms']:.2e} state {receipt['summary']['same_state_in']['state_max_abs_over_rms']:.2e} "
          f"carried {receipt['summary']['kernel_carried']['state_max_abs_over_rms']:.2e} per call loop {receipt['summary']['per_call_ms']['loop_median']:.1f} ms kernel {receipt['summary']['per_call_ms']['kernel_median']:.1f} ms", flush=True)


# ---------------------------------------------------------------- census

def full_head_rows(eng, ids, p_len, n_gen, path):
    """Chunked full-head prefill of ids through one arm; returns (float32 [n_gen, v_live] logits at the generated positions, cache, wall, selector)."""
    import mlx.core as mx
    from llmopt.lab import qprefill
    v_live = len(eng.tokenizer)
    out = np.zeros((n_gen, v_live), dtype=np.float32)
    lo, hi = p_len - 1, p_len - 1 + n_gen

    def cb(start, lg):
        S = lg.shape[1]
        a, b = max(start, lo), min(start + S, hi)
        if a < b:
            out[a - lo:b - lo] = np.asarray(lg[0, a - start:b - start, :v_live], dtype=np.float32)
    with selected(path) as sel:
        cache = eng.model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=eng.cfg.chunk, head="full", matmul=eng.cfg.matmul, on_chunk=cb)
        mx.eval(lg, *qprefill.cache_arrays(cache))
        wall = time.perf_counter() - t0
    return out, cache, wall, sel


def stage_census(eng, receipt):
    rolls, tinfo = teacher_rollouts(eng)
    frozen = [json.loads(l) for l in open(FROZEN_CENSUS)]
    receipt["teacher"] = tinfo
    receipt["frozen_census"] = {"path": FROZEN_CENSUS, "sha256": fsha(FROZEN_CENSUS), "n": len(frozen), "role": "descriptive cross-check and the near-tie definition (metal_* fields: arm C on Metal through the loop, generation 5, one-shot fused forward; cpu_* fields: the HF torch arm C of logs/qwen_e0/C)"}
    receipt["near_tie_definition"] = f"frozen generation-5 census metal_margin < {NEAR_TIE_MARGIN} logits (arm C on Metal through the loop; the cpu_* fields of that file are the HF torch arm C, not the mlx loop)"
    if SMOKE:
        rolls = rolls[:1]
        rolls[0]["gen"] = rolls[0]["gen"][:8]
    pos_path = os.path.join(OUTDIR, "census_positions.jsonl")
    if os.path.exists(pos_path):
        raise SystemExit(f"REFUSING: {pos_path} exists")
    os.makedirs(OUTDIR, exist_ok=True)
    pf = open(pos_path, "w")
    fi, per_prompt, mism, near, state_worst = 0, [], [], [], []
    v_live = len(eng.tokenizer)
    agree = {"LOOP": 0, "KERNEL": 0}
    full_max = []
    top16_max = []
    for pi, r in enumerate(rolls):
        ids, p_len, n_gen = r["prompt"] + r["gen"], len(r["prompt"]), len(r["gen"])
        order = ["LOOP", "KERNEL"] if pi % 2 == 0 else ["KERNEL", "LOOP"]
        got = {}
        for arm in order:
            lg, cache, wall, sel = full_head_rows(eng, ids, p_len, n_gen, ARMS[arm])
            record_dispatch(receipt, sel, f"census {r['id']} {arm}")
            got[arm] = (lg, cache, wall)
        L, K = got["LOOP"][0], got["KERNEL"][0]
        if not (np.isfinite(L).all() and np.isfinite(K).all()):
            raise SystemExit(f"REFUSING: non-finite logits on {r['id']}")
        serrs = state_errors(got["KERNEL"][1], got["LOOP"][1], eng.model)
        sw = worst(serrs)
        sw_conv = worst(serrs, "conv")
        state_worst.append({"prompt_id": r["id"], "state": sw, "conv": sw_conv, "per_layer_state_max_abs_over_rms": [e["state"]["max_abs_over_rms"] for e in serrs]})
        ident = 0
        for s in range(n_gen):
            fr = frozen[fi] if fi < len(frozen) else None
            if fr is not None and (fr["prompt_id"] != r["id"] or fr["step"] != s or fr["teacher_token"] != r["gen"][s]):
                raise SystemExit(f"REFUSING: frozen census row {fi} misaligned with {r['id']} step {s}")
            zl, zk = L[s].astype(np.float64), K[s].astype(np.float64)
            al, ml = top1_margin(zl)
            ak, mk = top1_margin(zk)
            top_ids = np.argsort(-zl)[:16]
            d16 = float(np.abs(zk[top_ids] - zl[top_ids]).max())
            dfull = float(np.abs(zk - zl).max())
            row = {"prompt_id": r["id"], "step": s, "teacher_token": r["gen"][s], "loop_argmax": al, "kernel_argmax": ak, "identical": al == ak,
                   "loop_margin": ml, "kernel_margin": mk, "top16_max_abs_diff": d16, "full_vocab_max_abs_diff": dfull,
                   "loop_agree": al == r["gen"][s], "kernel_agree": ak == r["gen"][s],
                   "frozen_metal_argmax": None if fr is None else fr["metal_argmax"], "frozen_metal_margin": None if fr is None else fr["metal_margin"],
                   "frozen_cpu_argmax": None if fr is None else fr["cpu_argmax"], "frozen_cpu_margin": None if fr is None else fr["cpu_margin"],
                   "near_tie": bool(fr is not None and fr["metal_margin"] < NEAR_TIE_MARGIN)}
            pf.write(json.dumps(row) + "\n")
            ident += int(row["identical"])
            agree["LOOP"] += int(row["loop_agree"])
            agree["KERNEL"] += int(row["kernel_agree"])
            full_max.append(dfull)
            top16_max.append(d16)
            if not row["identical"]:
                mism.append({k: row[k] for k in ("prompt_id", "step", "loop_argmax", "kernel_argmax", "loop_margin", "kernel_margin", "teacher_token", "near_tie")})
            if row["near_tie"]:
                near.append({k: row[k] for k in ("prompt_id", "step", "loop_argmax", "kernel_argmax", "identical", "loop_margin", "kernel_margin", "frozen_metal_margin", "teacher_token")})
            fi += 1
        per_prompt.append({"prompt_id": r["id"], "category": r["category"], "tokens": len(ids), "positions": n_gen, "identical": ident, "order": order,
                           "wall_s": {a: got[a][2] for a in ARMS}, "state_max_abs_over_rms": sw["max_abs_over_rms"],
                           "loop_vs_frozen_metal_identical": sum(1 for s in range(n_gen) if frozen[fi - n_gen + s]["metal_argmax"] == top1_margin(L[s].astype(np.float64))[0]) if fi <= len(frozen) else None})
        print(f"[gdn] census {r['id']}: {n_gen} positions, identical {ident}/{n_gen}, full-vocab max abs {max(full_max[-n_gen:]):.2e}, state {sw['max_abs_over_rms']:.2e}, "
              f"wall loop {got['LOOP'][2]:.1f}s kernel {got['KERNEL'][2]:.1f}s", flush=True)
        del got, L, K
    pf.close()
    n = fi
    if not SMOKE and n != len(frozen):
        raise SystemExit(f"REFUSING: {n} positions, frozen census has {len(frozen)}")
    receipt["positions_path"] = pos_path
    receipt["positions_sha256"] = fsha(pos_path)
    receipt["summary"] = {"n_positions": n, "argmax_identical": n - len(mism), "argmax_mismatches": mism, "n_mismatches": len(mism),
                          "near_ties": near, "n_near_ties": len(near), "near_tie_flips": sum(1 for x in near if not x["identical"]),
                          "teacher_agree": {a: agree[a] / n for a in ARMS}, "teacher_agree_counts": agree,
                          "full_vocab_max_abs_diff": {"max": max(full_max), "p99": float(np.percentile(full_max, 99)), "median": float(np.median(full_max))},
                          "top16_max_abs_diff": {"max": max(top16_max), "p99": float(np.percentile(top16_max, 99)), "median": float(np.median(top16_max))},
                          "state_max_abs_over_rms": max(x["state"]["max_abs_over_rms"] for x in state_worst), "state_max_rel_on_large": max(x["state"]["max_rel_on_large"] for x in state_worst),
                          "conv_state_max_abs_over_rms": max(x["conv"]["max_abs_over_rms"] for x in state_worst),
                          "all_states_finite": all(x["state"]["all_finite"] for x in state_worst),
                          "loop_vs_frozen_metal_identical": sum(p["loop_vs_frozen_metal_identical"] or 0 for p in per_prompt),
                          "wall_s": {a: sum(p["wall_s"][a] for p in per_prompt) for a in ARMS}}
    receipt["per_prompt"] = per_prompt
    receipt["state_errors"] = state_worst
    receipt["nonfinite_arrays"] = sum(1 for x in state_worst for k in ("state", "conv") if not x[k]["all_finite"])      # counted per (prompt, kind) over all 48 layers; the logits are refused non-finite above
    print(f"[gdn] census: identical {n - len(mism)}/{n}, near-tie flips {receipt['summary']['near_tie_flips']}/{len(near)}, state max {receipt['summary']['state_max_abs_over_rms']:.2e}", flush=True)


# ---------------------------------------------------------------- decode

SNAP_EVERY = 32 if not SMOKE else 8


def greedy_run(eng, prompt, n_steps, path, feed=None, snaps=None):
    """Prefill prompt (serving path), then n_steps greedy decode (or teacher-forced on feed); per step the argmax,
    margin and wall; every SNAP_EVERY steps the recurrent states of the LAYER_ORDINALS DeltaNet layers are copied into
    snaps[step] (a dict) when given; returns (rows, per-step float32 logits [n_steps, v_live], cache, selector, prefill_s)."""
    import mlx.core as mx
    from llmopt.lab import qprefill
    v_live = len(eng.tokenizer)
    logits = np.zeros((n_steps, v_live), dtype=np.float32)
    rows = []
    layer_ids = [i for i, _ in dnet_layers(eng.model)]
    snap_layers = [layer_ids[o] for o in LAYER_ORDINALS]
    with selected(path) as sel:
        cache = eng.model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(eng.model, prompt, cache, chunk=eng.cfg.chunk, head="final", matmul=eng.cfg.matmul)
        mx.eval(lg)
        t_pre = time.perf_counter() - t0
        row = lg[0, -1]
        for s in range(n_steps):
            z = np.asarray(row, dtype=np.float32)[:v_live]
            logits[s] = z
            am, mg = top1_margin(z.astype(np.float64))
            rows.append({"step": s, "argmax": am, "margin": mg})
            nxt = am if feed is None else int(feed[s])
            rows[-1]["fed"] = nxt
            ts = time.perf_counter()
            row = eng.model(mx.array([[nxt]]), cache=cache)[0, -1]
            mx.eval(row)
            rows[-1]["step_s"] = time.perf_counter() - ts
            if snaps is not None and (s + 1) % SNAP_EVERY == 0:
                snaps[s + 1] = {i: np.asarray(cache[i].cache[1], dtype=np.float32) for i in snap_layers}
    return rows, logits, cache, sel, t_pre


def stage_decode(eng, receipt):
    from llmopt.lab import qgdn
    rolls, tinfo = teacher_rollouts(eng)
    receipt["teacher"] = tinfo
    n_steps = LONG_STEPS
    out = []
    for r in rolls[:2 if not SMOKE else 1]:
        prompt = r["prompt"]
        res = {"prompt_id": r["id"], "prompt_tokens": len(prompt), "steps": n_steps}
        snaps_l, snaps_f = {}, {}
        rows_l, lg_l, cache_l, sel, pre_l = greedy_run(eng, prompt, n_steps, "loop", snaps=snaps_l)
        record_dispatch(receipt, sel, f"decode {r['id']} LOOP free-run")
        rows_k, lg_k, cache_k, sel, pre_k = greedy_run(eng, prompt, n_steps, "kernel")
        record_dispatch(receipt, sel, f"decode {r['id']} KERNEL free-run")
        rows_k2, lg_k2, cache_k2, sel, _ = greedy_run(eng, prompt, n_steps, "kernel")
        record_dispatch(receipt, sel, f"decode {r['id']} KERNEL free-run repeat")
        sl, sk, sk2 = [x["argmax"] for x in rows_l], [x["argmax"] for x in rows_k], [x["argmax"] for x in rows_k2]
        div = next((i for i in range(n_steps) if sl[i] != sk[i]), None)
        res["free_run"] = {"streams_equal": sl == sk, "first_divergence": div, "n_equal_prefix": n_steps if div is None else div,
                           "margins_at_divergence": None if div is None else {"loop": rows_l[div]["margin"], "kernel": rows_k[div]["margin"], "loop_argmax": sl[div], "kernel_argmax": sk[div]},
                           "kernel_repeat_equal": sk == sk2, "kernel_repeat_logits_max_abs": float(np.abs(lg_k2 - lg_k).max()),
                           "prefill_s": {"LOOP": pre_l, "KERNEL": pre_k}, "step_median_s": {"LOOP": statistics.median(x["step_s"] for x in rows_l), "KERNEL": statistics.median(x["step_s"] for x in rows_k)},
                           "loop_text": eng.tokenizer.decode(sl[:48]), "kernel_text": eng.tokenizer.decode(sk[:48])}
        # accumulated error: the loop's stream fed teacher-forced through the kernel arm
        rows_f, lg_f, cache_f, sel, _ = greedy_run(eng, prompt, n_steps, "kernel", feed=sl, snaps=snaps_f)
        record_dispatch(receipt, sel, f"decode {r['id']} KERNEL teacher-forced on the loop stream")
        curve = [float(np.abs(lg_f[s] - lg_l[s]).max()) for s in range(n_steps)]
        res["margins"] = {"loop_free_run": [x["margin"] for x in rows_l], "kernel_free_run": [x["margin"] for x in rows_k]}
        # the kernel's state (teacher-forced on the loop's tokens) against the loop's at the six ordinals, every SNAP_EVERY steps: the accumulated decode error
        res["state_by_step"] = [{"step": st, "per_ordinal_max_abs_over_rms": {str(o): qgdn.normalized_errors(snaps_f[st][i], snaps_l[st][i])["max_abs_over_rms"] for o, i in zip(LAYER_ORDINALS, list(snaps_l[st]))}}
                                for st in sorted(snaps_l) if st in snaps_f]
        res["teacher_forced"] = {"argmax_identical": sum(1 for s in range(n_steps) if int(lg_f[s].argmax()) == sl[s]), "max_abs_logit_diff_by_step": curve,
                                 "max_abs_logit_diff": max(curve), "max_abs_logit_diff_first_8": max(curve[:8]), "max_abs_logit_diff_last_8": max(curve[-8:]),
                                 "argmax_mismatches": [{"step": s, "loop_argmax": sl[s], "kernel_argmax": int(lg_f[s].argmax()), "loop_margin": rows_l[s]["margin"]} for s in range(n_steps) if int(lg_f[s].argmax()) != sl[s]]}
        serrs = state_errors(cache_f, cache_l, eng.model)
        res["final_state"] = {"state": worst(serrs), "conv": worst(serrs, "conv"), "per_layer_state_max_abs_over_rms": [e["state"]["max_abs_over_rms"] for e in serrs],
                              "nonfinite_arrays": sum(1 for e in serrs for k in ("state", "conv") if not e[k]["finite"]) + sum(1 for st in res["state_by_step"] for v in st["per_ordinal_max_abs_over_rms"].values() if not np.isfinite(v))}
        out.append(res)
        print(f"[gdn] decode {r['id']}: streams equal {sl == sk} (first divergence {div}), kernel repeat equal {sk == sk2}, teacher-forced max |dlogit| {max(curve):.2e}, "
              f"final state {res['final_state']['state']['max_abs_over_rms']:.2e}, step loop {res['free_run']['step_median_s']['LOOP'] * 1e3:.0f} ms kernel {res['free_run']['step_median_s']['KERNEL'] * 1e3:.0f} ms", flush=True)
        del lg_l, lg_k, lg_k2, lg_f, cache_l, cache_k, cache_k2, cache_f
    receipt["prompts"] = out
    receipt["summary"] = {"n_prompts": len(out), "stream_mismatch_prompts": sum(1 for x in out if not x["free_run"]["streams_equal"]),
                          "kernel_repeat_unequal_prompts": sum(1 for x in out if not x["free_run"]["kernel_repeat_equal"]),
                          "teacher_forced_max_abs_logit_diff": max(x["teacher_forced"]["max_abs_logit_diff"] for x in out),
                          "teacher_forced_argmax_mismatches": sum(len(x["teacher_forced"]["argmax_mismatches"]) for x in out),
                          "final_state_max_abs_over_rms": max(x["final_state"]["state"]["max_abs_over_rms"] for x in out),
                          "all_finite": all(x["final_state"]["state"]["all_finite"] and x["final_state"]["conv"]["all_finite"] for x in out),
                          "nonfinite_arrays": sum(x["final_state"]["nonfinite_arrays"] for x in out),
                          "state_by_step_max": max((v for x in out for st in x["state_by_step"] for v in st["per_ordinal_max_abs_over_rms"].values()), default=None),
                          "step_median_s": {a: statistics.median(x["free_run"]["step_median_s"][a] for x in out) for a in ARMS}}


# ---------------------------------------------------------------- heldout

def run_prefill(eng, ids, path, decode=0, instrument=False):
    import mlx.core as mx
    from llmopt.lab import qmetal, qprefill
    mx.clear_cache()
    active_before = int(mx.get_active_memory())
    mx.reset_peak_memory()
    with selected(path) as sel:
        cache = eng.model.make_cache()
        t0 = time.perf_counter()
        lg = qprefill.prefill_chunked(eng.model, ids, cache, chunk=eng.cfg.chunk, head="final", matmul=eng.cfg.matmul)
        mx.eval(lg)
        wall = time.perf_counter() - t0
        ids_top, vals_top = topk(lg[0, -1])
        m = mem()
        m["active_before_bytes"] = active_before
        m["transient_bytes"] = m["peak_bytes"] - active_before
        row = {"tokens": len(ids), "path": path, "ttft_s": wall, "prefill_tok_per_s": len(ids) / wall, "argmax": ids_top[0], "top16_ids": ids_top, "top16_logits": vals_top, "memory": m}
        if decode:
            cur, out, walls = ids_top[0], [], []
            t1 = time.perf_counter()
            for _ in range(decode):
                out.append(cur)
                ts = time.perf_counter()
                lz = eng.model(mx.array([[cur]]), cache=cache)[0, -1]
                mx.eval(lz)
                walls.append(time.perf_counter() - ts)
                cur = int(lz.argmax())
            row["decode_ids"], row["decode_s"] = out, time.perf_counter() - t1
            row["end_to_end_s"] = wall + row["decode_s"]
            row["decode_step_median_s"] = statistics.median(walls[4:]) if len(walls) > 8 else statistics.median(walls)
            row["peak_after_decode_bytes"] = int(mx.get_peak_memory())
        del cache
        if instrument:
            mx.clear_cache()
            cache2 = eng.model.make_cache()
            body, lm = qprefill.split_body_head(eng.model)
            chunks = []
            t2 = time.perf_counter()
            for s in range(0, len(ids), eng.cfg.chunk):
                c0 = time.perf_counter()
                with qmetal.gemm_scope(eng.cfg.matmul == "gemm"):
                    h = body(mx.array([ids[s:s + eng.cfg.chunk]]), cache2)
                    lg2 = lm(h[:, -1:, :])
                    t_issue = time.perf_counter() - c0
                    mx.eval(lg2, *qprefill.cache_arrays(cache2))
                chunks.append({"start": s, "issue_s": t_issue, "complete_s": time.perf_counter() - c0})
                del h
            row["instrumented"] = {"wall_s": time.perf_counter() - t2, "issue_s_total": sum(c["issue_s"] for c in chunks), "complete_s_total": sum(c["complete_s"] for c in chunks),
                                   "gpu_side_s_total": sum(c["complete_s"] - c["issue_s"] for c in chunks), "argmax_equal": int(lg2[0, -1].argmax()) == ids_top[0]}
            del cache2
    row["dispatch"] = sel.receipt()
    sel.count.assert_exclusive()
    mx.clear_cache()
    return row


def top16_diff(x, ref):
    a, b = dict(zip(x["top16_ids"], x["top16_logits"])), dict(zip(ref["top16_ids"], ref["top16_logits"]))
    common = set(a) & set(b)
    return max(abs(a[i] - b[i]) for i in common) if common else 99.0


def preflight_ok(extra_gb):
    import mlx.core as mx
    active = mx.get_active_memory() / 1e9
    return {"active_gb": active, "needed_gb": active + extra_gb, "limit_gb": 30.0, "ok": active + extra_gb < 30.0}


def stage_heldout(eng, receipt):
    if SMOKE:
        prompts = {256: fixed_prompt(eng, "gdn-heldout-smoke-256", 256)}
    else:
        frozen = json.load(open("scratch/evalsets/qwen_prefill_heldout_prompts.json"))["prompts"]
        prompts = {t: frozen[str(t)]["ids"] for t in TARGETS}
        for t in TARGETS:
            assert sha_ids(prompts[t]) == frozen[str(t)]["sha256"], f"held-out prompt {t} does not match its frozen sha"
    receipt["prompts"] = {str(t): {"tokens": len(p), "sha256": sha_ids(p)} for t, p in prompts.items()}
    reps = 1 if SMOKE else 3
    receipt["runs"], receipt["warmups"], receipt["instrumented"] = [], [], []
    for t in TARGETS:
        pf = preflight_ok(3.0)
        receipt.setdefault("preflight", {})[str(t)] = pf
        if not pf["ok"]:
            continue
        for arm in ARMS:
            w = run_prefill(eng, prompts[t], ARMS[arm])
            receipt["warmups"].append({"tokens": t, "arm": arm, "ttft_s": w["ttft_s"]})
        for _ in range(reps):
            for arm in ("LOOP", "KERNEL", "KERNEL", "LOOP"):
                r = run_prefill(eng, prompts[t], ARMS[arm], decode=DECODE_STEPS)
                r["arm"], r["target"] = arm, t
                receipt["runs"].append(r)
                print(f"[gdn] heldout {t} {arm:6s} ttft {r['ttft_s']:.2f}s e2e {r['end_to_end_s']:.2f}s peak {r['memory']['peak_bytes'] / 1e9:.2f} GB step {r['decode_step_median_s'] * 1e3:.0f} ms", flush=True)
        for arm in ARMS:
            r = run_prefill(eng, prompts[t], ARMS[arm], instrument=True)
            r["arm"], r["target"] = arm, t
            receipt["instrumented"].append(r)
    if not SMOKE:
        pf = preflight_ok(4.0)
        receipt["preflight"]["4096"] = pf
        if pf["ok"]:
            p4 = fixed_prompt(eng, "prefill-wall-dev-1-4096", 4096)       # the development 4,096 prompt of QWEN-PREFILL-WALL-0 (not held out), descriptive
            receipt["prompts"]["4096"] = {"tokens": len(p4), "sha256": sha_ids(p4), "role": "development prompt, descriptive"}
            receipt["runs_4096"] = []
            for arm in ("LOOP", "KERNEL", "KERNEL", "LOOP"):
                r = run_prefill(eng, p4, ARMS[arm], decode=DECODE_STEPS)
                r["arm"] = arm
                receipt["runs_4096"].append(r)
                print(f"[gdn] heldout 4096 {arm:6s} ttft {r['ttft_s']:.2f}s peak {r['memory']['peak_bytes'] / 1e9:.2f} GB", flush=True)
    # long generations: ABBA once per length
    receipt["long_runs"] = []
    for t in ([1024, 1024, 2048] if not SMOKE else [256]):       # two ABBA blocks at 1,024 (bars 9 and 10, 4 against 4), one at 2,048 (descriptive)
        if str(t) not in receipt["prompts"] or not receipt["preflight"].get(str(t), {}).get("ok", True):
            continue
        for arm in ("LOOP", "KERNEL", "KERNEL", "LOOP"):
            r = run_prefill(eng, prompts[t], ARMS[arm], decode=LONG_STEPS)
            r["arm"], r["target"] = arm, t
            receipt["long_runs"].append(r)
            print(f"[gdn] long {t} {arm:6s} e2e {r['end_to_end_s']:.1f}s decode {r['decode_s']:.1f}s step {r['decode_step_median_s'] * 1e3:.0f} ms", flush=True)
    summ = {}
    for t in TARGETS:
        rows = [r for r in receipt["runs"] if r["target"] == t and r["arm"] == "KERNEL"]
        base = [r for r in receipt["runs"] if r["target"] == t and r["arm"] == "LOOP"]
        if not rows or not base:
            continue
        ref = base[0]
        med = statistics.median
        summ[str(t)] = {"n": len(rows), "ttft_median_s": {"LOOP": med(x["ttft_s"] for x in base), "KERNEL": med(x["ttft_s"] for x in rows)},
                        "ttft_ratio": med(x["ttft_s"] for x in rows) / med(x["ttft_s"] for x in base),
                        "loop_spread_pct": 100 * (max(x["ttft_s"] for x in base) - min(x["ttft_s"] for x in base)) / med(x["ttft_s"] for x in base),
                        "kernel_spread_pct": 100 * (max(x["ttft_s"] for x in rows) - min(x["ttft_s"] for x in rows)) / med(x["ttft_s"] for x in rows),
                        "loop_decode16_spread_pct": 100 * (max(x["decode_s"] for x in base) - min(x["decode_s"] for x in base)) / med(x["decode_s"] for x in base),
                        "kernel_decode16_spread_pct": 100 * (max(x["decode_s"] for x in rows) - min(x["decode_s"] for x in rows)) / med(x["decode_s"] for x in rows),
                        "e2e16_ratio": med(x["end_to_end_s"] for x in rows) / med(x["end_to_end_s"] for x in base),
                        "decode16_s_ratio": med(x["decode_s"] for x in rows) / med(x["decode_s"] for x in base),
                        "decode_step_ratio": med(x["decode_step_median_s"] for x in rows) / med(x["decode_step_median_s"] for x in base),
                        "parity_mismatches": sum(1 for x in rows if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                        "loop_self_mismatches": sum(1 for x in base if x["argmax"] != ref["argmax"] or x["decode_ids"] != ref["decode_ids"]),
                        "top16_max_abs_diff": max(top16_diff(x, ref) for x in rows),
                        "peak_ratio": max(x["memory"]["peak_bytes"] for x in rows) / max(x["memory"]["peak_bytes"] for x in base),
                        "peak_after_decode_ratio": max(x["peak_after_decode_bytes"] for x in rows) / max(x["peak_after_decode_bytes"] for x in base),
                        "footprint_ratio": max(x["memory"]["footprint_bytes"] for x in rows) / max(x["memory"]["footprint_bytes"] for x in base),
                        "transient_ratio": max(x["memory"]["transient_bytes"] for x in rows) / max(1, max(x["memory"]["transient_bytes"] for x in base))}
        ins = {r["arm"]: r["instrumented"] for r in receipt["instrumented"] if r["target"] == t}
        if len(ins) == 2:
            summ[str(t)]["instrumented"] = {a: {"issue_s": ins[a]["issue_s_total"], "complete_s": ins[a]["complete_s_total"], "gpu_side_s": ins[a]["gpu_side_s_total"]} for a in ins}
    for t in sorted({r["target"] for r in receipt["long_runs"]}):
        rows = [r for r in receipt["long_runs"] if r["target"] == t and r["arm"] == "KERNEL"]
        base = [r for r in receipt["long_runs"] if r["target"] == t and r["arm"] == "LOOP"]
        med = statistics.median
        spread = lambda xs, k: 100 * (max(x[k] for x in xs) - min(x[k] for x in xs)) / med(x[k] for x in xs)
        summ[f"long:{t}"] = {"steps": LONG_STEPS, "n": len(rows), "e2e_ratio": med(x["end_to_end_s"] for x in rows) / med(x["end_to_end_s"] for x in base),
                             "spread_pct": {"loop_e2e": spread(base, "end_to_end_s"), "kernel_e2e": spread(rows, "end_to_end_s"), "loop_step": spread(base, "decode_step_median_s"), "kernel_step": spread(rows, "decode_step_median_s")},
                             "e2e_s": {"LOOP": med(x["end_to_end_s"] for x in base), "KERNEL": med(x["end_to_end_s"] for x in rows)},
                             "decode_step_ratio": med(x["decode_step_median_s"] for x in rows) / med(x["decode_step_median_s"] for x in base),
                             "decode_step_ms": {"LOOP": 1e3 * med(x["decode_step_median_s"] for x in base), "KERNEL": 1e3 * med(x["decode_step_median_s"] for x in rows)},
                             "streams_equal_to_loop": sum(1 for x in rows if x["decode_ids"] == base[0]["decode_ids"]), "loop_self_equal": sum(1 for x in base if x["decode_ids"] == base[0]["decode_ids"]),
                             "peak_after_decode_ratio": max(x["peak_after_decode_bytes"] for x in rows) / max(x["peak_after_decode_bytes"] for x in base)}
    if receipt.get("runs_4096"):
        rows = [r for r in receipt["runs_4096"] if r["arm"] == "KERNEL"]
        base = [r for r in receipt["runs_4096"] if r["arm"] == "LOOP"]
        summ["4096"] = {"ttft_ratio": statistics.median(x["ttft_s"] for x in rows) / statistics.median(x["ttft_s"] for x in base), "ttft_median_s": {"LOOP": statistics.median(x["ttft_s"] for x in base), "KERNEL": statistics.median(x["ttft_s"] for x in rows)},
                        "peak_ratio": max(x["memory"]["peak_bytes"] for x in rows) / max(x["memory"]["peak_bytes"] for x in base),
                        "parity_mismatches": sum(1 for x in rows if x["argmax"] != base[0]["argmax"] or x["decode_ids"] != base[0]["decode_ids"]), "role": "development prompt, descriptive"}
    receipt["summary"] = summ


# ---------------------------------------------------------------- cachehist

def stage_cachehist(eng, receipt):
    from llmopt.lab import qserve
    max_tokens = 32
    receipt["conversations"] = CONVERSATIONS
    receipt["max_tokens"] = max_tokens
    turns = {}
    for arm in ARMS:
        for mode in ("incremental", "replay"):
            with selected(ARMS[arm]) as sel:
                for ci, conv in enumerate(CONVERSATIONS):
                    s = qserve.Session(eng, mode=mode, enable_thinking=False)
                    for ti, user in enumerate(conv):
                        res = s.ask(user, max_tokens=max_tokens)
                        turns[(arm, mode, ci, ti)] = {"generated_ids": list(res.generated_ids), "mode": res.mode, "note": getattr(res, "note", None), "stop_reason": res.stop_reason,
                                                      "n_prompt": res.n_prompt, "cache_tokens": res.cache_tokens, "cache_offset_measured": res.cache_offset_measured,
                                                      "first_token_s": res.timings["first_token_s"], "decode_s": res.timings["decode_s"], "text": res.text[:120]}
                        print(f"[gdn] cachehist {arm} {mode} c{ci} t{ti}: {res.n_generated} tokens, mode {res.mode}, first token {res.timings['first_token_s']:.2f}s", flush=True)
                    del s
            record_dispatch(receipt, sel, f"cachehist {arm} {mode}")
    rows = []
    for ci, conv in enumerate(CONVERSATIONS):
        for ti in range(len(conv)):
            k_inc, k_rep, l_inc, l_rep = (turns[("KERNEL", "incremental", ci, ti)], turns[("KERNEL", "replay", ci, ti)], turns[("LOOP", "incremental", ci, ti)], turns[("LOOP", "replay", ci, ti)])
            rows.append({"conversation": ci, "turn": ti, "kernel_incremental_equals_kernel_replay": k_inc["generated_ids"] == k_rep["generated_ids"],
                         "kernel_incremental_equals_loop_incremental": k_inc["generated_ids"] == l_inc["generated_ids"],
                         "loop_incremental_equals_loop_replay": l_inc["generated_ids"] == l_rep["generated_ids"],
                         "modes": {"KERNEL": k_inc["mode"], "LOOP": l_inc["mode"]}, "notes": {"KERNEL": k_inc["note"], "LOOP": l_inc["note"]},
                         "n_generated": {"KERNEL": len(k_inc["generated_ids"]), "LOOP": len(l_inc["generated_ids"])},
                         "first_token_s": {"KERNEL_inc": k_inc["first_token_s"], "LOOP_inc": l_inc["first_token_s"], "KERNEL_replay": k_rep["first_token_s"], "LOOP_replay": l_rep["first_token_s"]},
                         "kernel_text": k_inc["text"], "loop_text": l_inc["text"]})
    receipt["turns"] = rows
    receipt["summary"] = {"n_turns": len(rows), "unequal_turns": sum(1 for r in rows if not (r["kernel_incremental_equals_kernel_replay"] and r["kernel_incremental_equals_loop_incremental"])),
                          "kernel_inc_vs_replay_unequal": sum(1 for r in rows if not r["kernel_incremental_equals_kernel_replay"]),
                          "kernel_vs_loop_unequal": sum(1 for r in rows if not r["kernel_incremental_equals_loop_incremental"]),
                          "loop_inc_vs_replay_unequal": sum(1 for r in rows if not r["loop_incremental_equals_loop_replay"]),
                          "incremental_reused_turns": {a: sum(1 for r in rows if r["modes"][a] == "incremental") for a in ARMS},
                          "fallback_turns": {a: sum(1 for r in rows if r["modes"][a] == "replay-fallback") for a in ARMS}}
    print(f"[gdn] cachehist: unequal turns {receipt['summary']['unequal_turns']}/{len(rows)}, reused {receipt['summary']['incremental_reused_turns']}", flush=True)


# ---------------------------------------------------------------- main

def main():
    from llmopt.lab import qgdn, qserve
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    t_proc = time.perf_counter()
    vendor = qgdn.check_vendor()
    eng = qserve.Engine()
    if eng.model.training is not True:
        raise SystemExit("REFUSING: the tower is expected in its default (training) mode; this study never flips it")
    receipt = {"commit": head, "tree_dirty": dirty, "smoke": SMOKE, "stage": STAGE, "cold_load_s": eng.load_s, "driver": os.path.relpath(__file__), "driver_sha256": fsha(__file__),
               "selector_sha256": fsha(qgdn.__file__), "layer_ordinals": LAYER_ORDINALS,
               "vendor": vendor, "model_training_flag": bool(eng.model.training),
               "engine": {"kernel_version": eng.info.get("kernel_version"), "chain_sha256": eng.info.get("chain_sha256"), "cache_limit_bytes": eng.cfg.cache_limit_bytes, "chunk": eng.cfg.chunk,
                          "matmul": eng.cfg.matmul, "tokenizer_files_sha256": getattr(eng, "tokenizer_files_sha256", {}), "families": eng.info.get("families"), "v_live": len(eng.tokenizer)}}
    {"layers": stage_layers, "census": stage_census, "decode": stage_decode, "heldout": stage_heldout, "cachehist": stage_cachehist}[STAGE](eng, receipt)
    receipt["wall_s"] = time.perf_counter() - t_proc
    receipt["process_peak_rss_bytes"] = mem().get("rss_peak_bytes")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(receipt, open(OUT, "w"), indent=1)
    print(f"[gdn] {STAGE} wrote {OUT} in {receipt['wall_s']:.0f}s", flush=True)


if __name__ == "__main__":
    main()
