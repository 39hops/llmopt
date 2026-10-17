"""Reference definitions for greedy speculative decoding and the quantities
the E0 census reports. Pure Python and numpy; no model, no device.

Three registers are kept apart and named in the code:

* identities that hold exactly (greedy exactness, tail-sum, acceptance
  definitions); the Lean file scratch/leancheck/SpecDecoding.lean states
  the same facts over abstract types;
* numerical estimates that depend on floating point (shift-aligned sup
  error, margins read from stored logits);
* the empirical accepted-prefix replay, which is an estimator over two
  recorded argmax streams, and the geometric closed form, which is kept
  only as a labelled model assumption.

Greedy versus sampled speculation are different acceptance laws:
greedy accepts a draft token iff it equals the teacher's argmax at the
same teacher-greedy context; exact sampling accepts with probability
sum_i min(p_i, q_i) = 1 - TV(p, q). They are never mixed here.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

GEOMETRIC_MODEL_LABEL = ("MODEL ASSUMPTION: independent per-position acceptance "
                         "with a constant rate; not a measurement")
REPLAY_LABEL = "EMPIRICAL REPLAY"


# ----------------------------------------------------------------- greedy law

def greedy(T: Callable[[list], object], prompt: Sequence, n: int, eos=None) -> list:
    """n tokens of plain greedy decoding under the deterministic next-token
    function T (stops early after emitting eos)."""
    ctx = list(prompt)
    out: list = []
    while len(out) < n:
        t = T(ctx)
        out.append(t)
        ctx.append(t)
        if eos is not None and t == eos:
            break
    return out


def greedy_speculative(T: Callable[[list], object],
                       propose: Callable[[list, int], list],
                       prompt: Sequence, n: int, k: int, eos=None):
    """Greedy speculative decoding: the proposer offers up to k tokens, the
    teacher is evaluated at every drafted context, the longest agreeing
    prefix is accepted and the teacher's own token at the first
    disagreement (or after the whole block) is emitted. Returns the
    emitted sequence and per-block statistics. The output equals
    greedy(T, prompt, n, eos) for any proposer (theorem B in the Lean
    file); the test suite checks it on random proposers."""
    ctx = list(prompt)
    out: list = []
    stats = {"blocks": 0, "accepted_per_block": [], "emitted_per_block": [],
             "drafted_per_block": []}
    while len(out) < n:
        draft = list(propose(list(ctx), k))[:k]
        accepted = 0
        block: list = []
        stop = False
        for d in draft:
            t = T(ctx + block)
            if t != d:
                break
            accepted += 1
            block.append(t)
            if eos is not None and t == eos:
                stop = True
                break
        if not stop:
            block.append(T(ctx + block))
        # truncate to the budget and to eos
        room = n - len(out)
        block = block[:room]
        if eos is not None and eos in block:
            block = block[:block.index(eos) + 1]
        out.extend(block)
        ctx.extend(block)
        stats["blocks"] += 1
        stats["accepted_per_block"].append(min(accepted, len(block)))
        stats["emitted_per_block"].append(len(block))
        stats["drafted_per_block"].append(len(draft))
        if eos is not None and block and block[-1] == eos:
            break
    return out, stats


# ------------------------------------------------ expected tokens per block

def emitted_from_tail(tail: Sequence[float]) -> float:
    """E[emitted] = sum_{j=0}^{k} P(A >= j), where tail[j] = P(A >= j) and
    tail[0] = 1. Holds for any acceptance distribution (no independence)."""
    return float(sum(tail))


def geometric_emitted(alpha: float, k: int) -> float:
    """(1 - alpha^(k+1)) / (1 - alpha): expected emitted tokens per block
    under GEOMETRIC_MODEL_LABEL. Equals k + 1 at alpha = 1."""
    if alpha == 1.0:
        return float(k + 1)
    return (1.0 - alpha ** (k + 1)) / (1.0 - alpha)


def speedup(alpha: float, k: int, c_draft: float, c_verify: float,
            c_teacher: float = 1.0) -> float:
    """Throughput ratio of speculative over plain decoding under the
    geometric model: E * c_teacher / (k * c_draft + c_verify). Speculation
    wins iff this exceeds 1. Monotone increasing in alpha."""
    return geometric_emitted(alpha, k) * c_teacher / (k * c_draft + c_verify)


# ------------------------------------------------- empirical prefix replay

def replay_blocks(teacher_argmax: Sequence, draft_argmax: Sequence, k: int) -> dict:
    """Accepted-prefix replay over two argmax streams recorded at the SAME
    teacher-greedy contexts (position i of both streams is the token
    predicted after the first i teacher-greedy tokens). Blocks of k
    drafts are walked in order; a block accepts the longest agreeing
    prefix and emits one more teacher token. The last block is truncated
    by the stream length. This is an estimator, not the geometric model:
    it keeps every dependence between neighbouring positions that the
    recorded streams contain."""
    if len(teacher_argmax) != len(draft_argmax):
        raise ValueError("streams must be aligned position by position")
    n = len(teacher_argmax)
    pos = 0
    acc: list[int] = []
    emit: list[int] = []
    drafted: list[int] = []
    rejected_blocks = 0
    while pos < n:
        kk = min(k, n - pos)
        a = 0
        while a < kk and draft_argmax[pos + a] == teacher_argmax[pos + a]:
            a += 1
        rejected = a < kk
        e = a + 1 if rejected else a + (1 if pos + a < n else 0)
        e = min(e, n - pos)
        if e == 0:  # cannot happen for n > pos, kept for safety
            e = 1
        acc.append(min(a, e))
        emit.append(e)
        drafted.append(kk)
        rejected_blocks += int(rejected)
        pos += e
    return {"model_label": REPLAY_LABEL, "k": k, "n_positions": n,
            "accepted_per_block": acc, "emitted_per_block": emit,
            "drafted_per_block": drafted, "blocks": len(emit),
            "rejected_blocks": rejected_blocks,
            "mean_emitted": float(np.mean(emit)) if emit else 0.0,
            "mean_accepted": float(np.mean(acc)) if acc else 0.0}


def replay_cost(replay: dict, c_draft: float, c_verify: Callable[[int], float],
                c_commit: float = 0.0, c_rollback: float = 0.0,
                c_overhead_per_block: float = 0.0) -> dict:
    """Charge every component of a speculative round to the replay:
    drafting (per drafted token), verification (per block, as a function
    of the block's drafted length), commit (per block), rollback (per
    block that rejected a draft), and a fixed per-block overhead. Returns
    the components, the total, and tokens per cost unit. Costs are in
    whatever unit the caller measured (seconds, or plain-teacher tokens)."""
    draft = c_draft * sum(replay["drafted_per_block"])
    verify = float(sum(c_verify(kk) for kk in replay["drafted_per_block"]))
    commit = c_commit * replay["blocks"]
    rollback = c_rollback * replay["rejected_blocks"]
    overhead = c_overhead_per_block * replay["blocks"]
    total = draft + verify + commit + rollback + overhead
    tokens = sum(replay["emitted_per_block"])
    return {"draft": draft, "verify": verify, "commit": commit,
            "rollback": rollback, "overhead": overhead, "total": total,
            "tokens": tokens,
            "tokens_per_unit": tokens / total if total > 0 else float("inf")}


# ------------------------------------------------ margins and perturbations

def top1_and_margin(z: np.ndarray) -> tuple[int, float]:
    """(argmax index, top-1 minus top-2). First index wins ties, matching
    numpy and torch argmax on CPU; margin 0 means a tie."""
    z = np.asarray(z, dtype=np.float64)
    i = int(z.argmax())
    rest = np.delete(z, i)
    return i, float(z[i] - rest.max()) if rest.size else float("inf")


def shift_aligned_sup_error(z: np.ndarray, zp: np.ndarray) -> float:
    """min over c of max_j |z_j + c - z'_j| = (max(d) - min(d)) / 2 with
    d = z' - z. Logits are defined up to a per-position shift, so this is
    the perturbation radius that the margin theorem should be fed."""
    d = np.asarray(zp, dtype=np.float64) - np.asarray(z, dtype=np.float64)
    return float((d.max() - d.min()) / 2.0)


def worst_case_perturbation(z: np.ndarray, eps: float) -> np.ndarray:
    """The tightness construction: lower the argmax by eps and raise the
    runner-up by eps. Flips the argmax whenever eps > margin / 2."""
    z = np.asarray(z, dtype=np.float64).copy()
    i = int(z.argmax())
    j = int(np.argsort(z)[-2])
    z[i] -= eps
    z[j] += eps
    return z


def softmax(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    e = np.exp(z - z.max())
    return e / e.sum()


def probability_margin(p: np.ndarray) -> float:
    """p_max minus the second largest probability (mu in the flip lemma)."""
    s = np.sort(np.asarray(p, dtype=np.float64))
    return float(s[-1] - s[-2])


# -------------------------------------------------- acceptance laws, TV, KL

def greedy_accept(p: np.ndarray, q: np.ndarray) -> bool:
    """Greedy law: accept iff the draft argmax equals the teacher argmax."""
    return int(np.asarray(p).argmax()) == int(np.asarray(q).argmax())


def total_variation(p: np.ndarray, q: np.ndarray) -> float:
    return float(0.5 * np.abs(np.asarray(p, dtype=np.float64) - np.asarray(q, dtype=np.float64)).sum())


def sampled_accept_prob(p: np.ndarray, q: np.ndarray) -> float:
    """Exact-sampling law (teacher p, draft q): acceptance probability is
    sum_i min(p_i, q_i) = 1 - TV(p, q)."""
    return float(np.minimum(np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)).sum())


def kl(p: np.ndarray, q: np.ndarray) -> float:
    """Forward KL(p || q) in nats over the support of p."""
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    m = p > 0
    return float((p[m] * (np.log(p[m]) - np.log(q[m]))).sum())


# ------------------------------------------------ per-position census row

def position_stats(z_teacher: np.ndarray, z_draft: np.ndarray, teacher_token: int,
                   topk: int = 16) -> dict:
    """One census row for a teacher-greedy position: both logit vectors are
    over the same live vocabulary. Greedy agreement, logit and probability
    margins, raw and shift-aligned sup error, TV, KL(teacher || draft),
    the sampled-law acceptance 1 - TV, the teacher token's rank under the
    draft, the draft's top-k, and whether the margin theorem certifies
    agreement at this position: not ambiguous (teacher_ambiguity) and
    sup_err_shift < margin_lower / 2, where margin_lower discounts one
    fp16 ulp from the stored margin."""
    zt = np.asarray(z_teacher, dtype=np.float64)
    zd = np.asarray(z_draft, dtype=np.float64)
    if zt.shape != zd.shape:
        raise ValueError("teacher/draft logit width mismatch")
    t_arg, t_margin = top1_and_margin(zt)
    d_arg, d_margin = top1_and_margin(zd)
    pt, pd = softmax(zt), softmax(zd)
    tv = total_variation(pt, pd)
    order = np.argsort(-zd)
    rank = int(np.nonzero(order == teacher_token)[0][0])
    sup_shift = shift_aligned_sup_error(zt, zd)
    amb = teacher_ambiguity(np.asarray(z_teacher), teacher_token)
    return {
        "teacher_token": int(teacher_token),
        "teacher_argmax": t_arg,
        "teacher_margin_logit": t_margin,
        "teacher_margin_lower": amb["margin_lower"],
        "fp16_ulp_at_top": amb["fp16_ulp_at_top"],
        "ambiguous": amb["ambiguous"],
        "recorded_equals_fp16_argmax": amb["recorded_equals_fp16_argmax"],
        "teacher_margin_prob": probability_margin(pt),
        "draft_argmax": d_arg,
        "draft_margin_logit": d_margin,
        "agree": bool(d_arg == teacher_token),
        "sup_err_raw": float(np.abs(zd - zt).max()),
        "sup_err_shift": sup_shift,
        "l2_err_shift": float(np.linalg.norm((zd - zt) - np.median(zd - zt))),
        "tv": tv,
        "sampled_accept": 1.0 - tv,
        "kl_teacher_draft": kl(pt, pd),
        "teacher_token_rank_in_draft": rank,
        "draft_topk_ids": [int(i) for i in order[:topk]],
        "draft_topk_logits": [float(zd[i]) for i in order[:topk]],
        "margin_certified": bool((not amb["ambiguous"]) and sup_shift < amb["margin_lower"] / 2.0),
    }


# ------------------------------------------ fp16-stored teacher ambiguity

def teacher_ambiguity(z_teacher_fp16: np.ndarray, recorded_token: int) -> dict:
    """Numerical register. The saved teacher logits are fp16 copies of the
    fp32 values that produced the recorded greedy token. Each stored value
    is within half an fp16 ulp of the true one, so the true top-1 margin
    is at least (stored margin - one ulp at the top logit): that is
    `margin_lower`. A position is `ambiguous` when the recorded token is
    not the stored fp16 argmax, or when the stored margin is at most one
    ulp (the true ordering of the top two cannot be read from the file).
    Ambiguous positions stay in the agreement analysis and are excluded
    from margin certification only."""
    z16 = np.asarray(z_teacher_fp16)
    z = z16.astype(np.float64)
    i, m = top1_and_margin(z)
    top = z[i]
    ulp = float(np.spacing(np.float16(top))) if z16.dtype == np.float16 else float(np.spacing(np.float16(top)))
    return {"fp16_argmax": i, "recorded_equals_fp16_argmax": bool(i == int(recorded_token)),
            "margin_fp16": m, "fp16_ulp_at_top": ulp, "margin_lower": m - ulp,
            "ambiguous": bool(i != int(recorded_token) or m <= ulp)}
