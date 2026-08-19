"""X/K scoring core for the Qwen artifact tree (lab adoption).

The pure math of the MODEL-1/MODEL-2 scorer, adopted verbatim from
the frozen results-cited driver scratch/qwen_model1_score.py after
its third in-place surface extension (auditor S3 bank; adoption
2026-08-19). While both files coexist, every adopted symbol stays
CHARACTER-IDENTICAL between them (tests/test_lab_qscore_adoption.py)
— a fix lands in both in the same commit or in neither.

Adopted quantities (PRE-REG QWEN-MODEL1-TREE + -METRIC/-PINS/-KFENCE):

  mean_ce           X-side: mean CE, logits[:-1] v ids[1:], live
                    vocab, fp64 log-softmax, non-finite REFUSE.
  mean_forward_kl   K-side: mean KL(teacher || arm) on the P-1 rows.
  sensitivity_floor f_X/f_K: max |shift| under +-1ulp fp16 record
                    perturbation (numerical floor, never significance).
  teacher_margins_top1 / flip_table / margin_bin
                    margin-stratified top1 flips on the frozen
                    TREE-PINS edges; n < SMALL_N strata carry no
                    directional claim.

teacher_receipt_block is NEW (not adopted): it resolves the banked
receipt-field rename (auditor N2) for future drivers — record shas
are named *_record_sha (they hash the saved logits arrays), and the
input-file shas ride alongside as *_input_sha.
"""
from __future__ import annotations

import hashlib

import numpy as np

MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
SMALL_N = 30


def log_softmax(logits: np.ndarray) -> np.ndarray:
    """Row-wise log-softmax in fp64 with max subtraction (-METRIC 7)."""
    x = logits.astype(np.float64)
    x = x - x.max(axis=-1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=-1, keepdims=True))


def mean_ce(logits: np.ndarray, ids, v_live: int) -> float:
    """Mean CE over logits[:-1] v ids[1:], live vocab. NaN/inf REFUSE."""
    lg = logits[:-1, :v_live]
    if not np.isfinite(lg).all():
        raise SystemExit("REFUSING: non-finite logits in CE input")
    ls = log_softmax(lg)
    tgt = np.asarray(ids[1:])
    if tgt.max() >= v_live:
        raise SystemExit("REFUSING: target id outside live vocab")
    return float(-ls[np.arange(len(tgt)), tgt].mean())


def mean_forward_kl(t_logits: np.ndarray, a_logits: np.ndarray,
                    v_live: int) -> float:
    """Mean over positions of KL(teacher || arm), live vocab, on the
    P-1 alignment rows (caller slices); fp64 reductions."""
    if t_logits.shape != a_logits.shape:
        raise SystemExit("REFUSING: teacher/arm logit shape mismatch")
    tl = t_logits[:, :v_live]
    al = a_logits[:, :v_live]
    if not (np.isfinite(tl).all() and np.isfinite(al).all()):
        raise SystemExit("REFUSING: non-finite logits in KL input")
    lt = log_softmax(tl)
    la = log_softmax(al)
    pt = np.exp(lt)
    return float((pt * (lt - la)).sum(axis=-1).mean())


def perturb_ulp(a_fp16: np.ndarray, up: bool) -> np.ndarray:
    """Move every fp16 value one ulp toward +/- inf (the registered
    +-1ulp record perturbation), returned as fp16."""
    direc = np.float16(np.inf) if up else np.float16(-np.inf)
    return np.nextafter(a_fp16, direc, dtype=np.float16)


def sensitivity_floor(fn, rec_fp16: np.ndarray) -> float:
    """max |fn(perturbed) - fn(record)| over the +-1ulp perturbations."""
    base = fn(rec_fp16.astype(np.float32))
    return max(abs(fn(perturb_ulp(rec_fp16, True).astype(np.float32)) - base),
               abs(fn(perturb_ulp(rec_fp16, False).astype(np.float32)) - base))


def margin_bin(m: float):
    for b in range(len(MARGIN_EDGES) - 1):
        if MARGIN_EDGES[b] <= m < MARGIN_EDGES[b + 1]:
            return b
    return len(MARGIN_EDGES) - 2


def teacher_margins_top1(t_logits_fp16: np.ndarray, v_live: int):
    """Teacher top1 ids and (top1-top2) logit margins, fp32 upcast,
    live vocab (TREE-PINS item 2)."""
    tl = t_logits_fp16[:, :v_live].astype(np.float32)
    part = np.argpartition(-tl, 1, axis=-1)[:, :2]
    rows = np.arange(len(tl))[:, None]
    vals = tl[rows, part]
    order = np.argsort(-vals, axis=-1)
    top2 = part[rows, order]
    m = vals[rows, order]
    return top2[:, 0], (m[:, 0] - m[:, 1])


def flip_table(t_top1, margins, a_top1):
    """Per-margin-bin flip counts: [n_positions, n_flips] per bin."""
    nb = len(MARGIN_EDGES) - 1
    tab = [[0, 0] for _ in range(nb)]
    for t, m, a in zip(t_top1, margins, a_top1):
        b = margin_bin(float(m))
        tab[b][0] += 1
        if int(t) != int(a):
            tab[b][1] += 1
    return tab


def sha_arr(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def fsha(p: str) -> str:
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def teacher_receipt_block(man_t: dict, teacher_dir: str) -> dict:
    """Teacher provenance block for a scorer receipt, with the renamed
    sha fields: *_record_sha hashes the SAVED LOGITS ARRAY (what the
    frozen drivers called corpus_sha/prefix_sha), *_input_sha hashes
    the eval input FILE the teacher consumed. All values derived from
    the manifest the run actually opened, never literals."""
    return {"dir": teacher_dir,
            "code_commit": man_t["code_commit"],
            "revision": man_t["revision"],
            "corpus_record_sha": man_t["records"]["corpus"]["sha256"],
            "prefix_record_sha": man_t["records"]["prefixes"]["sha256"],
            "corpus_input_sha": man_t["inputs"]["corpus_sha256"],
            "prefixes_input_sha": man_t["inputs"]["prefixes_sha256"]}
