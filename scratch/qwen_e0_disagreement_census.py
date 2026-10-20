"""E0 disagreement census: where and how the compressed arm C departs from
the teacher on the 4,722 preserved teacher-greedy positions (read-only
analysis of frozen evidence; nothing is trained, fitted or tuned here).

Inputs: the frozen E0 rows (logs/qwen_e0/C/positions.jsonl: teacher token,
margins, draft argmax, draft top-16, prompt family, step), the teacher's
full fp16 logits (logs/qwenteacher_v2/rollout_logits.npy, [256, 24, V],
sha pinned in the manifest) and, when present, the draft's full logits
(fp16, live vocabulary) and final-norm hidden states written by
scratch/qwen_metal_prefill_serve_ab.py MODE=census (the Metal hybrid path,
argmax-identical to the CPU census). With the dump the census compares
whole distributions; without it, it uses the rows' top-16 only and says
so in the receipt.

Sections of the receipt (one JSON, refuse-if-exists):
  margins      teacher margin against draft margin, agree and disagree,
               quantiles and the joint bins; disagreement rate per teacher
               margin bin and per draft margin bin
  top2         what the draft picks when it disagrees: the teacher's
               second choice (rank 2), a top-16 token, or outside; the
               teacher token's rank in the draft; whether the teacher's
               top-2 are the draft's top-2 in either order
  bias         per-logit statistics on teacher top-16 tokens (draft minus
               teacher logit after shift alignment), the logit-scale fit
               (slope of draft on teacher over the teacher top-16), the
               tokens over- and under-selected by the draft relative to
               the teacher, and the draft argmax's teacher rank
  families     agreement and KL by prompt family and by step bucket
  evaluable    which corrections the existing evidence can evaluate and
               which need vendor weights or more teacher inference (text)
Env: E0_DIR (logs/qwen_e0/C), TEACHER_DIR (logs/qwenteacher_v2),
CENSUS_RECEIPT (the MODE=census receipt whose dumps are used; the dump
is opened by the name and sha that receipt records, and every row's
dump argmax must equal the E0 row's draft_argmax), OUT
(logs/qwen_e0/C_disagreement_<commit>.json), SMOKE=1 (first 64 rows, OUT under
logs/qwen_e0_smoke/, OUT from the environment ignored).
"""
from __future__ import annotations

import collections
import hashlib
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from llmopt.lab.strict_json import write_strict  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
E0_DIR = os.environ.get("E0_DIR", "logs/qwen_e0/C")
TEACHER_DIR = os.environ.get("TEACHER_DIR", "logs/qwenteacher_v2")
CENSUS_RECEIPT = os.environ.get("CENSUS_RECEIPT", "")
def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


OUT = f"logs/qwen_e0_smoke/C_disagreement_{git_head()}.json" if SMOKE else os.environ.get("OUT", f"logs/qwen_e0/C_disagreement_{git_head()}.json")
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
STEP_EDGES = [0, 1, 4, 16, 64, 128, 256]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_array_file(path):
    """sha256 of the array's bytes (the teacher manifest hashes a.tobytes(), not the .npy file)."""
    a = np.load(path, mmap_mode="r")
    h = hashlib.sha256()
    step = max(1, (1 << 28) // max(1, a[0].nbytes))
    for i in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
    return h.hexdigest()


def bins(v, edges):
    return int(np.searchsorted(np.asarray(edges[1:], dtype=np.float64), v, side="right"))


def q(a):
    a = np.asarray(a, dtype=np.float64)
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "p10": float(np.percentile(a, 10)), "p50": float(np.median(a)),
            "p90": float(np.percentile(a, 90)), "min": float(a.min()), "max": float(a.max())}


def main():
    rows = [json.loads(line) for line in open(os.path.join(E0_DIR, "positions.jsonl"))]
    if SMOKE:
        rows = rows[:64]
    man = json.load(open(os.path.join(TEACHER_DIR, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    summary = json.load(open(os.path.join(E0_DIR, "summary.json")))
    v_live = int(summary["v_live"])
    teacher = np.load(os.path.join(TEACHER_DIR, "rollout_logits.npy"), mmap_mode="r")
    prompt_index = {pid: i for i, pid in enumerate(ro["prompt_ids"])}
    have_dump = bool(CENSUS_RECEIPT)
    draft, dump_logits, census_rec = None, None, None
    if have_dump:
        census_rec = json.load(open(CENSUS_RECEIPT))
        if census_rec.get("smoke") and not SMOKE:
            raise SystemExit("REFUSING: the census receipt is a smoke receipt")
        nm = next(k for k in census_rec["dumps"] if k.startswith("draft_logits"))
        dump_logits = os.path.join(census_rec["dump_dir"], nm)
        if sha_file(dump_logits) != census_rec["dumps"][nm]:
            raise SystemExit("REFUSING: dump sha differs from the census receipt")
        draft = np.load(dump_logits, mmap_mode="r")
        if draft.shape[0] != census_rec["n_positions"] or (not SMOKE and draft.shape[0] != len(rows)):
            raise SystemExit(f"REFUSING: dump has {draft.shape[0]} rows, census needs {len(rows)}")
        if SMOKE:
            rows = rows[:draft.shape[0]]          # the smoke census dumps one prompt's first positions only
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")

    n = len(rows)
    agree = np.array([r["agree"] for r in rows])
    tm = np.array([r["teacher_margin_logit"] for r in rows], dtype=np.float64)
    dm = np.array([r["draft_margin_logit"] for r in rows], dtype=np.float64)
    cat = np.array([r["category"] for r in rows])
    step = np.array([r["step"] for r in rows])
    rank_t_in_d = np.array([r["teacher_token_rank_in_draft"] for r in rows])     # 0-based (0 = draft argmax); 16 means outside the draft top-16
    kl = np.array([r["kl_teacher_draft"] for r in rows], dtype=np.float64)
    dis = ~agree

    # ---- teacher top-2 from the full record; draft side from the dump or the rows' top-16 ----
    t_top1 = np.zeros(n, np.int64)
    t_top2 = np.zeros(n, np.int64)
    t_logit_top1 = np.zeros(n)
    t_logit_top2 = np.zeros(n)
    t_top16 = np.zeros((n, 16), np.int64)
    d_pick_is_t2 = np.zeros(n, bool)
    d_top2_set_equal = np.zeros(n, bool)
    t2_rank_in_d = np.full(n, 99, np.int64)
    shift_err_top16 = np.full((n, 16), np.nan)         # draft minus teacher after shift alignment on teacher top-16
    slope = np.full(n, np.nan)
    d_argmax_t_rank = np.full(n, 99, np.int64)
    dump_argmax_mismatch = 0
    dump_mismatch_rows = []
    for i, r in enumerate(rows):
        zt = np.asarray(teacher[r["step"], prompt_index[r["prompt_id"]], :v_live], dtype=np.float32)
        order = np.argsort(-zt.astype(np.float64), kind="stable")
        t_top1[i], t_top2[i] = order[0], order[1]
        t_logit_top1[i], t_logit_top2[i] = zt[order[0]], zt[order[1]]
        t_top16[i] = order[:16]
        if t_top1[i] != r["teacher_token"]:
            # the recorded token is the argmax of the stored fp16 record except for the 26 ambiguous rows
            if not r["ambiguous"] and r["recorded_equals_fp16_argmax"]:
                raise SystemExit(f"REFUSING: teacher argmax mismatch at row {i}")
        d_ids = list(r["draft_topk_ids"])
        d_pick_is_t2[i] = r["draft_argmax"] == t_top2[i]
        d_top2_set_equal[i] = {d_ids[0], d_ids[1]} == {int(t_top1[i]), int(t_top2[i])}
        if int(t_top2[i]) in d_ids:
            t2_rank_in_d[i] = d_ids.index(int(t_top2[i])) + 1
        d_argmax_t_rank[i] = int(np.where(order == r["draft_argmax"])[0][0]) + 1
        if have_dump:
            zd = np.asarray(draft[i], dtype=np.float32)
            if int(zd.argmax()) != r["draft_argmax"]:
                # the dump is fp16; a float32 near-tie can round to the other order (the census receipt
                # compares in float32 and reports its own count); recorded with the fp16 gap, never silently
                top2 = np.sort(zd)[-2:]
                dump_argmax_mismatch += 1
                dump_mismatch_rows.append({"row": i, "prompt_id": r["prompt_id"], "step": r["step"], "e0_draft_argmax": r["draft_argmax"],
                                           "dump_argmax": int(zd.argmax()), "fp16_top2_gap": float(top2[1] - top2[0]),
                                           "e0_draft_margin": r["draft_margin_logit"]})
            a = zd[t_top16[i]].astype(np.float64)
            b = zt[t_top16[i]].astype(np.float64)
            shift = float(np.mean(zd.astype(np.float64)) - np.mean(zt.astype(np.float64)))
            shift_err_top16[i] = (a - shift) - b
            bc = b - b.mean()
            if float(bc @ bc) > 0:
                slope[i] = float(((a - a.mean()) @ bc) / (bc @ bc))
    # ---- margins ----
    tm_bin = np.array([bins(x, MARGIN_EDGES) for x in tm])
    dm_bin = np.array([bins(x, MARGIN_EDGES) for x in dm])
    joint = np.zeros((len(MARGIN_EDGES) - 1, len(MARGIN_EDGES) - 1, 2), np.int64)
    for i in range(n):
        joint[tm_bin[i], dm_bin[i], int(agree[i])] += 1
    margins = {
        "edges": MARGIN_EDGES,
        "teacher_margin": {"agree": q(tm[agree]), "disagree": q(tm[dis])},
        "draft_margin": {"agree": q(dm[agree]), "disagree": q(dm[dis])},
        "draft_minus_teacher_margin": {"agree": q(dm[agree] - tm[agree]), "disagree": q(dm[dis] - tm[dis])},
        "disagree_rate_by_teacher_margin_bin": [[int((tm_bin == b).sum()), int((dis & (tm_bin == b)).sum())] for b in range(len(MARGIN_EDGES) - 1)],
        "disagree_rate_by_draft_margin_bin": [[int((dm_bin == b).sum()), int((dis & (dm_bin == b)).sum())] for b in range(len(MARGIN_EDGES) - 1)],
        "joint_teacher_by_draft_bin_total_and_disagree": [[[int(joint[a, b].sum()), int(joint[a, b, 0])] for b in range(joint.shape[1])] for a in range(joint.shape[0])],
        "draft_margin_above_1_nat_while_wrong": int((dis & (dm > 1.0)).sum()),
        "draft_margin_above_2_nat_while_wrong": int((dis & (dm > 2.0)).sum()),
        "teacher_margin_above_2_nat_while_wrong": int((dis & (tm > 2.0)).sum()),
        "teacher_margin_above_5_nat_while_wrong": int((dis & (tm > 5.0)).sum()),
    }
    # ---- top-2 structure ----
    rk = np.minimum(rank_t_in_d, 16)
    top2 = {
        "n_disagree": int(dis.sum()),
        "draft_picks_teacher_top2": int((dis & d_pick_is_t2).sum()),
        "draft_picks_a_teacher_top16_token": int((dis & (d_argmax_t_rank <= 16)).sum()),
        "draft_picks_outside_teacher_top16": int((dis & (d_argmax_t_rank > 16)).sum()),
        "teacher_token_rank_in_draft_hist_0based": {str(k): int(v) for k, v in sorted(collections.Counter(rk[dis]).items())},
        "teacher_token_rank_note": "0-based as in the E0 rows: 1 = the draft's second choice; 16 = outside the draft's top-16",
        "teacher_token_is_draft_second": int((dis & (rank_t_in_d == 1)).sum()),
        "teacher_token_is_draft_third": int((dis & (rank_t_in_d == 2)).sum()),
        "teacher_token_outside_draft_top16": int((dis & (rank_t_in_d >= 16)).sum()),
        "swap_of_teacher_top2": int((dis & d_top2_set_equal).sum()),
        "draft_argmax_teacher_rank_hist_disagree": {str(k): int(v) for k, v in sorted(collections.Counter(np.minimum(d_argmax_t_rank[dis], 17)).items())},
        "teacher_top2_gap_when_swapped": q((t_logit_top1 - t_logit_top2)[dis & d_top2_set_equal]),
        "teacher_top2_gap_disagree_all": q((t_logit_top1 - t_logit_top2)[dis]),
    }
    # ---- bias ----
    bias = {"dump_present": have_dump, "dump_rows_whose_fp16_argmax_differs_from_e0_row": dump_argmax_mismatch if have_dump else None,
            "dump_fp16_argmax_mismatch_rows": dump_mismatch_rows,
            "note": "shift alignment subtracts the mean over the live vocabulary of (draft minus teacher); both records are fp16, so about 0.01 to 0.03 of rounding sits inside every per-logit difference; the census receipt's float32 argmax count is the decision-level comparison, the fp16 dump can flip a near-tie"}
    if have_dump and any(m["fp16_top2_gap"] > 0.0 for m in dump_mismatch_rows):      # fp16 rounding is monotone: a flip needs an exact fp16 tie
        raise SystemExit(f"REFUSING: a dump row disagrees with its E0 row beyond fp16 tie resolution: {dump_mismatch_rows}")
    if have_dump:
        e = shift_err_top16
        bias.update({
            "shift_aligned_err_on_teacher_top16": {"all": q(e[~np.isnan(e)]), "rank1": q(e[:, 0]), "rank2": q(e[:, 1]),
                                                   "rank3_16": q(e[:, 2:].ravel())},
            "err_rank1_minus_rank2": {"agree": q((e[:, 0] - e[:, 1])[agree]), "disagree": q((e[:, 0] - e[:, 1])[dis])},
            "mean_err_by_teacher_rank": [float(np.nanmean(e[:, k])) for k in range(16)],
            "logit_scale_slope_draft_on_teacher_top16": {"agree": q(slope[agree]), "disagree": q(slope[dis])},
        })
    tok_draft = collections.Counter(int(r["draft_argmax"]) for r in rows if not r["agree"])
    tok_teacher_missed = collections.Counter(int(r["teacher_token"]) for r in rows if not r["agree"])
    tok_teacher_all = collections.Counter(int(r["teacher_token"]) for r in rows)
    bias["draft_overselected_tokens_in_disagreements"] = [{"token": t, "draft_picks": c, "teacher_frequency": tok_teacher_all.get(t, 0)} for t, c in tok_draft.most_common(20)]
    bias["teacher_tokens_most_missed"] = [{"token": t, "missed": c, "teacher_frequency": tok_teacher_all[t], "miss_rate": c / tok_teacher_all[t]} for t, c in tok_teacher_missed.most_common(20)]
    # ---- families and positions ----
    fam = {}
    for c in sorted(set(cat)):
        m = cat == c
        fam[c] = {"n": int(m.sum()), "agree": int((agree & m).sum()), "agree_rate": float((agree & m).sum() / m.sum()),
                  "kl_mean": float(kl[m].mean()), "teacher_margin_p50": float(np.median(tm[m])),
                  "disagree_rate_at_teacher_margin_ge_2": float(((dis & m & (tm >= 2)).sum()) / max(1, (m & (tm >= 2)).sum()))}
    stepb = {}
    for a, b in zip(STEP_EDGES[:-1], STEP_EDGES[1:]):
        m = (step >= a) & (step < b)
        if m.sum():
            stepb[f"{a}-{b - 1}"] = {"n": int(m.sum()), "agree_rate": float((agree & m).sum() / m.sum()), "kl_mean": float(kl[m].mean()),
                                    "teacher_margin_p50": float(np.median(tm[m]))}
    per_prompt = {}
    for r in rows:
        d = per_prompt.setdefault(r["prompt_id"], {"category": r["category"], "n": 0, "agree": 0})
        d["n"] += 1
        d["agree"] += int(r["agree"])
    # the first generated position of every rollout (the token after the chat template's "<think>\n")
    tok = None
    tpath = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor")) + "/tokenizer.json"
    if os.path.exists(tpath):
        try:
            from tokenizers import Tokenizer
            tok = Tokenizer.from_file(tpath)
        except ImportError:
            tok = None
    def s(t):
        return tok.decode([int(t)]) if tok is not None else None
    first = [{"prompt_id": r["prompt_id"], "category": r["category"], "teacher_token": r["teacher_token"], "teacher_str": s(r["teacher_token"]),
              "teacher_margin": r["teacher_margin_logit"], "draft_argmax": r["draft_argmax"], "draft_str": s(r["draft_argmax"]),
              "draft_margin": r["draft_margin_logit"], "teacher_token_rank_in_draft": r["teacher_token_rank_in_draft"], "agree": r["agree"]}
             for r in rows if r["step"] == 0]
    first_pos = {"n": len(first), "agree": int(sum(f["agree"] for f in first)),
                 "teacher_tokens": dict(collections.Counter(f["teacher_token"] for f in first)),
                 "draft_tokens": dict(collections.Counter(f["draft_argmax"] for f in first)),
                 "strings": {str(t): s(t) for t in set(f["teacher_token"] for f in first) | set(f["draft_argmax"] for f in first)},
                 "rows": first, "tokenizer": tpath if tok is not None else None}
    evaluable = {
        "with_existing_evidence": [
            "disagreement structure, margins, biases, families, positions (this receipt)",
            "logit-scale and shift calibration of the draft against the teacher (diagnostic; a global temperature cannot change an argmax, so it cannot recover agreement)",
            "head-side experiments given a replacement head: the final-norm hidden states of all 4,722 positions are stored (dump), so any head matrix (vendor BF16, higher-precision requantization, low-rank residual) can be scored against the teacher logits without re-running the body",
            "independent validation sets that are not E0: the teacher's corpus record (356 positions) and prefix record (98 positions) carry full logits; arm C hidden states for them need one CPU or Metal forward each",
        ],
        "needs_vendor_weights": [
            "any restoration or requantization of a matrix (head, embeddings, attention, MLP): the frozen artifact holds only the compressed payloads; the vendor BF16 lm_head is in shard model-00018-of-00018.safetensors (3,392,197,344 bytes; the tensor is 2,542,796,800 bytes) and the embeddings in shard 3 (2,542,796,952 bytes)",
            "attributing the 17 percent disagreement to body versus head error: needs either the vendor head (teacher logits from draft hidden states) or the teacher's hidden states (not recorded)",
        ],
        "needs_more_teacher_inference": [
            "any held-out evaluation beyond the 24 rollouts, 356 corpus and 98 prefix positions; the teacher is not resident (76 GB vendor restore condition unmet)",
            "free-run (non teacher-forced) agreement and downstream task deltas",
        ],
    }
    import platform
    rollout_sha_disk = sha_array_file(os.path.join(TEACHER_DIR, "rollout_logits.npy")) if not SMOKE else None
    if rollout_sha_disk is not None and rollout_sha_disk != ro["sha256"]:
        raise SystemExit("REFUSING: teacher rollout record differs from the manifest sha")
    rec = {"code_commit": git_head(), "tree_dirty": git_dirty(), "numpy_version": np.__version__, "python": platform.python_version(),
           "smoke": SMOKE, "e0_dir": E0_DIR, "e0_rows_sha256": sha_file(os.path.join(E0_DIR, "positions.jsonl")),
           "teacher": {"dir": TEACHER_DIR, "code_commit": man["code_commit"], "revision": man["revision"], "rollout_sha": ro["sha256"],
                       "rollout_array_sha256_on_disk": rollout_sha_disk, "rollout_sha_verified": (rollout_sha_disk == ro["sha256"]) if not SMOKE else "skipped"},
           "dump": {"census_receipt": CENSUS_RECEIPT or None, "census_commit": census_rec["code_commit"] if have_dump else None,
                    "path": dump_logits, "present": have_dump, "sha256": census_rec["dumps"][os.path.basename(dump_logits)] if have_dump else None},
           "n_positions": n, "n_agree": int(agree.sum()), "agree_rate": float(agree.mean()), "v_live": v_live,
           "margins": margins, "top2": top2, "bias": bias, "families": fam, "step_buckets": stepb, "per_prompt": per_prompt,
           "first_generated_position": first_pos,
           "evaluable": evaluable}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    # strict JSON: the open margin-bin boundary is written as the string "inf",
    # never as a bare Infinity constant (llmopt.lab.strict_json)
    write_strict(rec, OUT, overwrite=True)
    print(f"[census] first generated position: agree {first_pos['agree']}/{first_pos['n']}, teacher {first_pos['teacher_tokens']}, draft {first_pos['draft_tokens']}")
    print(f"[census] {n} rows, agree {agree.mean():.4f}; disagree {dis.sum()}: teacher-top2 picked {top2['draft_picks_teacher_top2']}, teacher token draft-second {top2['teacher_token_is_draft_second']}, "
          f"swap {top2['swap_of_teacher_top2']}, teacher token outside draft top16 {top2['teacher_token_outside_draft_top16']}; "
          f"dump {have_dump}; wrote {OUT}")


if __name__ == "__main__":
    main()
