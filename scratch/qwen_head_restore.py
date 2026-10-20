"""QWEN-HEAD-RESTORE-0: does replacing only arm C's compressed S16 lm_head
with the vendor's original BF16 lm_head improve teacher agreement when
every preceding layer stays compressed? Head-only substitution on the
frozen arm C final-norm hidden states; nothing is trained, fitted,
calibrated or selected on any outcome.

Stages (MODE=...), one process each, receipts under logs/qwen_head_restore/
(refuse-if-exists; SMOKE=1 writes under logs/qwen_head_restore_smoke/ and
is never evidence):
  preflight  the frozen E0 hidden-state dump is present, shape- and
             sha-correct against the census receipt, and corresponds to
             the states fed to lm_head: the S16 head decoded to float32
             and applied to the dump reproduces the production fp16
             argmax (near-ties recorded with their fp16 gap); Metal
             decode slabs equal the CPU codec row for row; teacher
             record shas; storage and destination checks for the fetch.
  opener     the first-generated-position anomaly on the compressed
             model alone: the draft's opener under the teacher's exact
             template (teacher record exists) and under template and
             length variations (draft only; no teacher record exists for
             those, and none is claimed).
  corpus     arm C final-norm hidden states of the teacher's corpus (356
             positions, one raw-text document) and prefix (98 positions,
             six raw-text snippets) records through the qualified Metal
             body (prefill_chunked, matmul="gemm"); provenance against the
             manifest token lists, the eval payload shas and the CPU
             score receipt's top-1 agreement counts.
  fetch      the pinned vendor shard model-00018-of-00018.safetensors
             (lm_head.weight + the MTP block) to VENDOR_SHARD_DIR, outside
             the repository and never the config/tokenizer directory;
             size and sha256 against the vendor's LFS metadata (pinned in
             this source and re-read from the API at run time); the
             safetensors header's lm_head entry (BF16 [248320, 5120],
             2,542,796,800 bytes) verified. No other shard, no MTP
             instantiation, no teacher restore.
  score      both heads on the same hidden rows: s16_f32 (arm C head
             decoded to float32), bf16_f32 (vendor head widened to
             float32), bf16_h16 (vendor head, hidden rounded to bf16
             first: the precision-of-evaluation control) and, for E0,
             s16_prod (the production fp16 Metal logits). Per position:
             argmax, margin, teacher-token rank, logsumexp, teacher-token
             logit and cross-entropy, forward KL(teacher || candidate),
             the candidate's logits at the teacher's top-16, the top-8;
             paired counts (corrected, broken, unchanged) by set, block
             deltas (prompt, paragraph, prefix units), block bootstrap,
             slope medians, family and margin-bin breakdowns, the opener
             rows; observations JSON for docs/preregs/qwen-head-restore-0.json.
  runtime    (only if the registered validation fires) decode step, prefill
             and memory with a resident bf16 head against the S16 head.
Env: ART_DIR (~/qwen_whole0t/C), VENDOR_DIR (~/qwen_vendor: config,
tokenizer, shard index only), VENDOR_SHARD_DIR (~/qwen_vendor_shards/<rev8>),
CENSUS_RECEIPT (logs/qmetal/prefill_serve/census_d42cb215.json), OUT
(logs/qwen_head_restore), SMOKE=1.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import shutil
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from llmopt.lab import logit_stats as ls  # noqa: E402
from llmopt.lab.strict_json import write_strict  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
MODE = os.environ.get("MODE", "preflight")
ART_DIR = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
VENDOR_DIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
REPO = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SHARD = "model-00018-of-00018.safetensors"
SHARD_BYTES = 3392197344
SHARD_SHA256 = "1d3479509e21494658f9b64d317f5ea8e55c4025d28c702d6c4d0b356ce8ea06"   # vendor LFS metadata, read 2026-10-19
HEAD_SHAPE = (248320, 5120)
HEAD_BYTES = 2 * HEAD_SHAPE[0] * HEAD_SHAPE[1]
V_LIVE = 248077
WE, THE = 1596, 760
CHAIN = "logs/qwenwhole/artifact_digest_C.txt"
TEACHER_DIR = "logs/qwenteacher_v2"
E0_DIR = "logs/qwen_e0/C"
SCORE_C = "logs/qwenmodel1/score_C.json"
EV_DIR = "evals/qwen_model1"
CENSUS_RECEIPT = os.environ.get("CENSUS_RECEIPT", "logs/qmetal/prefill_serve/census_d42cb215.json")
VENDOR_SHARD_DIR = os.path.expanduser(os.environ.get("VENDOR_SHARD_DIR", f"~/qwen_vendor_shards/{REVISION[:8]}"))
OUT = "logs/qwen_head_restore_smoke" if SMOKE else os.environ.get("OUT", "logs/qwen_head_restore")
DUMP = os.path.join(OUT, "dump")
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
CHUNK_ROWS = 256
E0_HIDDEN_SHA256 = "01612fb6c99b7dcd87c03abc0b397741dcced5818cd34cc981d6e3ed62df74b4"   # census_d42cb215.json dumps
E0_LOGITS_SHA256 = "79162c5ddb7d974a445b3399a5bda75084ae2f4fd4117dd6c24f4ca37a0a1a3a"
PREREG = "docs/preregs/qwen-head-restore-0.json"
REGISTRATION_SHA256 = "ceeeb65490e0279abd77e646df1b2c6409881c5b450ebd781a92870b8b00b5ff"   # registration_digest of docs/preregs/qwen-head-restore-0.json
NEAR_TIE_GAP = 0.05           # fp16 top-2 gap below which a recompute argmax flip is a near-tie, not a defect
CORPUS_AGREE_TOL = 2          # positions by which the Metal top-1 agreement may differ from the CPU score receipt
PRIMARY = ("s16_f32", "bf16_f32")


def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def sha_array_file(path):
    """sha256 of the array's bytes (the teacher manifest hashes a.tobytes())."""
    a = np.load(path, mmap_mode="r")
    h = hashlib.sha256()
    step = max(1, (1 << 26) // max(1, a[0].nbytes))
    for i in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[i:i + step]).tobytes())
    return h.hexdigest()


def refuse(path):
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {path} exists")


def receipt_path(mode):
    os.makedirs(OUT, exist_ok=True)
    return os.path.join(OUT, f"{mode}.json")


def require_clean_tree():
    """A real receipt records code_commit; the tree must be that commit."""
    if not SMOKE and git_dirty():
        raise SystemExit("REFUSING: working tree is dirty; a real receipt's code_commit must name the code that ran")


def repo_root():
    return os.path.dirname(HERE)


def provenance(mode):
    rec = {"mode": mode, "code_commit": git_head(), "tree_dirty": git_dirty(), "smoke": SMOKE,
           "numpy_version": np.__version__, "python": sys.version.split()[0], "v_live": V_LIVE,
           "artifact_dir_basename": os.path.basename(ART_DIR.rstrip("/")), "artifact_chain": CHAIN,
           "census_receipt": CENSUS_RECEIPT, "teacher_dir": TEACHER_DIR}
    try:
        import mlx.core as mx
        rec["mlx_version"] = mx.__version__
        rec["device"] = str(mx.default_device())
    except ImportError:
        pass
    return rec


def rss_peak_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)   # bytes on macOS


# ---------------------------------------------------------------- frozen inputs

def load_manifest():
    from llmopt.lab import qartifact
    q = qartifact.qualify_artifact(ART_DIR, os.path.join(VENDOR_DIR, "model.safetensors.index.json"), CHAIN, preflight=False)
    return q["manifest"], q["report"]


def head_payload(man):
    e = man["lm_head.weight"]
    if e["codec"] != "s16" or tuple(e["shape"]) != HEAD_SHAPE:
        raise SystemExit(f"REFUSING: lm_head entry {e} is not the S16 [248320, 5120] head")
    with open(os.path.join(ART_DIR, e["shard"] + ".bin"), "rb") as f:
        f.seek(e["off"])
        buf = f.read(e["len"])
    if len(buf) != e["len"]:
        raise SystemExit("REFUSING: short head payload")
    return buf, e


def teacher_manifest():
    man = json.load(open(os.path.join(TEACHER_DIR, "teacher_manifest.json")))
    if man["revision"] != REVISION:
        raise SystemExit("REFUSING: teacher revision differs from the pinned vendor revision")
    return man


def e0_rows():
    rows = [json.loads(line) for line in open(os.path.join(E0_DIR, "positions.jsonl"))]
    return rows


def census_receipt():
    rec = json.load(open(CENSUS_RECEIPT))
    if rec.get("smoke") or rec.get("mode") != "census":
        raise SystemExit("REFUSING: the census receipt is not a real census receipt")
    return rec


def dump_paths(rec):
    names = list(rec["dumps"].keys())
    logits = [n for n in names if n.startswith("draft_logits_fp16")][0]
    hidden = [n for n in names if n.startswith("draft_hidden_fp32")][0]
    if rec["dumps"][hidden] != E0_HIDDEN_SHA256 or rec["dumps"][logits] != E0_LOGITS_SHA256:
        raise SystemExit("REFUSING: the census receipt's dump shas are not the pinned E0 dumps")
    return os.path.join(rec["dump_dir"], logits), os.path.join(rec["dump_dir"], hidden), rec["dumps"][logits], rec["dumps"][hidden]


# ---------------------------------------------------------------- heads

def s16_head_device(buf, entry):
    """Arm C's head decoded to float32 on the device through the kernels'
    decode path (slabs concatenated); returns (resident, W[V, 5120])."""
    import mlx.core as mx
    from llmopt.lab import qmetal
    res = qmetal.resident(buf, entry)
    if not res.gemm_ok:
        raise SystemExit("REFUSING: head is not normal-class; decode_f32 undefined")
    R, C = res.shape
    rows = max(1, min(R, qmetal.GEMM_SLAB_BYTES // (C * 4)))
    slabs = []
    for r0 in range(0, R, rows):
        s = res.decode_f32(r0, min(R, r0 + rows))
        mx.eval(s)
        slabs.append(s)
    W = mx.concatenate(slabs, axis=0)
    mx.eval(W)
    del slabs
    return res, W


def s16_head_f32(buf, entry):
    """The decoded head only; the resident's device payload is released."""
    res, W = s16_head_device(buf, entry)
    del res
    return W


def s16_rows_cpu(buf, entry, r0, r1):
    from llmopt.lab import qcodec
    return qcodec.dec_s16(ls.s16_row_slice_payload(buf, entry["shape"], r0, r1), (r1 - r0, entry["shape"][1]))


def vendor_shard_path():
    p = os.path.join(VENDOR_SHARD_DIR, SHARD)
    if SMOKE and not os.path.exists(p):
        return None
    if sha_file(p) != SHARD_SHA256:
        raise SystemExit("REFUSING: vendor shard sha256 differs from the pinned LFS sha")
    return p


def bf16_head_host():
    p = vendor_shard_path()
    return None if p is None else ls.read_bf16_tensor(p, "lm_head.weight", expect_shape=HEAD_SHAPE)


def logits_chunks(W, hidden, round_hidden_bf16=False):
    """Yields (start, logits[n, V_LIVE] float32 host) for CHUNK_ROWS of hidden rows."""
    import mlx.core as mx
    n = hidden.shape[0]
    for s in range(0, n, CHUNK_ROWS):
        h = np.ascontiguousarray(hidden[s:s + CHUNK_ROWS], dtype=np.float32)
        if round_hidden_bf16:
            h = ls.round_to_bf16(h)
        lg = mx.matmul(mx.array(h), W.T)
        mx.eval(lg)
        yield s, np.asarray(lg[:, :V_LIVE], dtype=np.float32)
        del lg


# ---------------------------------------------------------------- preflight

def mode_preflight():
    import mlx.core as mx
    rpath = receipt_path("preflight")
    refuse(rpath)
    require_clean_tree()
    t_all = time.time()
    rec = provenance("preflight")
    checks = {}
    # --- the census receipt and its dumps
    cen = census_receipt()
    lg_path, h_path, lg_sha, h_sha = dump_paths(cen)
    for p in (lg_path, h_path):
        if not os.path.exists(p):
            raise SystemExit(f"REFUSING: dump {p} absent")
    hid = np.load(h_path, mmap_mode="r")
    lgd = np.load(lg_path, mmap_mode="r")
    n = int(cen["n_positions"])
    checks["hidden_shape_ok"] = tuple(hid.shape) == (n, HEAD_SHAPE[1]) and hid.dtype == np.float32
    checks["logits_shape_ok"] = tuple(lgd.shape) == (n, V_LIVE) and lgd.dtype == np.float16
    checks["hidden_sha_ok"] = sha_file(h_path) == h_sha
    checks["logits_sha_ok"] = sha_file(lg_path) == lg_sha
    checks["census_argmax_equal_e0"] = int(cen["argmax_equal_e0"]) == n
    rows = e0_rows()
    checks["e0_rows_sha_ok"] = sha_file(os.path.join(E0_DIR, "positions.jsonl")) == cen["e0_rows_sha256"]
    checks["e0_rows_count_ok"] = len(rows) == n
    tman = teacher_manifest()
    ro = tman["records"]["rollouts"]
    order_ok = True
    k = 0
    for i, pid in enumerate(ro["prompt_ids"]):
        for s in range(ro["gen_lengths"][i]):
            r = rows[k]
            order_ok &= r["prompt_id"] == pid and r["step"] == s
            k += 1
    checks["e0_row_order_ok"] = bool(order_ok and k == n)
    # --- teacher records
    checks["teacher_rollout_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "rollout_logits.npy")) == ro["sha256"]
    checks["teacher_corpus_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "corpus_logits.npy")) == tman["records"]["corpus"]["sha256"]
    checks["teacher_prefix_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "prefix_logits.npy")) == tman["records"]["prefixes"]["sha256"]
    # --- arm C head: Metal decode slabs equal the CPU codec row for row
    man, report = load_manifest()
    buf, entry = head_payload(man)
    rec["artifact_qualification"] = {"rung0": report["rung0"], "conservation": report.get("conservation")}
    rec["head_entry"] = entry
    W = s16_head_f32(buf, entry)
    parity = []
    for r0, r1 in ((0, 2048), (123904, 124928), (HEAD_SHAPE[0] - 2048, HEAD_SHAPE[0])):
        cpu = s16_rows_cpu(buf, entry, r0, r1)
        dev = np.asarray(W[r0:r1])
        parity.append({"rows": [r0, r1], "bit_equal": bool(np.array_equal(cpu, dev)), "max_abs_diff": float(np.abs(cpu - dev).max())})
    checks["decode_parity_bit_equal"] = all(p["bit_equal"] for p in parity)
    rec["decode_parity"] = parity
    # --- correspondence: the decoded head on the hidden dump reproduces the production fp16 argmax
    n_use = min(64, n) if SMOKE else n
    mism = []
    max_abs = 0.0
    agree_prod = 0
    for s, lg in logits_chunks(W, hid[:n_use]):
        prod = np.asarray(lgd[s:s + lg.shape[0]], dtype=np.float32)
        max_abs = max(max_abs, float(np.abs(lg - prod).max()))
        am = lg.argmax(1)
        pm = prod.argmax(1)
        agree_prod += int((am == pm).sum())
        for j in np.nonzero(am != pm)[0]:
            top2 = np.sort(prod[j])[-2:]
            mism.append({"row": int(s + j), "prod_argmax": int(pm[j]), "recompute_argmax": int(am[j]),
                         "fp16_top2_gap": float(top2[1] - top2[0]), "recompute_gap": float(lg[j, pm[j]] - lg[j, am[j]])})
    bad = [m for m in mism if m["fp16_top2_gap"] > NEAR_TIE_GAP]
    checks["correspondence_ok"] = len(bad) == 0
    rec["correspondence"] = {"positions": n_use, "argmax_equal_production": agree_prod, "mismatches": mism,
                             "mismatches_beyond_near_tie": len(bad), "near_tie_gap": NEAR_TIE_GAP,
                             "max_abs_diff_f32_recompute_vs_fp16_dump": max_abs}
    # --- CPU oracle on a few positions: numpy float32 over CPU-decoded rows against the device logits
    idx = np.linspace(0, n_use - 1, 4).astype(int)
    h4 = np.asarray(hid[idx], dtype=np.float32)
    dev4 = np.asarray(mx.matmul(mx.array(h4), W.T))[:, :V_LIVE]
    cpu4 = np.zeros_like(dev4)
    step = 16384
    for r0 in range(0, V_LIVE, step):
        r1 = min(V_LIVE, r0 + step)
        cpu4[:, r0:r1] = h4.astype(np.float64) @ s16_rows_cpu(buf, entry, r0, r1).astype(np.float64).T
    rec["cpu_oracle"] = {"rows": idx.tolist(), "max_abs_diff": float(np.abs(cpu4 - dev4).max()),
                         "argmax_equal": int((cpu4.argmax(1) == dev4.argmax(1)).sum())}
    checks["cpu_oracle_argmax_ok"] = rec["cpu_oracle"]["argmax_equal"] == len(idx)
    del W
    # --- destination and storage for the fetch
    root = os.path.realpath(repo_root())
    dest = os.path.realpath(VENDOR_SHARD_DIR)
    probe = dest
    while not os.path.isdir(probe):          # the destination may not exist yet: measure its nearest existing ancestor
        probe = os.path.dirname(probe)
    free = shutil.disk_usage(probe).free
    checks["dest_outside_repo"] = not (dest == root or dest.startswith(root + os.sep))
    vreal = os.path.realpath(VENDOR_DIR)
    checks["dest_not_vendor_dir"] = not (dest == vreal or dest.startswith(vreal + os.sep))
    checks["no_shard_in_vendor_dir"] = not any(f.endswith(".safetensors") for f in os.listdir(VENDOR_DIR))
    checks["free_bytes_ok"] = free >= 2 * SHARD_BYTES
    rec["storage"] = {"free_bytes": int(free), "shard_bytes": SHARD_BYTES, "head_bytes": HEAD_BYTES,
                      "dest_role": "VENDOR_SHARD_DIR (outside the repository)", "shard_already_present": os.path.exists(os.path.join(dest, SHARD))}
    rec["checks"] = checks
    rec["ok"] = all(checks.values())
    rec["wall_s"] = time.time() - t_all
    rec["rss_peak_bytes"] = rss_peak_bytes()
    write_strict(rec, rpath)
    print(f"[hr] preflight ok={rec['ok']} checks={checks} correspondence {agree_prod}/{n_use} (beyond near-tie {len(bad)}), "
          f"max abs f32-vs-fp16 {max_abs:.4f}, cpu oracle max abs {rec['cpu_oracle']['max_abs_diff']:.2e}; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("preflight failed")


# ---------------------------------------------------------------- opener

FILLER = ("The reading room keeps its catalogue in three drawers. The first drawer lists books by author, the second by title, "
          "and the third by subject. A card in any drawer names the shelf and the position on the shelf. When a book is moved, "
          "all three cards are updated on the same afternoon, so the drawers never disagree with each other. ")


def load_model():
    from qwen_metal_qual import build_metal_model
    t0 = time.time()
    model, info = build_metal_model(ART_DIR, VENDOR_DIR)
    return model, info, time.time() - t0


def serving():
    from llmopt.lab import qprefill
    cfg = qprefill.ServingConfig()
    qprefill.configure_serving_memory(cfg)
    return {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes}


def mode_opener():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from transformers import AutoTokenizer
    from llmopt.lab import qprefill
    rpath = receipt_path("opener")
    refuse(rpath)
    require_clean_tree()
    rec = provenance("opener")
    rec["serving_applied"] = serving()
    tok = AutoTokenizer.from_pretrained(VENDOR_DIR)
    tman = teacher_manifest()
    ro = tman["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(TEACHER_DIR, "rollout_tokens.jsonl"))]
    rows = e0_rows()
    step0 = {r["prompt_id"]: r for r in rows if r["step"] == 0}
    teacher = np.load(os.path.join(TEACHER_DIR, "rollout_logits.npy"), mmap_mode="r")
    model, info, load_wall = load_model()
    rec["load_wall_s"] = load_wall
    n_prompts = 2 if SMOKE else len(recs)

    def filler_tokens(target):
        reps = 1
        while len(tok(FILLER * reps)["input_ids"]) < target:
            reps += 1
        return FILLER * reps

    conditions = [
        ("teacher_template", lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)),
        ("no_think", lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)),
        ("effort_low", lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, reasoning_effort="low")),
        ("effort_medium_no_system", lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, reasoning_effort="medium")),
        ("raw_no_system_think", lambda p: f"<|im_start|>user\n{p}<|im_end|>\n<|im_start|>assistant\n<think>\n"),
        ("filler_64", lambda p: tok.apply_chat_template([{"role": "user", "content": filler_tokens(64) + "\n\n" + p}], tokenize=False, add_generation_prompt=True)),
        ("filler_256", lambda p: tok.apply_chat_template([{"role": "user", "content": filler_tokens(256) + "\n\n" + p}], tokenize=False, add_generation_prompt=True)),
        ("filler_1024", lambda p: tok.apply_chat_template([{"role": "user", "content": filler_tokens(1024) + "\n\n" + p}], tokenize=False, add_generation_prompt=True)),
    ]
    results = {}
    t_all = time.time()
    for cname, build in conditions:
        per = []
        for i in range(n_prompts):
            pid = recs[i]["id"]
            text = build(recs[i]["prompt"])
            ids = tok(text, add_special_tokens=False)["input_ids"]
            if cname == "teacher_template":
                ref = [int(x) for x in np.asarray(ro["input_ids"][i])[np.asarray(ro["attention_mask"][i]) == 1]]
                if ids != ref:
                    raise SystemExit(f"REFUSING: rebuilt teacher template differs from the recorded prompt ids for {pid}")
            cache = make_prompt_cache(model)
            chunks = []
            qprefill.prefill_chunked(model, ids, cache, chunk=256, head="full", matmul="gemm",
                                     on_chunk=lambda s, lg: chunks.append(np.asarray(lg[0], dtype=np.float32)))
            lg = np.concatenate(chunks, axis=0)[:, :V_LIVE]
            z = lg[-1]
            order = np.argsort(-z.astype(np.float64), kind="stable")[:5]
            row = {"prompt_id": pid, "prompt_tokens": len(ids), "ids_sha256": hashlib.sha256(np.asarray(ids, np.int64).tobytes()).hexdigest(),
                   "argmax": int(order[0]), "argmax_str": tok.decode([int(order[0])]),
                   "top5": [[int(t), float(z[t]), tok.decode([int(t)])] for t in order],
                   "z_we_minus_the": float(z[WE] - z[THE]), "margin": float(z[order[0]] - z[order[1]])}
            # the two template positions before the opener: predicting the last two prompt tokens
            prev = []
            for back in (3, 2):
                zp = lg[-back]
                prev.append({"target": int(ids[-back + 1]), "target_str": tok.decode([int(ids[-back + 1])]), "argmax": int(zp.argmax()),
                             "agree": bool(zp.argmax() == ids[-back + 1]), "margin": float(np.sort(zp)[-1] - np.sort(zp)[-2])})
            row["template_positions"] = prev
            if cname == "teacher_template":
                e0 = step0[pid]
                zt = np.asarray(teacher[0, i, :V_LIVE], dtype=np.float32)
                row["e0_draft_argmax"] = e0["draft_argmax"]
                row["recompute_equals_e0"] = bool(row["argmax"] == e0["draft_argmax"])
                row["teacher_token"] = e0["teacher_token"]
                row["teacher_z_we_minus_the"] = float(zt[WE] - zt[THE])
                row["agree_teacher"] = bool(row["argmax"] == e0["teacher_token"])
            per.append(row)
            del cache, chunks, lg
        counts = {}
        for r in per:
            counts[r["argmax_str"]] = counts.get(r["argmax_str"], 0) + 1
        d = np.array([r["z_we_minus_the"] for r in per])
        summ = {"n": len(per), "argmax_counts": counts, "z_we_minus_the_mean": float(d.mean()), "z_we_minus_the_median": float(np.median(d)),
                "n_we_above_the": int((d > 0).sum()), "prompt_tokens_mean": float(np.mean([r["prompt_tokens"] for r in per])),
                "template_positions_agree": [int(sum(r["template_positions"][k]["agree"] for r in per)) for k in range(2)],
                "teacher_record": cname == "teacher_template"}
        if cname == "teacher_template":
            summ["agree_teacher"] = int(sum(r["agree_teacher"] for r in per))
            summ["recompute_equals_e0"] = int(sum(r["recompute_equals_e0"] for r in per))
            summ["teacher_z_we_minus_the_mean"] = float(np.mean([r["teacher_z_we_minus_the"] for r in per]))
        dup = [c for c, r in results.items() if [x["ids_sha256"] for x in r["rows"]] == [x["ids_sha256"] for x in per]]
        summ["duplicate_of"] = dup[0] if dup else None       # identical prompt ids as an earlier condition
        results[cname] = {"summary": summ, "rows": per}
        print(f"[hr] opener {cname}: {counts} mean z(We)-z(The) {summ['z_we_minus_the_mean']:+.2f}"
              + (f", teacher agree {summ['agree_teacher']}/{len(per)}" if cname == "teacher_template" else " (draft only)"), flush=True)
    rec["conditions"] = results
    rec["note"] = ("Teacher records exist for teacher_template only (the E0 step-0 rows and rollout_logits[0]); every other "
                   "condition reports the compressed model alone and claims no teacher agreement. The vendor template's default "
                   "reasoning effort is xhigh, so teacher_template is the xhigh system-prompt condition; reasoning_effort=medium "
                   "adds no system block and renders the same text as raw_no_system_think (summary.duplicate_of records it).")
    rec["wall_s"] = time.time() - t_all
    rec["memory_after"] = qprefill.memory_report()
    write_strict(rec, rpath)
    print(f"[hr] opener wrote {rpath}", flush=True)


# ---------------------------------------------------------------- corpus

class _RecordingHead:
    """Applies the model's head and keeps the hidden rows it saw (host float32)."""

    def __init__(self, lm):
        self.lm = lm
        self.hidden = []

    def __call__(self, h):
        self.hidden.append(np.asarray(h[0], dtype=np.float32))
        return self.lm(h)


def corpus_blocks(tokens, nl2):
    """Block of each position: the number of paragraph breaks at or before it."""
    return np.cumsum(np.asarray(tokens) == nl2).astype(np.int64)


def mode_corpus():
    import mlx.core as mx
    from mlx_lm.models.cache import make_prompt_cache
    from transformers import AutoTokenizer
    from llmopt.lab import qmetal, qprefill
    from qwen_metal_prefill_head import residents_of
    rpath = receipt_path("corpus")
    refuse(rpath)
    require_clean_tree()
    os.makedirs(DUMP, exist_ok=True)
    names = {"corpus_hidden": "corpus_hidden_fp32.npy", "prefix_hidden": "prefix_hidden_fp32.npy",
             "corpus_native_logits": "corpus_native_logits_fp16.npy", "prefix_native_logits": "prefix_native_logits_fp16.npy"}
    for nm in names.values():
        refuse(os.path.join(DUMP, nm))
    rec = provenance("corpus")
    rec["serving_applied"] = serving()
    tok = AutoTokenizer.from_pretrained(VENDOR_DIR)
    tman = teacher_manifest()
    rc, rp = tman["records"]["corpus"], tman["records"]["prefixes"]
    corpus_ids = [int(t) for t in rc["tokens"]]
    prefix_ids = [[int(t) for t in p] for p in rp["tokens"]]
    prov = {"corpus_txt_sha_ok": sha_file(os.path.join(EV_DIR, "corpus.txt")) == tman["inputs"]["corpus_sha256"],
            "prefixes_jsonl_sha_ok": sha_file(os.path.join(EV_DIR, "prefixes.jsonl")) == tman["inputs"]["prefixes_sha256"],
            "corpus_retokenize_equal": tok(open(os.path.join(EV_DIR, "corpus.txt")).read())["input_ids"] == corpus_ids,
            "prefix_retokenize_equal": [tok(json.loads(l)["text"])["input_ids"] for l in open(os.path.join(EV_DIR, "prefixes.jsonl"))] == prefix_ids,
            "prefix_bounds_ok": [len(p) for p in prefix_ids] == list(rp["bounds"]) and sum(rp["bounds"]) == rp["shape"][0],
            "corpus_len_ok": len(corpus_ids) == rc["shape"][0]}
    tc = np.load(os.path.join(TEACHER_DIR, "corpus_logits.npy"), mmap_mode="r")
    tp = np.load(os.path.join(TEACHER_DIR, "prefix_logits.npy"), mmap_mode="r")
    prov["teacher_corpus_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "corpus_logits.npy")) == rc["sha256"]
    prov["teacher_prefix_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "prefix_logits.npy")) == rp["sha256"]
    if SMOKE:
        corpus_ids = corpus_ids[:32]
        prefix_ids = prefix_ids[:1]
    model, info, load_wall = load_model()
    rec["load_wall_s"] = load_wall
    residents = residents_of(model)
    lm = model.language_model.lm_head
    head = _RecordingHead(lm)
    model.language_model.lm_head = head
    counts0 = qmetal.path_counts(residents)

    def run(ids):
        head.hidden = []
        chunks = []
        cache = make_prompt_cache(model)
        t0 = time.time()
        qprefill.prefill_chunked(model, ids, cache, chunk=256, head="full", matmul="gemm",
                                 on_chunk=lambda s, lg: chunks.append(np.asarray(lg[0], dtype=np.float32)))
        wall = time.time() - t0
        lg = np.concatenate(chunks, axis=0)[:, :V_LIVE]
        h = np.concatenate(head.hidden, axis=0)
        if lg.shape[0] != len(ids) or h.shape[0] != len(ids):
            raise SystemExit("REFUSING: rows != tokens")
        del cache
        return h, lg, wall

    t_all = time.time()
    h_c, lg_c, wall_c = run(corpus_ids)
    hs, lgs, walls = [], [], []
    for p in prefix_ids:
        h, lg, w = run(p)
        hs.append(h)
        lgs.append(lg)
        walls.append(w)
    h_p = np.concatenate(hs, axis=0)
    lg_p = np.concatenate(lgs, axis=0)
    # teacher argmax over the live vocabulary, fp16 as stored
    nc, npf = h_c.shape[0], h_p.shape[0]
    t_am_c = np.asarray(tc[:nc, :V_LIVE], dtype=np.float32).argmax(1)
    t_am_p = np.asarray(tp[:npf, :V_LIVE], dtype=np.float32).argmax(1)
    agree_c = int((lg_c.argmax(1) == t_am_c).sum())
    agree_p = int((lg_p.argmax(1) == t_am_p).sum())
    sc = json.load(open(SCORE_C))
    exp_c = int(round(sc["top1_agreement"]["corpus"] * rc["shape"][0]))
    exp_p = int(round(sc["top1_agreement"]["prefixes"] * rp["shape"][0]))
    prov["cpu_score_receipt"] = {"path": SCORE_C, "code_commit": sc["code_commit"], "device_actual": sc["device_actual"],
                                 "expected_agree_corpus": exp_c, "expected_agree_prefix": exp_p}
    prov["metal_agree_corpus"] = agree_c
    prov["metal_agree_prefix"] = agree_p
    prov["agree_within_tolerance"] = (abs(agree_c - exp_c) <= CORPUS_AGREE_TOL and abs(agree_p - exp_p) <= CORPUS_AGREE_TOL) if not SMOKE else None
    if not SMOKE and not prov["agree_within_tolerance"]:
        raise SystemExit(f"REFUSING: Metal top-1 agreement {agree_c}/{agree_p} differs from the CPU score receipt {exp_c}/{exp_p} by more than {CORPUS_AGREE_TOL}")
    nl2 = tok.convert_tokens_to_ids("ĊĊ")
    blocks_c = corpus_blocks(corpus_ids, nl2)
    blocks_p = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(prefix_ids)])
    if not SMOKE and int(blocks_c.max() + 1) != 5:
        raise SystemExit("REFUSING: the corpus does not split into the five registered paragraphs")
    np.save(os.path.join(DUMP, names["corpus_hidden"]), h_c)
    np.save(os.path.join(DUMP, names["prefix_hidden"]), h_p)
    np.save(os.path.join(DUMP, names["corpus_native_logits"]), lg_c.astype(np.float16))
    np.save(os.path.join(DUMP, names["prefix_native_logits"]), lg_p.astype(np.float16))
    counts1 = qmetal.path_counts(residents)
    rec.update({"provenance": prov, "n_corpus": nc, "n_prefix": npf, "prefix_bounds": [len(p) for p in prefix_ids],
                "corpus_blocks": {"paragraph_break_token": int(nl2), "n_blocks": int(blocks_c.max() + 1),
                                  "block_sizes": np.bincount(blocks_c).tolist()},
                "prefix_blocks": {"n_blocks": len(prefix_ids), "block_sizes": [len(p) for p in prefix_ids]},
                "walls_s": {"corpus": wall_c, "prefixes": walls}, "matrix_calls": {k: counts1[k] - counts0[k] for k in counts0},
                "dump_dir": DUMP, "dumps": {nm: sha_file(os.path.join(DUMP, nm)) for nm in names.values()},
                "memory_after": qprefill.memory_report(), "wall_s": time.time() - t_all,
                "ok": all(v for k, v in prov.items() if isinstance(v, bool)) and (prov["agree_within_tolerance"] in (True, None))})
    write_strict(rec, rpath)
    print(f"[hr] corpus: hidden {h_c.shape} + prefixes {h_p.shape}; Metal top-1 agreement corpus {agree_c}/{nc} (cpu receipt {exp_c}), "
          f"prefix {agree_p}/{npf} (cpu receipt {exp_p}); provenance {prov}; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("corpus stage failed")


# ---------------------------------------------------------------- fetch

def vendor_metadata():
    from huggingface_hub import HfApi
    info = HfApi().get_paths_info(REPO, [SHARD], revision=REVISION)
    if len(info) != 1 or info[0].lfs is None:
        raise SystemExit("REFUSING: vendor metadata for the shard is unavailable")
    return {"size": int(info[0].size), "lfs_sha256": info[0].lfs.sha256, "lfs_size": int(info[0].lfs.size)}


def mode_fetch():
    rpath = receipt_path("fetch")
    refuse(rpath)
    require_clean_tree()
    rec = provenance("fetch")
    pre = json.load(open(receipt_path("preflight")))
    if not pre.get("ok"):
        raise SystemExit("REFUSING: preflight receipt is not ok")
    meta = vendor_metadata()
    rec["vendor_metadata"] = meta
    if meta["lfs_sha256"] != SHARD_SHA256 or meta["size"] != SHARD_BYTES or meta["lfs_size"] != SHARD_BYTES:
        raise SystemExit("REFUSING: vendor metadata differs from the pinned sha256 / size")
    rec["repo"] = REPO
    rec["revision"] = REVISION
    rec["shard"] = SHARD
    rec["dest_role"] = "VENDOR_SHARD_DIR (outside the repository)"
    if SMOKE:
        rec["note"] = "smoke: metadata read only, no bytes downloaded"
        write_strict(rec, rpath)
        print(f"[hr] fetch smoke: metadata {meta}; wrote {rpath}", flush=True)
        return
    root = os.path.realpath(repo_root())
    dest = os.path.realpath(VENDOR_SHARD_DIR)
    vreal = os.path.realpath(VENDOR_DIR)
    if dest == root or dest.startswith(root + os.sep) or dest == vreal or dest.startswith(vreal + os.sep):
        raise SystemExit("REFUSING: destination inside the repository or the config/tokenizer directory")
    os.makedirs(dest, exist_ok=True)
    target = os.path.join(dest, SHARD)
    t0 = time.time()
    if os.path.exists(target) and os.path.getsize(target) == SHARD_BYTES and sha_file(target) == SHARD_SHA256:
        rec["download"] = {"skipped": True, "reason": "a byte-identical shard was already present"}
    else:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(REPO, SHARD, revision=REVISION, local_dir=dest)
        if os.path.realpath(p) != os.path.realpath(target):
            raise SystemExit(f"REFUSING: download landed at an unexpected path {os.path.basename(p)}")
        rec["download"] = {"skipped": False, "wall_s": time.time() - t0}
    size = os.path.getsize(target)
    sha = sha_file(target)
    header, start = ls.safetensors_header(target)
    names = sorted(k for k in header if k != "__metadata__")
    ent = header["lm_head.weight"]
    span = int(ent["data_offsets"][1]) - int(ent["data_offsets"][0])
    checks = {"size_ok": size == SHARD_BYTES, "sha256_ok": sha == SHARD_SHA256,
              "head_dtype_bf16": ent["dtype"] == "BF16", "head_shape_ok": [int(s) for s in ent["shape"]] == list(HEAD_SHAPE),
              "head_bytes_ok": span == HEAD_BYTES, "only_head_and_mtp": all(n == "lm_head.weight" or n.startswith("mtp.") for n in names),
              "data_end_ok": start + max(int(v["data_offsets"][1]) for k, v in header.items() if k != "__metadata__") == size}
    rec.update({"size_bytes": size, "sha256": sha, "header_bytes": start - 8, "tensor_names": names, "lm_head_entry": ent,
                "checks": checks, "ok": all(checks.values()), "whole_file_sha_verified": True, "wall_s": time.time() - t0})
    write_strict(rec, rpath)
    print(f"[hr] fetch: {size} bytes sha {sha[:12]} ok={rec['ok']} checks={checks}; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("fetch verification failed")


# ---------------------------------------------------------------- score

def teacher_side(zt, teacher_token):
    order = np.argsort(-zt.astype(np.float64), kind="stable")
    lt = ls.log_softmax64(zt)
    pt = np.exp(lt)
    return {"argmax": int(order[0]), "margin": float(zt[order[0]] - zt[order[1]]), "top16": order[:16].astype(np.int64),
            "top16_logits": zt[order[:16]].astype(np.float32), "entropy": float(-(pt * lt).sum()),
            "mean_logit": float(zt.astype(np.float64).mean()), "token_is_argmax": bool(order[0] == teacher_token)}


class _Set:
    """One scoring set: hidden rows, a teacher logits getter, teacher tokens, blocks, labels."""

    def __init__(self, name, hidden, teacher_row, teacher_tokens, blocks, meta):
        self.name, self.hidden, self.teacher_row, self.teacher_tokens, self.blocks, self.meta = name, hidden, teacher_row, teacher_tokens, blocks, meta
        self.n = hidden.shape[0]


def build_sets(pre, cor):
    rows = e0_rows()
    cen = census_receipt()
    lg_path, h_path, lg_sha, h_sha = dump_paths(cen)
    if pre["checks"]["hidden_sha_ok"] is not True or pre["checks"]["correspondence_ok"] is not True:
        raise SystemExit("REFUSING: preflight did not establish the hidden dump")
    hid = np.load(h_path, mmap_mode="r")
    prod = np.load(lg_path, mmap_mode="r")
    tman = teacher_manifest()
    ro = tman["records"]["rollouts"]
    pidx = {pid: i for i, pid in enumerate(ro["prompt_ids"])}
    roll = np.load(os.path.join(TEACHER_DIR, "rollout_logits.npy"), mmap_mode="r")
    n = len(rows) if not SMOKE else 64
    e0 = _Set("e0", hid[:n],
              lambda i: np.asarray(roll[rows[i]["step"], pidx[rows[i]["prompt_id"]], :V_LIVE], dtype=np.float32),
              np.array([r["teacher_token"] for r in rows[:n]], np.int64),
              np.array([pidx[r["prompt_id"]] for r in rows[:n]], np.int64),
              {"category": [r["category"] for r in rows[:n]], "step": [r["step"] for r in rows[:n]], "prompt_id": [r["prompt_id"] for r in rows[:n]],
               "e0_draft_argmax": [r["draft_argmax"] for r in rows[:n]], "teacher_margin_e0": [r["teacher_margin_logit"] for r in rows[:n]],
               "ambiguous": [r["ambiguous"] for r in rows[:n]]})
    e0.prod = prod
    d = cor["dump_dir"]
    for nm, sha in cor["dumps"].items():
        if sha_file(os.path.join(d, nm)) != sha:
            raise SystemExit(f"REFUSING: corpus dump {nm} does not match its receipt sha")
    hc = np.load(os.path.join(d, "corpus_hidden_fp32.npy"), mmap_mode="r")
    hp = np.load(os.path.join(d, "prefix_hidden_fp32.npy"), mmap_mode="r")
    nat_c = np.load(os.path.join(d, "corpus_native_logits_fp16.npy"), mmap_mode="r")
    nat_p = np.load(os.path.join(d, "prefix_native_logits_fp16.npy"), mmap_mode="r")
    tc = np.load(os.path.join(TEACHER_DIR, "corpus_logits.npy"), mmap_mode="r")
    tp = np.load(os.path.join(TEACHER_DIR, "prefix_logits.npy"), mmap_mode="r")
    nc, npf = hc.shape[0], hp.shape[0]
    corpus_tokens = [int(t) for t in tman["records"]["corpus"]["tokens"]][:nc]
    nl2 = cor["corpus_blocks"]["paragraph_break_token"]
    t_am_c = np.asarray(tc[:nc, :V_LIVE], dtype=np.float32).argmax(1)
    t_am_p = np.asarray(tp[:npf, :V_LIVE], dtype=np.float32).argmax(1)
    next_c = np.array(corpus_tokens[1:] + [-1], np.int64)
    bounds = cor["prefix_bounds"]
    next_p = []
    pref_tokens = [[int(t) for t in p] for p in tman["records"]["prefixes"]["tokens"]][:len(bounds)]
    for p in pref_tokens:
        next_p += p[1:] + [-1]
    corpus = _Set("corpus", hc, lambda i: np.asarray(tc[i, :V_LIVE], dtype=np.float32), t_am_c, corpus_blocks(corpus_tokens, nl2),
                  {"next_token": next_c.tolist()})
    prefix = _Set("prefix", hp, lambda i: np.asarray(tp[i, :V_LIVE], dtype=np.float32), t_am_p,
                  np.concatenate([np.full(b, i, np.int64) for i, b in enumerate(bounds)]), {"next_token": next_p})
    corpus.native = nat_c
    prefix.native = nat_p
    return [e0, corpus, prefix]


def correspondence(sets, arms_s16):
    """The float32 recompute of the S16 head on the stored hidden rows against the
    production fp16 logits of the same rows (E0: the census dump; corpus and
    prefixes: the corpus stage's native logits): argmax equal except at fp16 near-ties."""
    out = {}
    for st in sets:
        prod = st.prod if st.name == "e0" else st.native
        am = arms_s16[st.name]["argmax"]
        mism = []
        for i in range(st.n):
            row = np.asarray(prod[i], dtype=np.float32)
            pm = int(row.argmax())
            if pm != am[i]:
                top2 = np.sort(row)[-2:]
                mism.append({"row": i, "prod_argmax": pm, "recompute_argmax": int(am[i]), "fp16_top2_gap": float(top2[1] - top2[0])})
        bad = [m for m in mism if m["fp16_top2_gap"] > NEAR_TIE_GAP]
        out[st.name] = {"n": st.n, "mismatches": mism, "beyond_near_tie": len(bad), "ok": len(bad) == 0}
    return out


def score_arm(sets, W, round_hidden=False, prod=False):
    out = {}
    for st in sets:
        if prod and st.name != "e0":
            continue
        n = st.n
        arr = {"argmax": np.zeros(n, np.int64), "margin": np.zeros(n), "teacher_rank": np.zeros(n, np.int64), "lse": np.zeros(n),
               "z_teacher_token": np.zeros(n), "ce_teacher_token": np.zeros(n), "kl": np.zeros(n), "slope": np.zeros(n),
               "z_top16": np.zeros((n, 16), np.float32), "top8_ids": np.zeros((n, 8), np.int32), "top8_logits": np.zeros((n, 8), np.float32),
               "z_we_minus_the": np.zeros(n), "ce_next_token": np.full(n, np.nan), "mean_logit": np.zeros(n)}
        if prod:
            gen = ((s, np.asarray(st.prod[s:min(n, s + CHUNK_ROWS)], dtype=np.float32)) for s in range(0, n, CHUNK_ROWS))
        else:
            gen = logits_chunks(W, st.hidden, round_hidden)
        for s, lg in gen:
            for j in range(lg.shape[0]):
                i = s + j
                zt = st.teacher_row(i)
                tside = st.tside[i]
                ps = ls.position_stats(lg[j], zt, int(st.teacher_tokens[i]), tside["top16"], k_top=8)
                for k in ("argmax", "margin", "teacher_rank", "lse", "z_teacher_token", "ce_teacher_token", "mean_logit"):
                    arr[k][i] = ps[k]
                arr["kl"][i] = ps["kl_teacher_cand"]
                arr["z_top16"][i] = ps["z_top_ids"]
                arr["top8_ids"][i] = ps["top_ids"]
                arr["top8_logits"][i] = ps["top_logits"]
                arr["slope"][i] = ls.slope_on_teacher(ps["z_top_ids"], tside["top16_logits"])
                arr["z_we_minus_the"][i] = float(lg[j, WE] - lg[j, THE])
                nt = st.meta.get("next_token")
                if nt is not None and nt[i] >= 0:
                    arr["ce_next_token"][i] = float(ps["lse"] - lg[j, nt[i]])
        out[st.name] = arr
        print(f"[hr]   {st.name}: agree {int((arr['argmax'] == st.teacher_tokens).sum())}/{n}, mean KL {arr['kl'].mean():.4f}", flush=True)
    return out


def q(a):
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {"n": int(a.size), "mean": float(a.mean()), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)), "p90": float(np.percentile(a, 90))}


def bins(x, edges):
    for k in range(len(edges) - 1):
        if edges[k] <= x < edges[k + 1]:
            return k
    return len(edges) - 2


def summarize(sets, arms, tside_by_set):
    """Paired summaries of every arm against the primary reference s16_f32; the primary contrast is bf16_f32 v s16_f32."""
    summ = {"sets": {}, "contrasts": {}}
    ref = "s16_f32"
    for st in sets:
        tt = st.teacher_tokens
        s = {"n": st.n, "teacher": {"token_is_argmax": int(sum(t["token_is_argmax"] for t in tside_by_set[st.name])),
                                    "margin": q([t["margin"] for t in tside_by_set[st.name]]),
                                    "entropy": q([t["entropy"] for t in tside_by_set[st.name]])}, "arms": {}}
        for an, a in arms.items():
            if st.name not in a:
                continue
            r = a[st.name]
            agree = r["argmax"] == tt
            s["arms"][an] = {"agree": int(agree.sum()), "agree_rate": float(agree.mean()), "kl": q(r["kl"]), "ce_teacher_token": q(r["ce_teacher_token"]),
                             "ce_next_token": q(r["ce_next_token"]), "slope": q(r["slope"]), "margin": {"agree": q(r["margin"][agree]), "disagree": q(r["margin"][~agree])},
                             "teacher_rank_counts": {"0": int((r["teacher_rank"] == 0).sum()), "1": int((r["teacher_rank"] == 1).sum()),
                                                     "2": int((r["teacher_rank"] == 2).sum()), "3-15": int(((r["teacher_rank"] >= 3) & (r["teacher_rank"] < 16)).sum()),
                                                     "16+": int((r["teacher_rank"] >= 16).sum())},
                             "z_top16_minus_teacher_after_shift": None}
            # shift-aligned per-rank bias on the teacher top-16, the census statistic: (candidate - shift) - teacher with
            # shift = mean over the live vocabulary of candidate minus teacher logits
            tl = np.stack([t["top16_logits"] for t in tside_by_set[st.name]]).astype(np.float64)
            tmean = np.array([t["mean_logit"] for t in tside_by_set[st.name]])
            shift = r["mean_logit"] - tmean
            diff = (r["z_top16"].astype(np.float64) - shift[:, None]) - tl
            s["arms"][an]["z_top16_minus_teacher_after_shift"] = {"per_rank_mean": diff.mean(0).tolist(), "shift_mean": float(shift.mean())}
        summ["sets"][st.name] = s
        for an, a in arms.items():
            if an == ref or st.name not in a:
                continue
            ra, rb = arms[ref][st.name], a[st.name]
            aa, ab = ra["argmax"] == tt, rb["argmax"] == tt
            c = {"paired": ls.paired_counts(aa, ab), "blocks": ls.block_deltas(aa, ab, st.blocks),
                 "bootstrap": ls.block_bootstrap_delta_points(aa, ab, st.blocks, n_boot=10000, seed=0),
                 "kl_mean": {ref: float(ra["kl"].mean()), an: float(rb["kl"].mean()), "rel_delta": float(rb["kl"].mean() / ra["kl"].mean() - 1.0)},
                 "kl_paired_lower": int((rb["kl"] < ra["kl"]).sum()),
                 "ce_teacher_token_mean": {ref: float(ra["ce_teacher_token"].mean()), an: float(rb["ce_teacher_token"].mean())},
                 "slope_median": {ref: float(np.nanmedian(ra["slope"])), an: float(np.nanmedian(rb["slope"])),
                                  "delta": float(np.nanmedian(rb["slope"]) - np.nanmedian(ra["slope"]))},
                 "teacher_rank_improved": int((rb["teacher_rank"] < ra["teacher_rank"]).sum()), "teacher_rank_worsened": int((rb["teacher_rank"] > ra["teacher_rank"]).sum()),
                 "margin_delta_mean": {"agree_both": float((rb["margin"] - ra["margin"])[aa & ab].mean()) if (aa & ab).any() else None,
                                       "corrected": float((rb["margin"] - ra["margin"])[~aa & ab].mean()) if (~aa & ab).any() else None,
                                       "broken": float((rb["margin"] - ra["margin"])[aa & ~ab].mean()) if (aa & ~ab).any() else None},
                 "argmax_identical": int((ra["argmax"] == rb["argmax"]).sum())}
            if st.name == "e0":
                tm = np.array([t["margin"] for t in tside_by_set["e0"]])
                tb = np.array([bins(x, MARGIN_EDGES) for x in tm])
                c["by_teacher_margin_bin"] = [{"bin": [MARGIN_EDGES[k], MARGIN_EDGES[k + 1]], "n": int((tb == k).sum()),
                                               "agree_" + ref: int(aa[tb == k].sum()), "agree_" + an: int(ab[tb == k].sum())} for k in range(len(MARGIN_EDGES) - 1)]
                cats = np.array(sets[0].meta["category"])
                c["by_family"] = {f: {"n": int((cats == f).sum()), "agree_" + ref: int(aa[cats == f].sum()), "agree_" + an: int(ab[cats == f].sum())} for f in sorted(set(cats))}
                steps = np.array(sets[0].meta["step"])
                z0 = steps == 0
                c["opener_step0"] = {"n": int(z0.sum()), "argmax_" + ref: [int(x) for x in ra["argmax"][z0]], "argmax_" + an: [int(x) for x in rb["argmax"][z0]],
                                     "z_we_minus_the_" + ref: [float(x) for x in ra["z_we_minus_the"][z0]], "z_we_minus_the_" + an: [float(x) for x in rb["z_we_minus_the"][z0]],
                                     "teacher_token": [int(x) for x in tt[z0]], "agree_" + ref: int(aa[z0].sum()), "agree_" + an: int(ab[z0].sum()),
                                     "delta_z_we_minus_the_mean": float((rb["z_we_minus_the"] - ra["z_we_minus_the"])[z0].mean())}
            summ["contrasts"].setdefault(an, {})[st.name] = c
    # pooled validation (corpus + prefix) for every non-reference arm
    for an in arms:
        if an == ref or "corpus" not in arms[an]:
            continue
        aa = np.concatenate([arms[ref][s.name]["argmax"] == s.teacher_tokens for s in sets if s.name != "e0"])
        ab = np.concatenate([arms[an][s.name]["argmax"] == s.teacher_tokens for s in sets if s.name != "e0"])
        blocks = np.concatenate([sets[1].blocks, sets[2].blocks + int(sets[1].blocks.max()) + 1])
        ka = np.concatenate([arms[ref][s.name]["kl"] for s in sets if s.name != "e0"])
        kb = np.concatenate([arms[an][s.name]["kl"] for s in sets if s.name != "e0"])
        summ["contrasts"][an]["validation_pooled"] = {
            "paired": ls.paired_counts(aa, ab), "blocks": ls.block_deltas(aa, ab, blocks),
            "bootstrap": ls.block_bootstrap_delta_points(aa, ab, blocks, n_boot=10000, seed=0),
            "kl_mean": {ref: float(ka.mean()), an: float(kb.mean()), "rel_delta": float(kb.mean() / ka.mean() - 1.0)},
            "kl_paired_lower": int((kb < ka).sum())}
    return summ


def observations(summ, rec, prereg_path):
    from llmopt.lab.prereg import load, registration_digest
    doc = load(prereg_path)
    c = summ["contrasts"]["bf16_f32"]
    m1 = c["e0"]["paired"]["delta_points"]
    m2 = c["e0"]["slope_median"]["delta"]
    m3 = c["validation_pooled"]["paired"]["delta_points"]
    m4 = c["validation_pooled"]["kl_mean"]["rel_delta"]
    # bar 3 reads NO-FIRE when either subset net is negative, whatever the pooled delta; the refutation ratio folds that in
    m3_eff = m3 if (c["corpus"]["paired"]["net"] >= 0 and c["prefix"]["paired"]["net"] >= 0) else min(m3, 0.0)
    valid = bool(rec["inputs_ok"])
    digest = registration_digest(doc)
    if digest != REGISTRATION_SHA256:
        raise SystemExit(f"REFUSING: {prereg_path} digest {digest[:12]} is not the pinned registration {REGISTRATION_SHA256[:12]}")
    obs = {"note": f"Mechanical projection of {receipt_path('score')} (driver scratch/qwen_head_restore.py at {rec['code_commit']}); "
                   "primary contrast bf16_f32 against s16_f32 on identical hidden rows, float32 arithmetic, live vocabulary",
           "registration_sha256": digest, "measurement_valid": valid,
           "arms": {"S16": {"admissible": bool(rec["admissibility"]["s16"]), "reason": rec["admissibility"]["s16_reason"]},
                    "BF16": {"admissible": bool(rec["admissibility"]["bf16"]), "reason": rec["admissibility"]["bf16_reason"]},
                    "BF16_H16": {"admissible": bool(rec["admissibility"]["bf16"]), "reason": rec["admissibility"]["bf16_reason"] + "; hidden rounded to bf16 before the float32 product"},
                    "S16_PROD": {"admissible": True, "reason": "the census receipt's fp16 production logits of the same positions"}},
           "measurements": {
               "1": {"value": m1, "metric": "delta_agree_points_e0", "population": "positions:e0-4722", "aggregation": "paired_difference_points",
                     "provenance": f"E0 paired counts {c['e0']['paired']}"},
               "2": {"value": m2, "metric": "delta_median_slope_e0", "population": "positions:e0-4722", "aggregation": "paired_difference_of_medians",
                     "provenance": f"slope medians {c['e0']['slope_median']}"},
               "3": {"value": m3, "metric": "delta_agree_points_validation", "population": "positions:corpus-356+prefix-98", "aggregation": "paired_difference_points",
                     "provenance": f"validation paired counts {c['validation_pooled']['paired']}"},
               "3:corpus": {"value": c["corpus"]["paired"]["net"], "metric": "delta_agree_net_corpus", "population": "positions:corpus-356", "aggregation": "paired_net_count"},
               "3:prefix": {"value": c["prefix"]["paired"]["net"], "metric": "delta_agree_net_prefix", "population": "positions:prefix-98", "aggregation": "paired_net_count"},
               "4": {"value": m4, "metric": "rel_delta_mean_kl_validation", "population": "positions:corpus-356+prefix-98", "aggregation": "ratio_of_means_minus_one",
                     "provenance": f"validation KL means {c['validation_pooled']['kl_mean']}"},
               "max_bar13_ratio": {"value": max(m1 / 1.0, m3_eff / 1.0), "metric": "max_bar13_ratio", "population": "bars:1,3", "aggregation": "max_of_ratios"}},
           "contrasts": {str(k): {"admissible": valid, "reason": ("same hidden rows (sha-pinned), same vocabulary slice, same float32 arithmetic" if valid
                                                                  else rec["inputs_reason"])} for k in (1, 2, 3, 4)}}
    if not valid:
        obs["measurement_reason"] = rec["inputs_reason"]
        # the refutation clause reads a valid measurement only; without one it stays unadjudicated
        del obs["measurements"]["max_bar13_ratio"]
    return obs


def mode_score():
    import mlx.core as mx
    rpath = receipt_path("score")
    refuse(rpath)
    require_clean_tree()
    os.makedirs(DUMP, exist_ok=True)
    npz = os.path.join(DUMP, "score_arrays.npz")
    refuse(npz)
    opath = os.path.join(OUT, "observations.json")
    refuse(opath)
    rec = provenance("score")
    pre = json.load(open(receipt_path("preflight")))
    cor = json.load(open(receipt_path("corpus")))
    fetch = json.load(open(receipt_path("fetch"))) if os.path.exists(receipt_path("fetch")) else None
    inputs_ok = bool(pre.get("ok")) and bool(cor.get("ok"))
    reason = []
    if not pre.get("ok"):
        reason.append("preflight not ok")
    if not cor.get("ok"):
        reason.append("corpus stage not ok")
    rec["upstream"] = {"preflight_commit": pre["code_commit"], "corpus_commit": cor["code_commit"], "fetch_commit": fetch["code_commit"] if fetch else None}
    sets = build_sets(pre, cor)
    t_all = time.time()
    # teacher side once per set
    tside_by_set = {}
    for st in sets:
        st.tside = [teacher_side(st.teacher_row(i), int(st.teacher_tokens[i])) for i in range(st.n)]
        tside_by_set[st.name] = st.tside
    man, report = load_manifest()
    buf, entry = head_payload(man)
    arms = {}
    # --- arm C head, float32
    W = s16_head_f32(buf, entry)
    print("[hr] arm s16_f32", flush=True)
    arms["s16_f32"] = score_arm(sets, W)
    rec["correspondence"] = correspondence(sets, arms["s16_f32"])
    corr_ok = all(v["ok"] for v in rec["correspondence"].values())
    # CPU oracle for the S16 head: float64 over CPU-decoded rows at 8 positions of each set
    oracle = {}
    for st in sets:
        idx = np.linspace(0, st.n - 1, 8).astype(int)
        h = np.asarray(st.hidden[idx], dtype=np.float32)
        dev = np.asarray(mx.matmul(mx.array(h), W.T))[:, :V_LIVE]
        cpu = np.zeros_like(dev)
        for r0 in range(0, V_LIVE, 16384):
            r1 = min(V_LIVE, r0 + 16384)
            cpu[:, r0:r1] = h.astype(np.float64) @ s16_rows_cpu(buf, entry, r0, r1).astype(np.float64).T
        oracle[st.name] = {"rows": idx.tolist(), "max_abs_diff": float(np.abs(cpu - dev).max()), "argmax_equal": int((cpu.argmax(1) == dev.argmax(1)).sum()),
                           "recorded_argmax_equal": int((arms["s16_f32"][st.name]["argmax"][idx] == dev.argmax(1)).sum())}
    rec["s16_cpu_oracle"] = oracle
    s16_ok = all(o["argmax_equal"] == 8 and o["max_abs_diff"] < 0.05 for o in oracle.values())
    del W
    mx.clear_cache()
    print("[hr] arm s16_prod (E0 production fp16 logits)", flush=True)
    arms["s16_prod"] = score_arm(sets, None, prod=True)
    # --- vendor head
    Wh = bf16_head_host()
    standin = False
    if Wh is None:
        standin = True
        W2 = s16_head_f32(buf, entry)
        Wh = ls.round_to_bf16(np.asarray(W2))
        del W2
    bf16_meta = {"stand_in": standin}
    if fetch is not None and not standin:
        bf16_meta.update({"sha256": fetch["sha256"], "size_bytes": fetch["size_bytes"], "lm_head_entry": fetch["lm_head_entry"]})
    # head-level comparison: relative Frobenius error of the S16 head against the vendor head, by row slabs on the CPU
    num = den = 0.0
    maxrow = 0.0
    for r0 in range(0, HEAD_SHAPE[0], 16384):
        r1 = min(HEAD_SHAPE[0], r0 + 16384)
        d = s16_rows_cpu(buf, entry, r0, r1).astype(np.float64) - Wh[r0:r1].astype(np.float64)
        num += float((d * d).sum())
        den += float((Wh[r0:r1].astype(np.float64) ** 2).sum())
        maxrow = max(maxrow, float(np.sqrt((d * d).sum(1) / np.maximum((Wh[r0:r1].astype(np.float64) ** 2).sum(1), 1e-30)).max()))
    bf16_meta["s16_head_rel_fro_error"] = float(np.sqrt(num / den))
    bf16_meta["s16_head_max_row_rel_l2_error"] = maxrow
    Wd = mx.array(Wh)
    mx.eval(Wd)
    # CPU oracle for the vendor head: numpy float64 over row slabs at 8 positions of each set against the device product
    oracle_b = {}
    for st in sets:
        idx = np.linspace(0, st.n - 1, 8).astype(int)
        h = np.asarray(st.hidden[idx], dtype=np.float32)
        dev = np.asarray(mx.matmul(mx.array(h), Wd.T))[:, :V_LIVE]
        cpu = np.zeros_like(dev)
        for r0 in range(0, V_LIVE, 16384):
            r1 = min(V_LIVE, r0 + 16384)
            cpu[:, r0:r1] = h.astype(np.float64) @ Wh[r0:r1].astype(np.float64).T
        oracle_b[st.name] = {"rows": idx.tolist(), "max_abs_diff": float(np.abs(cpu - dev).max()), "argmax_equal": int((cpu.argmax(1) == dev.argmax(1)).sum())}
    rec["bf16_cpu_oracle"] = oracle_b
    bf16_ok = all(o["argmax_equal"] == 8 and o["max_abs_diff"] < 0.05 for o in oracle_b.values()) and not standin
    del Wh
    print("[hr] arm bf16_f32", flush=True)
    arms["bf16_f32"] = score_arm(sets, Wd)
    print("[hr] arm bf16_h16", flush=True)
    arms["bf16_h16"] = score_arm(sets, Wd, round_hidden=True)
    del Wd
    mx.clear_cache()
    summ = summarize(sets, arms, tside_by_set)
    # per-position arrays (machine-local, sha in the receipt)
    flat = {}
    for an, a in arms.items():
        for sn, r in a.items():
            for k, v in r.items():
                flat[f"{an}.{sn}.{k}"] = v
    for st in sets:
        flat[f"teacher.{st.name}.token"] = st.teacher_tokens
        flat[f"teacher.{st.name}.argmax"] = np.array([t["argmax"] for t in st.tside])
        flat[f"teacher.{st.name}.margin"] = np.array([t["margin"] for t in st.tside])
        flat[f"teacher.{st.name}.top16"] = np.stack([t["top16"] for t in st.tside])
        flat[f"teacher.{st.name}.top16_logits"] = np.stack([t["top16_logits"] for t in st.tside])
        flat[f"blocks.{st.name}"] = st.blocks
    np.savez(npz, **flat)
    rec.update({"inputs_ok": inputs_ok and s16_ok and bf16_ok and corr_ok,
                "inputs_reason": "; ".join(reason + ([] if s16_ok else ["s16 oracle"]) + ([] if bf16_ok else ["bf16 oracle or stand-in"]) + ([] if corr_ok else ["hidden/logit correspondence"])),
                "admissibility": {"s16": s16_ok, "s16_reason": f"decode parity (preflight) and float64 CPU oracle {oracle}",
                                  "bf16": bf16_ok, "bf16_reason": ("stand-in head (smoke)" if standin else f"vendor shard sha verified at fetch; header BF16 {list(HEAD_SHAPE)}; CPU oracle {oracle_b}")},
                "bf16_head": bf16_meta, "head_entry": entry, "artifact_qualification": {"rung0": report["rung0"]},
                "sets": {st.name: {"n": st.n, "blocks": int(len(set(st.blocks.tolist())))} for st in sets},
                "summary": summ, "arrays": {"path": npz, "sha256": sha_file(npz)}, "wall_s": time.time() - t_all, "rss_peak_bytes": rss_peak_bytes()})
    obs = observations(summ, rec, PREREG)
    rec["observations"] = opath
    write_strict(rec, rpath)
    write_strict(obs, opath)
    c = summ["contrasts"]["bf16_f32"]
    print(f"[hr] score: E0 {c['e0']['paired']} slope {c['e0']['slope_median']}; validation {c['validation_pooled']['paired']} "
          f"KL {c['validation_pooled']['kl_mean']}; wrote {rpath} and {opath}", flush=True)


# ---------------------------------------------------------------- runtime (conditional)

def mode_runtime():
    """Serving cost of a resident bfloat16 head against the S16 head: 1,024-token
    prefill (head="final"), 32 cached decode steps, memory. The bf16 head takes
    the hidden row in bfloat16 and multiplies in bfloat16 (the BF16_H16 arm's
    numerics, the realistic resident form); a float32 product would widen 5 GB
    per call and is not a serving option. Runs only on a VALIDATED verdict."""
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm.models.cache import make_prompt_cache
    from llmopt.lab import qprefill
    rpath = receipt_path("runtime")
    refuse(rpath)
    require_clean_tree()
    rec = provenance("runtime")
    rec["serving_applied"] = serving()
    shard = vendor_shard_path()
    man, report = load_manifest()
    buf, entry = head_payload(man)
    model, info, load_wall = load_model()
    rec["load_wall_s"] = load_wall
    tman = teacher_manifest()
    ids = [int(t) for t in tman["records"]["corpus"]["tokens"]]
    while len(ids) < 1024:
        ids = ids + ids
    ids = ids[:1024] if not SMOKE else ids[:64]
    n_rep = 3 if not SMOKE else 1
    n_dec = 32 if not SMOKE else 4
    s16_head = model.language_model.lm_head

    class Bf16Head(nn.Module):
        def __init__(self, W):
            super().__init__()
            self.W = W

        def __call__(self, h):
            lead = h.shape[:-1]
            y = mx.matmul(h.reshape(-1, h.shape[-1]).astype(mx.bfloat16), self.W.T)
            return y.astype(mx.float32).reshape(*lead, self.W.shape[0])

    def bf16_head():
        if shard is None:                       # smoke stand-in: the S16 head rounded to bfloat16 (never evidence)
            W = s16_head_f32(buf, entry)
            bits = ls.f32_to_bf16_bits(np.asarray(W))
            del W
        else:
            bits = ls.read_bf16_bits(shard, "lm_head.weight", expect_shape=HEAD_SHAPE)
        Wb = mx.array(bits).view(mx.bfloat16)
        mx.eval(Wb)
        return Bf16Head(Wb)

    out = {}
    for name in ("s16", "bf16"):
        head = s16_head if name == "s16" else bf16_head()
        model.language_model.lm_head = head
        mx.reset_peak_memory()
        mem0 = qprefill.memory_report()
        prefill_walls, dec = [], []
        last = None
        for rep in range(n_rep):
            cache = make_prompt_cache(model)
            t0 = time.time()
            lg = qprefill.prefill_chunked(model, ids, cache, chunk=256, head="final", matmul="gemm")
            prefill_walls.append(time.time() - t0)
            nxt = int(np.asarray(lg[0, -1]).argmax())
            toks = [nxt]
            for _ in range(n_dec):
                t1 = time.time()
                z = model(mx.array([[nxt]]), cache=cache)[0, -1]
                mx.eval(z)
                dec.append(time.time() - t1)
                nxt = int(np.asarray(z).argmax())
                toks.append(nxt)
            if last is not None and toks != last:
                rec.setdefault("decode_token_mismatch_between_reps", []).append(name)
            last = toks
            del cache
        out[name] = {"prefill_tok_s": [len(ids) / t for t in prefill_walls], "decode_ms_median": float(1000 * np.median(dec)),
                     "decode_ms_mean": float(1000 * np.mean(dec)), "decode_steps_timed": len(dec), "decode_tokens": last,
                     "memory_before": mem0, "memory_after": qprefill.memory_report()}
        print(f"[hr] runtime {name}: prefill {[round(len(ids) / t, 1) for t in prefill_walls]} tok/s, decode {1000 * np.median(dec):.1f} ms", flush=True)
        if name == "bf16":
            out[name]["head_bytes"] = int(head.W.nbytes)
            out[name]["numerics"] = "bfloat16 hidden x bfloat16 head, float32 output (the BF16_H16 arm)"
            out[name]["stand_in"] = shard is None
        del head
    rec["heads"] = out
    rec["prompt_tokens"] = len(ids)
    rec["decode_argmax_equal_s16_bf16"] = int(sum(a == b for a, b in zip(out["s16"]["decode_tokens"], out["bf16"]["decode_tokens"])))
    write_strict(rec, rpath)
    print(f"[hr] runtime wrote {rpath}", flush=True)


def main():
    modes = {"preflight": mode_preflight, "opener": mode_opener, "corpus": mode_corpus, "fetch": mode_fetch, "score": mode_score, "runtime": mode_runtime}
    if MODE not in modes:
        raise SystemExit(f"MODE must be one of {sorted(modes)}")
    modes[MODE]()


if __name__ == "__main__":
    main()
