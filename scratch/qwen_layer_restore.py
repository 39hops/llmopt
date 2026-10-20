"""QWEN-LAYER-RESTORE-0: one bounded layer-restoration pilot on arm C.

Replaces exactly two decoder layers of the compressed arm C tower (22,
gated DeltaNet; 23, full attention; both complete in vendor shard
model-00007-of-00018) with the vendor's original BF16 weights widened to
float32, inside an otherwise compressed network, and measures the
teacher-forced consequences on E0 (diagnostic, 4,722 positions) and the
corpus and prefix teacher records (validation, 454 positions). Nothing is
trained, fitted or selected on an outcome. Spec:
docs/superpowers/specs/2026-10-20-layer-restore-pilot.md.

Modes (MODE=...), one process each, receipts under
logs/qwen_layer_restore/<mode>[...].json, refuse-if-exists; SMOKE=1
writes under logs/qwen_layer_restore_smoke/ with a labelled stand-in
(the decoded arm C tensors as the "restored" weights) and tiny contexts:

  preflight   vendor index coverage of the target pair, arm C manifest
              entries, pinned shard metadata, storage and memory, frozen
              inputs (E0 rows, teacher records, census dump shas)
  capture     the compressed model's residual stream entering layer 22,
              leaving 22 and leaving 23 on the validation contexts (the
              frozen inputs of the one-layer oracle), dumped float32
  oracle_mlx  WEIGHTS=armc|vendor: standalone mlx-lm dense float32
              layers 22 and 23 (the tower's class with its l2norm
              override) on the captured inputs, one shot and chunk-256
              with the cache carried; outputs dumped for the torch side
  fetch       the one shard, hub metadata equal to the pins, sha and
              size verified, header parsed, target tensors present
  score       ARM=control|dense_c|l22|l23|group: the full tower with the
              arm's layers installed, teacher-forced over every E0
              rollout and the validation contexts, fresh cache per
              sequence, per-position statistics against the teacher
              logits; CAPTURE=1 also dumps nine residual-stream
              boundaries for the sensitivity instrument
  summarize   paired contrasts of every arm against control, the
              DENSE_C parity precondition, observations for the
              adjudicator
  sensitivity weight reconstruction error per tensor, the local
              activation difference and its evolution through the
              downstream boundaries, the final response (from the
              control and group captures)

The torch half of the oracle is scratch/qwen_layer_restore_torch.py
(run with .venv_teacher). Env: ART_DIR, VENDOR_DIR, VENDOR_SHARD_DIR,
OUT, SMOKE, MODE, WEIGHTS, ARM, CAPTURE.
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from llmopt.lab import layer_restore as lr  # noqa: E402
from llmopt.lab import logit_stats as ls  # noqa: E402
from llmopt.lab.strict_json import write_strict  # noqa: E402

SMOKE = os.environ.get("SMOKE") == "1"
MODE = os.environ.get("MODE", "preflight")
ART_DIR = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
VENDOR_DIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
REPO = "Qwen/Qwen3.8-27B"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
SHARD = "model-00007-of-00018.safetensors"
SHARD_BYTES = 2108759344
SHARD_SHA256 = "80b0c49033e9a0d5762562aa12f4acdb7f54da586f3d0110f28c48d91cf07892"
VENDOR_SHARD_DIR = os.path.expanduser(os.environ.get("VENDOR_SHARD_DIR", f"~/qwen_vendor_shards/{REVISION[:8]}"))
LAYERS = (22, 23)
FAMILY = {22: "linear_attn", 23: "self_attn"}
ARMS = {"control": (), "dense_c": (22, 23), "l22": (22,), "l23": (23,), "group": (22, 23)}
ARM_SOURCE = {"control": None, "dense_c": "armc", "l22": "vendor", "l23": "vendor", "group": "vendor"}
BOUNDARIES = ["in22", "out23", "out27", "out31", "out39", "out47", "out55", "out63", "final_norm"]
V_LIVE = 248077
WE, THE = 1596, 760
HIDDEN = 5120
TEACHER_DIR = "logs/qwenteacher_v2"
E0_DIR = "logs/qwen_e0/C"
EV_DIR = "evals/qwen_model1"
CENSUS_RECEIPT = "logs/qmetal/prefill_serve/census_d42cb215.json"
E0_HIDDEN_SHA256 = "01612fb6c99b7dcd87c03abc0b397741dcced5818cd34cc981d6e3ed62df74b4"
E0_LOGITS_SHA256 = "79162c5ddb7d974a445b3399a5bda75084ae2f4fd4117dd6c24f4ca37a0a1a3a"
OUT = "logs/qwen_layer_restore_smoke" if SMOKE else os.environ.get("OUT", "logs/qwen_layer_restore")
if not SMOKE and os.path.normpath(OUT) == "logs/qwen_layer_restore_smoke":
    raise SystemExit("REFUSING: a real run must not write into the smoke directory")
DUMP = os.path.join(OUT, "dump")
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
CHUNK = 256
ORACLE_REL_L2 = 1e-4          # registered: relative L2 of the layer output difference, per layer, all positions
ORACLE_MAX_ABS_OVER_RMS = 1e-3  # registered: max abs difference over the reference row RMS
PARITY_MARGIN = 0.01          # DENSE_C: argmax must equal control wherever the control margin exceeds this
PARITY_LOGIT_ABS = 1e-2       # DENSE_C: teacher top-16 logits within this of control
NEAR_TIE_GAP = 0.05           # CONTROL against the E0 rows: an argmax flip below this margin is an fp16 near-tie, not a defect
PREREG = "docs/preregs/qwen-layer-restore-0.json"
REGISTRATION_SHA256 = "ca108cd641ee525921e05957f462c552e8602aa167fa5f5cbc20cbed630d4f88"   # registration_digest of docs/preregs/qwen-layer-restore-0.json
MEM_BUDGET_BYTES = 24 << 30   # projected peak must stay under this (36 GiB machine)


# ---------------------------------------------------------------- house helpers

def git_head():
    return subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def git_dirty():
    return bool(subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True).stdout.strip())


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def sha_array_file(path):
    a = np.load(path, mmap_mode="r")
    h = hashlib.sha256()
    step = max(1, (1 << 26) // max(1, a[0].nbytes)) if a.ndim > 1 else a.shape[0]
    for s in range(0, a.shape[0], step):
        h.update(np.ascontiguousarray(a[s:s + step]).tobytes())
    return h.hexdigest()


def refuse(path):
    if os.path.exists(path):
        raise SystemExit(f"REFUSING: {path} exists")


def receipt_path(name):
    return os.path.join(OUT, f"{name}.json")


def require_clean_tree():
    if not SMOKE and git_dirty():
        raise SystemExit("REFUSING: dirty tree (tracked changes); a real receipt must name a commit that holds its code")


def repo_root():
    return os.path.dirname(HERE)


def provenance(mode):
    return {"mode": mode, "smoke": SMOKE, "code_commit": git_head(), "tree_dirty": git_dirty(),
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "driver": "scratch/qwen_layer_restore.py", "art_role": "ART_DIR (arm C)", "vendor_role": "VENDOR_DIR (config, tokenizer)",
            "vendor_shard_role": "VENDOR_SHARD_DIR (outside the repository)", "repo": REPO, "revision": REVISION, "shard": SHARD,
            "layers": list(LAYERS), "census_receipt": CENSUS_RECEIPT, "teacher_dir": TEACHER_DIR}


def rss_peak_bytes():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


# ---------------------------------------------------------------- artifacts

def load_manifest():
    from llmopt.lab import qartifact
    q = qartifact.qualify_artifact(ART_DIR, os.path.join(VENDOR_DIR, "model.safetensors.index.json"),
                                   f"logs/qwenwhole/artifact_digest_{os.path.basename(ART_DIR.rstrip('/'))}.txt")
    return q["manifest"], q["report"]


def vendor_index():
    return json.load(open(os.path.join(VENDOR_DIR, "model.safetensors.index.json")))["weight_map"]


def target_names(idx):
    return {L: sorted(lr.layer_tensor_map(idx, L)) for L in LAYERS}


_HANDLES: dict = {}


def payload(e):
    sh = e["shard"]
    if sh not in _HANDLES:
        _HANDLES[sh] = open(os.path.join(ART_DIR, sh + ".bin"), "rb")
    _HANDLES[sh].seek(e["off"])
    return _HANDLES[sh].read(e["len"])


def armc_tensor(man, name):
    """The arm C tensor decoded to float32 in the vendor layout (no load conventions applied)."""
    from llmopt.lab.qcodec import decode_entry
    e = man[name]
    if e["codec"] == "excluded":
        raise SystemExit(f"REFUSING: {name} is excluded from arm C")
    return np.ascontiguousarray(decode_entry(payload(e), e)).astype(np.float32)


def vendor_shard_path():
    return os.path.join(VENDOR_SHARD_DIR, SHARD)


def vendor_tensor(name, expect_shape):
    return ls.read_bf16_tensor(vendor_shard_path(), name, expect_shape=expect_shape)


def source_tensors(source, man, names):
    """{vendor name: float32 array in the vendor layout} for one layer from
    'armc' (decoded arm C, also the smoke stand-in) or 'vendor' (the shard)."""
    if source in ("armc", "standin"):
        return {n: armc_tensor(man, n) for n in names}
    if source == "vendor":
        if SMOKE:
            raise SystemExit("REFUSING: the vendor source is never read in smoke")
        return {n: vendor_tensor(n, tuple(man[n]["shape"])) for n in names}
    raise SystemExit(f"unknown source {source}")


def vendor_ready():
    """The fetch receipt says ok and the shard on disk still has the pinned sha."""
    fp = receipt_path("fetch")
    if not os.path.exists(fp):
        return False, "no fetch receipt"
    f = json.load(open(fp))
    if not f.get("ok"):
        return False, "fetch receipt not ok"
    p = vendor_shard_path()
    if not os.path.exists(p) or os.path.getsize(p) != SHARD_BYTES:
        return False, "shard missing or wrong size"
    if sha_file(p) != SHARD_SHA256:
        return False, "shard sha differs from the pin"
    return True, "fetch receipt ok; shard sha re-verified at stage start"


# ---------------------------------------------------------------- model

def l2norm_override():
    """The tower's exact-l2norm override of mlx-lm's GatedDeltaNet (the same
    two-line replacement scratch/qwen_metal_qual.build_metal_model applies,
    with the same refusal when the upstream source moves), so a standalone
    dense layer runs the code the tower runs."""
    import inspect
    import textwrap
    from mlx_lm.models import qwen3_5
    if getattr(qwen3_5, "_llmopt_l2norm_override", False):
        return
    src = textwrap.dedent(inspect.getsource(qwen3_5.GatedDeltaNet.__call__))
    a_line = "q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)"
    b_line = "k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)"
    if a_line not in src or b_line not in src:
        raise SystemExit("REFUSING: mlx-lm GatedDeltaNet source changed; re-derive the l2norm override")
    src = src.replace(a_line, "q = inv_scale * (q * mx.rsqrt((q * q).sum(-1, keepdims=True) + 1e-6))")
    src = src.replace(b_line, "k = k * mx.rsqrt((k * k).sum(-1, keepdims=True) + 1e-6)")
    ns: dict = {}
    exec(compile(src, "qwen3_5_gdn_exact_l2norm", "exec"), vars(qwen3_5), ns)
    qwen3_5.GatedDeltaNet.__call__ = ns["__call__"]
    qwen3_5._llmopt_l2norm_override = True


def load_model():
    from qwen_metal_qual import build_metal_model
    from mlx_lm.models import qwen3_5
    t0 = time.time()
    model, info = build_metal_model(ART_DIR, VENDOR_DIR)
    qwen3_5._llmopt_l2norm_override = True   # build_metal_model applied the same override
    return model, info, time.time() - t0


def serving():
    from llmopt.lab import qprefill
    cfg = qprefill.ServingConfig()
    qprefill.configure_serving_memory(cfg)
    return {"chunk": cfg.chunk, "matmul": cfg.matmul, "cache_limit_bytes": cfg.cache_limit_bytes}


def set_path(root, path, value):
    parts = path.split(".")
    obj = root
    for p in parts[:-1]:
        obj = obj[int(p)] if p.isdigit() else getattr(obj, p)
    leaf = parts[-1]
    if leaf.isdigit():
        obj[int(leaf)] = value
    else:
        setattr(obj, leaf, value)


def install_dense(model, tensors):
    """Install {vendor name: float32 array (vendor layout)} into the tower:
    two-dimensional weights become dense float32 nn.Linear modules (the
    fused residents they replace are released), everything else is set as
    an array after the mlx-lm load conventions. Returns the bytes installed."""
    import mlx.core as mx
    import mlx.nn as nn
    nbytes = 0
    for name, arr in tensors.items():
        path = lr.mlx_param_path(name)
        arr = lr.sanitize_vendor_tensor(name, arr)
        if arr.ndim == 2 and name.endswith(".weight"):
            lin = nn.Linear(arr.shape[1], arr.shape[0], bias=False)
            lin.weight = mx.array(arr)
            set_path(model, path[:-len(".weight")], lin)
        else:
            set_path(model, path, mx.array(arr))
        nbytes += arr.nbytes
    mx.eval(model.parameters())
    return nbytes


def _tap_class():
    import mlx.nn as nn

    class _Tap(nn.Module):
        """Wraps one decoder layer: records the layer's input and/or output rows
        (mx arrays, drained to host after the chunk is evaluated). An nn.Module
        so the tower's parameter tree and named_modules still traverse it."""

        def __init__(self, inner, want_in, want_out, store, key_in, key_out):
            super().__init__()
            self.inner = inner
            self.want_in, self.want_out, self.store, self.key_in, self.key_out = want_in, want_out, store, key_in, key_out
            self.is_linear = inner.is_linear

        def __call__(self, x, mask=None, cache=None):
            if self.want_in:
                self.store.setdefault(self.key_in, []).append(x)
            y = self.inner(x, mask, cache)
            if self.want_out:
                self.store.setdefault(self.key_out, []).append(y)
            return y
    return _Tap


class _RecordingHead:
    def __init__(self, lm, store, key):
        self.lm, self.store, self.key = lm, store, key

    def __call__(self, h):
        self.store.setdefault(self.key, []).append(h)
        return self.lm(h)


def install_taps(model, boundaries):
    """Taps for the named boundaries (inNN: input of layer NN; outNN: output;
    final_norm: the head's input). Returns the store."""
    store: dict = {}
    layers = model.language_model.model.layers
    want: dict = {}
    for b in boundaries:
        if b == "final_norm":
            continue
        kind, L = b[:-2], int(b[-2:])
        want.setdefault(L, {"in": False, "out": False})[kind] = True
    Tap = _tap_class()
    for L, w in want.items():
        layers[L] = Tap(layers[L], w["in"], w["out"], store, f"in{L}", f"out{L}")
    if "final_norm" in boundaries:
        model.language_model.lm_head = _RecordingHead(model.language_model.lm_head, store, "final_norm")
    return store


def drain(store, rows):
    """Move the stored chunk arrays to host as float32 [rows, D] and clear."""
    out = {}
    for k, arrs in store.items():
        out[k] = np.concatenate([np.asarray(a[0], dtype=np.float32) for a in arrs], axis=0)
        if out[k].shape[0] != rows:
            raise SystemExit(f"REFUSING: boundary {k} rows {out[k].shape[0]} != tokens {rows}")
    store.clear()
    return out


# ---------------------------------------------------------------- contexts

def teacher_manifest():
    return json.load(open(os.path.join(TEACHER_DIR, "teacher_manifest.json")))


def e0_rows():
    return [json.loads(line) for line in open(os.path.join(E0_DIR, "positions.jsonl"))]


def validation_contexts(tok=None):
    """corpus ids (356), prefix ids (6 lists), with the retokenisation and
    bounds checks the head-restore corpus stage made."""
    tman = teacher_manifest()
    rc, rp = tman["records"]["corpus"], tman["records"]["prefixes"]
    corpus_ids = [int(t) for t in rc["tokens"]]
    prefix_ids = [[int(t) for t in p] for p in rp["tokens"]]
    prov = {"corpus_txt_sha_ok": sha_file(os.path.join(EV_DIR, "corpus.txt")) == tman["inputs"]["corpus_sha256"],
            "prefixes_jsonl_sha_ok": sha_file(os.path.join(EV_DIR, "prefixes.jsonl")) == tman["inputs"]["prefixes_sha256"],
            "prefix_bounds_ok": [len(p) for p in prefix_ids] == list(rp["bounds"]) and sum(rp["bounds"]) == rp["shape"][0],
            "corpus_len_ok": len(corpus_ids) == rc["shape"][0]}
    if tok is not None:
        prov["corpus_retokenize_equal"] = tok(open(os.path.join(EV_DIR, "corpus.txt")).read())["input_ids"] == corpus_ids
        prov["prefix_retokenize_equal"] = [tok(json.loads(l)["text"])["input_ids"] for l in open(os.path.join(EV_DIR, "prefixes.jsonl"))] == prefix_ids
    if SMOKE:
        corpus_ids = corpus_ids[:32]
        prefix_ids = prefix_ids[:1]
    return corpus_ids, prefix_ids, prov, tman


def corpus_blocks(tokens, nl2):
    b = np.cumsum([1 if t == nl2 else 0 for t in tokens])
    b = np.concatenate([[0], b[:-1]])
    return b.astype(np.int64)


def e0_contexts(tman):
    """[(prompt_id, prompt ids, generated ids)] in E0 row order."""
    ro = tman["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(TEACHER_DIR, "rollout_tokens.jsonl"))]
    input_ids, mask = np.asarray(ro["input_ids"]), np.asarray(ro["attention_mask"])
    gl = list(ro["gen_lengths"])
    out = []
    if [r["id"] for r in recs] != list(ro["prompt_ids"]):
        raise SystemExit("REFUSING: rollout_tokens.jsonl order differs from the teacher manifest's prompt_ids")
    for i, r in enumerate(recs):
        prompt = [int(x) for x in input_ids[i][mask[i] == 1]]
        gen = [int(x) for x in r["tokens"][:gl[i]]]
        out.append((r["id"], prompt, gen))
    if SMOKE:
        out = [(out[0][0], out[0][1], out[0][2][:8])]
    return out


# ---------------------------------------------------------------- preflight

def mode_preflight():
    rpath = receipt_path("preflight")
    refuse(rpath)
    require_clean_tree()
    rec = provenance("preflight")
    t0 = time.time()
    idx = vendor_index()
    man, report = load_manifest()
    cov = lr.shard_coverage(idx)
    names = target_names(idx)
    checks = {}
    checks["targets_complete_in_shard"] = all(L in cov[SHARD]["complete"] for L in LAYERS)
    checks["targets_in_one_shard"] = all(set(lr.layer_tensor_map(idx, L).values()) == {SHARD} for L in LAYERS)
    checks["family_ok"] = all(any(FAMILY[L] in n for n in names[L]) for L in LAYERS)
    ents = {L: {n: {"codec": man[n]["codec"], "shape": man[n]["shape"], "dtype": man[n].get("dtype"), "len": man[n]["len"]} for n in names[L]} for L in LAYERS}
    checks["no_excluded_target"] = all(e["codec"] != "excluded" for L in LAYERS for e in ents[L].values())
    bf16_bytes = {L: int(sum(int(np.prod(e["shape"])) * 2 for e in ents[L].values())) for L in LAYERS}
    f32_bytes = {L: 2 * bf16_bytes[L] for L in LAYERS}
    comp_bytes = {L: int(sum(e["len"] for e in ents[L].values())) for L in LAYERS}
    # pinned shard metadata from the hub (read only)
    try:
        meta = vendor_metadata()
        checks["hub_metadata_equals_pins"] = meta["lfs_sha256"] == SHARD_SHA256 and meta["size"] == SHARD_BYTES and meta["lfs_size"] == SHARD_BYTES
    except Exception as ex:  # noqa: BLE001
        meta = {"error": type(ex).__name__}
        checks["hub_metadata_equals_pins"] = False
    # storage
    root, dest, vreal = os.path.realpath(repo_root()), os.path.realpath(VENDOR_SHARD_DIR), os.path.realpath(VENDOR_DIR)
    checks["dest_outside_repo"] = not (dest == root or dest.startswith(root + os.sep))
    checks["dest_not_vendor_dir"] = not (dest == vreal or dest.startswith(vreal + os.sep))
    probe = dest
    while not os.path.exists(probe):
        probe = os.path.dirname(probe)
    import shutil
    free = shutil.disk_usage(probe).free
    already = os.path.exists(vendor_shard_path()) and os.path.getsize(vendor_shard_path()) == SHARD_BYTES
    checks["free_bytes_ok"] = already or free > SHARD_BYTES + (4 << 30)
    # memory: census peak + float32 pair
    cen = json.load(open(CENSUS_RECEIPT))
    peak_census = int(cen["memory_after"]["peak_bytes"])
    projected = peak_census + sum(f32_bytes.values()) - sum(comp_bytes.values())
    checks["memory_projection_ok"] = projected < MEM_BUDGET_BYTES
    # frozen inputs
    d = cen["dump_dir"]
    h_name = [n for n in cen["dumps"] if n.startswith("draft_hidden")][0]
    l_name = [n for n in cen["dumps"] if n.startswith("draft_logits")][0]
    checks["census_dump_shas_pinned"] = cen["dumps"][h_name] == E0_HIDDEN_SHA256 and cen["dumps"][l_name] == E0_LOGITS_SHA256
    checks["census_dump_present"] = os.path.exists(os.path.join(d, h_name)) and os.path.exists(os.path.join(d, l_name))
    rows = e0_rows()
    tman = teacher_manifest()
    ro = tman["records"]["rollouts"]
    checks["e0_rows_count_ok"] = len(rows) == sum(ro["gen_lengths"]) == 4722
    checks["teacher_rollout_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "rollout_logits.npy")) == ro["sha256"]
    checks["teacher_corpus_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "corpus_logits.npy")) == tman["records"]["corpus"]["sha256"]
    checks["teacher_prefix_sha_ok"] = sha_array_file(os.path.join(TEACHER_DIR, "prefix_logits.npy")) == tman["records"]["prefixes"]["sha256"]
    _, _, vprov, _ = validation_contexts()
    checks.update({"validation_" + k: v for k, v in vprov.items()})
    checks["head_shard_present"] = os.path.exists(os.path.join(VENDOR_SHARD_DIR, "model-00018-of-00018.safetensors"))
    rec.update({"artifact_report": report, "shard_coverage_target": cov[SHARD], "n_layers_straddling": sum(len(v["partial"]) for v in cov.values()) // 2,
                "straddling_layers": sorted({L for v in cov.values() for L in v["partial"]}),
                "target_tensors": ents, "bytes": {"bf16": bf16_bytes, "f32_installed": f32_bytes, "arm_c_compressed": comp_bytes},
                "shard_pins": {"bytes": SHARD_BYTES, "sha256": SHARD_SHA256}, "hub_metadata": meta,
                "storage": {"dest_role": "VENDOR_SHARD_DIR", "free_bytes_at_dest": int(free), "shard_already_present": bool(already)},
                "memory": {"census_peak_bytes": peak_census, "projected_peak_bytes": int(projected), "budget_bytes": MEM_BUDGET_BYTES},
                "tolerances": {"oracle_rel_l2": ORACLE_REL_L2, "oracle_max_abs_over_rms": ORACLE_MAX_ABS_OVER_RMS,
                               "parity_margin": PARITY_MARGIN, "parity_logit_abs": PARITY_LOGIT_ABS},
                "checks": checks, "ok": all(checks.values()), "wall_s": time.time() - t0})
    write_strict(rec, rpath)
    print(f"[lr] preflight ok={rec['ok']} checks={checks} projected peak {projected / 1e9:.1f} GB; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("preflight failed")


def vendor_metadata():
    from huggingface_hub import HfApi
    info = HfApi().get_paths_info(REPO, [SHARD], revision=REVISION)
    if len(info) != 1 or info[0].lfs is None:
        raise SystemExit("REFUSING: vendor metadata for the shard is unavailable")
    return {"size": int(info[0].size), "lfs_sha256": info[0].lfs.sha256, "lfs_size": int(info[0].lfs.size)}


# ---------------------------------------------------------------- capture (oracle inputs)

def run_prefill(model, ids, store=None, chunk=CHUNK):
    """One sequence through prefill_chunked with a fresh cache; returns the
    float32 logits of every position [n, V_LIVE] (host) and the drained
    boundary rows when a store is given."""
    from mlx_lm.models.cache import make_prompt_cache
    from llmopt.lab import qprefill
    chunks = []
    cache = make_prompt_cache(model)
    t0 = time.time()
    qprefill.prefill_chunked(model, ids, cache, chunk=chunk, head="full", matmul="gemm",
                             on_chunk=lambda s, lg: chunks.append(np.asarray(lg[0], dtype=np.float32)[:, :V_LIVE]))
    wall = time.time() - t0
    lg = np.concatenate(chunks, axis=0)
    if lg.shape[0] != len(ids):
        raise SystemExit("REFUSING: logits rows != tokens")
    bnd = drain(store, len(ids)) if store is not None else None
    del cache, chunks
    return lg, bnd, wall


def mode_capture():
    rpath = receipt_path("capture")
    refuse(rpath)
    require_clean_tree()
    os.makedirs(DUMP, exist_ok=True)
    from transformers import AutoTokenizer
    rec = provenance("capture")
    rec["serving_applied"] = serving()
    tok = AutoTokenizer.from_pretrained(VENDOR_DIR)
    corpus_ids, prefix_ids, prov, tman = validation_contexts(tok)
    contexts = [("corpus", corpus_ids)] + [(f"prefix{i}", p) for i, p in enumerate(prefix_ids)]
    model, info, load_wall = load_model()
    store = install_taps(model, ["in22", "out22", "out23"])
    names = {}
    out = {}
    walls = {}
    for cname, ids in contexts:
        lg, bnd, wall = run_prefill(model, ids, store)
        walls[cname] = wall
        for k, arr in bnd.items():
            fn = f"capture_{cname}_{k}.npy"
            refuse(os.path.join(DUMP, fn))
            np.save(os.path.join(DUMP, fn), arr)
            names[f"{cname}:{k}"] = fn
        out[cname] = {"tokens": len(ids), "argmax": lg.argmax(1).tolist()}
        print(f"[lr] capture {cname}: {len(ids)} tokens {wall:.1f} s", flush=True)
    rec.update({"artifact": info, "load_wall_s": load_wall, "provenance": prov, "contexts": {c: len(i) for c, i in contexts},
                "argmax": out, "walls_s": walls, "dump_dir": DUMP, "dumps": {k: {"file": v, "sha256": sha_file(os.path.join(DUMP, v))} for k, v in names.items()},
                "boundaries": ["in22", "out22", "out23"], "ok": all(v for v in prov.values() if isinstance(v, bool))})
    write_strict(rec, rpath)
    print(f"[lr] capture: {len(names)} dumps; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("capture provenance failed")


# ---------------------------------------------------------------- oracle (mlx half)

def dense_layer(L, tensors):
    """A standalone mlx-lm DecoderLayer for layer L holding the given vendor
    tensors (float32, load conventions applied)."""
    import mlx.core as mx
    from mlx_lm.models import qwen3_5
    l2norm_override()
    cfg = json.load(open(os.path.join(VENDOR_DIR, "config.json")))
    targs = qwen3_5.TextModelArgs.from_dict(qwen3_5.ModelArgs.from_dict(cfg).text_config)
    layer = qwen3_5.DecoderLayer(targs, L)
    for name, arr in tensors.items():
        path = lr.mlx_param_path(name)
        rel = path.split(f"layers.{L}.", 1)[1]
        arr = lr.sanitize_vendor_tensor(name, arr)
        if arr.ndim == 2 and name.endswith(".weight"):
            import mlx.nn as nn
            lin = nn.Linear(arr.shape[1], arr.shape[0], bias=False)
            lin.weight = mx.array(arr)
            set_path(layer, rel[:-len(".weight")], lin)
        else:
            set_path(layer, rel, mx.array(arr))
    from mlx.utils import tree_flatten
    have = {k for k, _ in tree_flatten(layer.parameters())}
    given = {lr.mlx_param_path(n).split(f"layers.{L}.", 1)[1] for n in tensors}
    if have != given:
        raise SystemExit(f"REFUSING: layer {L} parameter set differs: missing {have - given}, extra {given - have}")
    mx.eval(layer.parameters())
    return layer


def layer_forward(layer, L, x_np, chunk=None):
    """Run one standalone layer over [n, D] float32 rows from a zero state:
    one shot (chunk None) or in chunks with the cache carried."""
    import mlx.core as mx
    from mlx_lm.models.base import create_attention_mask, create_ssm_mask
    from mlx_lm.models.cache import ArraysCache, KVCache
    cache = ArraysCache(size=2) if layer.is_linear else KVCache()
    n = x_np.shape[0]
    step = n if chunk is None else chunk
    outs = []
    for s in range(0, n, step):
        x = mx.array(x_np[s:s + step])[None]
        mask = create_ssm_mask(x, cache) if layer.is_linear else create_attention_mask(x, cache)
        y = layer(x, mask, cache)
        mx.eval(y)
        outs.append(np.asarray(y[0], dtype=np.float32))
    return np.concatenate(outs, axis=0)


def mode_oracle_mlx():
    src = os.environ.get("WEIGHTS", "armc")
    label = "standin" if (SMOKE and src == "vendor") else src
    if label == "standin":
        src = "armc"
    rpath = receipt_path(f"oracle_mlx_{label}")
    refuse(rpath)
    require_clean_tree()
    cap = json.load(open(receipt_path("capture")))
    if not cap.get("ok"):
        raise SystemExit("REFUSING: capture receipt is not ok")
    rec = provenance("oracle_mlx")
    rec["weights"] = label
    if src == "vendor":
        ok, why = vendor_ready()
        if not ok:
            raise SystemExit(f"REFUSING: {why}")
        rec["vendor_shard_check"] = why
    idx = vendor_index()
    man, _ = load_manifest()
    names = target_names(idx)
    t0 = time.time()
    layers = {L: dense_layer(L, source_tensors(src, man, names[L])) for L in LAYERS}
    rec["build_wall_s"] = time.time() - t0
    inputs = {22: "in22", 23: "out22"}
    results = {}
    dumps = {}
    pooled = {L: {"one_shot_vs_fused": [], "chunk_vs_one_shot": []} for L in LAYERS}
    for key, ent in cap["dumps"].items():
        cname, b = key.split(":")
        if b != "in22":
            continue
        for L in LAYERS:
            xin = np.load(os.path.join(cap["dump_dir"], cap["dumps"][f"{cname}:{inputs[L]}"]["file"]))
            fused = np.load(os.path.join(cap["dump_dir"], cap["dumps"][f"{cname}:out{L}"]["file"]))
            y1 = layer_forward(layers[L], L, xin)
            yc = layer_forward(layers[L], L, xin, chunk=CHUNK)
            fn = f"oracle_mlx_{label}_{cname}_out{L}.npy"
            refuse(os.path.join(DUMP, fn))
            np.save(os.path.join(DUMP, fn), y1)
            dumps[f"{cname}:out{L}"] = {"file": fn, "sha256": sha_file(os.path.join(DUMP, fn))}
            results[f"{cname}:out{L}"] = {"one_shot_vs_fused_tower": lr.boundary_stats(fused, y1),
                                          "chunk_vs_one_shot": lr.boundary_stats(y1, yc), "n": int(xin.shape[0])}
            pooled[L]["one_shot_vs_fused"].append((fused, y1))
            pooled[L]["chunk_vs_one_shot"].append((y1, yc))
            print(f"[lr] oracle_mlx {label} {cname} layer {L}: fused-vs-dense rel {results[f'{cname}:out{L}']['one_shot_vs_fused_tower']['rel_l2']:.2e}, "
                  f"chunk-vs-one-shot rel {results[f'{cname}:out{L}']['chunk_vs_one_shot']['rel_l2']:.2e}", flush=True)
    summary = {}
    for L in LAYERS:
        summary[str(L)] = {}
        for k, pairs in pooled[L].items():
            a = np.concatenate([p[0] for p in pairs]); b = np.concatenate([p[1] for p in pairs])
            summary[str(L)][k] = lr.boundary_stats(a, b)
    # the chunk-split check is held to the registered tolerance; the fused-vs-dense reading is a
    # parity reading for armc weights (same values, different kernels) and is reported, not gated
    chunk_ok = all(summary[str(L)]["chunk_vs_one_shot"]["rel_l2"] <= ORACLE_REL_L2 and
                   summary[str(L)]["chunk_vs_one_shot"]["max_abs_over_row_rms"] <= ORACLE_MAX_ABS_OVER_RMS for L in LAYERS)
    rec.update({"inputs": {str(L): inputs[L] for L in LAYERS}, "per_context": results, "pooled": summary, "dump_dir": DUMP, "dumps": dumps,
                "tolerance": {"rel_l2": ORACLE_REL_L2, "max_abs_over_row_rms": ORACLE_MAX_ABS_OVER_RMS},
                "chunk_split_within_tolerance": chunk_ok, "ok": chunk_ok, "wall_s": time.time() - t0, "rss_peak_bytes": rss_peak_bytes()})
    write_strict(rec, rpath)
    print(f"[lr] oracle_mlx {label}: pooled {json.dumps({L: {k: round(v['rel_l2'], 8) for k, v in s.items()} for L, s in summary.items()})}; "
          f"chunk split ok={chunk_ok}; wrote {rpath}", flush=True)
    if not chunk_ok:
        raise SystemExit("oracle_mlx: chunk split outside tolerance")


# ---------------------------------------------------------------- fetch

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
    rec["dest_role"] = "VENDOR_SHARD_DIR (outside the repository)"
    if SMOKE:
        rec["note"] = "smoke: metadata read only, no bytes downloaded"
        write_strict(rec, rpath)
        print(f"[lr] fetch smoke: metadata {meta}; wrote {rpath}", flush=True)
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
    idx = vendor_index()
    man, _ = load_manifest()
    tn = target_names(idx)
    want = [n for L in LAYERS for n in tn[L]]
    ents = {n: header.get(n) for n in want}
    checks = {"size_ok": size == SHARD_BYTES, "sha256_ok": sha == SHARD_SHA256,
              "targets_present": all(ents[n] is not None for n in want),
              "targets_bf16": all(ents[n] is not None and ents[n]["dtype"] == "BF16" for n in want),
              "targets_shape_ok": all(ents[n] is not None and [int(s) for s in ents[n]["shape"]] == list(man[n]["shape"]) for n in want),
              "index_names_equal_header": set(names) == {k for k, v in idx.items() if v == SHARD},
              "data_end_ok": start + max(int(v["data_offsets"][1]) for k, v in header.items() if k != "__metadata__") == size}
    rec.update({"size_bytes": size, "sha256": sha, "header_bytes": start - 8, "n_tensors": len(names), "target_entries": ents,
                "checks": checks, "ok": all(checks.values()), "whole_file_sha_verified": True, "wall_s": time.time() - t0})
    write_strict(rec, rpath)
    print(f"[lr] fetch: {size} bytes sha {sha[:12]} ok={rec['ok']} checks={checks}; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("fetch verification failed")


# ---------------------------------------------------------------- score

def census_dump_ok(cen, l_name):
    return cen["dumps"][l_name] == E0_LOGITS_SHA256


def teacher_side(zt, teacher_token):
    order = np.argsort(-zt.astype(np.float64), kind="stable")
    lt = ls.log_softmax64(zt)
    pt = np.exp(lt)
    return {"argmax": int(order[0]), "margin": float(zt[order[0]] - zt[order[1]]), "top16": order[:16].astype(np.int64),
            "top16_logits": zt[order[:16]].astype(np.float32), "entropy": float(-(pt * lt).sum()),
            "mean_logit": float(zt.astype(np.float64).mean()), "token_is_argmax": bool(order[0] == teacher_token)}


def new_stats(n):
    return {"argmax": np.zeros(n, np.int64), "margin": np.zeros(n), "teacher_rank": np.zeros(n, np.int64), "lse": np.zeros(n),
            "z_teacher_token": np.zeros(n), "ce_teacher_token": np.zeros(n), "kl": np.zeros(n), "slope": np.zeros(n),
            "z_top16": np.zeros((n, 16), np.float32), "top8_ids": np.zeros((n, 8), np.int32), "top8_logits": np.zeros((n, 8), np.float32),
            "z_we_minus_the": np.zeros(n), "ce_next_token": np.full(n, np.nan), "mean_logit": np.zeros(n),
            "t_argmax": np.zeros(n, np.int64), "t_margin": np.zeros(n), "t_top16": np.zeros((n, 16), np.int64), "t_top16_logits": np.zeros((n, 16), np.float32),
            "t_entropy": np.zeros(n), "t_mean_logit": np.zeros(n), "t_token": np.zeros(n, np.int64), "t_token_is_argmax": np.zeros(n, bool)}


def fill_stats(arr, i, z, zt, teacher_token, next_token=None):
    ts = teacher_side(zt, teacher_token)
    ps = ls.position_stats(z, zt, int(teacher_token), ts["top16"], k_top=8)
    for k in ("argmax", "margin", "teacher_rank", "lse", "z_teacher_token", "ce_teacher_token", "mean_logit"):
        arr[k][i] = ps[k]
    arr["kl"][i] = ps["kl_teacher_cand"]
    arr["z_top16"][i] = ps["z_top_ids"]
    arr["top8_ids"][i] = ps["top_ids"]
    arr["top8_logits"][i] = ps["top_logits"]
    arr["slope"][i] = ls.slope_on_teacher(ps["z_top_ids"], ts["top16_logits"])
    arr["z_we_minus_the"][i] = float(z[WE] - z[THE])
    if next_token is not None and next_token >= 0:
        arr["ce_next_token"][i] = float(ps["lse"] - z[next_token])
    for k in ("argmax", "margin", "top16", "top16_logits", "entropy", "mean_logit", "token_is_argmax"):
        arr["t_" + k][i] = ts[k]
    arr["t_token"][i] = teacher_token


def mode_score():
    import mlx.core as mx
    from transformers import AutoTokenizer
    from llmopt.lab import qmetal, qprefill
    from qwen_metal_prefill_head import residents_of
    arm = os.environ.get("ARM", "control")
    if arm not in ARMS:
        raise SystemExit(f"unknown ARM {arm}")
    capture = os.environ.get("CAPTURE", "0") == "1"
    rpath = receipt_path(f"score_{arm}")
    refuse(rpath)
    require_clean_tree()
    os.makedirs(DUMP, exist_ok=True)
    npz_path = os.path.join(DUMP, f"score_{arm}.npz")
    refuse(npz_path)
    rec = provenance("score")
    rec["arm"] = arm
    rec["serving_applied"] = serving()
    src = ARM_SOURCE[arm]
    label = src
    if src == "vendor":
        if SMOKE:
            label = "standin"
            src = "armc"
        else:
            ok, why = vendor_ready()
            if not ok:
                raise SystemExit(f"REFUSING: {why}")
            rec["vendor_shard_check"] = why
    rec["weights"] = label
    rec["layers_installed"] = list(ARMS[arm])
    tok = AutoTokenizer.from_pretrained(VENDOR_DIR)
    corpus_ids, prefix_ids, prov, tman = validation_contexts(tok)
    e0 = e0_contexts(tman)
    rows = e0_rows()
    ro = tman["records"]["rollouts"]
    pidx = {pid: i for i, pid in enumerate(ro["prompt_ids"])}
    roll = np.load(os.path.join(TEACHER_DIR, "rollout_logits.npy"), mmap_mode="r")
    tc = np.load(os.path.join(TEACHER_DIR, "corpus_logits.npy"), mmap_mode="r")
    tp = np.load(os.path.join(TEACHER_DIR, "prefix_logits.npy"), mmap_mode="r")
    idx = vendor_index()
    man, info_art = load_manifest()
    names = target_names(idx)
    model, info, load_wall = load_model()
    t_inst = time.time()
    installed = {}
    raw_equal = {}
    for L in ARMS[arm]:
        tensors = source_tensors(src, man, names[L])
        if src == "vendor":
            for n, a in tensors.items():
                if man[n]["codec"] == "raw":
                    raw_equal[n] = bool(np.array_equal(a, armc_tensor(man, n)))
        installed[str(L)] = {"tensors": {n: list(a.shape) for n, a in tensors.items()}, "bytes_f32": int(install_dense(model, tensors))}
        del tensors
    rec["install_wall_s"] = time.time() - t_inst
    rec["installed"] = installed
    rec["raw_tensors_equal_arm_c"] = raw_equal
    residents = residents_of(model)
    rec["fused_residents_after_install"] = len(residents)
    store = install_taps(model, BOUNDARIES) if capture else None
    counts0 = qmetal.path_counts(residents)
    mx.reset_peak_memory()
    # ---- E0
    n_e0 = sum(len(g) for _, _, g in e0)
    S = {"e0": new_stats(n_e0)}
    blocks_e0 = np.zeros(n_e0, np.int64)
    step_e0 = np.zeros(n_e0, np.int64)
    caps = {}
    if capture:
        n_total = n_e0 + len(corpus_ids) + sum(len(p) for p in prefix_ids)
        for b in BOUNDARIES:
            fn = os.path.join(DUMP, f"capture_{arm}_{b}.npy")
            refuse(fn)
            caps[b] = np.lib.format.open_memmap(fn, mode="w+", dtype=np.float32, shape=(n_total, HIDDEN))
    pos = 0
    row_i = 0
    walls = {}
    per_prompt = []
    t_all = time.time()
    for pid, prompt, gen in e0:
        ids = prompt + gen
        p_len = len(prompt)
        lg, bnd, wall = run_prefill(model, ids, store)
        a_t = 0
        for s in range(len(gen)):
            row = rows[row_i]
            if row["prompt_id"] != pid or row["step"] != s:
                raise SystemExit(f"REFUSING: E0 row order mismatch at {pid} step {s}")
            z = lg[p_len - 1 + s]
            zt = np.asarray(roll[s, pidx[pid], :V_LIVE], dtype=np.float32)
            fill_stats(S["e0"], pos, z, zt, int(row["teacher_token"]))
            blocks_e0[pos] = pidx[pid]
            step_e0[pos] = s
            a_t += int(S["e0"]["argmax"][pos] == row["teacher_token"])
            if capture:
                for b in BOUNDARIES:
                    caps[b][pos] = bnd[b][p_len - 1 + s]
            pos += 1
            row_i += 1
        if SMOKE:
            row_i = ro["gen_lengths"][0]
        per_prompt.append({"prompt_id": pid, "prompt_len": p_len, "gen_len": len(gen), "wall_s": wall, "tok_s": len(ids) / wall, "agree_teacher": a_t})
        print(f"[lr] {arm} {pid}: {len(ids)} tokens {wall:.1f} s, teacher {a_t}/{len(gen)}", flush=True)
        del lg, bnd
    walls["e0"] = time.time() - t_all
    # ---- validation
    t_v = time.time()
    nl2 = tok.convert_tokens_to_ids("ĊĊ")
    S["corpus"] = new_stats(len(corpus_ids))
    lg, bnd, wall = run_prefill(model, corpus_ids, store)
    t_am = np.asarray(tc[:len(corpus_ids), :V_LIVE], dtype=np.float32).argmax(1)
    nxt = corpus_ids[1:] + [-1]
    for i in range(len(corpus_ids)):
        fill_stats(S["corpus"], i, lg[i], np.asarray(tc[i, :V_LIVE], dtype=np.float32), int(t_am[i]), nxt[i])
        if capture:
            for b in BOUNDARIES:
                caps[b][pos + i] = bnd[b][i]
    pos += len(corpus_ids)
    walls["corpus"] = wall
    blocks_c = corpus_blocks(corpus_ids, nl2)
    del lg, bnd
    npf = sum(len(p) for p in prefix_ids)
    S["prefix"] = new_stats(npf)
    blocks_p = np.concatenate([np.full(len(p), i, np.int64) for i, p in enumerate(prefix_ids)])
    off = 0
    walls["prefixes"] = []
    for p in prefix_ids:
        lg, bnd, wall = run_prefill(model, p, store)
        walls["prefixes"].append(wall)
        nxt = p[1:] + [-1]
        for i in range(len(p)):
            zt = np.asarray(tp[off + i, :V_LIVE], dtype=np.float32)
            fill_stats(S["prefix"], off + i, lg[i], zt, int(zt.argmax()), nxt[i])
            if capture:
                for b in BOUNDARIES:
                    caps[b][pos + off + i] = bnd[b][i]
        off += len(p)
        del lg, bnd
    pos += npf
    walls["validation"] = time.time() - t_v
    counts1 = qmetal.path_counts(residents)
    census = None
    if arm == "control":
        # CONTROL must be arm C as booked: its E0 argmaxes against the frozen E0 rows' draft_argmax and its
        # logits against the census dump (fp16), beyond fp16 near-ties only
        cen = json.load(open(CENSUS_RECEIPT))
        l_name = [n for n in cen["dumps"] if n.startswith("draft_logits")][0]
        l_path = os.path.join(cen["dump_dir"], l_name)
        dump = np.load(l_path, mmap_mode="r")
        mism = []
        for i in range(n_e0):
            if int(S["e0"]["argmax"][i]) != int(rows[i]["draft_argmax"]):
                mism.append({"row": i, "draft_argmax": int(rows[i]["draft_argmax"]), "control_argmax": int(S["e0"]["argmax"][i]),
                             "control_margin": float(S["e0"]["margin"][i])})
        top8_diff = float(max(np.abs(np.asarray(dump[i], dtype=np.float32)[S["e0"]["top8_ids"][i]] - S["e0"]["top8_logits"][i]).max() for i in range(n_e0)))
        beyond = [m for m in mism if m["control_margin"] > NEAR_TIE_GAP]
        census = {"receipt": CENSUS_RECEIPT, "dump": l_name, "dump_sha256_pinned": cen["dumps"][l_name] == E0_LOGITS_SHA256,
                  "dump_sha256_on_disk_ok": (sha_array_file(l_path) == E0_LOGITS_SHA256) if not SMOKE else "skipped",
                  "argmax_mismatches_vs_e0_rows": mism, "beyond_near_tie": len(beyond), "top8_logit_max_abs_diff_vs_dump": top8_diff,
                  "near_tie_gap": NEAR_TIE_GAP, "ok": len(beyond) == 0 and census_dump_ok(cen, l_name)}
    mem = qprefill.memory_report()
    mem["mx_peak_bytes"] = int(mx.get_peak_memory())
    mem["rss_peak_bytes"] = rss_peak_bytes()
    out = {}
    for sname, arr in S.items():
        for k, v in arr.items():
            out[f"{sname}/{k}"] = v
    out["e0/blocks"] = blocks_e0
    out["e0/step"] = step_e0
    out["corpus/blocks"] = blocks_c
    out["prefix/blocks"] = blocks_p
    np.savez(npz_path, **out)
    dumps = {"score_npz": {"file": os.path.basename(npz_path), "sha256": sha_file(npz_path)}}
    if capture:
        for b, m in caps.items():
            m.flush()
            del m
        caps.clear()
        for b in BOUNDARIES:
            fn = f"capture_{arm}_{b}.npy"
            dumps[b] = {"file": fn, "sha256": sha_file(os.path.join(DUMP, fn))}
    agree = {s: int((S[s]["argmax"] == S[s]["t_token"]).sum()) for s in S}
    rec.update({"artifact": info, "load_wall_s": load_wall, "provenance": prov, "sets": {s: int(S[s]["argmax"].shape[0]) for s in S},
                "agree_teacher": agree, "mean_kl": {s: float(S[s]["kl"].mean()) for s in S}, "per_prompt": per_prompt,
                "walls_s": walls, "matrix_calls": {k: counts1[k] - counts0[k] for k in counts0}, "memory": mem,
                "capture": capture, "boundaries": BOUNDARIES if capture else [], "dump_dir": DUMP, "dumps": dumps,
                "e0_rows_sha256": sha_file(os.path.join(E0_DIR, "positions.jsonl")), "wall_s": time.time() - t_all,
                "control_vs_census": census,
                "teacher_token_convention": "E0: the recorded rollout token of the E0 row; corpus and prefixes: the teacher's fp16-stored argmax over the live vocabulary",
                "ok": all(v for v in prov.values() if isinstance(v, bool)) and (src != "vendor" or all(raw_equal.values())) and (census is None or census["ok"])})
    write_strict(rec, rpath)
    print(f"[lr] score {arm} ({label}): agree {agree}, KL {rec['mean_kl']}, peak {mem['mx_peak_bytes'] / 1e9:.1f} GB, "
          f"{rec['wall_s']:.0f} s; wrote {rpath}", flush=True)
    if not rec["ok"]:
        raise SystemExit("score: provenance or raw-tensor equality failed")


# ---------------------------------------------------------------- summarize

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


def load_scores():
    arms = {}
    recs = {}
    for arm in ARMS:
        rp = receipt_path(f"score_{arm}")
        if not os.path.exists(rp):
            continue
        r = json.load(open(rp))
        p = os.path.join(r["dump_dir"], r["dumps"]["score_npz"]["file"])
        if sha_file(p) != r["dumps"]["score_npz"]["sha256"]:
            raise SystemExit(f"REFUSING: {p} does not match its receipt sha")
        z = np.load(p)
        arms[arm] = {k: z[k] for k in z.files}
        recs[arm] = r
    if "control" not in arms:
        raise SystemExit("REFUSING: no control score")
    return arms, recs


def contrast(ra, rb, blocks, set_name):
    tt = ra["t_token"]
    aa, ab = ra["argmax"] == tt, rb["argmax"] == tt
    c = {"paired": ls.paired_counts(aa, ab), "blocks": ls.block_deltas(aa, ab, blocks),
         "bootstrap": ls.block_bootstrap_delta_points(aa, ab, blocks, n_boot=10000, seed=0),
         "kl_mean": {"control": float(ra["kl"].mean()), "arm": float(rb["kl"].mean()), "rel_delta": float(rb["kl"].mean() / ra["kl"].mean() - 1.0)},
         "kl_paired_lower": int((rb["kl"] < ra["kl"]).sum()),
         "ce_teacher_token_mean": {"control": float(ra["ce_teacher_token"].mean()), "arm": float(rb["ce_teacher_token"].mean())},
         "ce_next_token_mean": {"control": float(np.nanmean(ra["ce_next_token"])) if np.isfinite(ra["ce_next_token"]).any() else None,
                                "arm": float(np.nanmean(rb["ce_next_token"])) if np.isfinite(rb["ce_next_token"]).any() else None},
         "slope_median": {"control": float(np.nanmedian(ra["slope"])), "arm": float(np.nanmedian(rb["slope"])),
                          "delta": float(np.nanmedian(rb["slope"]) - np.nanmedian(ra["slope"]))},
         "teacher_rank_improved": int((rb["teacher_rank"] < ra["teacher_rank"]).sum()), "teacher_rank_worsened": int((rb["teacher_rank"] > ra["teacher_rank"]).sum()),
         "margin_delta_mean": {"agree_both": float((rb["margin"] - ra["margin"])[aa & ab].mean()) if (aa & ab).any() else None,
                               "corrected": float((rb["margin"] - ra["margin"])[~aa & ab].mean()) if (~aa & ab).any() else None,
                               "broken": float((rb["margin"] - ra["margin"])[aa & ~ab].mean()) if (aa & ~ab).any() else None},
         "argmax_identical": int((ra["argmax"] == rb["argmax"]).sum()),
         "z_top16_max_abs_diff": float(np.abs(rb["z_top16"].astype(np.float64) - ra["z_top16"]).max())}
    if set_name == "e0":
        tm = ra["t_margin"]
        tb = np.array([bins(x, MARGIN_EDGES) for x in tm])
        c["by_teacher_margin_bin"] = [{"bin": [MARGIN_EDGES[k], MARGIN_EDGES[k + 1]], "n": int((tb == k).sum()),
                                       "agree_control": int(aa[tb == k].sum()), "agree_arm": int(ab[tb == k].sum())} for k in range(len(MARGIN_EDGES) - 1)]
        z0 = ra["step"] == 0
        c["opener_step0"] = {"n": int(z0.sum()), "argmax_control": [int(x) for x in ra["argmax"][z0]], "argmax_arm": [int(x) for x in rb["argmax"][z0]],
                             "z_we_minus_the_control": [float(x) for x in ra["z_we_minus_the"][z0]], "z_we_minus_the_arm": [float(x) for x in rb["z_we_minus_the"][z0]],
                             "teacher_token": [int(x) for x in tt[z0]], "agree_control": int(aa[z0].sum()), "agree_arm": int(ab[z0].sum()),
                             "delta_z_we_minus_the_mean": float((rb["z_we_minus_the"] - ra["z_we_minus_the"])[z0].mean())}
    return c


def mode_summarize():
    from llmopt.lab.prereg import load, registration_digest
    rpath = receipt_path("summary")
    opath = receipt_path("observations")
    refuse(rpath)
    refuse(opath)
    require_clean_tree()
    rec = provenance("summarize")
    arms, recs = load_scores()
    rec["arms_present"] = sorted(arms)
    sets = ["e0", "corpus", "prefix"]
    # teacher side must be identical across arms (deterministic from the teacher logits)
    teacher_equal = all(np.array_equal(arms["control"][f"{s}/t_top16_logits"], arms[a][f"{s}/t_top16_logits"]) and
                        np.array_equal(arms["control"][f"{s}/t_token"], arms[a][f"{s}/t_token"]) for a in arms for s in sets)
    if not teacher_equal:
        raise SystemExit("REFUSING: teacher-side arrays differ between arms")
    summ = {"sets": {}, "contrasts": {}}
    for s in sets:
        tt = arms["control"][f"{s}/t_token"]
        ss = {"n": int(tt.shape[0]), "teacher": {"token_is_argmax": int(arms["control"][f"{s}/t_token_is_argmax"].sum()),
                                                   "margin": q(arms["control"][f"{s}/t_margin"]), "entropy": q(arms["control"][f"{s}/t_entropy"])}, "arms": {}}
        for a in arms:
            r = {k.split("/", 1)[1]: v for k, v in arms[a].items() if k.startswith(s + "/")}
            agree = r["argmax"] == tt
            ss["arms"][a] = {"agree": int(agree.sum()), "agree_rate": float(agree.mean()), "kl": q(r["kl"]), "ce_teacher_token": q(r["ce_teacher_token"]),
                             "ce_next_token": q(r["ce_next_token"]), "slope": q(r["slope"]),
                             "margin": {"agree": q(r["margin"][agree]), "disagree": q(r["margin"][~agree])},
                             "teacher_rank_counts": {"0": int((r["teacher_rank"] == 0).sum()), "1": int((r["teacher_rank"] == 1).sum()),
                                                     "2": int((r["teacher_rank"] == 2).sum()), "3-15": int(((r["teacher_rank"] >= 3) & (r["teacher_rank"] < 16)).sum()),
                                                     "16+": int((r["teacher_rank"] >= 16).sum())}}
            shift = r["mean_logit"] - r["t_mean_logit"]
            diff = (r["z_top16"].astype(np.float64) - shift[:, None]) - r["t_top16_logits"].astype(np.float64)
            ss["arms"][a]["z_top16_minus_teacher_after_shift"] = {"per_rank_mean": diff.mean(0).tolist(), "shift_mean": float(shift.mean())}
        summ["sets"][s] = ss
        for a in arms:
            if a == "control":
                continue
            ra = {k.split("/", 1)[1]: v for k, v in arms["control"].items() if k.startswith(s + "/")}
            rb = {k.split("/", 1)[1]: v for k, v in arms[a].items() if k.startswith(s + "/")}
            summ["contrasts"].setdefault(a, {})[s] = contrast(ra, rb, ra["blocks"], s)
    for a in arms:
        if a == "control":
            continue
        aa = np.concatenate([arms["control"][f"{s}/argmax"] == arms["control"][f"{s}/t_token"] for s in ("corpus", "prefix")])
        ab = np.concatenate([arms[a][f"{s}/argmax"] == arms[a][f"{s}/t_token"] for s in ("corpus", "prefix")])
        bc = arms["control"]["corpus/blocks"]
        blocks = np.concatenate([bc, arms["control"]["prefix/blocks"] + int(bc.max()) + 1])
        ka = np.concatenate([arms["control"][f"{s}/kl"] for s in ("corpus", "prefix")])
        kb = np.concatenate([arms[a][f"{s}/kl"] for s in ("corpus", "prefix")])
        summ["contrasts"][a]["validation_pooled"] = {
            "paired": ls.paired_counts(aa, ab), "blocks": ls.block_deltas(aa, ab, blocks),
            "bootstrap": ls.block_bootstrap_delta_points(aa, ab, blocks, n_boot=10000, seed=0),
            "kl_mean": {"control": float(ka.mean()), "arm": float(kb.mean()), "rel_delta": float(kb.mean() / ka.mean() - 1.0)},
            "kl_paired_lower": int((kb < ka).sum())}
    # DENSE_C parity precondition: same weights through dense float32 modules against the fused residents
    parity = None
    if "dense_c" in arms:
        parity = {}
        for s in sets:
            ra = {k.split("/", 1)[1]: v for k, v in arms["control"].items() if k.startswith(s + "/")}
            rb = {k.split("/", 1)[1]: v for k, v in arms["dense_c"].items() if k.startswith(s + "/")}
            decided = ra["margin"] > PARITY_MARGIN
            flips = int((ra["argmax"][decided] != rb["argmax"][decided]).sum())
            parity[s] = {"n": int(decided.size), "n_decided": int(decided.sum()), "argmax_flips_beyond_margin": flips,
                         "argmax_flips_total": int((ra["argmax"] != rb["argmax"]).sum()),
                         "z_top16_max_abs_diff": float(np.abs(rb["z_top16"].astype(np.float64) - ra["z_top16"]).max()),
                         "kl_mean_control": float(ra["kl"].mean()), "kl_mean_dense_c": float(rb["kl"].mean())}
        parity["ok"] = all(p["argmax_flips_beyond_margin"] == 0 and p["z_top16_max_abs_diff"] <= PARITY_LOGIT_ABS for p in parity.values() if isinstance(p, dict))
        parity["rule"] = f"argmax equal wherever the control margin exceeds {PARITY_MARGIN}; teacher top-16 logits within {PARITY_LOGIT_ABS}"
    # oracle preconditions (vendor weights): both halves within tolerance
    oracle = {}
    for half in ("oracle_mlx", "oracle_torch"):
        for w in ("vendor", "standin"):
            p = receipt_path(f"{half}_{w}")
            if os.path.exists(p):
                r = json.load(open(p))
                oracle[f"{half}_{w}"] = {"ok": bool(r.get("ok")), "code_commit": r.get("code_commit")}
    group_weights = recs.get("group", {}).get("weights")
    oracle_ok = all(v["ok"] for k, v in oracle.items() if k.endswith(group_weights or "vendor")) and len([k for k in oracle if k.endswith(group_weights or "vendor")]) == 2
    vendor_weights = group_weights == "vendor"
    valid = bool(parity and parity["ok"]) and oracle_ok and vendor_weights and all(recs[a].get("ok") for a in arms)
    reason = ("DENSE_C parity, both oracle halves within tolerance on the vendor weights, group at the vendor weights, every score receipt ok" if valid else
              f"parity {parity and parity['ok']}, oracle {oracle_ok}, group weights {group_weights!r} (vendor required; a stand-in is inadmissible), "
              f"receipts ok {[recs[a].get('ok') for a in arms]}")
    memory = {a: recs[a]["memory"] for a in arms}
    tok_s = {a: float(np.median([p["tok_s"] for p in recs[a]["per_prompt"]])) for a in arms}
    rec.update({"summary": summ, "parity_dense_c": parity, "oracle": oracle, "measurement_valid": valid, "measurement_reason": reason,
                "memory_by_arm": memory, "e0_prefill_tok_s_median_by_arm": tok_s, "weights_by_arm": {a: recs[a].get("weights") for a in arms},
                "multiplicity_note": "three vendor arms (l22, l23, group) are contrasted against control on the one validation set; only group is adjudicated; the single-layer contrasts are diagnostic readings"})
    write_strict(rec, rpath)
    obs = observations(summ, rec)
    write_strict(obs, opath)
    g = summ["contrasts"].get("group", {})
    print(f"[lr] summarize: valid={valid} ({reason}); group E0 {g.get('e0', {}).get('paired')}; validation {g.get('validation_pooled', {}).get('paired')}; "
          f"wrote {rpath} and {opath}", flush=True)


def observations(summ, rec):
    from llmopt.lab.prereg import load, registration_digest
    if REGISTRATION_SHA256 is None:
        raise SystemExit("REFUSING: no registration digest pinned in the driver; register first")
    doc = load(PREREG)
    digest = registration_digest(doc)
    if digest != REGISTRATION_SHA256:
        raise SystemExit(f"REFUSING: {PREREG} digest {digest[:12]} is not the pinned registration {REGISTRATION_SHA256[:12]}")
    c = summ["contrasts"].get("group")
    valid = bool(rec["measurement_valid"]) and c is not None
    gw = rec["weights_by_arm"].get("group")
    ns = {k: summ["sets"][k]["n"] for k in summ["sets"]}
    if not SMOKE and (ns.get("e0"), ns.get("corpus"), ns.get("prefix")) != (4722, 356, 98):
        raise SystemExit(f"REFUSING: set sizes {ns} are not the registered populations")
    obs = {"note": f"Mechanical projection of {receipt_path('summary')} (driver scratch/qwen_layer_restore.py at {rec['code_commit']}); "
                   f"primary contrast group (layers 22 and 23 at the {gw!r} weights, float32) against control (arm C as served), "
                   f"same contexts, fresh cache per sequence, live vocabulary; set sizes {ns}",
           "registration_sha256": digest, "measurement_valid": valid,
           "arms": {"CONTROL": {"admissible": True, "reason": "arm C as served through prefill_chunked (GEMM path), the production prefill"},
                    "DENSE_C": {"admissible": bool(rec["parity_dense_c"] and rec["parity_dense_c"]["ok"]), "reason": f"swap machinery with no weight change: {rec['parity_dense_c']}"},
                    "GROUP": {"admissible": valid, "reason": rec["measurement_reason"]},
                    "L22": {"admissible": valid, "reason": "diagnostic single-layer arm, same bytes; never adjudicated"},
                    "L23": {"admissible": valid, "reason": "diagnostic single-layer arm, same bytes; never adjudicated"}},
           "contrasts": {str(k): {"admissible": valid, "reason": rec["measurement_reason"]} for k in (1, 2, 3)}}
    if not valid:
        obs["measurement_reason"] = rec["measurement_reason"]
        obs["measurements"] = {}
        return obs
    m1 = c["e0"]["paired"]["delta_points"]
    m2 = c["validation_pooled"]["paired"]["delta_points"]
    m3 = c["validation_pooled"]["kl_mean"]["rel_delta"]
    # the refutation ratio reads bar 2 only as a RESOLVED fire: both conjunct nets at least 0 and the
    # block-bootstrap interval excluding zero; a FIRE-AT-LOW-RESOLUTION counts as NO-FIRE in the ratio
    resolved = (c["corpus"]["paired"]["net"] >= 0 and c["prefix"]["paired"]["net"] >= 0
                and c["validation_pooled"]["bootstrap"]["ci95_low"] > 0.0)
    m2_eff = m2 if resolved else min(m2, 0.0)
    obs["measurements"] = {
        "1": {"value": m1, "metric": "delta_agree_points_e0", "population": "positions:e0-4722", "aggregation": "paired_difference_points",
              "provenance": f"E0 paired counts {c['e0']['paired']}"},
        "2": {"value": m2, "metric": "delta_agree_points_validation", "population": "positions:corpus-356+prefix-98", "aggregation": "paired_difference_points",
              "provenance": f"validation paired counts {c['validation_pooled']['paired']}; bootstrap {c['validation_pooled']['bootstrap']}"},
        "2:corpus": {"value": c["corpus"]["paired"]["net"], "metric": "delta_agree_net_corpus", "population": "positions:corpus-356", "aggregation": "paired_net_count"},
        "2:prefix": {"value": c["prefix"]["paired"]["net"], "metric": "delta_agree_net_prefix", "population": "positions:prefix-98", "aggregation": "paired_net_count"},
        "3": {"value": m3, "metric": "rel_delta_mean_kl_validation", "population": "positions:corpus-356+prefix-98", "aggregation": "ratio_of_means_minus_one",
              "provenance": f"validation KL means {c['validation_pooled']['kl_mean']}"},
        "max_bar12_ratio": {"value": max(m1 / 1.0, m2_eff / 1.0), "metric": "max_bar12_ratio", "population": "bars:1,2", "aggregation": "max_of_ratios",
                            "provenance": f"bar 2 resolved (conjuncts and bootstrap interval excluding zero): {resolved}; bootstrap {c['validation_pooled']['bootstrap']}"}}
    return obs


# ---------------------------------------------------------------- sensitivity

def mode_sensitivity():
    rpath = receipt_path("sensitivity")
    refuse(rpath)
    require_clean_tree()
    rec = provenance("sensitivity")
    arms, recs = load_scores()
    if "group" not in arms or not recs["control"].get("capture") or not recs["group"].get("capture"):
        raise SystemExit("REFUSING: control and group captures are needed")
    idx = vendor_index()
    man, _ = load_manifest()
    names = target_names(idx)
    src = recs["group"]["weights"]
    t0 = time.time()
    # weight reconstruction error per tensor (vendor or stand-in against the decoded arm C)
    werr = {}
    for L in LAYERS:
        for n in names[L]:
            a = armc_tensor(man, n)
            b = vendor_tensor(n, tuple(man[n]["shape"])) if src == "vendor" else a
            werr[n] = {"codec": man[n]["codec"], "shape": list(a.shape), "rel_frobenius": lr.rel_frobenius(b, a)}
            del a, b
    n_e0 = int(arms["control"]["e0/t_token"].shape[0])
    n_c = int(arms["control"]["corpus/t_token"].shape[0])
    n_p = int(arms["control"]["prefix/t_token"].shape[0])
    spans = {"e0": (0, n_e0), "validation": (n_e0, n_e0 + n_c + n_p)}
    bnd = {}
    local = None
    for b in BOUNDARIES:
        fa = os.path.join(recs["control"]["dump_dir"], recs["control"]["dumps"][b]["file"])
        fb = os.path.join(recs["group"]["dump_dir"], recs["group"]["dumps"][b]["file"])
        if sha_file(fa) != recs["control"]["dumps"][b]["sha256"] or sha_file(fb) != recs["group"]["dumps"][b]["sha256"]:
            raise SystemExit(f"REFUSING: capture {b} does not match its receipt sha")
        A = np.load(fa, mmap_mode="r")
        B = np.load(fb, mmap_mode="r")
        bnd[b] = {}
        for sname, (s0, s1) in spans.items():
            st = lr.boundary_stats(np.asarray(A[s0:s1]), np.asarray(B[s0:s1]))
            bnd[b][sname] = st
        print(f"[lr] sensitivity {b}: e0 rel {bnd[b]['e0']['rel_l2']:.3e} (median row {bnd[b]['e0']['row_rel_l2_median']:.3e}), "
              f"validation rel {bnd[b]['validation']['rel_l2']:.3e}", flush=True)
    # evolution: ratio of each downstream boundary's median row relative L2 to the local (out23) one
    evo = {}
    for sname in spans:
        loc = bnd["out23"][sname]["row_rel_l2_median"]
        evo[sname] = {b: (bnd[b][sname]["row_rel_l2_median"] / loc if loc > 0 else None) for b in BOUNDARIES}
    # final response from the score arrays
    resp = {}
    for s in ("e0", "corpus", "prefix"):
        ra = arms["control"]; rb = arms["group"]
        resp[s] = {"argmax_flips": int((ra[f"{s}/argmax"] != rb[f"{s}/argmax"]).sum()), "n": int(ra[f"{s}/argmax"].shape[0]),
                   "kl_teacher_mean": {"control": float(ra[f"{s}/kl"].mean()), "group": float(rb[f"{s}/kl"].mean())},
                   "z_top16_mean_abs_diff": float(np.abs(rb[f"{s}/z_top16"].astype(np.float64) - ra[f"{s}/z_top16"]).mean())}
    rec.update({"weights": src, "weight_reconstruction_error": werr, "boundaries": bnd, "evolution_ratio_to_out23": evo, "final_response": resp,
                "in22_is_identical_by_construction": bnd["in22"]["e0"]["rel_l2"] == 0.0 and bnd["in22"]["validation"]["rel_l2"] == 0.0,
                "note": "one pair restored inside an otherwise compressed network; the propagation it measures is of this pair's local difference, "
                        "not an attribution of all compression error", "wall_s": time.time() - t0})
    write_strict(rec, rpath)
    print(f"[lr] sensitivity: wrote {rpath}", flush=True)


# ---------------------------------------------------------------- main

def main():
    modes = {"preflight": mode_preflight, "capture": mode_capture, "oracle_mlx": mode_oracle_mlx, "fetch": mode_fetch,
             "score": mode_score, "summarize": mode_summarize, "sensitivity": mode_sensitivity}
    if MODE not in modes:
        raise SystemExit(f"unknown MODE {MODE}; one of {sorted(modes)}")
    os.makedirs(OUT, exist_ok=True)
    modes[MODE]()


if __name__ == "__main__":
    main()
