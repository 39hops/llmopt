"""K2-HORIZON-TRANSPORT-0 instrument (prereg in docs/RESULTS.md): the
streaming TRAINING-STAGE census + frozen gate ladder for one K2 Horizon
size (SIZE=3.7B discovery, SIZE=7B prospective), every checkpoint frozen
by the immutable manifest docs/preregs/k2h-transport-0.manifest.json
(full commit + per-shard LFS sha256): tags are never identifiers here.

Streaming law (disk): a tag's shards are downloaded one at a time into a
per-tag scratch directory, each shard's sha256 checked against the
manifest before it is opened; the previous tag stays on disk until the
pair census is done, then is deleted; the endpoint pair re-downloads
pretrain_100000 shard by shard (two tags plus one shard on disk at any
time). Derived receipts only.

Per consecutive pair, per tensor: Frobenius norms, relative delta,
bit-identity, lag-1 cosine against the previous consecutive delta (kept
in RAM as float16); for the six STRUCTURAL pairs (five stage boundaries
and the pretraining endpoint pair) additionally stable rank before /
after, entropy effective rank of the delta, top-1 singular share, IPR of
the top-16 singular vectors, descriptive Hill alpha. Aggregates: 10-bin
normalized-depth profile (bin = floor(10 * l / L)), module-class shares,
depth centroid, whole-model lag-1 cosine.

Gate: the frozen 240-item ladder (scratch/k2h_gateladder.py, digest
2473cd7e) on the seven gated tags, bf16 on mps, one frozen tokenizer
(the pretrain_final variant of the size) for every tag, released rope
config per tag; plus a common-config control (pretraining rope_theta and
8,192 positions) at seed 0 on the five post-pretraining tags.

TR_SMOKE=1: two tags, first two shards for the census, tier 1 with 8
items; writes to logs/k2h/transport_<size>_smoke/.

Usage:
    SIZE=3.7B .venv-k2/bin/python scratch/k2h_transport.py
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


K = _load("k2h", "scratch/k2h_stagecensus.py")
LAD = _load("k2h_ladder", "scratch/k2h_gateladder.py")

SMOKE = os.environ.get("TR_SMOKE") == "1"
SIZE = os.environ["SIZE"]
MAN_PATH = "docs/preregs/k2h-transport-0.manifest.json"
MAN = json.load(open(os.path.join(ROOT, MAN_PATH)))["sizes"][SIZE]
REPO = MAN["repo"]
OUT = os.path.join(ROOT, f"logs/k2h/transport_{SIZE}" + ("_smoke" if SMOKE else ""))
SCRATCH = os.path.join(ROOT, f"logs/k2h/_tr_{SIZE}" + ("_smoke" if SMOKE else ""))
PREREG = "K2-HORIZON-TRANSPORT-0" + ("-SMOKE" if SMOKE else "")
PRETRAIN = [f"pretrain_{s}" for s in range(100000, 1100001, 100000)]
FINALS = ["mid_1_final", "mid_2_final", "mid_3_final", "mid_4_final", "sft_1_final"]
CHAIN = PRETRAIN + FINALS
STRUCTURAL = {"pretrain_1100000->mid_1_final": "B1", "mid_1_final->mid_2_final": "B2", "mid_2_final->mid_3_final": "B3",
              "mid_3_final->mid_4_final": "B4", "mid_4_final->sft_1_final": "B5", "pretrain_100000->pretrain_1100000": "P"}
GATED = ["pretrain_100000", "pretrain_1100000"] + FINALS
CONTROL_TAGS = FINALS
SEEDS = [0, 1]
TOKENIZER_TAG = "pretrain_final"
NBINS = 10
if SMOKE:
    # the smoke walks the one pair whose two tags use DIFFERENT shard file names
    # (model-*.safetensors v pytorch_model-*.safetensors), the failure of attempt 1
    CHAIN = ["mid_4_final", "sft_1_final"]
    GATED = ["sft_1_final"]
    CONTROL_TAGS = []
    SEEDS = [0]
    STRUCTURAL = {"mid_4_final->sft_1_final": "B5"}
SMOKE_SHARDS = 2


def sha_file(p):
    return K.sha_file(p)


# ------------------------------------------------------------ streaming fetch
def tag_dir(tag):
    return os.path.join(SCRATCH, tag)


def fetch_small(tag):
    from huggingface_hub import hf_hub_download
    e = MAN["selected"][tag]
    d = tag_dir(tag)
    os.makedirs(d, exist_ok=True)
    for f in ("config.json", "model.safetensors.index.json", "generation_config.json", "configuration_k2_horizon.py", "modeling_k2_horizon.py",
              "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        try:
            p = hf_hub_download(REPO, f, revision=e["commit"], local_dir=d)
        except Exception:  # noqa: BLE001 — optional files
            continue
        if f in e["small_sha256"] and sha_file(p) != e["small_sha256"][f]:
            raise SystemExit(f"REFUSING: {tag} {f} sha v manifest")
    return d


_PREFETCH = {}
PREFETCH_AHEAD = 3


def prefetch(tag, shards):
    """Download up to PREFETCH_AHEAD shards ahead in threads (disk law:
    two tags plus PREFETCH_AHEAD + 1 shards); every shard is still
    sha256-checked in fetch_shard before it is opened."""
    from concurrent.futures import ThreadPoolExecutor
    from huggingface_hub import hf_hub_download
    ex = ThreadPoolExecutor(max_workers=PREFETCH_AHEAD)
    e = MAN["selected"][tag]
    for s in shards:
        _PREFETCH[(tag, s)] = ex.submit(hf_hub_download, REPO, s, revision=e["commit"], local_dir=tag_dir(tag))
    ex.shutdown(wait=False)


def fetch_shard(tag, shard):
    from huggingface_hub import hf_hub_download
    e = MAN["selected"][tag]
    fut = _PREFETCH.pop((tag, shard), None)
    p = fut.result() if fut is not None else hf_hub_download(REPO, shard, revision=e["commit"], local_dir=tag_dir(tag))
    got = sha_file(p)
    if got != e["shards"][shard]["sha256"]:
        raise SystemExit(f"REFUSING: {tag} {shard} sha256 {got[:12]} v manifest {e['shards'][shard]['sha256'][:12]}")
    return p, got


def shard_list(tag):
    idx = json.load(open(os.path.join(tag_dir(tag), "model.safetensors.index.json")))
    shards = sorted(set(idx["weight_map"].values()))
    return shards, idx["weight_map"]


def census_shards(shards):
    """Smoke censuses only the first SMOKE_SHARDS shards; every shard is
    still downloaded so the gate can load the whole model."""
    return shards[:SMOKE_SHARDS] if SMOKE else shards


def load_shard(path):
    from safetensors import safe_open
    out = {}
    with safe_open(path, framework="pt") as f:
        for n in f.keys():
            out[n] = f.get_tensor(n)
    return out


class TagReader:
    """Tensor-name addressed reader over a tag directory: shard file names
    differ between tags (sft tags ship pytorch_model-*.safetensors), so the
    previous tag is always read by tensor name through its own index."""

    def __init__(self, tag):
        self.dir = tag_dir(tag)
        self.wmap = json.load(open(os.path.join(self.dir, "model.safetensors.index.json")))["weight_map"]
        self.handles = {}

    def get(self, name):
        from safetensors import safe_open
        shard = self.wmap[name]
        if shard not in self.handles:
            self.handles[shard] = safe_open(os.path.join(self.dir, shard), framework="pt")
        return self.handles[shard].get_tensor(name)

    def names(self):
        return set(self.wmap)

    def close(self):
        self.handles.clear()


def tokenizer_homologous(d, tok_dir):
    """Ordinary vocab (added-token names excluded) and merges of the tag's
    own tokenizer.json must equal the frozen tokenizer's; special-token
    NAMES are allowed to differ (the documented variant set)."""
    try:
        a = json.load(open(os.path.join(d, "tokenizer.json")))
        b = json.load(open(os.path.join(tok_dir, "tokenizer.json")))
    except FileNotFoundError:
        return False
    sa = {t["content"] for t in a.get("added_tokens", [])}
    sb = {t["content"] for t in b.get("added_tokens", [])}
    va = {k: v for k, v in a["model"]["vocab"].items() if k not in sa}
    vb = {k: v for k, v in b["model"]["vocab"].items() if k not in sb}
    return va == vb and a["model"].get("merges") == b["model"].get("merges") and {t["id"] for t in a.get("added_tokens", [])} == {t["id"] for t in b.get("added_tokens", [])}


# ------------------------------------------------------------ stats
def power_sigma_max(W, iters=30):
    g = torch.Generator().manual_seed(0)
    v = torch.randn(W.shape[1], generator=g)
    v = v / v.norm()
    for _ in range(iters):
        u = W @ v
        v = W.T @ u
        n = v.norm()
        if n == 0:
            return 0.0
        v = v / n
    return float((W @ v).norm())


def tensor_row(name, A, B, structural, prev_d):
    A32, B32 = A.float(), B.float()
    d = B32 - A32
    fa, fd = float(torch.linalg.vector_norm(A32)), float(torch.linalg.vector_norm(d))
    row = {"tensor": name, "cls": K.module_class(name), "layer": K.layer_of(name), "shape": list(A.shape), "fro_a": fa,
           "fro_b": float(torch.linalg.vector_norm(B32)), "fro_d": fd, "rel_d": fd / fa if fa > 0 else None, "identical": bool(torch.equal(A, B))}
    if prev_d is not None:
        pd = prev_d.float()
        den = float(torch.linalg.vector_norm(pd)) * fd
        row["cos_prev_delta"] = float(torch.dot(pd.flatten(), d.flatten()) / den) if den > 0 else None
    if structural and A.ndim == 2 and min(A.shape) >= 64:
        sa, sb = power_sigma_max(A32), power_sigma_max(B32)
        row["stable_rank_a"] = fa ** 2 / sa ** 2 if sa > 0 else None
        row["stable_rank_b"] = row["fro_b"] ** 2 / sb ** 2 if sb > 0 else None
        if fd > 0:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            S = S.numpy()
            row["effrank_d"] = K.entropy_effrank(S)
            row["effrank_d_norm"] = row["effrank_d"] / min(A.shape)
            row["top1_share_d"] = float(S[0] ** 2 / (S ** 2).sum())
            k = min(K.IPR_K, S.shape[0])
            row["ipr_left_d"] = K.ipr(U[:, :k].numpy())
            row["ipr_right_d"] = K.ipr(Vh[:k, :].T.numpy())
            row["ipr_left_ratio"] = row["ipr_left_d"] * A.shape[0]
            row["ipr_right_ratio"] = row["ipr_right_d"] * A.shape[1]
            row["hill_alpha_b"] = K.hill_alpha(torch.linalg.svdvals(B32).numpy())
    return row, d.half()


def aggregate(rows, L):
    tot = sum(r["fro_d"] ** 2 for r in rows)
    tot_a = sum(r["fro_a"] ** 2 for r in rows)
    layer = [0.0] * L
    cls = {}
    for r in rows:
        if r["layer"] is not None:
            layer[r["layer"]] += r["fro_d"] ** 2
        cls[r["cls"]] = cls.get(r["cls"], 0.0) + r["fro_d"] ** 2
    inl = sum(layer)
    prof = [v / inl if inl > 0 else 0.0 for v in layer]
    bins = [0.0] * NBINS
    for l, p in enumerate(prof):
        bins[min(NBINS - 1, (NBINS * l) // L)] += p
    cp = [r.get("cos_prev_delta") for r in rows if r.get("cos_prev_delta") is not None]
    num = sum(r["fro_d"] * r["fro_prev"] * r["cos_prev_delta"] for r in rows if r.get("cos_prev_delta") is not None and "fro_prev" in r)
    return {"total_rel_d": (tot / tot_a) ** 0.5 if tot_a > 0 else None, "layer_profile": prof, "depth_bins": bins,
            "depth_centroid": sum(p * l / (L - 1) for l, p in enumerate(prof)) if inl > 0 else None,
            "share_by_class": {k: v / tot if tot > 0 else 0.0 for k, v in cls.items()},
            "class_order": sorted(cls, key=lambda k: -cls[k]),
            "n_identical": sum(r["identical"] for r in rows), "n_tensors": len(rows),
            "lag1_median_cos": float(np.median(cp)) if cp else None, "lag1_frac_negative": float(np.mean([c < 0 for c in cp])) if cp else None,
            "lag1_whole_model_cos": (num / ((sum(r["fro_d"] ** 2 for r in rows if "fro_prev" in r) ** 0.5) * (sum(r["fro_prev"] ** 2 for r in rows if "fro_prev" in r) ** 0.5))) if cp else None,
            "medians_2d": {k: float(np.median([r[k] for r in rows if k in r and r[k] is not None])) for k in
                           ("stable_rank_a", "stable_rank_b", "effrank_d", "effrank_d_norm", "top1_share_d", "ipr_left_ratio", "ipr_right_ratio", "hill_alpha_b")
                           if any(k in r and r[k] is not None for r in rows)}}


# ------------------------------------------------------------ gate
def load_model_from_dir(d, tok_dir, common_config=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(tok_dir, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(d, trust_remote_code=True)
    rope = dict(getattr(cfg, "rope_parameters", None) or {})
    if common_config:
        base = MAN["selected"]["pretrain_100000"]["config"]
        cfg.rope_parameters = dict(base["rope_parameters"])
        cfg.max_position_embeddings = base["max_position_embeddings"]
    if not torch.backends.mps.is_available():
        raise SystemExit("REFUSING: mps unavailable")
    model = AutoModelForCausalLM.from_pretrained(d, config=cfg, trust_remote_code=True, dtype=torch.bfloat16, low_cpu_mem_usage=True).to("mps").eval()
    return model, tok, rope, dict(cfg.rope_parameters), cfg.max_position_embeddings


def run_ladder(model, tok, items, seed, tag, rows_f, counters, label):
    tiers = [1] if SMOKE else sorted(LAD.TIERS)
    out = {}
    for tier in tiers:
        sub_items = items if not SMOKE else [it for it in items if it["tier"] == 1][:8]
        g = LAD.run_tier(model, tok, sub_items, tier, seed, tag, rows_f, counters)
        out[tier] = g
        print(f"[gate] {SIZE} {tag} {label} seed {seed} T{tier} {g['correct']}/{g['n']} {g['by_family']}", flush=True)
    return out


def gate_tag(tag, tok_dir, items, rec):
    counters = {"timeout": 0, "empty": 0}
    rows_f = open(os.path.join(OUT, "gate_rows.jsonl"), "a")
    tg = time.time()
    model, tok, rope_rel, rope_used, mp = load_model_from_dir(tag_dir(tag), tok_dir)
    rec["gate"][tag] = {"rope_released": rope_rel, "rope_used": rope_used, "max_position_embeddings": mp, "seeds": {}}
    for sd in SEEDS:
        rec["gate"][tag]["seeds"][sd] = run_ladder(model, tok, items, sd, tag, rows_f, counters, "released")
    del model
    torch.mps.empty_cache()
    if tag in CONTROL_TAGS:
        model, tok, _, rope_used, mp = load_model_from_dir(tag_dir(tag), tok_dir, common_config=True)
        rec["gate_control"][tag] = {"rope_used": rope_used, "max_position_embeddings": mp,
                                    "seeds": {SEEDS[0]: run_ladder(model, tok, items, SEEDS[0], tag + "|common", rows_f, counters, "common")}}
        del model
        torch.mps.empty_cache()
    rec["gate"][tag]["oracle_counters"] = counters
    rec["timing"][f"gate_{tag}"] = round(time.time() - tg, 1)
    rows_f.close()


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    START = start_provenance(["scratch/k2h_transport.py", "scratch/k2h_gateladder.py", "scratch/k2h_stagecensus.py", MAN_PATH])
    os.makedirs(OUT)
    os.makedirs(SCRATCH, exist_ok=True)  # leftovers are re-verified by sha256 before use, never trusted
    t0 = time.time()
    items = LAD.make_items()
    rec = {"prereg": PREREG, "smoke": SMOKE, "size": SIZE, "repo": REPO, "start": START, "chain": CHAIN, "gated": GATED, "seeds": SEEDS,
           "items_sha256": LAD.digest(items), "n_items": len(items), "tags": {}, "pairs": {}, "gate": {}, "gate_control": {}, "timing": {},
           "device": "mps" if torch.backends.mps.is_available() else "none",
           "versions": {"python": sys.version, "torch": torch.__version__, "transformers": __import__("transformers").__version__}}
    if rec["device"] != "mps":
        raise SystemExit("REFUSING: mps unavailable")
    # frozen tokenizer for the size
    tok_dir = fetch_small(TOKENIZER_TAG)
    rec["tokenizer"] = {"tag": TOKENIZER_TAG, "commit": MAN["selected"][TOKENIZER_TAG]["commit"], "tokenizer_sha256": MAN["selected"][TOKENIZER_TAG]["small_sha256"]["tokenizer.json"]}
    census_f = open(os.path.join(OUT, "census.jsonl"), "w")
    prev_tag, prev_delta = None, {}
    held = None  # no tag is held: the endpoint pair re-downloads CHAIN[0] shard by shard (disk law: two tags + one shard)
    p_first = CHAIN[0]
    L = None
    for tag in CHAIN:
        td = time.time()
        fetch_small(tag)
        shards, wmap = shard_list(tag)
        cfg_file = json.load(open(os.path.join(tag_dir(tag), "config.json")))
        rec["tags"][tag] = {"commit": MAN["selected"][tag]["commit"], "n_shards": len(shards), "shard_sha256": {},
                            "config": {k: cfg_file.get(k) for k in ("num_hidden_layers", "hidden_size", "max_position_embeddings", "rope_parameters", "vocab_size")},
                            "config_sha256": sha_file(os.path.join(tag_dir(tag), "config.json")),
                            "tokenizer_ordinary_vocab_identical": tokenizer_homologous(tag_dir(tag), tok_dir)}
        if not rec["tags"][tag]["tokenizer_ordinary_vocab_identical"]:
            raise SystemExit(f"REFUSING: {tag} tokenizer ordinary vocab or merges differ from the frozen tokenizer")
        prefetch(tag, shards)
        if prev_tag is None:
            for s in shards:
                rec["tags"][tag]["shard_sha256"][s] = fetch_shard(tag, s)[1]
            rec["timing"][f"download_{tag}"] = round(time.time() - td, 1)
            prev_tag = tag
            if tag in GATED:
                gate_tag(tag, tok_dir, items, rec)
            continue
        pair = f"{prev_tag}->{tag}"
        structural = pair in STRUCTURAL
        keep_delta = tag in PRETRAIN and tag != "pretrain_1100000"  # the next pair is a 100k lag pair
        rows, cur_delta = [], {}
        prev_reader = TagReader(prev_tag)
        if prev_reader.names() != set(wmap):
            raise SystemExit(f"REFUSING: tensor sets differ {prev_tag} v {tag}")
        for s in shards:
            rec["tags"][tag]["shard_sha256"][s] = fetch_shard(tag, s)[1]
            if s not in census_shards(shards):
                continue
            B = load_shard(os.path.join(tag_dir(tag), s))
            A = {n: prev_reader.get(n) for n in B}
            for n in sorted(A):
                row, d16 = tensor_row(n, A[n], B[n], structural, prev_delta.get(n))
                if n in prev_delta:
                    row["fro_prev"] = float(torch.linalg.vector_norm(prev_delta[n].float()))
                row.update({"pair": pair, "structural": structural, "smoke": SMOKE})
                rows.append(row)
                census_f.write(json.dumps(row) + "\n")
                if keep_delta:
                    cur_delta[n] = d16
                del d16
            census_f.flush()
            del A, B
        prev_reader.close()
        L = L or (1 + max(r["layer"] for r in rows if r["layer"] is not None))
        agg = aggregate(rows, L)
        rec["pairs"][pair] = {"structural": structural, "kind": STRUCTURAL.get(pair, "LAG-PRETRAIN-100k"), **agg}
        rec["timing"][f"pair_{pair}"] = round(time.time() - td, 1)
        print(f"[pair] {SIZE} {pair} rel={agg['total_rel_d']:.4g} centroid={agg['depth_centroid']} lag1={agg['lag1_median_cos']} ident={agg['n_identical']}/{agg['n_tensors']} {round(time.time() - t0)}s", flush=True)
        # lag-1 deltas are kept (float16, RAM) only while the NEXT pair is a 100k-step pretraining pair
        prev_delta = cur_delta if keep_delta else {}
        del cur_delta
        if prev_tag != held:
            shutil.rmtree(tag_dir(prev_tag), ignore_errors=True)
        prev_tag = tag
        # endpoint pair when the last pretraining tag lands: CHAIN[0] is re-downloaded one shard at a time
        if tag == "pretrain_1100000":
            tp = time.time()
            pair = f"{p_first}->{tag}"
            rows = []
            first_shards, _ = shard_list(p_first) if os.path.exists(os.path.join(tag_dir(p_first), "model.safetensors.index.json")) else (None, None)
            if first_shards is None:
                fetch_small(p_first)
                first_shards, _ = shard_list(p_first)
            cur_reader = TagReader(tag)
            prefetch(p_first, first_shards)
            for s in first_shards:
                pth, got = fetch_shard(p_first, s)
                rec["tags"][p_first]["shard_sha256"][s] = got
                if s in census_shards(first_shards):
                    A = load_shard(pth)
                    if not set(A) <= cur_reader.names():
                        raise SystemExit(f"REFUSING: tensor sets differ in {s} (endpoint pair)")
                    B = {n: cur_reader.get(n) for n in A}
                    for n in sorted(A):
                        row, _ = tensor_row(n, A[n], B[n], True, None)
                        row.update({"pair": pair, "structural": True, "smoke": SMOKE})
                        rows.append(row)
                        census_f.write(json.dumps(row) + "\n")
                    census_f.flush()
                    del A, B
                os.remove(pth)
            cur_reader.close()
            agg = aggregate(rows, L)
            rec["pairs"][pair] = {"structural": True, "kind": "P", **agg}
            rec["timing"][f"pair_{pair}"] = round(time.time() - tp, 1)
            print(f"[pair] {SIZE} {pair} rel={agg['total_rel_d']:.4g} centroid={agg['depth_centroid']} {round(time.time() - t0)}s", flush=True)
            shutil.rmtree(tag_dir(p_first), ignore_errors=True)
        if tag in GATED:
            gate_tag(tag, tok_dir, items, rec)
    census_f.close()
    shutil.rmtree(SCRATCH, ignore_errors=True)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["completion_commit"] = completion_commit()
    json.dump(rec, open(os.path.join(OUT, "receipt.json"), "w"), indent=1, default=str)
    print(json.dumps({"timing": rec["timing"], "wall": rec["wall_s"]}, indent=1))


if __name__ == "__main__":
    main()
