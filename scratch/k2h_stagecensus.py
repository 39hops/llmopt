"""K2-HORIZON-STAGE-DELTA-CENSUS-0 instrument (prereg in docs/RESULTS.md):
the TRAINING-STAGE / CHECKPOINT-LIFECYCLE census of IFM/K2-Horizon-0.9B.
Zero training. For every pair in the frozen ancestry pin
(docs/preregs/k2h-stagecensus-0.ancestry.json) it downloads both tags,
computes per-tensor weight-delta statistics (normalized Frobenius delta,
stable rank before/after, entropy effective rank of the delta, IPR of the
delta's top singular vectors, a descriptive Hill alpha of the after-tensor,
cosine alignment of consecutive deltas), aggregates them into per-layer
and per-module share profiles, and scores every tag on a raw-completion
120-item sympy-verified mathgen gate (one prompt format for all tags,
three fixed few-shot seeds, greedy). The rl-mopd_final -> main pair is the
registered NO-OP precondition (documented weights unchanged): bit-identical
tensors and identical gate token ids, or every reading books behind a
failed precondition.

Receipts (streamed, refuse-if-exists on the output directory):
  logs/k2h/stagecensus/census.jsonl     one row per (pair, tensor)
  logs/k2h/stagecensus/gate_rows.jsonl  one row per (tag, seed, item)
  logs/k2h/stagecensus/receipt.json     pins, profiles, bars, provenance
K2H_SMOKE=1 writes to logs/k2h/stagecensus_smoke/ on two tags and a
12-item gate; smoke rows carry smoke: true.

Usage (the rung's own venv, transformers >= 5 for the remote code):
    .venv-k2/bin/python scratch/k2h_stagecensus.py
"""
import hashlib
import json
import multiprocessing as mp
import os
import random
import re
import sys
import time

import numpy as np
import sympy
import torch
import transformers

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

SMOKE = os.environ.get("K2H_SMOKE") == "1"
OUT = os.path.join(ROOT, "logs/k2h/stagecensus_smoke" if SMOKE else "logs/k2h/stagecensus")
ANC_PATH = "docs/preregs/k2h-stagecensus-0.ancestry.json"
MODEL = "IFM/K2-Horizon-0.9B"
PREREG = "K2-HORIZON-STAGE-DELTA-CENSUS-0" + ("-SMOKE" if SMOKE else "")
GATE_SEEDS = [0] if SMOKE else [0, 1, 2]
N_PER_CELL = 2 if SMOKE else 20
FAMILIES = ("diff", "int", "expand")
TIERS = (1, 2)
N_SHOTS = 4
MAX_NEW = 64
BATCH = 8
HILL_K_MAX = 50
IPR_K = 16
ORACLE_WALL = 10.0
ALLOW = ["*.json", "*.py", "*.safetensors", "*.jinja", "LICENSE"]


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- gate items
def _item(fam, tier, rng):
    import sympy as sp
    X = sp.Symbol("x")
    a, b, c, d = (rng.randint(1, 9) for _ in range(4))
    k = rng.randint(2, 4)
    if fam == "diff":
        f = a * X**k + b * X if tier == 1 else a * X**k * sp.sin(b * X) + c * X
        truth = sp.diff(f, X)
        q = f"Differentiate with respect to x: {sp.sstr(f)}"
    elif fam == "int":
        f = a * X**k + b if tier == 1 else a * X * sp.exp(b * X)
        truth = sp.integrate(f, X)
        q = f"Find an antiderivative with respect to x (no constant): {sp.sstr(f)}"
    else:
        f = (a * X + b) * (c * X + d) if tier == 1 else (a * X + b) * (c * X + d) * (X - k)
        truth = sp.expand(f)
        q = f"Expand fully: {sp.sstr(f)}"
    return {"family": fam, "tier": tier, "prompt": q, "truth": sp.sstr(truth)}


def make_items():
    items = []
    for fam in FAMILIES:
        for tier in TIERS:
            seen = set()
            i = 0
            while len([it for it in items if it["family"] == fam and it["tier"] == tier]) < N_PER_CELL:
                it = _item(fam, tier, random.Random(f"k2h-gate-{fam}-{tier}-{i}"))
                i += 1
                if it["prompt"] in seen:
                    continue
                seen.add(it["prompt"])
                it["id"] = len(items)
                items.append(it)
    return items


def make_shots(seed, exclude):
    """Fixed few-shot block per seed: N_SHOTS worked examples per family
    (tier alternating), excluded from the gate items by prompt string."""
    shots = []
    for fam in FAMILIES:
        j = 0
        got = 0
        while got < N_SHOTS:
            it = _item(fam, 1 + (got % 2), random.Random(f"k2h-shot-{seed}-{fam}-{j}"))
            j += 1
            if it["prompt"] in exclude:
                continue
            shots.append(it)
            got += 1
    random.Random(f"k2h-shot-order-{seed}").shuffle(shots)
    return shots


def render(shots, item):
    body = "".join(f"Problem: {s['prompt']}\nAnswer: {s['truth']}\n\n" for s in shots)
    return body + f"Problem: {item['prompt']}\nAnswer:"


def parse_answer(text):
    line = text.split("\n", 1)[0].strip()
    return line.rstrip(".") if line else None


def _check_worker(conn, ans, truth):
    try:
        import sympy as sp
        X = sp.Symbol("x")
        got = sp.sympify(ans, locals={"x": X})
        want = sp.sympify(truth, locals={"x": X})
        d = sp.simplify(got - want)
        conn.send(bool(d == 0))
    except Exception:
        conn.send(False)


def check_boxed(ans, truth, counters):
    """Fork-isolated sympy equivalence (never SIGALRM); timeout = reject."""
    if not ans:
        counters["empty"] += 1
        return False
    ctx = mp.get_context("fork")
    parent, child = ctx.Pipe(duplex=False)
    p = ctx.Process(target=_check_worker, args=(child, ans, truth))
    p.start()
    child.close()
    p.join(ORACLE_WALL)
    if p.is_alive():
        p.kill()
        p.join()
        counters["timeout"] += 1
        return False
    ok = parent.recv() if parent.poll() else False
    parent.close()
    return ok


# ---------------------------------------------------------------- weights
def download(tag, pin):
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL, revision=tag, allow_patterns=ALLOW)
    commit = os.path.basename(path)
    if not commit.startswith(pin):
        raise SystemExit(f"REFUSING: tag {tag} resolved to {commit}, ancestry pin {pin}")
    return path, commit


def load_weights(path):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    W = {}
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for name in f.keys():
                W[name] = f.get_tensor(name).to(torch.float32)
    return W


def module_class(name):
    if name == "model.embed_tokens.weight":
        return "embed"
    if name == "lm_head.weight":
        return "lm_head"
    if name.endswith("norm.weight"):
        return "norm"
    m = re.search(r"(self_attn|mlp)\.(\w+_proj)", name)
    return f"{m.group(1)}.{m.group(2)}" if m else "other"


def layer_of(name):
    m = re.search(r"model\.layers\.(\d+)\.", name)
    return int(m.group(1)) if m else None


def entropy_effrank(s):
    p = s**2
    p = p / p.sum()
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def hill_alpha(s):
    lam = np.sort(s**2)[::-1]
    k = min(HILL_K_MAX, len(lam) // 10)
    if k < 5 or lam[k - 1] <= 0:
        return None
    return float(1.0 + k / np.log(lam[:k] / lam[k - 1]).sum())


def ipr(vecs):
    v = vecs / np.linalg.norm(vecs, axis=0, keepdims=True)
    return float((v**4).sum(axis=0).mean())


def tensor_row(name, A, B, prev_d):
    d = B - A
    fa = float(torch.linalg.vector_norm(A))
    fd = float(torch.linalg.vector_norm(d))
    row = {"tensor": name, "cls": module_class(name), "layer": layer_of(name),
           "shape": list(A.shape), "fro_a": fa, "fro_b": float(torch.linalg.vector_norm(B)),
           "fro_d": fd, "rel_d": fd / fa if fa > 0 else None,
           "identical": bool(torch.equal(A, B))}
    if prev_d is not None:
        den = float(torch.linalg.vector_norm(prev_d)) * fd
        row["cos_prev_delta"] = float((prev_d.flatten() @ d.flatten()) / den) if den > 0 else None
    if A.ndim == 2 and min(A.shape) >= 64:
        sa = torch.linalg.svdvals(A).numpy()
        sb = torch.linalg.svdvals(B).numpy()
        row["stable_rank_a"] = float((sa**2).sum() / sa[0] ** 2)
        row["stable_rank_b"] = float((sb**2).sum() / sb[0] ** 2)
        row["hill_alpha_b"] = hill_alpha(sb)
        if fd > 0:
            U, S, Vh = torch.linalg.svd(d, full_matrices=False)
            S = S.numpy()
            row["effrank_d"] = entropy_effrank(S)
            row["top1_share_d"] = float(S[0] ** 2 / (S**2).sum())
            k = min(IPR_K, S.shape[0])
            row["ipr_left_d"] = ipr(U[:, :k].numpy())
            row["ipr_right_d"] = ipr(Vh[:k, :].T.numpy())
            row["ipr_left_uniform"] = 1.0 / A.shape[0]
            row["ipr_right_uniform"] = 1.0 / A.shape[1]
    return row, d


def aggregate(rows, n_layers):
    tot = sum(r["fro_d"] ** 2 for r in rows)
    tot_a = sum(r["fro_a"] ** 2 for r in rows)
    layer = [0.0] * n_layers
    cls = {}
    for r in rows:
        if r["layer"] is not None:
            layer[r["layer"]] += r["fro_d"] ** 2
        cls[r["cls"]] = cls.get(r["cls"], 0.0) + r["fro_d"] ** 2
    share_layer = [v / tot if tot > 0 else 0.0 for v in layer]
    in_layers = sum(layer)
    prof = [v / in_layers if in_layers > 0 else 0.0 for v in layer]
    centroid = sum(p * l / (n_layers - 1) for l, p in enumerate(prof)) if in_layers > 0 else None
    return {"total_rel_d": (tot / tot_a) ** 0.5 if tot_a > 0 else None,
            "share_by_layer_of_total": share_layer,
            "layer_profile": prof, "depth_centroid": centroid,
            "share_by_class": {k: v / tot if tot > 0 else 0.0 for k, v in cls.items()},
            "n_identical": sum(r["identical"] for r in rows), "n_tensors": len(rows)}


def rankdata(x):
    x = np.asarray(x, dtype=float)
    order = np.argsort(x)
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return ranks


def spearman(a, b):
    ra, rb = rankdata(a), rankdata(b)
    if ra.std() == 0 or rb.std() == 0:
        return None
    return float(np.corrcoef(ra, rb)[0, 1])


# ---------------------------------------------------------------- gate
def load_model(path, force_no_yarn=False):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(path, trust_remote_code=True)
    rope = dict(getattr(cfg, "rope_parameters", None) or {})
    if force_no_yarn:
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": rope["rope_theta"]}
        cfg.max_position_embeddings = 8192
    if not torch.backends.mps.is_available():
        raise SystemExit("REFUSING: mps unavailable (one device for every tag)")
    dev = "mps"
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, trust_remote_code=True,
                                                 dtype=torch.float32, low_cpu_mem_usage=True)
    model.to(dev).eval()
    return model, tok, dev, rope


def run_gate(model, tok, dev, items, seed, tag, rows_f, counters, extra):
    shots = make_shots(seed, {it["prompt"] for it in items})
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    nl = tok("\n", add_special_tokens=False)["input_ids"]
    correct = 0
    by_cell = {}
    ids_all = []
    for lo in range(0, len(items), BATCH):
        batch = items[lo:lo + BATCH]
        prompts = [render(shots, it) for it in batch]
        enc = tok(prompts, return_tensors="pt", padding=True)
        enc.pop("token_type_ids", None)
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                                 pad_token_id=tok.pad_token_id, eos_token_id=nl[-1])
        n_in = enc["input_ids"].shape[1]
        for it, seq in zip(batch, out):
            gen = seq[n_in:].tolist()
            text = tok.decode(gen, skip_special_tokens=True)
            ans = parse_answer(text)
            ok = check_boxed(ans, it["truth"], counters)
            correct += ok
            key = f"{it['family']}{it['tier']}"
            by_cell[key] = by_cell.get(key, 0) + int(ok)
            ids_all.append(gen)
            rows_f.write(json.dumps({"tag": tag, "seed": seed, "item": it["id"], "family": it["family"],
                                     "tier": it["tier"], "answer": ans, "truth": it["truth"],
                                     "correct": bool(ok), "gen_ids": gen, "smoke": SMOKE, **extra}) + "\n")
        rows_f.flush()
    return {"correct": correct, "n": len(items), "by_cell": by_cell,
            "gen_sha256": hashlib.sha256(json.dumps(ids_all).encode()).hexdigest()}


# ---------------------------------------------------------------- main
def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    anc = json.load(open(os.path.join(ROOT, ANC_PATH)))
    ladder = ["rl-mopd_final", "main"] if SMOKE else anc["ladder"]
    pairs = [p for p in anc["pairs"] if p["a"] in ladder and p["b"] in ladder]
    endpoints = {} if SMOKE else anc["stage_endpoints_for_profiles"]
    START = start_provenance(["scratch/k2h_stagecensus.py", ANC_PATH])
    os.makedirs(OUT)
    t0 = time.time()
    items = make_items()
    rec = {"prereg": PREREG, "smoke": SMOKE, "model": MODEL, "start": START, "ladder": ladder,
           "pairs": pairs, "gate_seeds": GATE_SEEDS, "n_items": len(items), "items_sha256":
           hashlib.sha256(json.dumps(items).encode()).hexdigest(), "shots": {},
           "tags": {}, "pair_stats": {}, "gate": {}, "oracle_counters": {},
           "versions": {"python": sys.version, "torch": torch.__version__, "transformers": transformers.__version__,
                        "sympy": sympy.__version__, "numpy": np.__version__}}
    for s in GATE_SEEDS:
        rec["shots"][s] = [sh["prompt"] for sh in make_shots(s, {it["prompt"] for it in items})]
    # 1. download + pin every tag
    paths = {}
    for tag in ladder:
        path, commit = download(tag, anc["tags"][tag])
        paths[tag] = path
        rec["tags"][tag] = {"commit": commit, "config_sha256": sha_file(os.path.join(path, "config.json")),
                            "index_sha256": sha_file(os.path.join(path, "model.safetensors.index.json")),
                            "shard_sha256": {}}
        print(f"[tag] {tag} {commit} {round(time.time() - t0)}s", flush=True)
    # 2. weight census, consecutive pairs first (carrying dW for alignment), then stage endpoints
    census_f = open(os.path.join(OUT, "census.jsonl"), "w")
    cache = {}

    def W(tag):
        if tag not in cache:
            if len(cache) >= 2:
                cache.pop(next(iter(cache)))
            cache[tag] = load_weights(paths[tag])
            idx = json.load(open(os.path.join(paths[tag], "model.safetensors.index.json")))
            for shard in sorted(set(idx["weight_map"].values())):
                rec["tags"][tag]["shard_sha256"][shard] = sha_file(os.path.join(paths[tag], shard))
        return cache[tag]

    n_layers = None
    prev_d = None
    all_pairs = [(p["a"], p["b"], p["class"], True) for p in pairs]
    all_pairs += [(a, b, f"ENDPOINT-{k}", False) for k, (a, b) in endpoints.items()
                  if (a, b) not in {(p["a"], p["b"]) for p in pairs}]
    for a, b, cls, chain in all_pairs:
        Wa, Wb = W(a), W(b)
        if set(Wa) != set(Wb):
            raise SystemExit(f"REFUSING: tensor sets differ {a} v {b}")
        n_layers = n_layers or (1 + max(layer_of(n) for n in Wa if layer_of(n) is not None))
        rows = []
        cur_d = {}
        for name in sorted(Wa):
            row, d = tensor_row(name, Wa[name], Wb[name], prev_d.get(name) if (chain and prev_d) else None)
            row.update({"pair": f"{a}->{b}", "pair_class": cls, "smoke": SMOKE})
            rows.append(row)
            census_f.write(json.dumps(row) + "\n")
            if chain:
                cur_d[name] = d
        census_f.flush()
        if chain:
            prev_d = cur_d
        agg = aggregate(rows, n_layers)
        rec["pair_stats"][f"{a}->{b}"] = {"class": cls, **agg}
        print(f"[pair] {a}->{b} {cls} total_rel_d={agg['total_rel_d']} centroid={agg['depth_centroid']} "
              f"identical={agg['n_identical']}/{agg['n_tensors']} {round(time.time() - t0)}s", flush=True)
    census_f.close()
    cache.clear()
    prev_d = None
    # 3. profiles + Spearman
    prof = {k: rec["pair_stats"][f"{a}->{b}"]["layer_profile"] for k, (a, b) in endpoints.items()}
    keys = list(prof)
    rec["profile_spearman"] = {f"{x}|{y}": spearman(prof[x], prof[y]) for i, x in enumerate(keys) for y in keys[i + 1:]}
    rec["profiles"] = prof
    # 4. gates
    counters = {"timeout": 0, "empty": 0}
    rows_f = open(os.path.join(OUT, "gate_rows.jsonl"), "w")
    for tag in ladder:
        model, tok, dev, rope = load_model(paths[tag])
        rec["tags"][tag]["rope_parameters"] = rope
        rec["tags"][tag]["device"] = dev
        rec["gate"][tag] = {}
        for s in GATE_SEEDS:
            g = run_gate(model, tok, dev, items, s, tag, rows_f, counters, {"rider": None})
            rec["gate"][tag][s] = g
            print(f"[gate] {tag} seed {s} {g['correct']}/{g['n']} {g['by_cell']} {round(time.time() - t0)}s", flush=True)
        del model
        if dev == "mps":
            torch.mps.empty_cache()
    # 5. descriptive rider: main with YaRN disabled, seed 0 only
    rec["rider_no_yarn"] = None
    try:
        model, tok, dev, rope = load_model(paths["main"], force_no_yarn=True)
        g = run_gate(model, tok, dev, items, GATE_SEEDS[0], "main", rows_f, counters, {"rider": "no_yarn"})
        rec["rider_no_yarn"] = {"seed": GATE_SEEDS[0], **g}
        print(f"[rider] main no_yarn {g['correct']}/{g['n']}", flush=True)
        del model
    except Exception as e:  # rider is descriptive; its failure is booked, never hidden
        rec["rider_no_yarn"] = {"status": "NOT-RUN", "error": repr(e)[:300]}
        print(f"[rider] FAILED {e!r}", flush=True)
    rows_f.close()
    rec["oracle_counters"] = counters
    # 6. precondition + bars
    noop = rec["pair_stats"].get("rl-mopd_final->main")
    gate_same = all(rec["gate"]["rl-mopd_final"][s]["gen_sha256"] == rec["gate"]["main"][s]["gen_sha256"]
                    for s in GATE_SEEDS)
    ti = bool(noop and noop["n_identical"] == noop["n_tensors"] == 255)
    rec["precondition_noop"] = {"tensors_identical": ti, "gate_ids_identical": gate_same, "fires": ti and gate_same}
    if not SMOKE and not rec["precondition_noop"]["fires"]:
        rec["bars"] = {"status": "PRECONDITION-FAILED"}
    elif not SMOKE:
        G = lambda t, s: rec["gate"][t][s]["correct"]
        deltas = [G("rl-mopd_final", s) - G("mid_2_final", s) for s in GATE_SEEDS]
        sp_ = rec["profile_spearman"]
        stage3 = [sp_[k] for k in sp_ if all(x in ("PRETRAIN-INTERNAL", "MID_1-INTERNAL", "MID_2-INTERNAL")
                                             for x in k.split("|"))]
        rec["bars"] = {
            "a_post_training_gate_move": {"deltas": deltas,
                                          "fires": all(d > 7 for d in deltas) or all(d < -7 for d in deltas)},
            "b_stage_profiles_distinct": {"spearman": stage3, "fires": len(stage3) == 3 and all(v is not None and v <= 0.5 for v in stage3)},
            "c_distill_output_proximal": {"centroid": rec["pair_stats"]["rl_rl-merged->rl-mopd_final"]["depth_centroid"],
                                          "fires": (rec["pair_stats"]["rl_rl-merged->rl-mopd_final"]["depth_centroid"] or 0) > 0.5},
        }
        rec["refuted_if"] = {"context_extension_distinct_refuted": bool(
            sp_["PRETRAIN-INTERNAL|MID_1-INTERNAL"] is not None and sp_["PRETRAIN-INTERNAL|MID_1-INTERNAL"] > 0.5
            and sp_["PRETRAIN-INTERNAL|MID_2-INTERNAL"] is not None and sp_["PRETRAIN-INTERNAL|MID_2-INTERNAL"] > 0.5)}
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["completion_commit"] = completion_commit()
    json.dump(rec, open(os.path.join(OUT, "receipt.json"), "w"), indent=1, default=str)
    print(json.dumps({k: rec[k] for k in ("precondition_noop", "profile_spearman", "oracle_counters", "wall_s")
                      if k in rec}, indent=1, default=str))
    if "bars" in rec:
        print(json.dumps(rec["bars"], indent=1, default=str))


if __name__ == "__main__":
    main()
