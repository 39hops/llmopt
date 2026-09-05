"""K2-HORIZON-RESIDUES-0 instrument (prereg in docs/RESULTS.md): the two
cheap residues of VERDICT K2-HORIZON-STAGE-DELTA-CENSUS-0 (RESULTS L66337).

A. FACTORIAL — weights {mid_2_final, rl_rl-merged} x rope config {none,
   yarn} on the frozen 120-item raw-completion gate at the three frozen
   shot seeds; the two cells already booked at L66337 must reproduce
   their generation digests bit-exactly (gate); main effects and the
   interaction per seed.
B. AUTOCORRELATION — the full published 0.9B tag chain (pretrain
   500000..600000 by 10k, mid_1 50000..75000 by 5k, mid_2 5000..45000 by
   5k then 47684) walked in order; for every delta between consecutive
   tags the per-tensor cosine against the previous one, two and three
   deltas (lags 1-3), plus coarse-spacing cosines inside each stage. Tags
   not already in the HF cache are downloaded to a per-tag scratch cache
   and deleted after use (disk fence). Weights held as bf16 (exact),
   deltas as float16 for the lag deque.

Imports the frozen scratch/k2h_stagecensus.py for items, shots, gate and
loading (never copied). RES_SMOKE=1 writes to logs/k2h/residues_smoke/,
sets K2H_SMOKE for the imported module (12-item gate), and walks a
three-tag cached chain.

Usage:
    .venv-k2/bin/python scratch/k2h_residues.py
"""
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import time

SMOKE = os.environ.get("RES_SMOKE") == "1"
if SMOKE:
    os.environ["K2H_SMOKE"] = "1"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from llmopt.lab.provenance import completion_commit, start_provenance  # noqa: E402

_spec = importlib.util.spec_from_file_location("k2h", os.path.join(ROOT, "scratch/k2h_stagecensus.py"))
K = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(K)

OUT = os.path.join(ROOT, "logs/k2h/residues_smoke" if SMOKE else "logs/k2h/residues")
PREREG = "K2-HORIZON-RESIDUES-0" + ("-SMOKE" if SMOKE else "")
BOOKED = os.path.join(ROOT, "logs/k2h/stagecensus/receipt.json")
LOCK = os.path.join(ROOT, "docs/receipts.lock.json")
ANC = os.path.join(ROOT, "docs/preregs/k2h-stagecensus-0.ancestry.json")
SEEDS = [0] if SMOKE else [0, 1, 2]
SCRATCH_CACHE = os.path.join(ROOT, "logs/k2h/_tagcache")
PRETRAIN = [f"pretrain_{s}" for s in range(500000, 600001, 10000)]
MID1 = [f"mid_1_{s}" for s in range(50000, 75001, 5000)]
MID2 = [f"mid_2_{s}" for s in range(5000, 45001, 5000)] + ["mid_2_47684"]
CHAIN = PRETRAIN + MID1 + MID2
COARSE = {"pretrain": ("pretrain_500000", "pretrain_550000", "pretrain_600000"),
          "mid_1": ("mid_1_50000", "mid_1_60000", "mid_1_75000"),
          "mid_2": ("mid_2_5000", "mid_2_25000", "mid_2_47684")}
LAGS = (1, 2, 3)
if SMOKE:
    CHAIN = ["pretrain_500000", "pretrain_600000", "mid_1_50000"]
    COARSE = {}


def stage_of(tag):
    return "pretrain" if tag.startswith("pretrain") else ("mid_1" if tag.startswith("mid_1") else "mid_2")


def cos(a, b):
    num = float(torch.dot(a.float().flatten(), b.float().flatten()))
    den = float(torch.linalg.vector_norm(a.float())) * float(torch.linalg.vector_norm(b.float()))
    return num / den if den > 0 else None


# ------------------------------------------------------------ part A
def rope_override(cfg, mode, yarn):
    theta = cfg.rope_parameters["rope_theta"]
    if mode == "none":
        cfg.rope_parameters = {"rope_type": "default", "rope_theta": theta}
        cfg.max_position_embeddings = 8192
    else:
        cfg.rope_parameters = dict(yarn, rope_theta=theta)
        cfg.max_position_embeddings = 131072
    return cfg


def load_cell(path, mode, yarn):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    cfg = rope_override(AutoConfig.from_pretrained(path, trust_remote_code=True), mode, yarn)
    if not torch.backends.mps.is_available():
        raise SystemExit("REFUSING: mps unavailable")
    model = AutoModelForCausalLM.from_pretrained(path, config=cfg, trust_remote_code=True,
                                                 dtype=torch.float32, low_cpu_mem_usage=True).to("mps").eval()
    return model, tok, dict(cfg.rope_parameters), cfg.max_position_embeddings


def part_a(rec, anc, items, t0):
    booked = json.load(open(BOOKED))
    yarn = {k: v for k, v in booked["tags"]["main"]["rope_parameters"].items() if k != "rope_theta"}
    rec["A"] = {"yarn_source": "booked receipt tags.main.rope_parameters", "yarn": yarn, "cells": {}}
    counters = {"timeout": 0, "empty": 0}
    rows_f = open(os.path.join(OUT, "factorial_rows.jsonl"), "w")
    paths = {}
    for tag in ("mid_2_final", "rl_rl-merged"):
        paths[tag], commit = K.download(tag, anc["tags"][tag])
        rec["A"][f"commit_{tag}"] = commit
    for tag in ("mid_2_final", "rl_rl-merged"):
        for mode in ("none", "yarn"):
            model, tok, rp, mp = load_cell(paths[tag], mode, yarn)
            cell = f"{tag}|{mode}"
            rec["A"]["cells"][cell] = {"rope_parameters": rp, "max_position_embeddings": mp, "seeds": {}}
            for s in SEEDS:
                g = K.run_gate(model, tok, "mps", items, s, tag, rows_f, counters, {"rider": None, "rope": mode, "cell": cell})
                rec["A"]["cells"][cell]["seeds"][s] = g
                print(f"[A] {cell} seed {s} {g['correct']}/{g['n']} {round(time.time() - t0)}s", flush=True)
            del model
            torch.mps.empty_cache()
    rows_f.close()
    rec["A"]["oracle_counters"] = counters
    C = lambda tag, mode, s: rec["A"]["cells"][f"{tag}|{mode}"]["seeds"][s]["correct"]
    rec["A"]["effects"] = {}
    for s in SEEDS:
        m2n, m2y, mgn, mgy = C("mid_2_final", "none", s), C("mid_2_final", "yarn", s), C("rl_rl-merged", "none", s), C("rl_rl-merged", "yarn", s)
        rec["A"]["effects"][s] = {"weight_main": ((mgn - m2n) + (mgy - m2y)) / 2, "rope_main": ((m2y - m2n) + (mgy - mgn)) / 2,
                                  "interaction": mgy - mgn - m2y + m2n, "weight_at_none": mgn - m2n, "weight_at_yarn": mgy - m2y,
                                  "rope_at_mid2": m2y - m2n, "rope_at_merged": mgy - mgn}
    # reproduction gate against the booked cells
    repro = {}
    for tag, mode in (("mid_2_final", "none"), ("rl_rl-merged", "yarn")):
        for s in SEEDS:
            repro[f"{tag}|{mode}|{s}"] = rec["A"]["cells"][f"{tag}|{mode}"]["seeds"][s]["gen_sha256"] == booked["gate"][tag][str(s)]["gen_sha256"]
    rec["A"]["reproduction"] = {"cells": repro, "fires": all(repro.values())}
    if not SMOKE:
        E = rec["A"]["effects"]
        rec["A"]["bars"] = {"A1_weight_main_positive": {"values": [E[s]["weight_main"] for s in SEEDS], "fires": all(E[s]["weight_main"] > 0 for s in SEEDS)},
                            "A2_rope_main_positive": {"values": [E[s]["rope_main"] for s in SEEDS], "fires": all(E[s]["rope_main"] > 0 for s in SEEDS)},
                            "A3_additive": {"values": [E[s]["interaction"] for s in SEEDS], "fires": all(abs(E[s]["interaction"]) <= 3 for s in SEEDS)}}
        if not rec["A"]["reproduction"]["fires"]:
            rec["A"]["bars"] = {"status": "REPRODUCTION-FAILED"}


# ------------------------------------------------------------ part B
def fetch(tag, pin, rec):
    """Cached tags come from the default cache; others go to a per-tag
    scratch cache that is deleted after use."""
    from huggingface_hub import snapshot_download
    try:
        path = snapshot_download(K.MODEL, revision=tag, allow_patterns=K.ALLOW, local_files_only=True)
        scratch = None
    except Exception:
        scratch = os.path.join(SCRATCH_CACHE, tag)
        path = snapshot_download(K.MODEL, revision=tag, allow_patterns=K.ALLOW, cache_dir=scratch)
    commit = os.path.basename(path)
    if not commit.startswith(pin):
        raise SystemExit(f"REFUSING: {tag} resolved to {commit}, pin {pin}")
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    shards = {s: K.sha_file(os.path.join(path, s)) for s in sorted(set(idx["weight_map"].values()))}
    rec["B"]["tags"][tag] = {"commit": commit, "scratch": scratch is not None, "shard_sha256": shards}
    return path, scratch


def load_bf16(path):
    from safetensors import safe_open
    idx = json.load(open(os.path.join(path, "model.safetensors.index.json")))
    W = {}
    for shard in sorted(set(idx["weight_map"].values())):
        with safe_open(os.path.join(path, shard), framework="pt") as f:
            for n in f.keys():
                W[n] = f.get_tensor(n)
    return W


def part_b(rec, pins, t0):
    rows_f = open(os.path.join(OUT, "autocorr_rows.jsonl"), "w")
    rec["B"] = {"chain": CHAIN, "tags": {}, "deltas": [], "coarse": {}}
    prev_tag, prev_W = None, None
    deque = []  # list of (delta_tag_pair, {tensor: fp16 delta}) newest last
    keep = {t for trip in COARSE.values() for t in trip}
    held = {}
    for tag in CHAIN:
        path, scratch = fetch(tag, pins[tag], rec)
        W = load_bf16(path)
        if tag in keep:
            held[tag] = W
        if prev_W is not None:
            if set(W) != set(prev_W):
                raise SystemExit("REFUSING: tensor sets differ")
            pair = f"{prev_tag}->{tag}"
            cls = "WITHIN" if stage_of(prev_tag) == stage_of(tag) else "BOUNDARY"
            d = {n: (W[n].float() - prev_W[n].float()) for n in W}
            tot = float(sum(float(torch.linalg.vector_norm(v)) ** 2 for v in d.values())) ** 0.5
            entry = {"pair": pair, "stage": stage_of(tag), "class": cls, "fro_total": tot, "lags": {}}
            for lag in LAGS:
                if len(deque) < lag:
                    continue
                ppair, pd = deque[-lag]
                cs = []
                num = den_a = den_b = 0.0
                for n in W:
                    c = cos(d[n], pd[n])
                    fa, fb = float(torch.linalg.vector_norm(d[n])), float(torch.linalg.vector_norm(pd[n].float()))
                    num += float(torch.dot(d[n].flatten(), pd[n].float().flatten()))
                    den_a += fa ** 2
                    den_b += fb ** 2
                    rows_f.write(json.dumps({"pair": pair, "prev_pair": ppair, "lag": lag, "stage": stage_of(tag), "class": cls,
                                             "tensor": n, "cls": K.module_class(n), "layer": K.layer_of(n), "cos": c,
                                             "fro_d": fa, "fro_prev": fb, "smoke": SMOKE}) + "\n")
                    if c is not None:
                        cs.append(c)
                entry["lags"][lag] = {"prev_pair": ppair, "median_cos": float(np.median(cs)) if cs else None,
                                      "frac_negative": float(np.mean([c < 0 for c in cs])) if cs else None,
                                      "whole_model_cos": num / (den_a ** 0.5 * den_b ** 0.5) if den_a > 0 and den_b > 0 else None, "n": len(cs)}
            rows_f.flush()
            rec["B"]["deltas"].append(entry)
            print(f"[B] {pair} {cls} fro={tot:.4g} " + " ".join(f"lag{l}={v['median_cos']:.3f}/{v['frac_negative']:.2f}" for l, v in entry["lags"].items()) + f" {round(time.time() - t0)}s", flush=True)
            deque.append((pair, {n: v.half() for n, v in d.items()}))
            deque = deque[-max(LAGS):]
            del d
        prev_tag, prev_W = tag, W
        if scratch:
            shutil.rmtree(scratch, ignore_errors=True)
    rows_f.close()
    for st, (a, b, c) in COARSE.items():
        d1 = {n: held[b][n].float() - held[a][n].float() for n in held[a]}
        d2 = {n: held[c][n].float() - held[b][n].float() for n in held[a]}
        cs = [cos(d1[n], d2[n]) for n in d1]
        cs = [x for x in cs if x is not None]
        num = sum(float(torch.dot(d1[n].flatten(), d2[n].flatten())) for n in d1)
        den = (sum(float(torch.linalg.vector_norm(v)) ** 2 for v in d1.values()) * sum(float(torch.linalg.vector_norm(v)) ** 2 for v in d2.values())) ** 0.5
        rec["B"]["coarse"][st] = {"triple": [a, b, c], "median_cos": float(np.median(cs)), "frac_negative": float(np.mean([x < 0 for x in cs])), "whole_model_cos": num / den if den > 0 else None}
        print(f"[B-coarse] {st} {a}->{b} v {b}->{c} median={rec['B']['coarse'][st]['median_cos']:.3f}", flush=True)
    held.clear()
    # aggregates + bars
    agg = {}
    for st in ("pretrain", "mid_1", "mid_2"):
        D = rec["B"]["deltas"]
        w1 = [e["lags"][1]["median_cos"] for i, e in enumerate(D) if i > 0 and e["stage"] == st and e["class"] == "WITHIN"
              and 1 in e["lags"] and D[i - 1]["class"] == "WITHIN" and D[i - 1]["stage"] == st]
        agg[st] = {"within_lag1_medians": w1, "n_pairs": len(w1), "frac_pairs_negative": float(np.mean([x < 0 for x in w1])) if w1 else None,
                   "median_abs_within_lag1": float(np.median([abs(x) for x in w1])) if w1 else None}
    bnd = {}
    for i, e in enumerate(rec["B"]["deltas"]):
        if i > 0 and rec["B"]["deltas"][i - 1]["class"] == "BOUNDARY" and 1 in e["lags"]:
            bnd[e["stage"]] = {"pair": e["pair"], "boundary_pair": e["lags"][1]["prev_pair"], "cos_first_within_v_boundary": e["lags"][1]["median_cos"]}
    rec["B"]["aggregates"] = agg
    rec["B"]["boundary"] = bnd
    if not SMOKE:
        rec["B"]["bars"] = {
            "B1_repeatable_anti_alignment": {"frac_negative_by_stage": {s: agg[s]["frac_pairs_negative"] for s in agg},
                                             "fires": all(agg[s]["frac_pairs_negative"] is not None and agg[s]["frac_pairs_negative"] >= 0.8 for s in agg)},
            "B2_spacing_sign_agrees": {"by_stage": {s: {"coarse": rec["B"]["coarse"][s]["median_cos"], "fine_median": float(np.median(agg[s]["within_lag1_medians"]))} for s in agg},
                                       "fires": all(np.sign(rec["B"]["coarse"][s]["median_cos"]) == np.sign(np.median(agg[s]["within_lag1_medians"])) for s in agg)},
            "B3_boundary_specific": {"by_stage": {s: {"boundary_abs": abs(bnd[s]["cos_first_within_v_boundary"]), "within_median_abs": agg[s]["median_abs_within_lag1"]} for s in bnd},
                                     "fires": len(bnd) == 2 and all(abs(bnd[s]["cos_first_within_v_boundary"]) > agg[s]["median_abs_within_lag1"] for s in bnd)}}


def main():
    if os.path.exists(OUT):
        raise SystemExit(f"REFUSING: {OUT} exists")
    lock = json.load(open(LOCK))["receipts"]
    rel = os.path.relpath(BOOKED, ROOT)
    if K.sha_file(BOOKED) != lock[rel]["sha256"]:
        raise SystemExit("REFUSING: booked receipt sha v lock")
    anc = json.load(open(ANC))
    START = start_provenance(["scratch/k2h_residues.py", "scratch/k2h_stagecensus.py", "docs/preregs/k2h-stagecensus-0.ancestry.json", rel])
    os.makedirs(OUT)
    t0 = time.time()
    items = K.make_items()
    rec = {"prereg": PREREG, "smoke": SMOKE, "model": K.MODEL, "start": START, "seeds": SEEDS, "n_items": len(items),
           "items_sha256": hashlib.sha256(json.dumps(items).encode()).hexdigest(), "booked_receipt_sha256": K.sha_file(BOOKED),
           "versions": {"python": sys.version, "torch": torch.__version__}}
    pins = dict(anc["tags"])
    from huggingface_hub import HfApi
    refs = HfApi().list_repo_refs(K.MODEL)
    for t in refs.tags:
        pins.setdefault(t.name, t.target_commit[:8])
    rec["pins_from_refs_api"] = {t: pins[t] for t in CHAIN}
    part_a(rec, anc, items, t0)
    part_b(rec, pins, t0)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["completion_commit"] = completion_commit()
    json.dump(rec, open(os.path.join(OUT, "receipt.json"), "w"), indent=1, default=str)
    print(json.dumps({"A_effects": rec["A"]["effects"], "A_repro": rec["A"]["reproduction"], "A_bars": rec["A"].get("bars"),
                      "B_agg": rec["B"]["aggregates"], "B_coarse": rec["B"]["coarse"], "B_bars": rec["B"].get("bars"), "wall": rec["wall_s"]}, indent=1, default=str))


if __name__ == "__main__":
    main()
