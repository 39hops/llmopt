"""QWEN-CHEAP-READOUT-0 driver: arm-state candidate-set census
(PRE-REG in docs/RESULTS.md; bank CHEAP-READOUT-CENSUS).

Reruns arm A's streamed CPU forward on the frozen MODEL-1 surface,
capturing h_A = the literal tensor entering the (patched) lm_head
forward. Three objects at identical positions:
  T   frozen teacher logits, A_W = w4 head @ h_A (native),
  A_T = vendor bf16 head @ h_A (control).
Identity fixture gates everything: A_W recomputed from captured h_A
must reproduce the run's own native logits (top1 identical at every
position, max abs diff <= 1e-3) or the run books INSTRUMENT-NOT-RUN.

Model-build machinery and pure math are REUSED from
scratch/qwen_model1_score.py by module load (no copy); this driver
adds only capture + candidate-set readings.

    ART_DIR=~/qwen_whole0t/A ARM=A .venv/bin/python \
        scratch/qwen_cheap_readout.py          (mac, cpu only)
    SMOKE=1 ...   # truncated, *_smoke paths only

Receipt: logs/qwencheapread/census_<ARM>.json (refuse-if-exists);
h/logit arrays untracked npz, sha-pinned in the receipt.
"""
import hashlib
import importlib.util
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab.provenance import (completion_commit,  # noqa: E402
                                   start_provenance)

KS = (16, 64, 256, 1024)
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name, rel):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(_ROOT, rel))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def rk_mk(t_logits_f32, obj_logits, v_live, ks=KS):
    """R_k = P(teacher top1 in obj top-k); M_k = mean teacher mass
    captured by obj's top-k set. fp64 softmax for mass."""
    t = t_logits_f32[:, :v_live]
    o = obj_logits[:, :v_live]
    t_top1 = t.argmax(1)
    x = t.astype(np.float64)
    x = x - x.max(1, keepdims=True)
    p = np.exp(x)
    p /= p.sum(1, keepdims=True)
    out = {}
    order = np.argsort(-o, axis=1)
    for k in ks:
        cand = order[:, :k]
        hit = (cand == t_top1[:, None]).any(1)
        mass = np.take_along_axis(p, cand, 1).sum(1)
        out[f"R_{k}"] = float(hit.mean())
        out[f"M_{k}"] = float(mass.mean())
        out[f"n_miss_{k}"] = int((~hit).sum())
    out["n"] = int(len(t))
    return out


def rk_by_margin(t_logits_fp16, obj_logits, v_live, ms):
    """Margin-stratified R_256 on the frozen MARGIN_EDGES bins;
    raw counts always included (small-n fence rides on the reader)."""
    t_top1, margins = ms.teacher_margins_top1(t_logits_fp16, v_live)
    order = np.argsort(-obj_logits[:, :v_live], axis=1)[:, :256]
    hit = (order == np.asarray(t_top1)[:, None]).any(1)
    table = {}
    for m, h in zip(margins, hit):
        b = ms.margin_bin(float(m))
        d = table.setdefault(str(b), {"n": 0, "hits": 0})
        d["n"] += 1
        d["hits"] += int(h)
    return table


def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    suf = "_smoke" if smoke else ""
    arm = os.environ["ARM"]
    assert arm in ("A",), arm   # widen deliberately, arm by arm
    out_dir = "logs/qwencheapread" + suf
    os.makedirs(out_dir, exist_ok=True)
    rcpt = os.path.join(out_dir, f"census_{arm}.json")
    npz_path = os.path.join(out_dir, f"census_arrays_{arm}.npz")
    for p in (rcpt, npz_path):
        if os.path.exists(p):
            raise SystemExit(f"REFUSING: {p} exists")
    START = start_provenance(
        ["scratch/qwen_cheap_readout.py",
         "scratch/qwen_model1_score.py",
         "llmopt/lab/qcodec.py", "llmopt/lab/qcodec_fast.py",
         "llmopt/lab/qartifact.py"])

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())
    if os.environ.get("DEVICE", "cpu") != "cpu":
        raise SystemExit("REFUSING: census runs on cpu only")

    ms = _load("qwen_model1_score", "scratch/qwen_model1_score.py")
    from transformers import AutoTokenizer
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR",
                                             "~/qwen_vendor"))
    tok = AutoTokenizer.from_pretrained(vdir)
    man_t = json.load(open(os.path.join(ms.TEACHER_DIR,
                                        "teacher_manifest.json")))
    corpus_ids, ptoks = ms.refuse_list_checks(man_t, tok, smoke)
    tl_corpus = ms.load_record("corpus_logits.npy",
                               man_t["records"]["corpus"])
    tl_prefix = ms.load_record("prefix_logits.npy",
                               man_t["records"]["prefixes"])
    bounds = man_t["records"]["prefixes"]["bounds"]
    v_live = len(tok)

    # ---- build the streamed model exactly as the scorer does ----
    # (executes the scorer's own build path via a temporary main-less
    # replication is NOT done; instead we inline the minimal calls)
    from llmopt.lab import qartifact
    from llmopt.lab.qcodec import decode_entry
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows
    art = os.path.expanduser(os.environ["ART_DIR"])
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    if not os.path.exists(chain):
        raise SystemExit(f"REFUSING: no digest chain {chain}")
    q = qartifact.qualify_artifact(
        art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    print(f"[cr] qualified {arm}: {q['report']}", flush=True)
    handles = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def decode(name):
        e = MAN[name]
        if e["codec"] == "excluded":
            raise SystemExit(f"REFUSING: excluded tensor {name}")
        return torch.from_numpy(np.ascontiguousarray(
            decode_entry(payload(e), e)))

    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(vdir)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.float32)
    model.eval()

    def shard_key(nm):
        if nm.startswith("model."):
            return "model.language_model." + nm[len("model."):]
        return nm

    for nm, _ in list(model.named_parameters()):
        if nm.startswith("model.layers."):
            continue
        if nm in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        mod = model.get_submodule(nm.rsplit(".", 1)[0])
        setattr(mod, nm.rsplit(".", 1)[1],
                torch.nn.Parameter(decode(shard_key(nm)),
                                   requires_grad=False))

    def rows_for(key):
        e = MAN[key]
        cls = {"w4": W4Rows, "s16": S16Rows}.get(e["codec"])
        if cls is None:
            raise SystemExit(f"REFUSING: io codec {e['codec']}")
        return cls(payload(e), e["shape"])

    emb = rows_for("model.language_model.embed_tokens.weight")
    head = rows_for("lm_head.weight")

    def emb_fwd(input_ids):
        flat = input_ids.reshape(-1)
        out = torch.empty(flat.shape[0], emb.C)
        for j, t in enumerate(flat.tolist()):
            out[j] = torch.from_numpy(emb.rows(t, t + 1)[0])
        return out.reshape(*input_ids.shape, emb.C)

    model.model.embed_tokens.forward = emb_fwd

    captured = []   # h entering the head, per forward call

    def w4_head_mm(x):
        outs = []
        for lo in range(0, head.R, 16384):
            hi = min(lo + 16384, head.R)
            outs.append(x @ torch.from_numpy(head.rows(lo, hi)).T)
        return torch.cat(outs, -1)

    def head_fwd(x):
        captured.append(x.detach().to(torch.float32).clone())
        return w4_head_mm(x)

    model.lm_head.forward = head_fwd
    if [nm for nm, b in model.named_buffers() if b.is_meta]:
        raise SystemExit("REFUSING: meta buffers after build")
    from llmopt.lab import qrope
    rp = cfg.text_config.rope_parameters
    qrope.check_inv_freq(model.model.rotary_emb.inv_freq.numpy(),
                         float(rp["rope_theta"]),
                         int(cfg.text_config.head_dim
                             * rp.get("partial_rotary_factor", 1.0)))
    layers = model.model.layers

    def make_pre(i):
        def pre(module, args, kwargs):
            for nm, _ in module.named_parameters():
                full = f"model.language_model.layers.{i}.{nm}"
                m2 = module.get_submodule(nm.rsplit(".", 1)[0]) \
                    if "." in nm else module
                leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
                m2._parameters[leaf] = torch.nn.Parameter(
                    decode(full), requires_grad=False)
            return None
        return pre

    def post(module, args, kwargs, output):
        for nm, p in list(module.named_parameters()):
            m2 = module.get_submodule(nm.rsplit(".", 1)[0]) \
                if "." in nm else module
            leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
            m2._parameters[leaf] = torch.nn.Parameter(
                p.to("meta"), requires_grad=False)
        return output

    for i, lyr in enumerate(layers):
        lyr.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        lyr.register_forward_hook(post, with_kwargs=True)

    def fwd(ids):
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        lg = out.logits[0].float().numpy()
        if not np.isfinite(lg).all():
            raise SystemExit("REFUSING: non-finite arm logits")
        return lg

    if smoke:
        corpus_ids = corpus_ids[:12]
        tl_corpus = tl_corpus[:12]
        ptoks = ptoks[:1]
        bounds = bounds[:1]
        tl_prefix = tl_prefix[:bounds[0]]

    # ---- forwards with capture ----
    t = time.time()
    al_corpus = fwd(corpus_ids)
    h_corpus = captured[-1][0].numpy()
    print(f"[cr] corpus forward {time.time()-t:.0f}s "
          f"h {h_corpus.shape}", flush=True)
    h_prefix, al_prefix = [], []
    for pi, ids in enumerate(ptoks):
        t = time.time()
        lg = fwd(ids)
        al_prefix.append(lg)
        h_prefix.append(captured[-1][0].numpy())
        print(f"[cr] prefix {pi} forward {time.time()-t:.0f}s",
              flush=True)
    h_prefix_cat = np.concatenate(h_prefix)
    al_prefix_cat = np.concatenate(al_prefix)
    if len(h_corpus) != len(tl_corpus) or \
            len(h_prefix_cat) != len(tl_prefix):
        raise SystemExit("REFUSING: captured position count mismatch")

    # ---- identity fixture (gate) ----
    rec_c = w4_head_mm(torch.from_numpy(h_corpus)).numpy()
    rec_p = w4_head_mm(torch.from_numpy(h_prefix_cat)).numpy()
    fix = {}
    for tag, rec, nat in (("corpus", rec_c, al_corpus),
                          ("prefix", rec_p, al_prefix_cat)):
        top1_same = bool((rec[:, :v_live].argmax(1)
                          == nat[:, :v_live].argmax(1)).all())
        maxabs = float(np.abs(rec - nat).max())
        fix[tag] = {"top1_identical_all": top1_same,
                    "max_abs_diff": maxabs}
        if not top1_same or maxabs > 1e-3:
            with open(rcpt, "w") as f:
                f.write(json.dumps({"INSTRUMENT-NOT-RUN": fix,
                                    "start": START}, indent=1))
            raise SystemExit(f"IDENTITY FIXTURE FAILED: {fix}")
    print(f"[cr] identity fixture PASS {fix}", flush=True)
    del rec_c, rec_p   # ~4.7 GB each at full vocab

    # ---- vendor head control (A_T) ----
    from safetensors import safe_open
    idx = json.load(open(os.path.join(
        vdir, "model.safetensors.index.json")))
    with safe_open(os.path.join(vdir, idx["weight_map"]["lm_head.weight"]),
                   framework="pt", device="cpu") as h5:
        Wv = h5.get_tensor("lm_head.weight").float()
    t = time.time()

    def vendor_mm(x):
        outs = []
        for lo in range(0, Wv.shape[0], 16384):
            outs.append(x @ Wv[lo:lo + 16384].T)
        return torch.cat(outs, -1).numpy()

    at_corpus = vendor_mm(torch.from_numpy(h_corpus))
    at_prefix = vendor_mm(torch.from_numpy(h_prefix_cat))
    print(f"[cr] vendor head control {time.time()-t:.0f}s", flush=True)

    # ---- readings ----
    res = {"arm": arm, "surface": ms.TEACHER_DIR,
           "teacher_commit_pin": ms.TEACHER_COMMIT,
           "v_live": v_live, "smoke": smoke,
           "identity_fixture": fix, "ks": list(KS),
           "readings": {}}
    tc32 = tl_corpus.astype(np.float32)
    tp32 = tl_prefix.astype(np.float32)
    for cls, t16, t32, aw, at in (
            ("corpus_X", tl_corpus, tc32, al_corpus, at_corpus),
            ("prefix_K", tl_prefix, tp32, al_prefix_cat, at_prefix)):
        res["readings"][cls] = {
            "A_W": rk_mk(t32, aw, v_live),
            "A_T": rk_mk(t32, at, v_live),
            "R256_by_margin_A_W": rk_by_margin(t16, aw, v_live, ms),
            "R256_by_margin_A_T": rk_by_margin(t16, at, v_live, ms)}
        print(f"[cr] {cls} A_W {res['readings'][cls]['A_W']}",
              flush=True)
        print(f"[cr] {cls} A_T {res['readings'][cls]['A_T']}",
              flush=True)
    np.savez_compressed(npz_path, h_corpus=h_corpus,
                        h_prefix=h_prefix_cat)
    res["arrays_npz"] = npz_path
    res["arrays_npz_sha256"] = hashlib.sha256(
        open(npz_path, "rb").read()).hexdigest()
    res["start"] = START
    res["completion_commit"] = completion_commit()
    with open(rcpt, "w") as f:
        f.write(json.dumps(res, indent=1) + "\n")
    print(f"[cr] -> {rcpt}", flush=True)


if __name__ == "__main__":
    main()
