"""QWEN-MODEL1-TREE scorer: X/K for one artifact arm v the locked teacher.

Registered quantities (PRE-REG QWEN-MODEL1-TREE + -METRIC/-PRIORS/
-PINS/-KFENCE; projection docs/preregs/qwen-model1-tree.json):

  X_arm = mean per-position excess CE over the corpus,
          CE_arm - CE_teacher, logits[:-1] v ids[1:] (P-1 terms),
          live vocab, fp32-minimum log-softmax (fp64 here).
  K_arm = mean forward KL(teacher || arm) over prefix positions,
          same alignment convention, live vocab.
  f_X, f_K = fp16_record_sensitivity_floor(+_K): max |shift| of the
          pipeline output when every teacher fp16 logit moves +-1ulp.
  Margin-stratified flip rates: P(arm top1 != teacher top1) per
          frozen margin bin (TREE-PINS edges), per stream, raw
          counts always printed; n < 30 strata carry NO directional
          claim (small-n fence).

Tree measurements are CPU-ONLY (-METRIC 6): this driver refuses any
accelerator residency. Manifests are consumed only through
llmopt.lab.qartifact; decode only through llmopt.lab.qcodec(_fast).

    ART_DIR=~/qwen_whole0t/A ARM=A .venv/bin/python \
        scratch/qwen_model1_score.py
    SMOKE=1 ART_DIR=... ARM=A ...   # truncated, *_smoke paths only

Receipt: logs/qwenmodel1/score_<ARM>.json (append-refused).
"""
import hashlib
import json
import os
import sys
import time

import numpy as np

TEACHER_DIR = "logs/qwenteacher_v2"
TEACHER_COMMIT = "0ca4151"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
EV = "evals/qwen_model1"
MARGIN_EDGES = [0.0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, float("inf")]
SMALL_N = 30


# ---------------------------------------------------------------- math
# Pure functions, no I/O: tests/test_qwen_model1_scorer.py pins them
# with hand-computable fixtures before any 27B forward runs.

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


# ------------------------------------------------------------- refuse
def refuse_list_checks(man: dict, tok, smoke: bool):
    """The -METRIC (8) scorer refuse-list, fail-closed."""
    if "smoke" in TEACHER_DIR or man.get("smoke"):
        raise SystemExit("REFUSING: smoke teacher records")
    if man.get("code_commit") != TEACHER_COMMIT:
        raise SystemExit(f"REFUSING: teacher code_commit "
                         f"{man.get('code_commit')} != {TEACHER_COMMIT}")
    if man.get("revision") != REVISION:
        raise SystemExit("REFUSING: teacher revision mismatch")
    trav = man.get("traversal") or {}
    if (trav.get("linear_attn_layers_executed"),
            trav.get("full_attn_layers_executed")) != (48, 16):
        raise SystemExit(f"REFUSING: teacher traversal census {trav}")
    if trav.get("rope_calls", 0) < 16:
        raise SystemExit("REFUSING: teacher rope_calls < 16")
    # single-teacher rule: exactly one non-smoke manifest outside
    # quarantine anywhere under logs/
    import glob
    mans = [p for p in glob.glob("logs/**/teacher_manifest.json",
                                 recursive=True)
            if "quarantine" not in p and "_smoke" not in p]
    if mans != [os.path.join(TEACHER_DIR, "teacher_manifest.json")]:
        raise SystemExit(f"REFUSING: teacher manifests found {mans}")
    # eval payload identity
    for f, k in (("corpus.txt", "corpus_sha256"),
                 ("prefixes.jsonl", "prefixes_sha256"),
                 ("prompts.jsonl", "prompts_sha256")):
        if fsha(os.path.join(EV, f)) != man["inputs"][k]:
            raise SystemExit(f"REFUSING: {f} sha != teacher manifest")
    # re-tokenization drift: re-encoding must reproduce the locked lists
    corpus_ids = tok(open(os.path.join(EV, "corpus.txt")).read())["input_ids"]
    if corpus_ids != man["records"]["corpus"]["tokens"]:
        raise SystemExit("REFUSING: corpus re-tokenization drift")
    pref_rows = [json.loads(l) for l in open(os.path.join(EV,
                 "prefixes.jsonl"))]
    ptoks = [tok(r["text"])["input_ids"] for r in pref_rows]
    if ptoks != man["records"]["prefixes"]["tokens"]:
        raise SystemExit("REFUSING: prefix re-tokenization drift")
    return corpus_ids, ptoks


def load_record(name: str, man_rec: dict) -> np.ndarray:
    p = os.path.join(TEACHER_DIR, name)
    a = np.load(p)
    if sha_arr(a) != man_rec["sha256"]:
        raise SystemExit(f"REFUSING: {name} sha != manifest")
    if list(a.shape) != man_rec["shape"]:
        raise SystemExit(f"REFUSING: {name} shape != manifest")
    return a


# ---------------------------------------------------------------- main
def main():
    smoke = os.environ.get("SMOKE", "0") == "1"
    suf = "_smoke" if smoke else ""
    out_dir = f"logs/qwenmodel1{suf}"
    arm = os.environ["ARM"]
    # A/B/C = WHOLE-0T compile arms; F/L/Q = ATTN-ATTRIB-1; D/E =
    # IO-ATTRIB-1; BL*/FL* = LBAND-1 (chains emitted by
    # scratch/qwen_recompose.py)
    assert arm in ("A", "B", "C", "F", "L", "Q", "D", "E",
                   "BLe", "BLm", "BLl", "FLe", "FLm", "FLl"), arm
    art = os.path.expanduser(os.environ["ART_DIR"])
    os.makedirs(out_dir, exist_ok=True)
    # RESCORE=1 writes score_<ARM>_rescore.json alongside the frozen
    # original (scorer-repeatability + provenance-closure runs; the
    # frozen receipt is never opened for write)
    resuf = "_rescore" if os.environ.get("RESCORE", "0") == "1" else ""
    rcpt_path = os.path.join(out_dir, f"score_{arm}{resuf}.json")
    if os.path.exists(rcpt_path):
        raise SystemExit(f"REFUSING: {rcpt_path} exists")

    import torch
    torch.set_grad_enabled(False)
    torch.set_num_threads(os.cpu_count())
    # device rule (-METRIC 6): tree measurements exist on cpu only
    if os.environ.get("DEVICE", "cpu") != "cpu":
        raise SystemExit("REFUSING: tree scorer runs on cpu only")

    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from llmopt.lab import qartifact
    from llmopt.lab.qcodec import decode_entry
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows
    from transformers import AutoTokenizer

    t0 = time.time()
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tok = AutoTokenizer.from_pretrained(vdir)
    man_t = json.load(open(os.path.join(TEACHER_DIR,
                                        "teacher_manifest.json")))
    corpus_ids, ptoks = refuse_list_checks(man_t, tok, smoke)
    tl_corpus = load_record("corpus_logits.npy", man_t["records"]["corpus"])
    tl_prefix = load_record("prefix_logits.npy",
                            man_t["records"]["prefixes"])
    bounds = man_t["records"]["prefixes"]["bounds"]
    v_rec = tl_corpus.shape[1]
    v_live = len(tok)
    if v_live > v_rec:
        raise SystemExit("REFUSING: live vocab exceeds record width")

    # artifact qualification: chain REQUIRED for a tree measurement
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    if not os.path.exists(chain):
        raise SystemExit(f"REFUSING: no digest chain {chain}")
    q = qartifact.qualify_artifact(
        art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    # the exact chain the qualification bound to, in the receipt
    # (receipt-audit adoption: rung0 recorded only checked/unchained)
    q["report"]["chain_sha256"] = fsha(chain)
    MAN = q["manifest"]
    print(f"[m1] qualified {arm}: {q['report']}", flush=True)

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

    # decode-oracle fixture (-METRIC 8 round-trip class, projected to
    # the scoring machine: the frozen per-tensor encoders train on the
    # compile device; cross-device codebook bit-identity is not a
    # property the registration guarantees. The zero-model-cost oracle
    # here catches the same defect class — nibble order, codebook
    # offset, exponent bias all produce garbage against vendor bytes):
    # one coded tensor per codec present, canonical decode v vendor
    # bf16, relative L2 must stay under the family tripwire class.
    from safetensors import safe_open
    idx = json.load(open(os.path.join(vdir,
                                      "model.safetensors.index.json")))
    fixture = {}
    want = ["model.language_model.layers.0.linear_attn.in_proj_z.weight",
            "lm_head.weight"]
    for nm in want:
        e = MAN.get(nm)
        if e is None or e["codec"] not in ("w4", "s16"):
            continue
        with safe_open(os.path.join(vdir, idx["weight_map"][nm]),
                       framework="pt", device="cpu") as h:
            Wv = h.get_tensor(nm).float().numpy()
        Wd = decode_entry(payload(e), e)
        rel = float(np.linalg.norm(Wd - Wv) / np.linalg.norm(Wv))
        rows_dec = (W4Rows if e["codec"] == "w4" else S16Rows)(
            payload(e), e["shape"])
        if not np.array_equal(rows_dec.rows(0, min(64, e["shape"][0])),
                              Wd[:min(64, e["shape"][0])]):
            raise SystemExit(f"REFUSING: rows-decoder != canonical {nm}")
        if rel > 0.5:
            raise SystemExit(f"REFUSING: decode fixture {nm} rel {rel}")
        fixture[nm] = {"codec": e["codec"], "rel_l2_vs_vendor": rel}
        del Wv, Wd
    if not fixture:
        raise SystemExit("REFUSING: no codec fixture tensor found")
    print(f"[m1] decode fixture: {fixture}", flush=True)

    # ---- streamed model (runtime0r pattern, codec-general io) ----
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(vdir)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg,
                                                 torch_dtype=torch.float32)
    model.eval()

    def shard_key(nm):
        if nm.startswith("model."):
            return "model.language_model." + nm[len("model."):]
        return nm

    layer_pref = "model.layers."
    for nm, _ in list(model.named_parameters()):
        if nm.startswith(layer_pref):
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
            raise SystemExit(f"REFUSING: io codec {e['codec']} for {key}")
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

    def head_fwd(x):
        outs = []
        for lo in range(0, head.R, 16384):
            hi = min(lo + 16384, head.R)
            outs.append(x @ torch.from_numpy(head.rows(lo, hi)).T)
        return torch.cat(outs, -1)

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
    trav = {"layer_calls": [0] * len(layers),
            "families": ["full_attn" if hasattr(l, "self_attn")
                         else "linear_attn" for l in layers]}

    def make_pre(i):
        def pre(module, args, kwargs):
            trav["layer_calls"][i] += 1
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

    # ---- corpus: X ----
    t = time.time()
    al_corpus = fwd(corpus_ids)
    print(f"[m1] corpus forward {time.time()-t:.0f}s", flush=True)
    if al_corpus.shape[1] != v_rec:
        raise SystemExit("REFUSING: arm logit width != teacher record "
                         f"({al_corpus.shape[1]} v {v_rec})")
    ce_t = mean_ce(tl_corpus.astype(np.float32), corpus_ids, v_live)
    ce_a = mean_ce(al_corpus, corpus_ids, v_live)
    if ce_t >= 10.0:
        print(f"[m1] INSTRUMENT-ALARM: CE_teacher {ce_t:.3f} nats",
              flush=True)
    X = ce_a - ce_t
    f_X = sensitivity_floor(
        lambda a: mean_ce(a, corpus_ids, v_live), tl_corpus)
    print(f"[m1] CE_teacher {ce_t:.4f} CE_{arm} {ce_a:.4f} "
          f"X_{arm} {X:.5f} f_X {f_X:.2e}", flush=True)

    # ---- prefixes: K ----
    al_prefix = []
    off = 0
    K_terms_t16, K_terms_a = [], []
    for pi, ids in enumerate(ptoks):
        t = time.time()
        lg = fwd(ids)
        al_prefix.append(lg)
        n = bounds[pi]
        K_terms_t16.append(tl_prefix[off:off + n - 1])
        K_terms_a.append(lg[:n - 1])
        off += n
        print(f"[m1] prefix {pi} len {n} forward {time.time()-t:.0f}s",
              flush=True)
    tt16 = np.concatenate(K_terms_t16)
    aa = np.concatenate(K_terms_a)
    K = mean_forward_kl(tt16.astype(np.float32), aa, v_live)
    f_K = sensitivity_floor(
        lambda a: mean_forward_kl(a, aa, v_live), tt16)
    print(f"[m1] K_{arm} {K:.5f} f_K {f_K:.2e}", flush=True)

    # ---- margin-stratified flips (all recorded positions/stream) ----
    flips = {}
    for stream, tl, al in (
            ("corpus", tl_corpus, al_corpus),
            ("prefixes", tl_prefix if not smoke else tl_prefix,
             np.concatenate(al_prefix))):
        t_top1, marg = teacher_margins_top1(tl, v_live)
        a_top1 = al[:, :v_live].argmax(axis=-1)
        tab = flip_table(t_top1, marg, a_top1)
        flips[stream] = tab
        for b, (n, f) in enumerate(tab):
            fence = " [small-n: raw counts only]" if n < SMALL_N else ""
            lo = MARGIN_EDGES[b]
            hi = MARGIN_EDGES[b + 1]
            print(f"[m1] {stream} bin [{lo},{hi}) n={n} flips={f}{fence}",
                  flush=True)

    top1_agree = {
        s: 1.0 - sum(f for _, f in tab) / max(sum(n for n, _ in tab), 1)
        for s, tab in flips.items()}

    import subprocess
    rcpt = {
        "arm": arm, "artifact": art, "smoke": smoke,
        "rescore": bool(resuf),
        # DERIVED from the resident parameters, never a literal
        # (receipt-audit adoption 2026-08-17; RULE-ABLATE-1 class)
        "device_actual": str(next(p.device for _, p
                                  in model.named_parameters()
                                  if not p.is_meta)),
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "teacher": {"dir": TEACHER_DIR,
                    "code_commit": man_t["code_commit"],
                    "revision": man_t["revision"],
                    "corpus_sha": man_t["records"]["corpus"]["sha256"],
                    "prefix_sha": man_t["records"]["prefixes"]["sha256"]},
        "qualification": q["report"],
        "decode_fixture": fixture,
        "v_record": int(v_rec), "v_live": int(v_live),
        "vocab_masked": bool(v_live != v_rec),
        "n_corpus_terms": len(corpus_ids) - 1,
        "n_prefix_terms": int(tt16.shape[0]),
        "ce_teacher_nats": ce_t, "ce_arm_nats": ce_a,
        "X": X, "K": K, "f_X": f_X, "f_K": f_K,
        "margin_edges": MARGIN_EDGES[:-1] + ["inf"],
        "flip_table": flips, "small_n_fence": SMALL_N,
        "top1_agreement": top1_agree,
        "traversal": {"layers": len(trav["layer_calls"]),
                      "min_calls": min(trav["layer_calls"]),
                      "linear_attn": sum(
                          1 for i, f in enumerate(trav["families"])
                          if f == "linear_attn"
                          and trav["layer_calls"][i] > 0),
                      "full_attn": sum(
                          1 for i, f in enumerate(trav["families"])
                          if f == "full_attn"
                          and trav["layer_calls"][i] > 0)},
        "wall_s": round(time.time() - t0, 1)}
    if not smoke and (rcpt["traversal"]["linear_attn"],
                      rcpt["traversal"]["full_attn"]) != (48, 16):
        raise SystemExit(f"REFUSING receipt: traversal {rcpt['traversal']}")
    with open(rcpt_path, "w") as f:
        f.write(json.dumps(rcpt) + "\n")
    print(f"[m1] receipt -> {rcpt_path} wall {rcpt['wall_s']}s",
          flush=True)


if __name__ == "__main__":
    main()
