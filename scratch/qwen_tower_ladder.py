"""qcuda-tower qualification ladder, steps d-g (3080; spec
2026-08-19-qcuda-tower-runtime; GO 2026-08-19).

Old runtime = frozen scratch/qwen_cuda_rung4.py build (w4 fused, s16
layer tensors DENSE FP32 — the abort's path). New runtime = the
qcuda_tower dispatcher (w4 + s16 both fused, io on dedicated
compressed paths, raw dense FP32 unchanged: s16 routing is the ONLY
variable on this equivalence rung).

Steps (STEP env; old/new full towers run in SEPARATE processes —
never co-resident on the 10 GiB card):
  d      2-layer hidden-state stub, old-v-new IN one process
         (dense-decoded s16 v FusedS16Linear on the same layer
         stack; io excluded by construction). CALIBRATES the
         frozen step-e tolerances via the precommitted formula
         T_rel = max(10 * d_rel, 1e-5), T_abs = max(10 * d_abs,
         1e-4)  -> logs/qcudatower/ladder_d.json
  e_old  full rung4 BLe build, forward-1 on the frozen 50-token
         prompt, logits saved  -> ladder_e_old.json + e_old_logits.npy
  e_new  full tower build: verify_routes (exact conservation,
         dedicated io routes) + plan_residency (GPU subset; CPU
         io bytes receipted separately) + forward-1; numerical
         parity v e_old under ladder_d.json tolerances
         -> ladder_e.json
  f_old  rung4 cached 2-token greedy, token2 logits saved
         -> ladder_f_old.json + f_old_tok2.npy
  f_new  tower: (A) cached token2 == uncached token2 (cache
         correctness), (B) cached token2 v old cached token2
         (decode/GEMV-path equivalence)  -> ladder_f.json
  g      tower gen32: tok/s + alloc/reserved/free observed
         -> ladder_g.json

    STEP=d ART_DIR=~/qwen_whole0t/BLe .venv/bin/python \
        scratch/qwen_tower_ladder.py
All receipts append-refused; PROMPT_IDS frozen in-source.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
import torch  # noqa: E402

from llmopt.lab import qartifact, qcuda_tower as qt  # noqa: E402
from llmopt.lab.qcodec import decode_entry  # noqa: E402

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/BLe"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
OUT = "logs/qcudatower"
STEP = os.environ["STEP"]
# frozen probe prompt (64 tokens after tokenization is NOT required —
# the ids below are the frozen surface, tokenizer-independent)
PROMPT_TEXT = ("The derivative of x**3 * sin(x) with respect to x is "
               "computed by the product rule: d/dx[u*v] = u'v + uv'. "
               "Here u = x**3 and v = sin(x), so the answer is")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _fsha(rel):
    return hashlib.sha256(
        open(os.path.join(_ROOT, rel), "rb").read()).hexdigest()


START = {"start_commit": subprocess.check_output(
             ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
         "start_status_porcelain": subprocess.check_output(
             ["git", "status", "--porcelain"]).decode(),
         "interpreter": sys.executable,
         "file_sha256": {p: _fsha(p) for p in (
             "llmopt/lab/qcuda.py", "llmopt/lab/qcuda_tower.py",
             "llmopt/lab/qcodec.py", "scratch/qwen_cuda_rung4.py",
             "scratch/qwen_tower_ladder.py")}}


def rcpt_path(name):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, name)
    if os.path.exists(p):
        raise SystemExit(f"REFUSING: {p} exists")
    return p


def write_rcpt(p, obj):
    obj["start"] = START
    obj["completion_commit"] = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    with open(p, "w") as f:
        f.write(json.dumps(obj) + "\n")
    print(f"[tl] receipt -> {p}", flush=True)


def load_r4():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "qwen_cuda_rung4",
        os.path.join(_ROOT, "scratch", "qwen_cuda_rung4.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["qwen_cuda_rung4"] = mod
    spec.loader.exec_module(mod)
    return mod


def man_and_payload():
    arm = os.path.basename(ART.rstrip("/"))
    chain = os.path.join(_ROOT, "logs", "qwenwhole",
                         f"artifact_digest_{arm}.txt")
    q = qartifact.qualify_artifact(
        ART, VDIR + "/model.safetensors.index.json", chain)
    man = q["manifest"]
    handles = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])
    return q, man, payload


def prompt_ids():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(VDIR)
    ids = tok(PROMPT_TEXT)["input_ids"]
    return ids, hashlib.sha256(
        np.asarray(ids, np.int64).tobytes()).hexdigest()


# ---------------------------------------------------------------- d
def step_d():
    """2-layer hidden-state stub: the SAME two early layers built
    twice — s16 tensors dense-decoded (old path) v FusedS16Linear
    (new path); w4 fused in both; io excluded by construction.
    Calibrates the frozen e tolerances (formula precommitted here)."""
    p = rcpt_path("ladder_d.json")
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    q, man, payload = man_and_payload()
    cfg = AutoConfig.from_pretrained(VDIR)
    cfg.text_config.num_hidden_layers = 2
    torch.manual_seed(0)

    def build_stub(new_path: bool):
        with init_empty_weights(include_buffers=False):
            m = AutoModelForCausalLM.from_config(
                cfg, torch_dtype=torch.float32)
        m.eval()
        routes = {}
        for i, lyr in enumerate(m.model.layers):
            for sub_nm, sub in list(lyr.named_modules()):
                if not isinstance(sub, torch.nn.Linear):
                    continue
                full = (f"model.language_model.layers.{i}."
                        f"{sub_nm}.weight")
                e = man.get(full)
                if e is None:
                    continue
                parent = lyr.get_submodule(sub_nm.rsplit(".", 1)[0]) \
                    if "." in sub_nm else lyr
                leaf = sub_nm.rsplit(".", 1)[-1]
                if e["codec"] == "w4" or (e["codec"] == "s16"
                                          and new_path):
                    setattr(parent, leaf, qt.fused_module(
                        e, payload(e)))
                    routes[full] = e["codec"]
                elif e["codec"] == "s16":     # old path: dense fp32
                    W = torch.from_numpy(np.ascontiguousarray(
                        decode_entry(payload(e), e)))
                    setattr(parent, leaf, torch.nn.Linear(
                        e["shape"][1], e["shape"][0], bias=False))
                    getattr(parent, leaf).weight = torch.nn.Parameter(
                        W.cuda(), requires_grad=False)
                    routes[full] = "dense_fp32(old)"
        # remaining meta params under model.* (norms/convs/raw incl.
        # the final norm): dense once. embed/head stay meta — the
        # stub is driven by inputs_embeds and never reaches lm_head.
        for nm, prm in list(m.named_parameters()):
            if not prm.is_meta or not nm.startswith("model.") \
                    or nm == "model.embed_tokens.weight":
                continue
            e = man.get("model.language_model." + nm[len("model."):])
            if e is None:
                raise SystemExit(f"REFUSING: no manifest entry {nm}")
            mod2 = m.get_submodule(nm.rsplit(".", 1)[0])
            setattr(mod2, nm.rsplit(".", 1)[1], torch.nn.Parameter(
                torch.from_numpy(np.ascontiguousarray(decode_entry(
                    payload(e), e))).cuda(), requires_grad=False))
        m.model.rotary_emb.to("cuda")
        return m, routes

    H = cfg.text_config.hidden_size
    x = (torch.randn(1, 16, H, generator=torch.Generator()
                     .manual_seed(7)) * 0.05).cuda()
    outs = {}
    n_s16_by = {}
    for tag, newp in (("old", False), ("new", True)):
        m, routes = build_stub(newp)
        with torch.no_grad():
            h = m.model(inputs_embeds=x,
                        use_cache=False).last_hidden_state
        assert torch.isfinite(h).all(), f"non-finite stub out ({tag})"
        outs[tag] = h.float().cpu().numpy()
        n_s16_by[tag] = sum(1 for v in routes.values()
                            if v in ("s16", "dense_fp32(old)"))
        print(f"[tl] d:{tag} routes {len(routes)} "
              f"(s16-class {n_s16_by[tag]})", flush=True)
        del m
        torch.cuda.empty_cache()
    assert n_s16_by["old"] == n_s16_by["new"] and n_s16_by["old"] > 0, \
        n_s16_by  # the stub must actually exercise s16 tensors
    diff = np.abs(outs["new"] - outs["old"])
    d_abs = float(diff.max())
    d_rel = float(np.linalg.norm(outs["new"] - outs["old"])
                  / np.linalg.norm(outs["old"]))
    # PRECOMMITTED tolerance formula (spec r2 item 3): frozen here,
    # BEFORE step e ever runs
    tol = {"T_abs": max(10 * d_abs, 1e-4),
           "T_rel_l2": max(10 * d_rel, 1e-5)}
    print(f"[tl] d: abs {d_abs:.3e} rel_l2 {d_rel:.3e} -> frozen "
          f"{tol}", flush=True)
    write_rcpt(p, {"step": "d", "hidden_abs_max": d_abs,
                   "hidden_rel_l2": d_rel, "frozen_tolerances": tol,
                   "n_stub_layers": 2, "qualification": q["report"]})


# ------------------------------------------------------------ e_old
def step_e_old():
    p = rcpt_path("ladder_e_old.json")
    r4 = load_r4()
    model, trav, n_fused = r4.build()
    ids, ids_sha = prompt_ids()
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]).cuda(),
                    use_cache=False)
    lg = out.logits[0].float().cpu().numpy()
    assert np.isfinite(lg).all()
    np.save(os.path.join(OUT, "e_old_logits.npy"), lg)
    write_rcpt(p, {"step": "e_old", "runtime": "rung4-dense-s16",
                   "n_fused_w4": n_fused, "prompt_ids_sha": ids_sha,
                   "n_ids": len(ids), "wall_s": round(time.time()-t0, 1),
                   "traversal": trav["attn_exec"],
                   "logits_sha": hashlib.sha256(lg.tobytes()).hexdigest(),
                   "top1_last": int(lg[-1].argmax())})


# ------------------------------------------------------------ e_new
def build_tower():
    """Full BLe build through the qcuda_tower dispatcher."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from llmopt.lab.qcodec_fast import S16Rows, W4Rows
    q, man, payload = man_and_payload()
    cfg = AutoConfig.from_pretrained(VDIR)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.float32)
    model.eval()

    def shard_key(nm):
        if nm.startswith("model."):
            return "model.language_model." + nm[len("model."):]
        return nm

    # residency plan BEFORE any payload lands (free_bytes measured
    # against an empty card; planning after surgery would double-count
    # the already-resident payloads and refuse spuriously). GPU subset
    # = every compressed 2D tensor except the CPU-row embed; io embed
    # bytes receipted separately.
    gpu_keys = [k for k in qt.expected_compressed(man)
                if k != "model.language_model.embed_tokens.weight"]
    free, total = torch.cuda.mem_get_info()
    plan = qt.plan_residency([man[k] for k in gpu_keys], free)
    ee = man["model.language_model.embed_tokens.weight"]
    plan["cpu_bytes"] = {"embed_compressed": ee["len"]}

    n_by = {"w4": 0, "s16": 0}
    for i, lyr in enumerate(model.model.layers):
        for sub_nm, sub in list(lyr.named_modules()):
            if not isinstance(sub, torch.nn.Linear):
                continue
            full = (f"model.language_model.layers.{i}."
                    f"{sub_nm}.weight")
            e = man.get(full)
            if e is None or e["codec"] not in ("w4", "s16"):
                continue
            assert sub.bias is None, f"unexpected bias on {full}"
            parent = lyr.get_submodule(sub_nm.rsplit(".", 1)[0]) \
                if "." in sub_nm else lyr
            setattr(parent, sub_nm.rsplit(".", 1)[-1],
                    qt.fused_module(e, payload(e)))
            n_by[e["codec"]] += 1
    print(f"[tl] tower fused {n_by}", flush=True)

    for nm, prm in list(model.named_parameters()):
        if not prm.is_meta:
            continue
        if nm in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        e = man[shard_key(nm)]
        mod2 = model.get_submodule(nm.rsplit(".", 1)[0])
        setattr(mod2, nm.rsplit(".", 1)[1], torch.nn.Parameter(
            torch.from_numpy(np.ascontiguousarray(decode_entry(
                payload(e), e))).cuda(), requires_grad=False))

    he = man["lm_head.weight"]
    assert ee["codec"] == "s16" and he["codec"] == "s16", \
        (ee["codec"], he["codec"])
    emb = S16Rows(payload(ee), ee["shape"])

    def emb_fwd(input_ids):
        flat = input_ids.reshape(-1)
        out = torch.empty(flat.shape[0], emb.C)
        for j, t in enumerate(flat.tolist()):
            out[j] = torch.from_numpy(emb.rows(t, t + 1)[0])
        return out.reshape(*input_ids.shape, emb.C).cuda()

    model.model.embed_tokens.forward = emb_fwd

    class LastPosHead(torch.nn.Module):
        """Real module replacement (not a forward patch) so the head
        participates in verify_routes as a fused module."""

        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:]
            return self.inner(x)

    model.lm_head = LastPosHead(
        qt.FusedS16Linear(qt.S16Gpu(payload(he), he["shape"])))

    meta_left = [nm for nm, prm in model.named_parameters()
                 if prm.is_meta
                 and nm != "model.embed_tokens.weight"]
    if meta_left:
        raise SystemExit(f"REFUSING: meta params left {meta_left[:4]}")
    model.model.rotary_emb.to("cuda")

    def name_fn(path):
        if path == "lm_head.inner":
            return "lm_head.weight"
        if path.startswith("model.layers."):
            return "model.language_model." + path[len("model."):] \
                + ".weight"
        return path + ".weight"

    routes = qt.verify_routes(
        model, man, name_fn=name_fn,
        dedicated_routes={
            "model.language_model.embed_tokens.weight":
                "cpu_compressed_rows"})
    n_routes = {"FusedW4Linear": 0, "FusedS16Linear": 0,
                "cpu_compressed_rows": 0}
    for v in routes.values():
        n_routes[v] += 1
    # exact conservation, counts derived from the manifest
    assert len(routes) == len(qt.expected_compressed(man))
    print(f"[tl] verify_routes OK: {n_routes}", flush=True)
    W4Rows  # keep import explicit for io-w4 arms (unused for BLe)
    return model, plan, routes, n_routes


def mem_obs():
    free, total = torch.cuda.mem_get_info()
    return {"alloc": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "free": int(free), "total": int(total)}


def step_e_new():
    p = rcpt_path("ladder_e.json")
    tol = json.load(open(os.path.join(
        OUT, "ladder_d.json")))["frozen_tolerances"]
    old = np.load(os.path.join(OUT, "e_old_logits.npy"))
    model, plan, routes, n_routes = build_tower()
    ids, ids_sha = prompt_ids()
    old_r = json.load(open(os.path.join(OUT, "ladder_e_old.json")))
    assert old_r["prompt_ids_sha"] == ids_sha, "prompt drift"
    t0 = time.time()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]).cuda(),
                    use_cache=False)
    lg = out.logits[0].float().cpu().numpy()
    assert np.isfinite(lg).all()
    assert lg.shape == old.shape, (lg.shape, old.shape)
    abs_max = float(np.abs(lg - old).max())
    rel_l2 = float(np.linalg.norm(lg - old) / np.linalg.norm(old))
    top1_same = bool((lg.argmax(-1) == old.argmax(-1)).all())
    ok = (abs_max <= tol["T_abs"] and rel_l2 <= tol["T_rel_l2"]
          and top1_same)
    print(f"[tl] e: abs {abs_max:.3e} (T {tol['T_abs']:.1e}) rel_l2 "
          f"{rel_l2:.3e} (T {tol['T_rel_l2']:.1e}) top1_same "
          f"{top1_same} -> {'PASS' if ok else 'FAIL'}", flush=True)
    write_rcpt(p, {"step": "e", "abs_max": abs_max, "rel_l2": rel_l2,
                   "top1_identical_all_positions": top1_same,
                   "tolerances": tol, "pass": ok,
                   "prompt_ids_sha": ids_sha,
                   "residency_plan": plan, "route_counts": n_routes,
                   "mem_observed": mem_obs(),
                   "wall_s": round(time.time() - t0, 1)})
    if not ok:
        raise SystemExit("ladder e FAIL")


# --------------------------------------------------------------- f
def greedy2(model, ids):
    """Two greedy tokens WITH cache; returns (tok1, tok2, logits2)."""
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]).cuda(),
                    use_cache=True)
        t1 = int(out.logits[0, -1].argmax())
        out2 = model(input_ids=torch.tensor([[t1]]).cuda(),
                     past_key_values=out.past_key_values,
                     use_cache=True)
        lg2 = out2.logits[0, -1].float().cpu().numpy()
    return t1, int(lg2.argmax()), lg2


def step_f_old():
    p = rcpt_path("ladder_f_old.json")
    r4 = load_r4()
    model, trav, _ = r4.build()
    ids, ids_sha = prompt_ids()
    t1, t2, lg2 = greedy2(model, ids)
    np.save(os.path.join(OUT, "f_old_tok2.npy"), lg2)
    write_rcpt(p, {"step": "f_old", "tok1": t1, "tok2": t2,
                   "prompt_ids_sha": ids_sha,
                   "logits2_sha": hashlib.sha256(
                       lg2.tobytes()).hexdigest()})


def step_f_new():
    p = rcpt_path("ladder_f.json")
    tol = json.load(open(os.path.join(
        OUT, "ladder_d.json")))["frozen_tolerances"]
    old = json.load(open(os.path.join(OUT, "ladder_f_old.json")))
    old_lg2 = np.load(os.path.join(OUT, "f_old_tok2.npy"))
    model, plan, routes, n_routes = build_tower()
    ids, ids_sha = prompt_ids()
    assert old["prompt_ids_sha"] == ids_sha
    t1, t2, lg2 = greedy2(model, ids)
    # (A) cache correctness inside the new runtime: uncached token2
    with torch.no_grad():
        out_u = model(input_ids=torch.tensor([ids + [t1]]).cuda(),
                      use_cache=False)
    lg2_u = out_u.logits[0, -1].float().cpu().numpy()
    a_abs = float(np.abs(lg2 - lg2_u).max())
    a_rel = float(np.linalg.norm(lg2 - lg2_u) / np.linalg.norm(lg2_u))
    # (B) tower equivalence through the decode/GEMV path v old
    b_abs = float(np.abs(lg2 - old_lg2).max())
    b_rel = float(np.linalg.norm(lg2 - old_lg2)
                  / np.linalg.norm(old_lg2))
    ok = (t1 == old["tok1"] and t2 == old["tok2"]
          and int(lg2_u.argmax()) == t2
          and a_abs <= tol["T_abs"] and a_rel <= tol["T_rel_l2"]
          and b_abs <= tol["T_abs"] and b_rel <= tol["T_rel_l2"])
    print(f"[tl] f: A(cached=uncached) abs {a_abs:.3e} rel {a_rel:.3e}"
          f"; B(new=old) abs {b_abs:.3e} rel {b_rel:.3e}; toks "
          f"{t1},{t2} v {old['tok1']},{old['tok2']} -> "
          f"{'PASS' if ok else 'FAIL'}", flush=True)
    write_rcpt(p, {"step": "f", "tok1": t1, "tok2": t2,
                   "old_tok1": old["tok1"], "old_tok2": old["tok2"],
                   "A_cache_abs": a_abs, "A_cache_rel_l2": a_rel,
                   "B_oldnew_abs": b_abs, "B_oldnew_rel_l2": b_rel,
                   "tolerances": tol, "pass": ok,
                   "mem_observed": mem_obs()})
    if not ok:
        raise SystemExit("ladder f FAIL")


# --------------------------------------------------------------- g
def step_g():
    p = rcpt_path("ladder_g.json")
    model, plan, routes, n_routes = build_tower()
    ids, ids_sha = prompt_ids()
    m0 = mem_obs()
    with torch.no_grad():
        out = model(input_ids=torch.tensor([ids]).cuda(),
                    use_cache=True)
        past = out.past_key_values
        t = int(out.logits[0, -1].argmax())
        toks = [t]
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(32):
            out = model(input_ids=torch.tensor([[t]]).cuda(),
                        past_key_values=past, use_cache=True)
            past = out.past_key_values
            t = int(out.logits[0, -1].argmax())
            toks.append(t)
        torch.cuda.synchronize()
    wall = time.time() - t0
    m1 = mem_obs()
    print(f"[tl] g: 32 tokens in {wall:.1f}s = {32/wall:.2f} tok/s; "
          f"alloc {m1['alloc']/2**30:.2f} GiB free "
          f"{m1['free']/2**30:.2f} GiB", flush=True)
    write_rcpt(p, {"step": "g", "n_tokens": 32,
                   "wall_s": round(wall, 2),
                   "tok_s": round(32 / wall, 3),
                   "tokens": toks, "prompt_ids_sha": ids_sha,
                   "residency_plan": plan, "route_counts": n_routes,
                   "mem_before": m0, "mem_after": m1})


if __name__ == "__main__":
    {"d": step_d, "e_old": step_e_old, "e_new": step_e_new,
     "f_old": step_f_old, "f_new": step_f_new, "g": step_g}[STEP]()
