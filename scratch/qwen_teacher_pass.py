"""QWEN-MODEL-1 teacher-baseline pass (frozen procedure, SPEC.md).

One deliberately slow layer-streaming CPU pass of the VENDOR
Qwen3.8-27B (revision 1d4bf0f2) over exactly the frozen eval
payload: corpus.txt positions, every prefixes.jsonl position, and
the prompts.jsonl greedy rollouts. Records are hashed and LOCKED;
every later backend scores against these, never a re-run.

Streaming mechanism: the model is built on the meta device; each
decoder layer gets a forward PRE-hook that materializes its
parameters from the mmap'd vendor shards (bf16 -> fp32) and a
POST-hook that returns them to meta. Embeddings, lm_head, final
norm, and all sub-1M tensors stay resident fp32. Vendor forward
code runs untouched; only weight residency is managed.

Outputs (logs/qwenteacher/):
  corpus_logits.npy    [P, V] fp16 full logits at every corpus position
  prefix_logits.npy    [sum_i len_i, V] fp16, prefixes concatenated
  rollout_tokens.jsonl per-prompt greedy tokens (256 max) + text
  rollout_logits.npy   [sum_steps, V] fp16, per-step logits
  teacher_manifest.json  shapes, boundaries, sha256 of each record,
                         model revision, code_commit, wall

Runs in .venv_teacher (transformers>=5; the lab venv pins <5 for
mlx-lm). SMOKE=1 truncates to 2 layers + 8 tokens and writes to
*_smoke paths — never evidence.

    SMOKE=1 .venv_teacher/bin/python scratch/qwen_teacher_pass.py
    .venv_teacher/bin/python scratch/qwen_teacher_pass.py
"""
import hashlib
import json
import os
import subprocess
import time

import numpy as np
import torch

SMOKE = os.environ.get("SMOKE", "0") == "1"
SUF = "_smoke" if SMOKE else ""
VDIR = os.path.expanduser("~/qwen_vendor")
OUT = f"logs/qwenteacher_v2{SUF}"       # v1 quarantined (zeroed RoPE)
EV = "evals/qwen_model1"
REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"
MAX_NEW = 8 if SMOKE else 256

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())


def build_streamed_model():
    from accelerate import init_empty_weights
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(VDIR)
    if SMOKE:
        cfg.text_config.num_hidden_layers = 2
    # init_empty_weights with include_buffers=False (the default):
    # parameters land on meta, BUFFERS ARE COMPUTED FOR REAL by each
    # module's init. v1 of this driver built under
    # torch.device("meta") and zero-filled every meta buffer, which
    # silenced RoPE's inv_freq — non-vendor logits, caught before
    # any lock was used (v1 outputs quarantined).
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg,
                                                 torch_dtype=torch.float32)
    model.eval()

    idx = json.load(open(os.path.join(VDIR, "model.safetensors.index.json")))
    wmap = idx["weight_map"]
    handles = {}

    def get(name):
        sh = wmap[name]
        if sh not in handles:
            handles[sh] = safe_open(os.path.join(VDIR, sh),
                                    framework="pt", device="cpu")
        return handles[sh].get_tensor(name).float()

    # AutoModelForCausalLM builds the TEXT tower only, so module
    # names lack the multimodal wrapper's "language_model." segment;
    # shard keys carry it (lm_head is top-level both sides)
    def shard_key(name):
        if name.startswith("model."):
            return "model.language_model." + name[len("model."):]
        return name

    layer_pref = "model.layers."
    resident = []
    for name, _ in model.named_parameters():
        if not name.startswith(layer_pref):
            resident.append(name)
    for name in resident:
        if shard_key(name) not in wmap:
            raise SystemExit(f"resident param missing from shards: {name}")
        mod = model.get_submodule(name.rsplit(".", 1)[0])
        setattr(mod, name.rsplit(".", 1)[1],
                torch.nn.Parameter(get(shard_key(name)),
                                   requires_grad=False))
    # FAIL-CLOSED: no meta buffer may survive to a forward pass —
    # a silently-defaulted buffer produces non-vendor logits
    meta_bufs = [n for n, b in model.named_buffers() if b.is_meta]
    if meta_bufs:
        raise SystemExit(f"REFUSING: meta buffers after build: "
                         f"{meta_bufs[:5]}")
    iv = model.model.rotary_emb.inv_freq
    if float(iv.abs().sum()) == 0.0:
        raise SystemExit("REFUSING: rotary inv_freq is all-zero")
    layers = model.model.layers

    def make_pre(i):
        def pre(module, args, kwargs):
            sd = {}
            for n, p in module.named_parameters():
                full = f"model.language_model.layers.{i}.{n}"
                sd[n] = get(full)
            module._load_from_state_dict_shim = None
            for n, t in sd.items():
                mod = module.get_submodule(n.rsplit(".", 1)[0]) \
                    if "." in n else module
                leaf = n.rsplit(".", 1)[1] if "." in n else n
                mod._parameters[leaf] = torch.nn.Parameter(
                    t, requires_grad=False)
            return None
        return pre

    def post(module, args, kwargs, output):
        for n, p in list(module.named_parameters()):
            mod = module.get_submodule(n.rsplit(".", 1)[0]) \
                if "." in n else module
            leaf = n.rsplit(".", 1)[1] if "." in n else n
            mod._parameters[leaf] = torch.nn.Parameter(
                p.to("meta"), requires_grad=False)
        return output

    # TRAVERSAL PROOF (incident: v1's zeroed-RoPE bug escaped smoke
    # because the sliced model never reached a consuming layer —
    # the receipt must PROVE every registered layer family ran):
    trav = {"layer_calls": [0] * len(layers), "rope_calls": 0,
            "families": ["full_attn" if hasattr(l, "self_attn")
                         else "linear_attn" for l in layers]}

    def count_layer(i):
        def h(module, args, kwargs):
            trav["layer_calls"][i] += 1
            return None
        return h

    def count_rope(module, args, kwargs):
        trav["rope_calls"] += 1
        return None

    model.model.rotary_emb.register_forward_pre_hook(
        count_rope, with_kwargs=True)
    for i, lyr in enumerate(layers):
        lyr.register_forward_pre_hook(count_layer(i), with_kwargs=True)
        lyr.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        lyr.register_forward_hook(post, with_kwargs=True)
    return model, trav


def sha(a: np.ndarray) -> str:
    return hashlib.sha256(a.tobytes()).hexdigest()


def main():
    from transformers import AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "teacher_manifest.json")
    if os.path.exists(man_path):
        raise SystemExit(f"REFUSING: {man_path} exists (teacher is locked)")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(VDIR)
    model, trav = build_streamed_model()
    commit = subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    man = {"revision": REVISION, "code_commit": commit, "smoke": SMOKE,
           "records": {}}

    def fwd(ids):
        out = model(input_ids=torch.tensor([ids]), use_cache=False)
        return out.logits[0].to(torch.float16).numpy()

    # 1. corpus — teacher-forced full logits
    corpus_ids = tok(open(os.path.join(EV, "corpus.txt")).read())["input_ids"]
    if SMOKE:
        corpus_ids = corpus_ids[:8]
    t = time.time()
    cl = fwd(corpus_ids)
    np.save(os.path.join(OUT, "corpus_logits.npy"), cl)
    man["records"]["corpus"] = {"tokens": corpus_ids, "shape": list(cl.shape),
                                "sha256": sha(cl),
                                "wall_s": round(time.time() - t, 1)}
    print(f"[tp] corpus {cl.shape} {time.time()-t:.0f}s", flush=True)

    # 2. prefixes — full logits at every position
    pref_rows = [json.loads(l) for l in open(os.path.join(EV,
                 "prefixes.jsonl"))]
    plogits, bounds, ptoks = [], [], []
    for r in pref_rows:
        ids = tok(r["text"])["input_ids"]
        if SMOKE:
            ids = ids[:8]
        t = time.time()
        lg = fwd(ids)
        bounds.append(len(lg))
        ptoks.append(ids)
        plogits.append(lg)
        print(f"[tp] prefix len {len(ids)} {time.time()-t:.0f}s", flush=True)
    pl = np.concatenate(plogits)
    np.save(os.path.join(OUT, "prefix_logits.npy"), pl)
    man["records"]["prefixes"] = {"tokens": ptoks, "bounds": bounds,
                                  "shape": list(pl.shape), "sha256": sha(pl)}

    # 3. rollouts — greedy, layer-streaming per step, all prompts
    #    BATCHED so each 52GB weight sweep serves every prompt
    prows = [json.loads(l) for l in open(os.path.join(EV, "prompts.jsonl"))]
    if SMOKE:
        prows = prows[:2]
    texts = [tok.apply_chat_template(
        [{"role": "user", "content": r["prompt"]}],
        tokenize=False, add_generation_prompt=True) for r in prows]
    enc = tok(texts, padding=True, padding_side="left", return_tensors="pt")
    ids, mask = enc["input_ids"], enc["attention_mask"]
    eos = tok.eos_token_id
    done = torch.zeros(len(prows), dtype=torch.bool)
    steps, step_logits = [], []
    for step in range(MAX_NEW):
        t = time.time()
        out = model(input_ids=ids, attention_mask=mask, use_cache=False)
        lg = out.logits[:, -1, :]
        nxt = lg.argmax(-1)
        nxt[done] = eos
        step_logits.append(lg.to(torch.float16).numpy())
        steps.append(nxt.tolist())
        done |= nxt.eq(eos)
        ids = torch.cat([ids, nxt[:, None]], 1)
        mask = torch.cat([mask, torch.ones_like(nxt[:, None])], 1)
        print(f"[tp] step {step+1}/{MAX_NEW} done={int(done.sum())}"
              f"/{len(prows)} {time.time()-t:.0f}s", flush=True)
        if bool(done.all()):
            break
    rl = np.stack(step_logits)          # [steps, B, V]
    np.save(os.path.join(OUT, "rollout_logits.npy"), rl)
    rollout_rows = []
    for b, r in enumerate(prows):
        toks = [s[b] for s in steps]
        if eos in toks:
            toks = toks[:toks.index(eos) + 1]
        rollout_rows.append({"prompt": r["prompt"],
                             "category": r.get("category"),
                             "tokens": toks,
                             "text": tok.decode(toks,
                                                skip_special_tokens=True)})
    with open(os.path.join(OUT, "rollout_tokens.jsonl"), "w") as f:
        for row in rollout_rows:
            f.write(json.dumps(row) + "\n")
    man["records"]["rollouts"] = {"n_prompts": len(prows),
                                  "steps": len(steps),
                                  "shape": list(rl.shape),
                                  "sha256": sha(rl)}
    # TRAVERSAL GATE: refuse the lock unless every layer ran and
    # every registered family (incl. the RoPE path) was exercised
    calls = trav["layer_calls"]
    fams = trav["families"]
    n_lin = sum(1 for f in fams if f == "linear_attn")
    n_full = sum(1 for f in fams if f == "full_attn")
    idle = [i for i, c in enumerate(calls) if c == 0]
    if idle:
        raise SystemExit(f"REFUSING LOCK: layers never executed: "
                         f"{idle[:8]}")
    if not SMOKE and (n_lin, n_full) != (48, 16):
        raise SystemExit(f"REFUSING LOCK: family census "
                         f"lin={n_lin} full={n_full}, expected 48/16")
    if trav["rope_calls"] == 0:
        raise SystemExit("REFUSING LOCK: rotary path never called")
    man["traversal"] = {
        "layers_executed": len(calls),
        "linear_attn_layers_executed": n_lin,
        "full_attn_layers_executed": n_full,
        "min_layer_calls": min(calls), "rope_calls": trav["rope_calls"]}
    man["wall_s"] = round(time.time() - t0, 1)
    with open(man_path, "w") as f:
        f.write(json.dumps(man) + "\n")
    print(f"[tp] LOCKED -> {man_path} wall {man['wall_s']}s", flush=True)


if __name__ == "__main__":
    main()
