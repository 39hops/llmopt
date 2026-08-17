"""QWEN-RUNTIME-0R minimal CPU reference: decode a WHOLE-0T
artifact per layer and generate.

The "does it talk" oracle, first cut: model built on the meta
device (teacher-pass architecture), decoder-layer pre/post hooks
materialize weights by DECODING the compressed artifact instead of
mmapping vendor shards. Resident set (embed/head/norms/small)
decodes once at build. Vendor forward code untouched.

Payload layouts (frozen by scratch/qwen_whole0t.py, verified by
the scorer scout 2026-08-17):
  w4 : exps u8[R*C/128] ++ codebook fp16[256,4] ++ idxs u8[R*C/4]
       scale = 2^(exps-127) per block-128; groups of 4 consecutive
       along the row.
  s16: exps u8[R*C/128] ++ levels fp16[16] ++ codes u8[R*C/2]
       HIGH nibble = even element (opposite of GPTQ convention).
  raw: vendor bytes (BF16); excluded: never loaded (vision/MTP —
       absent from the text tower anyway).

    ART_DIR=~/qwen_whole0t/A PROMPT="..." N_NEW=32 \
        .venv/bin/python scratch/qwen_runtime0r.py

Descriptive sanity peek only: neutral prompts, never the frozen
MODEL-1 payload; no number from this script feeds the tree.
"""
import json
import os
import time

import numpy as np
import torch

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
N_NEW = int(os.environ.get("N_NEW", "32"))
CHAT = os.environ.get("CHAT", "1") == "1"

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())

MAN = json.load(open(os.path.join(ART, "manifest.json")))
_handles = {}


def _payload(e):
    sh = e["shard"]
    if sh not in _handles:
        _handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
    f = _handles[sh]
    f.seek(e["off"])
    return f.read(e["len"])


def decode(name):
    """SINGLE decode path: llmopt.lab.qcodec (the canonical module
    shared with the sidecar and scorer; golden fixtures in
    tests/test_qwen_codec.py). No local decode logic lives here."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))
    from llmopt.lab.qcodec import decode_entry
    e = MAN[name]
    if e["codec"] == "excluded":
        raise SystemExit(f"REFUSING: excluded tensor requested: {name}")
    W = decode_entry(_payload(e), e)
    return torch.from_numpy(np.ascontiguousarray(W))


def build():
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(VDIR)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg,
                                                 torch_dtype=torch.float32)
    model.eval()

    def shard_key(nm):
        if nm.startswith("model."):
            return "model.language_model." + nm[len("model."):]
        return nm

    # RESIDENT SET, memory-shaped for a 13GB host (first peek
    # OOM'd holding embed+lm_head fp32 = ~9.5 GiB): io tensors are
    # held fp16 ONLY when the exact fp32->fp16->fp32 round-trip
    # oracle passes on the actual tensor (dec16 below REFUSES
    # otherwise — a correctness reference must never run on
    # silently altered weights). embed gathers fp16 rows and casts
    # fp32 at lookup; lm_head matmuls in fp32 per row-chunk.
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

    def dec16(key):
        # fp16 residency must be PROVEN lossless on the actual
        # tensor, not argued from representability: exact
        # fp32 -> fp16 -> fp32 round-trip equality is the oracle.
        # On failure the storage is fp16-APPROXIMATED and says so.
        W32 = decode(key)
        W16 = W32.to(torch.float16)
        if not torch.equal(W32, W16.float()):
            bad = int((W32 != W16.float()).sum())
            raise SystemExit(
                f"REFUSING: fp16 residency NOT lossless for {key} "
                f"({bad}/{W32.numel()} entries change) — a reference "
                f"runtime never runs on altered weights")
        return W16

    emb16 = dec16("model.language_model.embed_tokens.weight")
    head16 = dec16("lm_head.weight")

    def emb_fwd(input_ids):
        return torch.nn.functional.embedding(input_ids, emb16).float()

    model.model.embed_tokens.forward = emb_fwd

    def head_fwd(x):
        outs = []
        for lo in range(0, head16.shape[0], 16384):
            outs.append(x @ head16[lo:lo + 16384].float().T)
        return torch.cat(outs, -1)

    model.lm_head.forward = head_fwd
    meta_bufs = [nm for nm, b in model.named_buffers() if b.is_meta]
    if meta_bufs:
        raise SystemExit(f"REFUSING: meta buffers: {meta_bufs[:4]}")
    if float(model.model.rotary_emb.inv_freq.abs().sum()) == 0.0:
        raise SystemExit("REFUSING: zero inv_freq")
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

    # traversal census (QWEN-TEACHER-0-TRAVERSAL pattern, reused
    # not reinvented): forward1 asserts 64/48/16 + rope like the
    # teacher's lock gate — a smoke must prove mechanism execution
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


def main():
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(VDIR)
    model, trav = build()
    print(f"[0r] built from {ART} in {time.time()-t0:.0f}s", flush=True)
    text = PROMPT
    if CHAT:
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
    if os.environ.get("MODE") == "forward1":
        # ladder rung 4: ONE full-tower forward, mechanism-complete
        t = time.time()
        out = model(input_ids=ids["input_ids"], use_cache=False)
        lg = out.logits[0, -1].float()
        assert torch.isfinite(lg).all(), "non-finite logits"
        calls = trav["layer_calls"]
        fams = trav["families"]
        n_lin = sum(1 for i, f in enumerate(fams)
                    if f == "linear_attn" and calls[i] > 0)
        n_full = sum(1 for i, f in enumerate(fams)
                     if f == "full_attn" and calls[i] > 0)
        assert (len(calls), n_lin, n_full) == (64, 48, 16), \
            f"traversal {len(calls)}/{n_lin}/{n_full} != 64/48/16"
        assert trav["rope_calls"] >= n_full, "rope under-called"
        assert min(calls) > 0, "idle layer"
        top = torch.topk(lg, 5)
        print(f"[0r] forward1 {time.time()-t:.0f}s | vocab "
              f"{lg.shape[-1]} | traversal 64/{n_lin}/{n_full} "
              f"rope={trav['rope_calls']} | top5:", flush=True)
        for v, i in zip(top.values.tolist(), top.indices.tolist()):
            print(f"[0r]   {v:8.3f}  {tok.decode([i])!r}", flush=True)
        import resource
        import sys as _s
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_b = rss if _s.platform == "darwin" else rss * 1024
        print(f"[0r] peak RSS {rss_b/2**30:.2f} GiB (observed, v "
              f"qualifier estimate)", flush=True)
        return
    t = time.time()
    out = model.generate(**ids, max_new_tokens=N_NEW, do_sample=False,
                         use_cache=True, pad_token_id=tok.eos_token_id)
    gen = out[0][ids["input_ids"].shape[1]:]
    wall = time.time() - t
    print(f"[0r] {len(gen)} tokens in {wall:.0f}s "
          f"({wall/max(len(gen),1):.1f}s/tok)", flush=True)
    print("[0r] PROMPT:", PROMPT, flush=True)
    print("[0r] OUTPUT:", tok.decode(gen, skip_special_tokens=True),
          flush=True)


if __name__ == "__main__":
    main()
