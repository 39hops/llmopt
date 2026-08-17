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
    e = MAN[name]
    codec = e["codec"]
    if codec == "excluded":
        raise SystemExit(f"REFUSING: excluded tensor requested: {name}")
    shape = e["shape"]
    n = 1
    for d in shape:
        n *= d
    buf = _payload(e)
    if codec == "raw":
        u16 = np.frombuffer(buf, np.uint16)
        W = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
        return torch.from_numpy(W.copy())
    nb = n // 128
    exps = np.frombuffer(buf, np.uint8, nb, 0).astype(np.int32)
    scale = np.exp2(exps - 127).astype(np.float32)
    if codec == "w4":
        cb = np.frombuffer(buf, np.float16, 256 * 4, nb).reshape(256, 4)
        idx = np.frombuffer(buf, np.uint8, n // 4, nb + 2048)
        Wn = cb.astype(np.float32)[idx].reshape(nb, 128)
    elif codec == "s16":
        lv = np.frombuffer(buf, np.float16, 16, nb).astype(np.float32)
        codes = np.frombuffer(buf, np.uint8, n // 2, nb + 32)
        c = np.empty(n, np.uint8)
        c[0::2] = codes >> 4
        c[1::2] = codes & 0xF
        Wn = lv[c].reshape(nb, 128)
    else:
        raise SystemExit(f"unknown codec {codec}")
    return torch.from_numpy((Wn * scale[:, None]).reshape(shape))


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
    # OOM'd holding embed+lm_head fp32 = ~9.5 GiB): decoded W4/S16
    # values are fp16_codebook * 2^k, so an fp16 resident copy is
    # BIT-LOSSLESS (power-of-two scale moves only the exponent;
    # scale range asserted). embed gathers fp16 rows and casts
    # fp32 at lookup; lm_head matmuls in fp32 per row-chunk. Same
    # decoded weights, same function, ~5.5GB peak.
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
        e = MAN[key]
        exps = np.frombuffer(_payload(e), np.uint8,
                             (e["shape"][0] * e["shape"][1]) // 128, 0)
        assert exps.max() < 127 + 15, "scale would overflow fp16"
        return decode(key).to(torch.float16)

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

    for i, lyr in enumerate(layers):
        lyr.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        lyr.register_forward_hook(post, with_kwargs=True)
    return model


def main():
    from transformers import AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(VDIR)
    model = build()
    print(f"[0r] built from {ART} in {time.time()-t0:.0f}s", flush=True)
    text = PROMPT
    if CHAT:
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")
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
