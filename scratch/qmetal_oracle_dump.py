"""Dump the CPU oracle's full fp32 logits for one prompt plus recorded
rollout tokens (the census runtime: transformers streamed decode of
arm C, scratch/qwen_e0_spec_census.py construction). Receipt for the
Metal-path parity check; refuse-if-exists. Env: PROMPT_IDX (default 0),
STEPS (default 8), ART_DIR, VENDOR_DIR, TEACHER_DIR, OUT.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))


def main() -> None:
    import torch
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from llmopt.lab import qartifact, qrope
    from llmopt.lab.qcodec import decode_entry

    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tdir = os.environ.get("TEACHER_DIR", "logs/qwenteacher_v2")
    pi = int(os.environ.get("PROMPT_IDX", "0"))
    steps = int(os.environ.get("STEPS", "8"))
    out = os.environ.get("OUT", f"logs/qmetal/oracle_{arm}_p{pi}_s{steps}.npz")
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    prompt = [int(x) for x in input_ids[pi][mask[pi] == 1]]
    gen = [int(x) for x in recs[pi]["tokens"][:steps]]
    ids = prompt + gen
    chain = f"logs/qwenwhole/artifact_digest_{arm}.txt"
    q = qartifact.qualify_artifact(art, os.path.join(vdir, "model.safetensors.index.json"), chain)
    MAN = q["manifest"]
    handles: dict = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(art, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def decode(name):
        return torch.from_numpy(np.ascontiguousarray(decode_entry(payload(MAN[name]), MAN[name])))

    cfg = AutoConfig.from_pretrained(vdir)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
    model.eval()
    for nm, _ in list(model.named_parameters()):
        if nm.startswith("model.layers."):
            continue
        mod = model.get_submodule(nm.rsplit(".", 1)[0])
        leaf = nm.rsplit(".", 1)[1]
        full = nm if not nm.startswith("model.") else "model.language_model." + nm[len("model."):]
        mod._parameters[leaf] = torch.nn.Parameter(decode(full), requires_grad=False)
    rp = cfg.text_config.rope_parameters
    qrope.check_inv_freq(model.model.rotary_emb.inv_freq.numpy(), float(rp["rope_theta"]),
                         int(cfg.text_config.head_dim * rp.get("partial_rotary_factor", 1.0)))

    def make_pre(i):
        def pre(module, args, kwargs):
            for nm, _ in module.named_parameters():
                full = f"model.language_model.layers.{i}.{nm}"
                m2 = module.get_submodule(nm.rsplit(".", 1)[0]) if "." in nm else module
                leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
                m2._parameters[leaf] = torch.nn.Parameter(decode(full), requires_grad=False)
        return pre

    def post(module, args, kwargs, output):
        for nm, p in list(module.named_parameters()):
            m2 = module.get_submodule(nm.rsplit(".", 1)[0]) if "." in nm else module
            leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
            m2._parameters[leaf] = torch.nn.Parameter(p.to("meta"), requires_grad=False)
        return output

    hidden = {}
    want_hidden = os.environ.get("HIDDEN") == "1"

    def make_cap(i):
        def cap(module, args, kwargs, output):
            h = output[0] if isinstance(output, (tuple, list)) else output
            hidden[f"layer_{i}"] = h[0].detach().float().numpy().copy()
            return None
        return cap

    for i, ly in enumerate(model.model.layers):
        ly.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        ly.register_forward_hook(post, with_kwargs=True)
        if want_hidden:
            ly.register_forward_hook(make_cap(i), with_kwargs=True)
    if want_hidden:
        def cap_in(module, args, kwargs, output):
            hidden["embed"] = output[0].detach().float().numpy().copy()
        model.model.embed_tokens.register_forward_hook(cap_in, with_kwargs=True)
        def cap_norm(module, args, kwargs, output):
            hidden["final_norm"] = output[0].detach().float().numpy().copy()
        model.model.norm.register_forward_hook(cap_norm, with_kwargs=True)
    t0 = time.time()
    with torch.no_grad():
        lg = model(input_ids=torch.tensor([ids]), use_cache=False).logits[0].float().numpy()
    wall = time.time() - t0
    np.savez(out, ids=np.asarray(ids), logits=lg.astype(np.float32), prompt_len=len(prompt),
             wall_s=wall, arm=arm, chain=open(chain).read(), **hidden)
    print(f"[oracle] {recs[pi]['id']}: {len(ids)} tokens, {wall:.0f}s, logits {lg.shape}, wrote {out}")


if __name__ == "__main__":
    main()
