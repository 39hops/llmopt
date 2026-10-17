"""E0 preflight (instrument only, no bar is read): on the Mac CPU
reference runtime, show that the single-sequence unpadded forward the
census uses reproduces the left-padded batched forward the teacher
rollouts were generated under, on arm C itself, for two prompts of
different lengths; re-derive every prompt from the chat template and
compare with the stored input_ids; and account eos on every rollout.

Writes logs/qwen_e0_preflight/<arm>/preflight.json (refuse-if-exists).
Env: ART_DIR, VENDOR_DIR, TEACHER_DIR as the census; STEPS (default 16
generated tokens per prompt in the comparison); PROMPTS "i,j" indices
(default: shortest and longest prompt).
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
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from llmopt.lab import qartifact, qrope
    from llmopt.lab.qcodec import decode_entry

    art = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/C"))
    arm = os.path.basename(art.rstrip("/"))
    vdir = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
    tdir = os.environ.get("TEACHER_DIR", "logs/qwenteacher_v2")
    steps = int(os.environ.get("STEPS", "16"))
    out = f"logs/qwen_e0_preflight/{arm}"
    if os.path.exists(out):
        raise SystemExit(f"REFUSING: {out} exists")
    man = json.load(open(os.path.join(tdir, "teacher_manifest.json")))
    ro = man["records"]["rollouts"]
    recs = [json.loads(line) for line in open(os.path.join(tdir, "rollout_tokens.jsonl"))]
    input_ids = np.asarray(ro["input_ids"])
    mask = np.asarray(ro["attention_mask"])
    plen = mask.sum(1)
    tok = AutoTokenizer.from_pretrained(vdir)
    gcfg = json.load(open(os.path.join(vdir, "generation_config.json")))
    eos_ids = set(gcfg["eos_token_id"] if isinstance(gcfg["eos_token_id"], list) else [gcfg["eos_token_id"]])
    pad_id = gcfg.get("pad_token_id")

    # ---- 1. prompt reconstruction: chat template -> stored input_ids ----
    prows = [json.loads(line) for line in open("evals/qwen_model1/prompts.jsonl")]
    assert [r["id"] for r in prows] == ro["prompt_ids"] == [r["id"] for r in recs], "prompt order"
    texts = [tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                     tokenize=False, add_generation_prompt=True) for r in prows]
    enc = tok(texts, padding=True, padding_side="left", return_tensors="np")
    recon_ok = bool((enc["input_ids"] == input_ids).all() and (enc["attention_mask"] == mask).all())
    pads_left_only = all((mask[i][np.argmax(mask[i] == 1):] == 1).all() for i in range(len(recs)))
    pad_in_prompt = any(pad_id in input_ids[i][mask[i] == 1].tolist() for i in range(len(recs)))

    # ---- 2. eos accounting ----
    eos_rows = []
    for i, r in enumerate(recs):
        g = r["tokens"]
        eos_rows.append({"id": r["id"], "gen_len": len(g), "stopped_early": len(g) < ro["steps"],
                         "last_is_eos": g[-1] in eos_ids,
                         "eos_inside": any(t in eos_ids for t in g[:-1])})
    eos_ok = all((row["stopped_early"] == row["last_is_eos"]) and not row["eos_inside"]
                 for row in eos_rows)

    # ---- 3. padded batch v unpadded singles on the arm forward ----
    sel = os.environ.get("PROMPTS")
    idx = [int(x) for x in sel.split(",")] if sel else [int(plen.argmin()), int(plen.argmax())]
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
        e = MAN[name]
        return torch.from_numpy(np.ascontiguousarray(decode_entry(payload(e), e)))

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

    for i, ly in enumerate(model.model.layers):
        ly.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        ly.register_forward_hook(post, with_kwargs=True)

    seqs = []
    for i in idx:
        prompt = [int(x) for x in input_ids[i][mask[i] == 1]]
        gen = [int(x) for x in recs[i]["tokens"][:steps]]
        seqs.append((i, prompt, gen))
    # left-padded batch exactly as generate held it: pads, prompt, generated
    width = max(len(p) + len(g) for _, p, g in seqs)
    b_ids = np.full((len(seqs), width), pad_id, dtype=np.int64)
    b_mask = np.zeros((len(seqs), width), dtype=np.int64)
    for r, (_, p, g) in enumerate(seqs):
        s = p + g
        b_ids[r, width - len(s):] = s
        b_mask[r, width - len(s):] = 1
    t0 = time.time()
    with torch.no_grad():
        lb = model(input_ids=torch.tensor(b_ids), attention_mask=torch.tensor(b_mask),
                   use_cache=False).logits.float().numpy()
    wall_b = time.time() - t0
    comp = []
    for r, (i, p, g) in enumerate(seqs):
        t0 = time.time()
        with torch.no_grad():
            ls = model(input_ids=torch.tensor([p + g]), use_cache=False).logits[0].float().numpy()
        wall_s = time.time() - t0
        off = width - len(p) - len(g)
        a = lb[r, off + len(p) - 1: off + len(p) - 1 + len(g)]
        b = ls[len(p) - 1: len(p) - 1 + len(g)]
        comp.append({"prompt_idx": i, "id": recs[i]["id"], "prompt_len": len(p), "steps": len(g),
                     "pad_cols": int(off), "argmax_identical": bool((a.argmax(1) == b.argmax(1)).all()),
                     "max_abs_diff": float(np.abs(a - b).max()),
                     "max_rel_l2": float(max(np.linalg.norm(a[t] - b[t]) / np.linalg.norm(b[t])
                                             for t in range(len(g)))),
                     "wall_batch_s": wall_b, "wall_single_s": wall_s})
        print(f"[pf] {recs[i]['id']}: pads {off}, argmax identical {comp[-1]['argmax_identical']}, "
              f"max abs {comp[-1]['max_abs_diff']:.3e}, max rel l2 {comp[-1]['max_rel_l2']:.3e}", flush=True)
    os.makedirs(out, exist_ok=True)
    json.dump({"arm": arm, "prompt_reconstruction_exact": recon_ok, "pads_left_only": pads_left_only,
               "pad_in_prompt": pad_in_prompt, "eos_accounting_ok": eos_ok, "eos_rows": eos_rows,
               "eos_ids": sorted(eos_ids), "padded_v_unpadded": comp, "steps": steps},
              open(os.path.join(out, "preflight.json"), "w"), indent=1)
    print(f"[pf] reconstruction {recon_ok}, pads left only {pads_left_only}, pad in prompt "
          f"{pad_in_prompt}, eos ok {eos_ok}; wrote {out}", flush=True)


if __name__ == "__main__":
    main()
