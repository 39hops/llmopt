"""CUDA leg rungs 3+4: artifact A resident in VRAM, per-layer GPU
decode, full-tower forward — per-layer hidden-state error and
backend-agreement KL against the CPU reference computed in the SAME
process.

Residency plan (measured budget: 8.86 GiB free on the 3080/WSL,
logs/qwencuda/rung0.json): all 64 layers' w4 payloads live on
device as u8 buffers (~6.3 GiB; io tensors excluded), a pre-layer
hook decodes each layer's weights to fp32 with the Triton kernel
(peak one-layer scratch ~1.5 GiB), a post-layer hook frees them.
embed stays CPU-compressed (row gather, exact fp32); lm_head
payload lives on device and decodes in row chunks per use, LAST
POSITION ONLY (fp32 logits for a full prefill would be
vocab 248320 x seq x 4 B — the reviewer's B4).

Scale exactness without resident fp32 scales: the kernel rebuilds
scale = 2^(exp-127) from the u8 exponent by BIT CONSTRUCTION
((exp << 23) bitcast to fp32 — exact for 1 <= exp <= 254; exp = 0
is the fp32 subnormal 2^-127, special-cased as a literal; exp =
255 bitcasts to +inf, matching np.exp2 overflow). No exp2
evaluation on device. The rung-1 fixture set (4 random shapes +
exp 0/127/255 edges) re-runs IN-PROCESS against this kernel and
must be bit-exact before any payload is uploaded.

Backend rule (registered): numbers here are backend-agreement
quantities against the CPU reference ONLY — never tree quantities
(the scorer refuses device != cpu). Receipt key is
backend_agreement_kl_vs_cpu_ref, a name the tree scorer does not
consume.

    MODE=forward1 .venv/bin/python scratch/qwen_cuda_rung3.py
    N_NEW=32 PROMPT=... .venv/bin/python scratch/qwen_cuda_rung3.py

Receipt: logs/qwencuda/rung3_forward1.json (refuse-if-exists).
"""
import json
import os
import subprocess
import sys
import time

import numpy as np
import torch
import triton
import triton.language as tl

ART = os.path.expanduser(os.environ.get("ART_DIR", "~/qwen_whole0t/A"))
VDIR = os.path.expanduser(os.environ.get("VENDOR_DIR", "~/qwen_vendor"))
PROMPT = os.environ.get("PROMPT", "The capital of France is")
N_NEW = int(os.environ.get("N_NEW", "0"))
CHAT = os.environ.get("CHAT", "1") == "1"
CPU_REF = os.environ.get("CPU_REF", "1") == "1"
OUT = os.environ.get("RECEIPT", "logs/qwencuda/rung3_forward1.json")

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab import qartifact, qrope  # noqa: E402
from llmopt.lab.qcodec import BLOCK, dec_w4, decode_entry  # noqa: E402

SUBNORMAL_2_M127 = tl.constexpr(5.877471754111438e-39)  # fp32 2^-127


@triton.jit
def w4_decode_kernel(idx_ptr, cb_ptr, exp_ptr, out_ptr, n,
                     BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    e = tl.load(exp_ptr + offs // 128, mask=m, other=127).to(tl.int32)
    sbits = e << 23
    s = tl.where(e == 0, SUBNORMAL_2_M127,
                 sbits.to(tl.float32, bitcast=True))
    byte = tl.load(idx_ptr + offs // 4, mask=m, other=0)
    lane = offs % 4
    val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + lane, mask=m,
                  other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, val * s, mask=m)


class W4Gpu:
    """One w4 payload resident on device as u8/fp16 buffers."""

    def __init__(self, buf: bytes, shape):
        n = int(np.prod(shape))
        nb = n // BLOCK
        self.shape, self.n = list(shape), n
        self.exps = torch.from_numpy(
            np.frombuffer(buf, np.uint8, nb, 0).copy()).cuda()
        self.cb = torch.from_numpy(
            np.frombuffer(buf, np.float16, 1024, nb).copy()).cuda()
        self.idx = torch.from_numpy(
            np.frombuffer(buf, np.uint8, n // 4, nb + 2048).copy()).cuda()

    def decode(self, lo=0, hi=None):
        """Decode rows [lo, hi) to fp32 cuda (row-aligned: C % 128
        == 0 asserted at build)."""
        R, C = self.shape
        hi = R if hi is None else hi
        n = (hi - lo) * C
        out = torch.empty(n, dtype=torch.float32, device="cuda")
        # row-sliced launch: offset payload views by whole rows
        i0, e0 = lo * C // 4, lo * C // BLOCK
        w4_decode_kernel[(triton.cdiv(n, 1024),)](
            self.idx[i0:], self.cb, self.exps[e0:], out, n, BLK=1024)
        return out.reshape(hi - lo, C)


def _fixture_gate():
    """rung-1 parity set re-run against THIS kernel (in-kernel
    scales): bit-exact or refuse to touch the artifact."""
    rng = np.random.default_rng(3)
    cases = []
    for R, C in ((8, 256), (16, 128), (5, 640), (32, 512)):
        nb = R * C // BLOCK
        buf = (rng.integers(120, 132, nb, dtype=np.uint8).tobytes()
               + (rng.standard_normal((256, 4)) * 0.3)
               .astype(np.float16).tobytes()
               + rng.integers(0, 256, R * C // 4,
                              dtype=np.uint8).tobytes())
        cases.append((f"random-{R}x{C}", buf, [R, C]))
    for ev in (0, 127, 255):
        nb = 8 * 256 // BLOCK
        buf = (np.full(nb, ev, np.uint8).tobytes()
               + (rng.standard_normal((256, 4)) * 0.3)
               .astype(np.float16).tobytes()
               + rng.integers(0, 256, 8 * 64,
                              dtype=np.uint8).tobytes())
        cases.append((f"exp-{ev}", buf, [8, 256]))
    for name, buf, shape in cases:
        got = W4Gpu(buf, shape).decode().cpu().numpy()
        ref = dec_w4(buf, shape)
        if not np.array_equal(got, ref, equal_nan=True):
            raise SystemExit(f"KERNEL PARITY FAIL: {name}")
    print(f"[r3] kernel parity gate: {len(cases)} fixtures bit-exact",
          flush=True)


def build(device: str):
    """Meta model + per-layer decode hooks. device='cpu' is the
    reference lane (canonical qcodec decode); device='cuda' decodes
    from resident W4Gpu payloads. Hidden states are captured after
    every decoder layer into taps[]."""
    from accelerate import init_empty_weights
    from transformers import AutoConfig, AutoModelForCausalLM
    from llmopt.lab.qcodec_fast import W4Rows

    arm = os.path.basename(ART.rstrip("/"))
    chain = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "logs", "qwenwhole",
        f"artifact_digest_{arm}.txt")
    q = qartifact.qualify_artifact(
        ART, VDIR + "/model.safetensors.index.json",
        chain if os.path.exists(chain) else None,
        allow_unchained=os.environ.get("ALLOW_UNCHAINED") == "1")
    man = q["manifest"]
    handles = {}

    def payload(e):
        sh = e["shard"]
        if sh not in handles:
            handles[sh] = open(os.path.join(ART, sh + ".bin"), "rb")
        handles[sh].seek(e["off"])
        return handles[sh].read(e["len"])

    def cpu_decode(name):
        e = man[name]
        return torch.from_numpy(
            np.ascontiguousarray(decode_entry(payload(e), e)))

    cfg = AutoConfig.from_pretrained(VDIR)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForCausalLM.from_config(
            cfg, torch_dtype=torch.float32)
    model.eval()

    def shard_key(nm):
        if nm.startswith("model."):
            return "model.language_model." + nm[len("model."):]
        return nm

    layer_pref = "model.layers."
    for nm, _ in list(model.named_parameters()):
        if nm.startswith(layer_pref) or nm in (
                "model.embed_tokens.weight", "lm_head.weight"):
            continue
        mod = model.get_submodule(nm.rsplit(".", 1)[0])
        setattr(mod, nm.rsplit(".", 1)[1], torch.nn.Parameter(
            cpu_decode(shard_key(nm)).to(device),
            requires_grad=False))

    # io codec gate: this driver implements a w4 GPU/row path ONLY.
    # S16 io (artifacts B/C) decoded here as w4 produced non-finite
    # logits (measured 2026-08-17) — refuse, never reinterpret.
    ee = man["model.language_model.embed_tokens.weight"]
    he_chk = man["lm_head.weight"]
    for _nm, _e in (("embed_tokens", ee), ("lm_head", he_chk)):
        if _e["codec"] != "w4":
            raise SystemExit(
                f"REFUSING: {_nm} codec {_e['codec']!r} — this "
                f"driver supports w4 io only (S16 GPU path not "
                f"built; artifacts B/C need it)")
    emb = W4Rows(payload(ee), ee["shape"])

    def emb_fwd(input_ids):
        flat = input_ids.reshape(-1)
        out = torch.empty(flat.shape[0], emb.C)
        for j, t in enumerate(flat.tolist()):
            out[j] = torch.from_numpy(emb.rows(t, t + 1)[0])
        return out.reshape(*input_ids.shape, emb.C).to(device)

    model.model.embed_tokens.forward = emb_fwd

    he = man["lm_head.weight"]
    if device == "cuda":
        head_gpu = W4Gpu(payload(he), he["shape"])

        def head_fwd(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:]          # last position only (B4)
            outs = []
            for lo in range(0, he["shape"][0], 16384):
                hi = min(lo + 16384, he["shape"][0])
                outs.append(x @ head_gpu.decode(lo, hi).T)
            return torch.cat(outs, -1)
    else:
        head_cpu = W4Rows(payload(he), he["shape"])

        def head_fwd(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:]
            outs = []
            for lo in range(0, head_cpu.R, 16384):
                hi = min(lo + 16384, head_cpu.R)
                outs.append(
                    x @ torch.from_numpy(head_cpu.rows(lo, hi)).T)
            return torch.cat(outs, -1)

    model.lm_head.forward = head_fwd

    meta_bufs = [nm for nm, b in model.named_buffers() if b.is_meta]
    if meta_bufs:
        raise SystemExit(f"REFUSING: meta buffers: {meta_bufs[:4]}")
    rp = cfg.text_config.rope_parameters
    qrope.check_inv_freq(
        model.model.rotary_emb.inv_freq.cpu().numpy(),
        float(rp["rope_theta"]),
        int(cfg.text_config.head_dim
            * rp.get("partial_rotary_factor", 1.0)))
    if device == "cuda":
        model.model.rotary_emb.to("cuda")

    layers = model.model.layers
    # resident compressed layer payloads (cuda lane only)
    gpu_payloads = {}
    if device == "cuda":
        t0 = time.time()
        for i in range(len(layers)):
            for nm, _ in layers[i].named_parameters():
                full = f"model.language_model.layers.{i}.{nm}"
                e = man[full]
                if e["codec"] == "w4":
                    assert e["shape"][1] % BLOCK == 0 \
                        and e["shape"][1] % 4 == 0, full
                    gpu_payloads[full] = W4Gpu(payload(e), e["shape"])
                else:                   # raw/s16: tiny, decode once
                    gpu_payloads[full] = cpu_decode(full).cuda()
        torch.cuda.synchronize()
        free, _ = torch.cuda.mem_get_info()
        print(f"[r3] payloads resident in {time.time()-t0:.0f}s, "
              f"free now {free/2**30:.2f} GiB", flush=True)

    taps = []
    trav = {"layer_calls": [0] * len(layers),
            "attn_exec": {"linear_attn": 0, "full_attn": 0},
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
                if device == "cuda":
                    src = gpu_payloads[full]
                    W = src.decode() if isinstance(src, W4Gpu) else src
                else:
                    W = cpu_decode(full)
                m2._parameters[leaf] = torch.nn.Parameter(
                    W, requires_grad=False)
            return None
        return pre

    def post(module, args, kwargs, output):
        h = output[0] if isinstance(output, tuple) else output
        taps.append(float(h.detach().float().norm()))
        taps_full.append(h.detach().float().cpu()
                         if len(taps_full) < 64 else None)
        for nm, p in list(module.named_parameters()):
            m2 = module.get_submodule(nm.rsplit(".", 1)[0]) \
                if "." in nm else module
            leaf = nm.rsplit(".", 1)[1] if "." in nm else nm
            m2._parameters[leaf] = torch.nn.Parameter(
                p.to("meta"), requires_grad=False)
        return output

    taps_full = []

    def count_attn(fam):
        def h(module, args, kwargs):
            trav["attn_exec"][fam] += 1
            return None
        return h

    for lyr in layers:
        if hasattr(lyr, "self_attn"):
            lyr.self_attn.register_forward_pre_hook(
                count_attn("full_attn"), with_kwargs=True)
        else:
            lyr.linear_attn.register_forward_pre_hook(
                count_attn("linear_attn"), with_kwargs=True)
    for i, lyr in enumerate(layers):
        lyr.register_forward_pre_hook(make_pre(i), with_kwargs=True)
        lyr.register_forward_hook(post, with_kwargs=True)
    return model, trav, taps_full


def forward1(model, ids, device):
    t = time.time()
    out = model(input_ids=ids.to(device), use_cache=False)
    if device == "cuda":
        torch.cuda.synchronize()
    lg = out.logits[0, -1].float().cpu()
    assert torch.isfinite(lg).all(), "non-finite logits"
    return lg, time.time() - t


def main():
    from transformers import AutoTokenizer
    if os.path.exists(OUT):
        raise SystemExit(f"refuse: {OUT} exists — new run, new path")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("[r3] code_commit " + subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        flush=True)
    _fixture_gate()

    tok = AutoTokenizer.from_pretrained(VDIR)
    text = PROMPT
    if CHAT:
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"]

    free0 = torch.cuda.mem_get_info()[0]
    model, trav, taps_g = build("cuda")
    lg_g, wall_g = forward1(model, ids, "cuda")
    calls, fams = trav["layer_calls"], trav["families"]
    n_lin = sum(1 for i, f in enumerate(fams)
                if f == "linear_attn" and calls[i] > 0)
    n_full = sum(1 for i, f in enumerate(fams)
                 if f == "full_attn" and calls[i] > 0)
    assert (len(calls), n_lin, n_full) == (64, 48, 16), \
        f"traversal {len(calls)}/{n_lin}/{n_full}"
    assert trav["attn_exec"]["linear_attn"] >= 48 \
        and trav["attn_exec"]["full_attn"] >= 16, trav["attn_exec"]
    peak = torch.cuda.max_memory_allocated()
    peak_res = torch.cuda.max_memory_reserved()
    print(f"[r3] cuda forward1 {wall_g:.1f}s | peak alloc "
          f"{peak/2**30:.2f} GiB reserved {peak_res/2**30:.2f}",
          flush=True)
    top = torch.topk(lg_g, 5)
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"[r3]   {v:8.3f}  {tok.decode([i])!r}", flush=True)

    rec = {
        "rung": 3,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short",
             "HEAD"]).decode().strip(),
        "device_actual": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "wsl": "microsoft" in os.uname().release.lower(),
        "artifact": os.path.basename(ART.rstrip("/")),
        "prompt": PROMPT, "seq_len": int(ids.shape[1]),
        "kernel_form": "per_layer_decode_then_dense",
        "expanded_operand_bytes_per_layer": "one layer fp32, freed",
        "logits_positions": "last_only",
        "traversal": {"layers": 64, "linear": n_lin, "full": n_full,
                      "attn_exec": trav["attn_exec"]},
        "vram_free_at_start_bytes": int(free0),
        "vram_peak_alloc_bytes": int(peak),
        "vram_peak_reserved_bytes": int(peak_res),
        "cuda_forward_s": round(wall_g, 2),
        "env": {k: os.environ.get(k) for k in
                ("TORCH_DISABLE_NATIVE_JIT",
                 "PYTORCH_CUDA_ALLOC_CONF")},
    }

    if CPU_REF:
        del model
        torch.cuda.empty_cache()
        model_c, trav_c, taps_c = build("cpu")
        lg_c, wall_c = forward1(model_c, ids, "cpu")
        rels = []
        for k, (hg, hc) in enumerate(zip(taps_g, taps_c)):
            if hg is None or hc is None:
                continue
            d = (hg - hc).norm() / hc.norm().clamp_min(1e-30)
            rels.append(float(d))
        pg = torch.softmax(lg_g, -1)
        pc = torch.softmax(lg_c, -1)
        kl = float((pc * (pc.clamp_min(1e-12).log()
                          - pg.clamp_min(1e-12).log())).sum())
        agree = bool(int(lg_g.argmax()) == int(lg_c.argmax()))
        # near-tie fence: margin at CPU argmax
        s = torch.sort(lg_c, descending=True)
        margin = float(s.values[0] - s.values[1])
        print(f"[r3] per-layer rel err: max {max(rels):.3e} "
              f"median {float(np.median(rels)):.3e} | "
              f"backend KL {kl:.3e} | argmax agree {agree} "
              f"(cpu margin {margin:.3f}) | cpu {wall_c:.0f}s",
              flush=True)
        rec["cpu_forward_s"] = round(wall_c, 1)
        rec["per_layer_hidden_rel_err"] = {
            "max": max(rels), "median": float(np.median(rels)),
            "per_layer": [round(r, 9) for r in rels]}
        rec["backend_agreement_kl_vs_cpu_ref"] = kl
        rec["argmax_agree"] = agree
        rec["cpu_top1_margin"] = margin

    if N_NEW > 0:
        # KV-cached greedy generation on the cuda lane (rung 4
        # smoke): weights already resident; per-token wall includes
        # per-layer decode (kernel_form unchanged). Prefill+decode
        # reported separately per the registered protocol.
        if CPU_REF:
            # the cuda model was torn down for the reference lane;
            # rebuild it (payloads re-upload, ~2s)
            del model_c
            model, trav, _ = build("cuda")
        # else: REUSE the live cuda model — a second build would
        # double the resident payloads (~13 GiB) and drive WSL into
        # host-memory oversubscription (measured: gen hung >10 min)
        torch.cuda.synchronize()
        tg = time.time()
        out_ids = model.generate(
            input_ids=ids.cuda(), max_new_tokens=N_NEW,
            do_sample=False, use_cache=True,
            pad_token_id=tok.eos_token_id)
        torch.cuda.synchronize()
        wall = time.time() - tg
        gen = out_ids[0][ids.shape[1]:]
        txt = tok.decode(gen, skip_special_tokens=True)
        print(f"[r3] gen {len(gen)} tokens in {wall:.1f}s "
              f"({wall/max(len(gen),1):.2f}s/tok)", flush=True)
        print("[r3] OUTPUT:", txt, flush=True)
        rec["gen"] = {"n_new": int(len(gen)),
                      "wall_s": round(wall, 2),
                      "s_per_tok": round(wall / max(len(gen), 1), 3),
                      "output": txt}

    with open(OUT, "x") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("[r3] receipt ->", OUT, flush=True)


if __name__ == "__main__":
    main()
