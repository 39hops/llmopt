"""CUDA leg rung 4: FUSED tower — every w4 Linear executes as a
fused decode+GEMV from the resident compressed payload; no fp32
weight materialization in the decode phase.

Module surgery instead of hooks: each nn.Linear inside the decoder
layers whose manifest entry is w4 is REPLACED by FusedW4Linear
holding the payload as resident u8/fp16 device buffers. Its
forward:
  - single row (decode step): fused GEMV kernel (rung-2 lineage,
    scales rebuilt in-kernel by exponent bit-construction as
    validated bit-exact in rung 3) — only compressed bytes touch
    DRAM (kernel_form: fused).
  - multi-row (prefill): decode row-chunks (transient, freed) and
    matmul — prefill happens once; decode-phase traffic is what
    the ladder is buying down.
Non-w4 layer params (norms, convs, biases — raw) decode once at
build and stay resident fp32. embed stays CPU-compressed row
gather; lm_head is a FusedW4Linear evaluated at the LAST position
only.

Parity gates, in-process before the artifact is touched:
  - decode kernel: the 7-fixture bit-exact set (rung 3's gate).
  - GEMV kernel: random-payload y=Wx v canonical decode @ x in
    float64, rel bar 1e-5 (accumulation order differs; bit
    exactness is not expected for GEMV), plus exp-edge payloads.
Tower gates: traversal census 64/48/16, finite logits, and
backend agreement v the rung-3 receipt's CPU-lane top-1.

    MODE=forward1 [CPU_REF=1] .venv/bin/python scratch/qwen_cuda_rung4.py
    N_NEW=32 .venv/bin/python scratch/qwen_cuda_rung4.py

Receipt: RECEIPT env (refuse-if-exists), default
logs/qwencuda/rung4_forward1.json.
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
CPU_REF = os.environ.get("CPU_REF", "0") == "1"
OUT = os.environ.get("RECEIPT", "logs/qwencuda/rung4_forward1.json")

torch.set_grad_enabled(False)
torch.set_num_threads(os.cpu_count())
sys.path.insert(0, os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
from llmopt.lab import qartifact, qrope  # noqa: E402
from llmopt.lab.qcodec import (BLOCK, dec_s16, dec_w4,  # noqa: E402
                               decode_entry)

SUB127 = tl.constexpr(5.877471754111438e-39)  # exact fp32 2^-127


@triton.jit
def w4_decode_kernel(idx_ptr, cb_ptr, exp_ptr, out_ptr, n,
                     BLK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLK + tl.arange(0, BLK)
    m = offs < n
    e = tl.load(exp_ptr + offs // 128, mask=m, other=127).to(tl.int32)
    s = tl.where(e == 0, SUB127,
                 (e << 23).to(tl.float32, bitcast=True))
    byte = tl.load(idx_ptr + offs // 4, mask=m, other=0)
    val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + offs % 4,
                  mask=m, other=0.0).to(tl.float32)
    tl.store(out_ptr + offs, val * s, mask=m)


@triton.jit
def s16_gemv_kernel(code_ptr, lv_ptr, exp_ptr, x_ptr, y_ptr, C,
                    BLK_C: tl.constexpr):
    """s16 fused GEMV: HIGH nibble = EVEN element (qcodec
    convention — opposite of GPTQ); scale by bit-construction."""
    r = tl.program_id(0)
    acc = tl.zeros((BLK_C,), tl.float32)
    for c0 in range(0, C, BLK_C):
        offs = c0 + tl.arange(0, BLK_C)
        m = offs < C
        flat = r * C + offs
        e = tl.load(exp_ptr + flat // 128, mask=m,
                    other=127).to(tl.int32)
        s = tl.where(e == 0, SUB127,
                     (e << 23).to(tl.float32, bitcast=True))
        byte = tl.load(code_ptr + flat // 2, mask=m,
                       other=0).to(tl.int32)
        nib = tl.where(offs % 2 == 0, byte >> 4, byte & 0xF)
        val = tl.load(lv_ptr + nib, mask=m,
                      other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        acc += val * s * x
    tl.store(y_ptr + r, tl.sum(acc, 0))


@triton.jit
def w4_gemv_kernel(idx_ptr, cb_ptr, exp_ptr, x_ptr, y_ptr, C,
                   BLK_C: tl.constexpr):
    """One program per output row, fused decode+dot, fp32 acc.
    Scales rebuilt in-kernel (bit-construction, rung-3 validated)."""
    r = tl.program_id(0)
    acc = tl.zeros((BLK_C,), tl.float32)
    for c0 in range(0, C, BLK_C):
        offs = c0 + tl.arange(0, BLK_C)
        m = offs < C
        flat = r * C + offs
        e = tl.load(exp_ptr + flat // 128, mask=m,
                    other=127).to(tl.int32)
        s = tl.where(e == 0, SUB127,
                     (e << 23).to(tl.float32, bitcast=True))
        byte = tl.load(idx_ptr + flat // 4, mask=m, other=0)
        val = tl.load(cb_ptr + byte.to(tl.int32) * 4 + offs % 4,
                      mask=m, other=0.0).to(tl.float32)
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        acc += val * s * x
    tl.store(y_ptr + r, tl.sum(acc, 0))


class W4Gpu:
    def __init__(self, buf: bytes, shape):
        n = int(np.prod(shape))
        nb = n // BLOCK
        self.shape, self.n = list(shape), n
        self.exps = torch.from_numpy(
            np.frombuffer(buf, np.uint8, nb, 0).copy()).cuda()
        self.cb = torch.from_numpy(
            np.frombuffer(buf, np.float16, 1024, nb).copy()).cuda()
        self.idx = torch.from_numpy(
            np.frombuffer(buf, np.uint8, n // 4,
                          nb + 2048).copy()).cuda()

    def decode_rows(self, lo, hi):
        R, C = self.shape
        n = (hi - lo) * C
        out = torch.empty(n, dtype=torch.float32, device="cuda")
        w4_decode_kernel[(triton.cdiv(n, 1024),)](
            self.idx[lo * C // 4:], self.cb,
            self.exps[lo * C // BLOCK:], out, n, BLK=1024)
        return out.reshape(hi - lo, C)

    def gemv(self, x):
        R, C = self.shape
        y = torch.empty(R, dtype=torch.float32, device="cuda")
        w4_gemv_kernel[(R,)](self.idx, self.cb, self.exps,
                             x.contiguous(), y, C, BLK_C=512)
        return y


class S16Gpu:
    """One s16 payload resident on device. GEMV-only (io use); the
    prefill path decodes rows via S16Rows on CPU."""

    def __init__(self, buf: bytes, shape):
        n = int(np.prod(shape))
        nb = n // BLOCK
        self.shape = list(shape)
        self.exps = torch.from_numpy(
            np.frombuffer(buf, np.uint8, nb, 0).copy()).cuda()
        self.lv = torch.from_numpy(
            np.frombuffer(buf, np.float16, 16, nb).copy()).cuda()
        self.codes = torch.from_numpy(
            np.frombuffer(buf, np.uint8, n // 2,
                          nb + 32).copy()).cuda()

    def gemv(self, x):
        R, C = self.shape
        y = torch.empty(R, dtype=torch.float32, device="cuda")
        s16_gemv_kernel[(R,)](self.codes, self.lv, self.exps,
                              x.contiguous(), y, C, BLK_C=512)
        return y


class FusedW4Linear(torch.nn.Module):
    CHUNK = 8192

    def __init__(self, pay: W4Gpu):
        super().__init__()
        self.pay = pay
        self.out_features, self.in_features = pay.shape

    def forward(self, x):
        lead = x.shape[:-1]
        C = x.shape[-1]
        flat = x.reshape(-1, C)
        if flat.shape[0] == 1:
            y = self.pay.gemv(flat[0].float())
            return y.reshape(*lead, -1).to(x.dtype)
        outs = []
        R = self.pay.shape[0]
        for lo in range(0, R, self.CHUNK):
            hi = min(lo + self.CHUNK, R)
            W = self.pay.decode_rows(lo, hi)
            outs.append(flat.float() @ W.T)
        return torch.cat(outs, -1).reshape(*lead, -1).to(x.dtype)


def _gates():
    rng = np.random.default_rng(3)
    cases = []
    for R, C in ((8, 256), (16, 128), (5, 640), (32, 512)):
        nb = R * C // BLOCK
        cases.append((f"random-{R}x{C}",
                      rng.integers(120, 132, nb,
                                   dtype=np.uint8).tobytes()
                      + (rng.standard_normal((256, 4)) * 0.3)
                      .astype(np.float16).tobytes()
                      + rng.integers(0, 256, R * C // 4,
                                     dtype=np.uint8).tobytes(),
                      [R, C]))
    for ev in (0, 127, 254):
        nb = 8 * 256 // BLOCK
        cases.append((f"exp-{ev}",
                      np.full(nb, ev, np.uint8).tobytes()
                      + (rng.standard_normal((256, 4)) * 0.3)
                      .astype(np.float16).tobytes()
                      + rng.integers(0, 256, 8 * 64,
                                     dtype=np.uint8).tobytes(),
                      [8, 256]))
    for name, buf, shape in cases:
        pay = W4Gpu(buf, shape)
        ref = dec_w4(buf, shape)
        got = pay.decode_rows(0, shape[0]).cpu().numpy()
        if not np.array_equal(got, ref, equal_nan=True):
            raise SystemExit(f"DECODE PARITY FAIL: {name}")
        x = np.random.default_rng(17).standard_normal(
            shape[1]).astype(np.float32)
        y = pay.gemv(torch.from_numpy(x).cuda()).cpu().numpy()
        r64 = ref.astype(np.float64) @ x.astype(np.float64)
        denom = max(1e-30, float(np.abs(r64).max()))
        rel = float(np.abs(y - r64).max() / denom)
        if rel > 1e-5:
            raise SystemExit(f"GEMV PARITY FAIL: {name} rel={rel:.3e}")
    # s16 GEMV gate (io path for B/C): same shapes + exp edges
    n_s16 = 0
    for R, C in ((8, 256), (16, 128), (5, 640)):
        for ev in (None, 0, 127, 254):
            nb = R * C // BLOCK
            exps = (rng.integers(120, 132, nb, dtype=np.uint8)
                    if ev is None else np.full(nb, ev, np.uint8))
            buf = (exps.tobytes()
                   + (rng.standard_normal(16) * 0.3)
                   .astype(np.float16).tobytes()
                   + rng.integers(0, 256, R * C // 2,
                                  dtype=np.uint8).tobytes())
            ref = dec_s16(buf, [R, C])
            x = np.random.default_rng(29).standard_normal(
                C).astype(np.float32)
            y = S16Gpu(buf, [R, C]).gemv(
                torch.from_numpy(x).cuda()).cpu().numpy()
            r64 = ref.astype(np.float64) @ x.astype(np.float64)
            denom = max(1e-30, float(np.abs(r64).max()))
            rel = float(np.abs(y - r64).max() / denom)
            if rel > 1e-5:
                raise SystemExit(f"S16 GEMV FAIL {R}x{C} exp={ev} "
                                 f"rel={rel:.3e}")
            n_s16 += 1
    print(f"[r4] gates: {len(cases)} w4 decode bit-exact + gemv "
          f"<=1e-5; {n_s16} s16 gemv <=1e-5", flush=True)


def build():
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

    # module surgery: replace w4 Linears with fused modules
    n_fused = 0
    for i, lyr in enumerate(model.model.layers):
        for sub_nm, sub in list(lyr.named_modules()):
            if not isinstance(sub, torch.nn.Linear):
                continue
            full = (f"model.language_model.layers.{i}."
                    f"{sub_nm}.weight")
            e = man.get(full)
            if e is None or e["codec"] != "w4":
                continue
            assert sub.bias is None, f"unexpected bias on {full}"
            assert e["shape"][1] % BLOCK == 0 and e["shape"][1] % 4 == 0
            parent = lyr.get_submodule(sub_nm.rsplit(".", 1)[0]) \
                if "." in sub_nm else lyr
            setattr(parent, sub_nm.rsplit(".", 1)[-1],
                    FusedW4Linear(W4Gpu(payload(e), e["shape"])))
            n_fused += 1
    print(f"[r4] fused {n_fused} linears", flush=True)

    # everything else (non-layer resident set + remaining layer
    # params: norms/convs/etc) decodes once, resident fp32 cuda
    for nm, p in list(model.named_parameters()):
        if not p.is_meta:
            continue
        if nm in ("model.embed_tokens.weight", "lm_head.weight"):
            continue
        mod = model.get_submodule(nm.rsplit(".", 1)[0])
        setattr(mod, nm.rsplit(".", 1)[1], torch.nn.Parameter(
            cpu_decode(shard_key(nm)).cuda(), requires_grad=False))

    # io codec dispatch: w4 (artifact A) or s16 (B/C). Any other
    # codec refuses — never reinterpret bytes (the B incident).
    from llmopt.lab.qcodec_fast import S16Rows
    ee = man["model.language_model.embed_tokens.weight"]
    he = man["lm_head.weight"]
    io_rows = {"w4": W4Rows, "s16": S16Rows}
    for _nm, _e in (("embed_tokens", ee), ("lm_head", he)):
        if _e["codec"] not in io_rows:
            raise SystemExit(f"REFUSING: {_nm} codec {_e['codec']!r}"
                             " — w4/s16 io only")
    emb = io_rows[ee["codec"]](payload(ee), ee["shape"])

    def emb_fwd(input_ids):
        flat = input_ids.reshape(-1)
        out = torch.empty(flat.shape[0], emb.C)
        for j, t in enumerate(flat.tolist()):
            out[j] = torch.from_numpy(emb.rows(t, t + 1)[0])
        return out.reshape(*input_ids.shape, emb.C).cuda()

    model.model.embed_tokens.forward = emb_fwd
    if he["codec"] == "w4":
        head = FusedW4Linear(W4Gpu(payload(he), he["shape"]))

        def head_fwd(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:]          # last position only
            return head(x)
    else:                               # s16: GEMV-only head
        head_pay = S16Gpu(payload(he), he["shape"])

        def head_fwd(x):
            if x.dim() == 3 and x.shape[1] > 1:
                x = x[:, -1:]          # last position only
            lead = x.shape[:-1]
            flat = x.reshape(-1, x.shape[-1])
            assert flat.shape[0] == 1, "s16 head is GEMV-only"
            y = head_pay.gemv(flat[0].float())
            return y.reshape(*lead, -1).to(x.dtype)

    model.lm_head.forward = head_fwd

    # embed/lm_head params stay meta by design — their forwards are
    # overridden to compressed-resident paths above
    meta_left = [nm for nm, p in model.named_parameters()
                 if p.is_meta and nm not in (
                     "model.embed_tokens.weight", "lm_head.weight")]
    if meta_left:
        raise SystemExit(f"REFUSING: meta params left: {meta_left[:4]}")
    meta_bufs = [nm for nm, b in model.named_buffers() if b.is_meta]
    if meta_bufs:
        raise SystemExit(f"REFUSING: meta buffers: {meta_bufs[:4]}")
    rp = cfg.text_config.rope_parameters
    qrope.check_inv_freq(
        model.model.rotary_emb.inv_freq.cpu().numpy(),
        float(rp["rope_theta"]),
        int(cfg.text_config.head_dim
            * rp.get("partial_rotary_factor", 1.0)))
    model.model.rotary_emb.to("cuda")

    trav = {"attn_exec": {"linear_attn": 0, "full_attn": 0},
            "layer_calls": [0] * len(model.model.layers)}

    def count_attn(fam):
        def h(module, args, kwargs):
            trav["attn_exec"][fam] += 1
            return None
        return h

    def count_layer(i):
        def h(module, args, kwargs):
            trav["layer_calls"][i] += 1
            return None
        return h

    for i, lyr in enumerate(model.model.layers):
        lyr.register_forward_pre_hook(count_layer(i),
                                      with_kwargs=True)
        if hasattr(lyr, "self_attn"):
            lyr.self_attn.register_forward_pre_hook(
                count_attn("full_attn"), with_kwargs=True)
        else:
            lyr.linear_attn.register_forward_pre_hook(
                count_attn("linear_attn"), with_kwargs=True)
    return model, trav, n_fused


def main():
    from transformers import AutoTokenizer
    if os.path.exists(OUT):
        raise SystemExit(f"refuse: {OUT} exists — new run, new path")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    print("[r4] code_commit " + subprocess.check_output(
        ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        flush=True)
    _gates()

    tok = AutoTokenizer.from_pretrained(VDIR)
    text = PROMPT
    if CHAT:
        text = tok.apply_chat_template(
            [{"role": "user", "content": PROMPT}],
            tokenize=False, add_generation_prompt=True)
    ids = tok(text, return_tensors="pt")["input_ids"]

    free0 = torch.cuda.mem_get_info()[0]
    t0 = time.time()
    model, trav, n_fused = build()
    torch.cuda.synchronize()
    print(f"[r4] built in {time.time()-t0:.0f}s, free "
          f"{torch.cuda.mem_get_info()[0]/2**30:.2f} GiB", flush=True)

    t = time.time()
    out = model(input_ids=ids.cuda(), use_cache=False)
    torch.cuda.synchronize()
    wall_f = time.time() - t
    lg = out.logits[0, -1].float().cpu()
    assert torch.isfinite(lg).all(), "non-finite logits"
    calls = trav["layer_calls"]
    assert len(calls) == 64 and min(calls) > 0, "idle layer"
    assert trav["attn_exec"]["linear_attn"] >= 48 \
        and trav["attn_exec"]["full_attn"] >= 16, trav["attn_exec"]
    top = torch.topk(lg, 5)
    print(f"[r4] forward1 {wall_f:.2f}s | top5:", flush=True)
    for v, i in zip(top.values.tolist(), top.indices.tolist()):
        print(f"[r4]   {v:8.3f}  {tok.decode([i])!r}", flush=True)

    rec = {
        "rung": 4,
        "code_commit": subprocess.check_output(
            ["git", "rev-parse", "--short",
             "HEAD"]).decode().strip(),
        "device_actual": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "wsl": "microsoft" in os.uname().release.lower(),
        "artifact": os.path.basename(ART.rstrip("/")),
        "prompt": PROMPT, "seq_len": int(ids.shape[1]),
        "kernel_form": "fused",
        "n_fused_linears": n_fused,
        "logits_positions": "last_only",
        "attn_exec": trav["attn_exec"],
        "vram_free_at_start_bytes": int(free0),
        "forward1_s": round(wall_f, 3),
        "top1_id": int(top.indices[0]),
        "top1_margin": float(top.values[0] - top.values[1]),
        "env": {k: os.environ.get(k) for k in
                ("TORCH_DISABLE_NATIVE_JIT",
                 "PYTORCH_CUDA_ALLOC_CONF")},
    }

    if N_NEW > 0:
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
        spt = wall / max(len(gen), 1)
        print(f"[r4] gen {len(gen)} tok in {wall:.1f}s "
              f"({spt:.3f}s/tok = {1/spt:.2f} tok/s)", flush=True)
        print("[r4] OUTPUT:", txt, flush=True)
        rec["gen"] = {"n_new": int(len(gen)),
                      "wall_s": round(wall, 2),
                      "s_per_tok": round(spt, 4),
                      "tok_s": round(1 / spt, 2), "output": txt}

    rec["vram_peak_alloc_bytes"] = int(torch.cuda.max_memory_allocated())
    rec["vram_peak_reserved_bytes"] = int(
        torch.cuda.max_memory_reserved())
    with open(OUT, "x") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    print("[r4] receipt ->", OUT, flush=True)


if __name__ == "__main__":
    main()
