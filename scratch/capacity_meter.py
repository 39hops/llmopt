"""THE CAPACITY METER (pre-reg 2026-07-29 late night): M =
span_bits - code_entropy at per-row step sigma/2 — the fixed-width
penalty the sigma grid pays to the worst outlier. At-capacity
(Gaussian) weights: M ~ 1.5-2 bits; heavy tails inflate M.
Decision rule: M small -> sigma-law allocator; M large ->
max-anchored/calibrated. Cells: house crystals, SmolLM2-1.7B,
DeepSeek-V3 layer-30 routed experts (fp8 block-dequant, as the
07-17 gauge). Desk only. Run on 3080 with MODELS=qwen for the
Qwen cell. __main__-guarded.
"""
import glob
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, ".")


def meter(w):
    """w [out, in] fp -> (M bits, kurtosis), per-row sigma/2 step."""
    wf = w.float()
    s = wf.std(dim=1, keepdim=True).clamp(min=1e-8)
    codes = torch.round(wf * torch.ceil(2.0 / s))
    span = int(codes.max() - codes.min()) + 1
    span_bits = max(1, math.ceil(math.log2(span)))
    _, cnt = np.unique(codes.numpy().ravel(), return_counts=True)
    p = cnt / cnt.sum()
    ent = float(-(p * np.log2(p)).sum())
    k = float(((wf - wf.mean()) ** 4).mean() / wf.var() ** 2)
    return span_bits - ent, k


def report(tag, tensors):
    tot = wm = wk = 0
    for w in tensors:
        m, k = meter(w)
        n = w.numel()
        wm += m * n
        wk += k * n
        tot += n
    print(f"METER {tag}: M = {wm / tot:.2f} bits | "
          f"kurtosis {wk / tot:.2f} | {tot / 1e6:.1f}M params",
          flush=True)


def house(path, keys=None):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return [v for k, v in sd.items()
            if v.ndim == 2 and k.startswith("blocks.")
            and (keys is None or any(s in k for s in keys))]


def hf_linears(st_path, want, limit=64):
    from safetensors import safe_open
    out = []
    with safe_open(st_path, framework="pt") as f:
        names = [n for n in f.keys() if want(n)]
        for n in names[:limit]:
            out.append((n, f.get_tensor(n)))
    return out


def main():
    which = os.environ.get("MODELS", "mac")
    if which == "qwen":
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)
        report("Qwen2.5-0.5B", [
            mm.weight.detach() for n, mm in m.named_modules()
            if isinstance(mm, torch.nn.Linear) and "lm_head" not in n])
        return

    report("crystal d64h8", house(
        "checkpoints/sym_birth_dense_mps_h8_ema.pt"))
    report("crystal L4d56", house(
        "checkpoints/sym_birth_dense_mps_L4_ema.pt"))
    report("crystal cplx_none(gate/up)", house(
        "checkpoints/cplx_none.pt", keys=("gate", "up")))

    smol = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--HuggingFaceTB--"
        "SmolLM2-1.7B-Instruct/snapshots/*/model.safetensors"))[0]
    ts = hf_linears(smol, lambda n: n.endswith(".weight")
                    and ("mlp" in n or "self_attn" in n), limit=48)
    report("SmolLM2-1.7B", [w.float() for _, w in ts])

    ds = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V3/"
        "snapshots/*/model-00076-of-000163.safetensors"))[0]
    from safetensors import safe_open
    with safe_open(ds, framework="pt") as f:
        names = [n for n in f.keys()
                 if "experts" in n and n.endswith("weight")
                 and "scale_inv" not in n and "shared" not in n]
        exp = []
        for n in names[:48]:
            w = f.get_tensor(n).float()
            try:  # fp8 block dequant (128x128 block scales, as 07-17)
                sc = f.get_tensor(n + "_scale_inv").float()
                bo = (w.shape[0] + sc.shape[0] - 1) // sc.shape[0]
                bi = (w.shape[1] + sc.shape[1] - 1) // sc.shape[1]
                w = w * sc.repeat_interleave(bo, 0)[:w.shape[0]] \
                    .repeat_interleave(bi, 1)[:, :w.shape[1]]
            except Exception:
                pass
            exp.append(w)
    report("DeepSeek-V3 L30 experts", exp)


if __name__ == "__main__":
    main()
