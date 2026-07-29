"""PACKED CRYSTAL C6 (pre-reg 2026-07-29 night, Artin's GO): external
validity on Qwen2.5-0.5B (3080). Arms: sigma-pack (q=ceil(2/sigma)
per tensor, closed form) v HQQ (matched bits, group 64) v RTN.
Score: mean DeltaKL/token v fp16 on 16 fixed prompts + perplexity
on a fixed README slice + per-arm quantization wall-time.
Fake-quant only. __main__-guarded.
"""
import math
import sys
import time

sys.path.insert(0, ".")
import torch  # noqa: E402

from llmopt.quantize.methods import hqq, rtn  # noqa: E402

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "The derivative of x**3 is",
    "Newton's second law states that",
    "The integral of cos(x) dx equals",
    "In a vacuum, all objects fall",
    "The eigenvalues of a symmetric matrix are",
    "Energy conservation means that",
    "The Taylor series of exp(x) is",
    "A prime number is defined as",
    "The speed of light in vacuum is",
    "Angular momentum is conserved when",
    "The determinant of a 2x2 matrix",
    "Simple harmonic motion has period",
    "The chain rule says d/dx f(g(x)) =",
    "Entropy of an isolated system",
    "The quadratic formula gives roots",
    "Maxwell's equations describe",
]


def sigma_pack(w):
    s = float(w.float().std())
    q = math.ceil(2.0 / max(s, 1e-8))
    codes = torch.round(w.float() * q)
    span = int(codes.max()) - int(codes.min()) + 1
    return codes / q, max(1, math.ceil(math.log2(span)))


def sigma_pack_row(w):
    """C6b: per-output-row sigma — still closed-form, zero
    calibration; metadata = one scale per row (rtn/hqq class)."""
    wf = w.float()
    s = wf.std(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.ceil(2.0 / s)
    codes = torch.round(wf * q)
    span = int(codes.max() - codes.min()) + 1
    return codes / q, max(1, math.ceil(math.log2(span)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()

    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    orig = {n: m.weight.detach().clone() for n, m in lin.items()}
    print(f"C6 {len(lin)} linears, "
          f"{sum(w.numel() for w in orig.values()) / 1e6:.1f}M params",
          flush=True)

    text = open("README.md", encoding="utf-8").read()[:8000]
    ids = tok(text, return_tensors="pt").input_ids[:, :2048].to(dev)
    pids = [tok(p, return_tensors="pt").input_ids.to(dev)
            for p in PROMPTS]

    @torch.no_grad()
    def score(tag, t_q):
        kl = n = 0.0
        for p, ref in zip(pids, fp_logits):
            lg = model(p).logits.float()
            lp, rp = lg.log_softmax(-1), ref.log_softmax(-1)
            kl += float((rp.exp() * (rp - lp)).sum())
            n += lg.shape[1]
        loss = float(model(ids, labels=ids).loss)
        print(f"C6 {tag}: DeltaKL {kl / n:.4f}/tok | "
              f"ppl {math.exp(loss):.3f} | quant {t_q:.1f}s",
              flush=True)

    with torch.no_grad():
        fp_logits = [model(p).logits.float() for p in pids]
        fp_loss = float(model(ids, labels=ids).loss)
    print(f"C6 fp16 control: ppl {math.exp(fp_loss):.3f}", flush=True)

    # sigma-pack (measures its own avg bits -> sets matched bits)
    import os
    arm = os.environ.get("ARM", "tensor")  # tensor | row (C6b)
    pack_fn = sigma_pack_row if arm == "row" else sigma_pack
    t0 = time.time()
    bits_sum = pn = 0
    packs = {}
    for name, w in orig.items():
        wq, b = pack_fn(w.cpu())
        packs[name] = wq
        bits_sum += b * w.numel()
        pn += w.numel()
    t_sig = time.time() - t0
    avg_bits = bits_sum / pn
    mb = max(2, round(avg_bits))
    print(f"C6 sigma-pack[{arm}] avg raw bits {avg_bits:.2f} "
          f"-> matched bits {mb}", flush=True)
    for name, m in lin.items():
        m.weight.data.copy_(packs[name].to(dev).half())
    score(f"sigma-pack[{arm}]", t_sig)

    for tag, fn in (("rtn", lambda w: rtn(w, mb)),
                    ("hqq", lambda w: hqq(w, mb, group_size=64))):
        t0 = time.time()
        for name, m in lin.items():
            m.weight.data.copy_(fn(orig[name].float().cpu()).to(dev).half())
        score(tag, time.time() - t0)

    for name, m in lin.items():  # restore
        m.weight.data.copy_(orig[name])


if __name__ == "__main__":
    main()
