"""P2a-v2 THE ANALYTIC-CLIP ALLOCATOR (pre-reg 2026-07-29 close):
zero-calibration span attack on SmolLM2-1.7B (Mac). Arms: rtn
per-row absmax | sigma-clip k in {4,6,8} (grid over +-min(absmax,
k*sigma), outliers saturate) | hqq. DeltaKL + README ppl +
wall-time, C6 harness form. __main__-guarded.
"""
import math
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from llmopt.quantize.methods import hqq  # noqa: E402
from pack_c6 import PROMPTS  # noqa: E402

MODEL = "HuggingFaceTB/SmolLM2-1.7B-Instruct"
BITS = 6


def grid_q(w, rng, bits):
    """Per-row symmetric uniform grid over [-rng, rng]; saturate."""
    qmax = 2 ** (bits - 1) - 1
    scale = rng.clamp(min=1e-8) / qmax
    return (w / scale).round().clamp(-qmax, qmax) * scale


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()
    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    orig = {n: m.weight.detach().to("cpu", torch.float16)
            for n, m in lin.items()}
    print(f"P2a {len(lin)} linears, "
          f"{sum(w.numel() for w in orig.values()) / 1e6:.0f}M | "
          f"bits {BITS}", flush=True)

    text = open("README.md", encoding="utf-8").read()[:8000]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to(dev)
    pids = [tok(p, return_tensors="pt").input_ids.to(dev)
            for p in PROMPTS]
    with torch.no_grad():
        fp_logits = [model(p).logits.float().cpu() for p in pids]
        fp_loss = float(model(ids, labels=ids).loss)
    print(f"P2a fp16 control: ppl {math.exp(fp_loss):.3f}", flush=True)

    @torch.no_grad()
    def score(tag, t_q):
        kl = n = 0.0
        for p, ref in zip(pids, fp_logits):
            lg = model(p).logits.float().cpu()
            lp, rp = lg.log_softmax(-1), ref.log_softmax(-1)
            kl += float((rp.exp() * (rp - lp)).sum())
            n += lg.shape[1]
        loss = float(model(ids, labels=ids).loss)
        print(f"P2a {tag}: DeltaKL {kl / n:.4f}/tok | "
              f"ppl {math.exp(loss):.3f} | quant {t_q:.1f}s",
              flush=True)

    def arm(tag, fn):
        t0 = time.time()
        for n, m in lin.items():
            w = orig[n].float()
            m.weight.data.copy_(fn(w).to(dev).half())
        score(tag, time.time() - t0)

    def clip_fn(k):
        def f(w):
            am = w.abs().amax(dim=1, keepdim=True)
            s = w.std(dim=1, keepdim=True)
            return grid_q(w, torch.minimum(am, k * s), BITS)
        return f

    arm("rtn(absmax)", lambda w: grid_q(
        w, w.abs().amax(dim=1, keepdim=True), BITS))
    for k in (4.0, 6.0, 8.0):
        arm(f"sigma-clip k={k:g}", clip_fn(k))
    arm("hqq", lambda w: hqq(w, BITS, group_size=64))


if __name__ == "__main__":
    main()
