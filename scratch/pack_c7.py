"""PACKED CRYSTAL C7 (pre-reg 2026-07-29 late, Artin's GO): the
at-capacity transport claim. OLMoE-1B-7B: sigma[row] v rtn v hqq
fake-quant on ROUTED EXPERT tensors (control arm: same on dense
attention tensors). Capacity meter reads both groups first.
DeltaKL on 16 fixed prompts + README-slice ppl + wall-times.
Mac 36GB / MPS. __main__-guarded.
"""
import math
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scratch")
import torch  # noqa: E402

from capacity_meter import meter  # noqa: E402
from llmopt.quantize.methods import hqq, rtn  # noqa: E402
from pack_c6 import PROMPTS  # noqa: E402

MODEL = "allenai/OLMoE-1B-7B-0924-Instruct"


def sigma_row(w):
    wf = w.float()
    s = wf.std(dim=1, keepdim=True).clamp(min=1e-8)
    q = torch.ceil(2.0 / s)
    codes = torch.round(wf * q)
    span = int(codes.max() - codes.min()) + 1
    return codes / q, max(1, math.ceil(math.log2(span)))


def main():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.float16).to(dev).eval()

    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    groups = {
        "EXPERTS": {n: m for n, m in lin.items() if ".experts." in n},
        "ATTN": {n: m for n, m in lin.items() if "self_attn" in n},
    }
    for g, d in groups.items():
        print(f"C7 group {g}: {len(d)} linears, "
              f"{sum(m.weight.numel() for m in d.values()) / 1e6:.0f}M",
              flush=True)

    # capacity meter per group (prediction 1) — sample for wall-time
    for g, d in groups.items():
        tot = wm = wk = 0
        for n, m in list(d.items())[:96]:
            mm, kk = meter(m.weight.detach().float().cpu())
            c = m.weight.numel()
            wm += mm * c
            wk += kk * c
            tot += c
        print(f"C7 METER {g}: M = {wm / tot:.2f} bits | "
              f"kurt {wk / tot:.2f} ({tot / 1e6:.0f}M read)", flush=True)

    text = open("README.md", encoding="utf-8").read()[:8000]
    ids = tok(text, return_tensors="pt").input_ids[:, :1024].to(dev)
    pids = [tok(p, return_tensors="pt").input_ids.to(dev)
            for p in PROMPTS]
    with torch.no_grad():
        fp_logits = [model(p).logits.float().cpu() for p in pids]
        fp_loss = float(model(ids, labels=ids).loss)
    print(f"C7 fp16 control: ppl {math.exp(fp_loss):.3f}", flush=True)

    @torch.no_grad()
    def score(tag, t_q):
        kl = n = 0.0
        for p, ref in zip(pids, fp_logits):
            lg = model(p).logits.float().cpu()
            lp, rp = lg.log_softmax(-1), ref.log_softmax(-1)
            kl += float((rp.exp() * (rp - lp)).sum())
            n += lg.shape[1]
        loss = float(model(ids, labels=ids).loss)
        print(f"C7 {tag}: DeltaKL {kl / n:.4f}/tok | "
              f"ppl {math.exp(loss):.3f} | quant {t_q:.1f}s", flush=True)

    for g, d in groups.items():
        orig = {n: m.weight.detach().clone() for n, m in d.items()}
        # sigma arm measures bits -> sets matched bits for rtn/hqq
        t0 = time.time()
        bits_sum = pn = 0
        packs = {}
        for n, w in orig.items():
            wq, b = sigma_row(w.cpu())
            packs[n] = wq
            bits_sum += b * w.numel()
            pn += w.numel()
        t_sig = time.time() - t0
        mb = max(2, round(bits_sum / pn))
        print(f"C7 {g} sigma[row] avg raw bits {bits_sum / pn:.2f} "
              f"-> matched {mb}", flush=True)
        for n, m in d.items():
            m.weight.data.copy_(packs[n].to(dev).half())
        score(f"{g} sigma[row]", t_sig)
        del packs
        for tag, fn in (("rtn", lambda w: rtn(w, mb)),
                        ("hqq", lambda w: hqq(w, mb, group_size=64))):
            t0 = time.time()
            for n, m in d.items():
                m.weight.data.copy_(
                    fn(orig[n].float().cpu()).to(dev).half())
            score(f"{g} {tag}", time.time() - t0)
        for n, m in d.items():  # restore before next group
            m.weight.data.copy_(orig[n])
        del orig


if __name__ == "__main__":
    main()
