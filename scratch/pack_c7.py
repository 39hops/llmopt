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


def group_of(name):
    if ".experts." in name:
        return "EXPERTS"
    if "self_attn" in name:
        return "ATTN"
    return None


def main():
    """Streaming design (v2 after the OOM kill): the model is
    RELOADED from disk for every arm and quantized IN PLACE tensor
    by tensor — no clones of the 6.4B expert params ever exist."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = "mps"
    tok = AutoTokenizer.from_pretrained(MODEL)
    text = open("README.md", encoding="utf-8").read()[:8000]

    def load():
        return AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.float16).to(dev).eval()

    def prompts_ids():
        return ([tok(p, return_tensors="pt").input_ids.to(dev)
                 for p in PROMPTS],
                tok(text, return_tensors="pt").input_ids[:, :1024].to(dev))

    model = load()
    lin = {n: m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    for g in ("EXPERTS", "ATTN"):
        d = [m for n, m in lin.items() if group_of(n) == g]
        tot = wm = wk = 0
        for m in d[:96]:
            mm, kk = meter(m.weight.detach().float().cpu())
            c = m.weight.numel()
            wm += mm * c
            wk += kk * c
            tot += c
        print(f"C7 METER {g}: M = {wm / tot:.2f} bits | "
              f"kurt {wk / tot:.2f} ({tot / 1e6:.0f}M of "
              f"{sum(m.weight.numel() for m in d) / 1e6:.0f}M, "
              f"{len(d)} linears)", flush=True)

    pids, ids = prompts_ids()
    with torch.no_grad():
        fp_logits = [model(p).logits.float().cpu() for p in pids]
        fp_loss = float(model(ids, labels=ids).loss)
    print(f"C7 fp16 control: ppl {math.exp(fp_loss):.3f}", flush=True)
    del model, lin

    MB = {"EXPERTS": 6, "ATTN": 6}  # measured 5.85 first pass

    def run_arm(g, tag, fn):
        m = load()
        t0 = time.time()
        bits_sum = pn = 0
        for n, mod in m.named_modules():
            if not (isinstance(mod, torch.nn.Linear)
                    and group_of(n) == g):
                continue
            w = mod.weight.detach().float().cpu()
            if tag == "sigma[row]":
                wq, b = sigma_row(w)
                bits_sum += b * w.numel()
                pn += w.numel()
            else:
                wq = fn(w)
            mod.weight.data.copy_(wq.to(dev).half())
            del w, wq
        t_q = time.time() - t0
        if pn:
            print(f"C7 {g} sigma[row] avg raw bits "
                  f"{bits_sum / pn:.2f}", flush=True)
        pids, ids = prompts_ids()
        kl = n = 0.0
        with torch.no_grad():
            for p, ref in zip(pids, fp_logits):
                lg = m(p).logits.float().cpu()
                lp, rp = lg.log_softmax(-1), ref.log_softmax(-1)
                kl += float((rp.exp() * (rp - lp)).sum())
                n += lg.shape[1]
            loss = float(m(ids, labels=ids).loss)
        print(f"C7 {g} {tag}: DeltaKL {kl / n:.4f}/tok | "
              f"ppl {math.exp(loss):.3f} | quant {t_q:.1f}s",
              flush=True)
        del m

    for g in ("EXPERTS", "ATTN"):
        run_arm(g, "sigma[row]", None)
        run_arm(g, "rtn", lambda w, g=g: rtn(w, MB[g]))
        run_arm(g, "hqq", lambda w, g=g: hqq(w, MB[g], group_size=64))


if __name__ == "__main__":
    main()
