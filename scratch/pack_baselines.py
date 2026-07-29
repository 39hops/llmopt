"""PACKED CRYSTAL C3 (pre-reg 2026-07-29 eve): GPTQ/AWQ/HQQ honest
table on d64h8 EMA. Baselines from llmopt/quantize/methods.py on
every block Linear; calibration activations hooked from 24 prompts
at GATE_BAND+500_000 seed offsets (never the gate band). Arms:
{rtn,gptq,awq,hqq} x {5,3} bits -> full gate + mean DeltaKL v fp
logits + calibration wall-time. C1 controls reused (fp 58, packed
58). __main__-guarded.
"""
import sys
import time

sys.path.insert(0, ".")
sys.path.insert(0, "scripts")
import sympy as sp  # noqa: E402
import torch  # noqa: E402

import step_grpo_micro as G  # noqa: E402
from bench_step_tokens import _gen_isolated  # noqa: E402
from llmopt.quantize.methods import awq, gptq, hqq, rtn  # noqa: E402
from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

CKPT = "checkpoints/sym_birth_dense_mps_h8_ema.pt"
CFG = dict(d=64, layers=8, heads=8, ffn=256)
LIN = ("qkv", "o", "gate", "up", "down")
CAL_OFF = 500_000


def calib_prompts(tok):
    ids = []
    for lv in G.GATE_LEVELS:
        for i in range(5):
            p = _gen_isolated(lv, G.GATE_BAND + CAL_OFF + 1000 * lv + i)
            if p is None:
                continue
            cur = f"Integral({sp.sstr(p._expr)}, x)"
            ids.append(tok.encode(f"Current: {cur}\nHints: none\nStep: "))
    return ids[:24]


def capture(model, prompts, dev):
    """-> {key: X [n, in]} inputs of every block Linear + fp logits."""
    acts, hooks = {}, []
    for name, mod in model.named_modules():
        if any(name.endswith("." + s) for s in LIN):
            def pre(m, inp, key=name):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1]).float()
                acts.setdefault(key, []).append(x.cpu())
            hooks.append(mod.register_forward_pre_hook(pre))
    logits = []
    with torch.no_grad():
        for p in prompts:
            out = model(torch.tensor([p], device=dev))
            logits.append(out.detach().float().cpu())
    for h in hooks:
        h.remove()
    return {k: torch.cat(v) for k, v in acts.items()}, logits


def main():
    tok = MathTokenizer()
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    base = torch.load(CKPT, map_location="cpu", weights_only=True)

    def build(sd):
        m = build_model(len(tok.vocab), **CFG).to(dev)
        m.load_state_dict({k: v.to(dev) for k, v in sd.items()})
        m.eval()
        return m

    prompts = calib_prompts(tok)
    t0 = time.time()
    fp = build(base)
    acts, fp_logits = capture(fp, prompts, dev)
    del fp
    t_cal = time.time() - t0
    print(f"C3 calibration pass: {len(prompts)} prompts, "
          f"{sum(a.shape[0] for a in acts.values())} rows, "
          f"{t_cal:.1f}s", flush=True)

    def quantize(method, bits):
        sd = {k: v.clone() for k, v in base.items()}
        t0 = time.time()
        for k in list(sd):
            if not (k.endswith(".weight") and k.startswith("blocks.")
                    and sd[k].ndim == 2):
                continue
            key = k[:-len(".weight")]
            w = sd[k].float()
            if method == "rtn":
                sd[k] = rtn(w, bits)
            elif method == "hqq":
                sd[k] = hqq(w, bits)
            elif method == "gptq":
                x = acts[key]
                sd[k] = gptq(w, x.T @ x, bits)
            elif method == "awq":
                sd[k] = awq(w, acts[key], bits)[0]
        return sd, time.time() - t0

    def dkl(sd):
        m = build(sd)
        tot = n = 0.0
        with torch.no_grad():
            for p, ref in zip(prompts, fp_logits):
                lg = m(torch.tensor([p], device=dev)).float().cpu()
                lp, rp = lg.log_softmax(-1), ref.log_softmax(-1)
                tot += float((rp.exp() * (rp - lp)).sum())
                n += lg.shape[1]
        del m
        return tot / n

    for bits in (5, 3):
        for method in ("rtn", "gptq", "awq", "hqq"):
            sd, t_m = quantize(method, bits)
            kl = dkl(sd)
            m = build(sd)
            with torch.no_grad():
                solves, valid = G.gate_eval(m, tok, dev)
            del m
            print(f"C3 {method}{bits}: {sum(solves.values())}/120 "
                  f"@ {valid:.2f}% | DeltaKL {kl:.4f}/tok | "
                  f"opt {t_m:.1f}s (+cal {t_cal:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
