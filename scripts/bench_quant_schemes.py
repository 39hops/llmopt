"""Three-lane 4-bit quantization race on REAL model weights, scored in
FUNCTION space (the house law: never score weights by weight distance).

Lanes (all 16 levels, group_size along the input dim):
  uniform : per-group min/max affine grid — minimax-optimal in weight
            space (caps the worst-case distance; Artin's reallocation)
  nf4     : code points at gaussian quantiles, per-group absmax —
            allocate accuracy where the weight MASS is (QLoRA's prior)
  awq_lite: per-input-channel rescale s_c = mean|x_c|^alpha before a
            uniform quant — allocate accuracy where the OUTPUT cares
            (activation-aware; the law's own answer)

Score: y = x @ W.T on real captured activations; rel-err vs fp32.
Origin: 2026-07-11 Cerebras riff -> int4 dequant-GEMV rung; toy round
measured uniform > NF4 in function space on random gaussians — this
is the real-weights rematch with the third lane added.
"""
from __future__ import annotations

import numpy as np

GROUP = 128          # Artin's call (practice_7): metadata nearly free
ALPHA = 0.5          # AWQ's default importance exponent
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PROMPTS = [
    "Find an antiderivative of: 3*x**2*cos(x**3)",
    "Explain why the sky is blue in one sentence.",
    "def fib(n):",
    "The integral of 1/x is",
]


def capture(n_layers: int = 6):
    """fp32 weights + real input activations for a spread of linears."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
    model.eval()
    names = []
    for i in (0, 6, 12, 18, 23):
        names += [f"model.layers.{i}.self_attn.q_proj",
                  f"model.layers.{i}.mlp.down_proj"]
    mods = dict(model.named_modules())
    acts: dict = {}
    hooks = []
    for n in names:
        def mk(n):
            def h(_m, inp, _out):
                x = inp[0].detach().reshape(-1, inp[0].shape[-1])
                acts.setdefault(n, []).append(x)
            return h
        hooks.append(mods[n].register_forward_hook(mk(n)))
    with torch.no_grad():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            model(ids)
    for h in hooks:
        h.remove()
    out = []
    for n in names:
        w = mods[n].weight.detach().numpy().astype(np.float32)
        x = torch.cat(acts[n]).numpy().astype(np.float32)
        out.append((n, w, x))
    return out


def _group(w):  # (N, D) -> (N, D//GROUP, GROUP)
    return w.reshape(w.shape[0], -1, GROUP)


def quant_uniform(w):
    g = _group(w)
    mn = g.min(2, keepdims=True)
    sc = (g.max(2, keepdims=True) - mn) / 15.0
    sc[sc == 0] = 1.0
    q = np.clip(np.round((g - mn) / sc), 0, 15)
    return (q * sc + mn).reshape(w.shape)


_NF4 = None


def _nf4_codes():
    global _NF4
    if _NF4 is None:
        big = np.sort(np.random.default_rng(0).normal(size=2_000_000))
        qs = np.linspace(0.5 / 16, 1 - 0.5 / 16, 16)
        c = big[(qs * len(big)).astype(int)].astype(np.float32)
        _NF4 = c / np.abs(c).max()
    return _NF4


def quant_nf4(w):
    codes = _nf4_codes()
    g = _group(w)
    am = np.abs(g).max(2, keepdims=True)
    am[am == 0] = 1.0
    idx = np.abs((g / am)[..., None] - codes).argmin(-1)
    return (codes[idx] * am).reshape(w.shape)


def quant_awq_lite(w, x):
    # per-input-channel importance from real calibration activations
    s = np.abs(x).mean(0) ** ALPHA
    s[s == 0] = 1.0
    wd = quant_uniform(w * s)          # protected channels shrink error
    return wd / s                      # fold the scale back (exact fp)


def main():
    rows = capture()
    print(f"{'layer':44s} {'uniform':>9s} {'nf4':>9s} {'awq_lite':>9s}")
    tot = {"uniform": [], "nf4": [], "awq_lite": []}
    for n, w, x in rows:
        y0 = x @ w.T
        den = np.abs(y0).mean()
        errs = {}
        for name, wd in [("uniform", quant_uniform(w)),
                         ("nf4", quant_nf4(w)),
                         ("awq_lite", quant_awq_lite(w, x))]:
            e = np.abs(x @ wd.T - y0).mean() / den * 100
            errs[name] = e
            tot[name].append(e)
        print(f"{n:44s} {errs['uniform']:8.3f}% {errs['nf4']:8.3f}% "
              f"{errs['awq_lite']:8.3f}%")
    print("-" * 76)
    means = {k: float(np.mean(v)) for k, v in tot.items()}
    print(f"{'MEAN output rel-err':44s} "
          + " ".join(f"{means[k]:8.3f}%" for k in ("uniform", "nf4", "awq_lite")))
    best = min(means, key=means.get)
    print(f"WINNER (function space, real weights + real activations): {best}")
    print("bar: the winning lane is the packing the Metal kernel carries")


if __name__ == "__main__":
    main()
