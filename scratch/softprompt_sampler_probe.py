"""Why a bit-identical model gates differently behind the virtual-token
harness (AMENDMENT SOFT-PROMPT-1-SAMPLER).

Two independent checks, both cheap and CPU-only:

PART 1 — the model. Build a stock mathnative model, save it, load it
back through softprompt1.with_virtual_tokens, and compare. Reports the
max logit difference on ordinary ids, argmax agreement, the virtual
logit ceiling, and a per-tensor state-dict comparison. Expected: exact
agreement, which clears the model of the discrepancy.

PART 2 — the sampler. Compare torch.multinomial over a 40-category
probability vector against the same vector with 8 exact zeros appended.
The distribution over real tokens is identical; the question is whether
the DRAW is. A single draw from a fresh generator agrees; sequential
draws from one generator do not, because the number of random values
consumed depends on the category count. sample_wave_lp reuses one
generator for up to max_new=120 tokens per rollout.

Usage: .venv/bin/python scratch/softprompt_sampler_probe.py
"""
import importlib.util
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, "scripts")

from llmopt.train.mathnative import MathTokenizer, build_model  # noqa: E402

D, LAYERS, FFN, HEADS = 64, 8, 256, 4


def _load_softprompt():
    spec = importlib.util.spec_from_file_location(
        "sp1", os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "softprompt1.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def part1_model() -> None:
    sp1 = _load_softprompt()
    tok = MathTokenizer()
    V = len(tok.vocab)
    torch.manual_seed(0)
    stock = build_model(V, d=D, layers=LAYERS, heads=HEADS, ffn=FFN).eval()
    f = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    torch.save(stock.state_dict(), f.name)
    try:
        big, _, V2, pref = sp1.with_virtual_tokens(
            f.name, D, LAYERS, FFN, HEADS, "cpu")
    finally:
        os.unlink(f.name)
    big.eval()
    assert V == V2

    ids = torch.tensor([tok.encode("Integral(x**2, x)", False)])
    mask = torch.ones_like(ids)
    with torch.no_grad():
        a, b = stock(ids, mask), big(ids, mask)
    print(f"[model] V={V} P={len(pref)} "
          f"stock{tuple(a.shape)} big{tuple(b.shape)}")
    print(f"[model] max |stock - big[:V]| = "
          f"{(a - b[..., :V]).abs().max().item()}")
    print(f"[model] virtual logit ceiling = {b[..., V:].max().item()}")
    print(f"[model] argmax equal everywhere = "
          f"{bool((a.argmax(-1) == b[..., :V].argmax(-1)).all())}")

    sa, sb = stock.state_dict(), big.state_dict()
    differing = []
    for k in sorted(sa):
        x, y = sa[k], sb[k]
        if y.shape != x.shape and y.shape[0] == V + len(pref):
            y = y[:V]
        if not torch.equal(x, y):
            differing.append(k)
    print(f"[model] tensors compared = {len(sa)}, differing = {differing}")
    print(f"[model] emb tied to head = "
          f"{sa.get('emb.weight') is sa.get('head.weight')}")


def part2_sampler(n_seeds: int = 200, rollout: int = 30) -> None:
    torch.manual_seed(1)
    p40 = torch.softmax(torch.randn(40) * 3 / 0.7, -1)
    p48 = torch.cat([p40, torch.zeros(8)])
    print(f"\n[sampler] first-40 probabilities identical = "
          f"{torch.equal(p40, p48[:40])}")

    mism = sum(
        int(torch.multinomial(
            p40, 1, generator=torch.Generator("cpu").manual_seed(s)))
        != int(torch.multinomial(
            p48, 1, generator=torch.Generator("cpu").manual_seed(s)))
        for s in range(n_seeds))
    print(f"[sampler] fresh-generator single draws mismatching: "
          f"{mism}/{n_seeds}")

    g1 = torch.Generator("cpu").manual_seed(7)
    g2 = torch.Generator("cpu").manual_seed(7)
    s1 = [int(torch.multinomial(p40, 1, generator=g1)) for _ in range(rollout)]
    s2 = [int(torch.multinomial(p48, 1, generator=g2)) for _ in range(rollout)]
    print(f"[sampler] sequential rollout identical = {s1 == s2}")
    if s1 != s2:
        i = next(i for i in range(rollout) if s1[i] != s2[i])
        print(f"[sampler] first divergence at draw {i}: {s1[i]} v {s2[i]}")
        print(f"[sampler]   40-way {s1[:12]}")
        print(f"[sampler]   48-way {s2[:12]}")


if __name__ == "__main__":
    part1_model()
    part2_sampler()
