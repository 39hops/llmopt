"""Booked-number battery for the gate module (Phase 3 module 5).

Replays AMENDMENT SOFT-PROMPT-1-SAMPLER (RESULTS.md L27693 block,
2026-08-12, Mac): the sampler at scripts/step_grpo_micro.py:65 is
category-count-sensitive — a single draw from a FRESH generator
agrees 0/200 when the probability vector is widened with exact
zeros, while consecutive draws from ONE shared generator
desynchronize (CPU multinomial consumes an amount of the random
stream that depends on the category count). Plus the gate
problem-set pins: the standard 120 gate's seed grid
(GATE_BAND + 1000*lv + i) is deterministic under the string-seed
law; the literals below were frozen 2026-08-12 by
scratch/gatepins_freeze.py.
"""
import pytest

torch = pytest.importorskip("torch")


def _probs(n, seed):
    g = torch.Generator().manual_seed(seed)
    p = torch.rand(n, generator=g)
    return p / p.sum()


def test_fresh_generator_single_draw_agrees_200():
    # booked: "a single draw from a FRESH generator agrees:
    # 0 mismatches in 200" (RESULTS.md, SOFT-PROMPT-1-SAMPLER)
    mismatches = 0
    for s in range(200):
        p40 = _probs(40, 1000 + s)
        p48 = torch.cat([p40, torch.zeros(8)])
        a = int(torch.multinomial(
            p40, 1, generator=torch.Generator().manual_seed(s)))
        b = int(torch.multinomial(
            p48, 1, generator=torch.Generator().manual_seed(s)))
        mismatches += a != b
    assert mismatches == 0


def test_shared_generator_stream_desyncs():
    # booked mechanism: category count changes how much of the
    # random stream each draw consumes, so identical distributions
    # diverge on LATER draws from a shared generator
    ga = torch.Generator().manual_seed(7)
    gb = torch.Generator().manual_seed(7)
    seq_a, seq_b = [], []
    for s in range(12):
        p40 = _probs(40, 2000 + s)
        p48 = torch.cat([p40, torch.zeros(8)])
        seq_a.append(int(torch.multinomial(p40, 1, generator=ga)))
        seq_b.append(int(torch.multinomial(p48, 1, generator=gb)))
    assert seq_a != seq_b


GATE_PINS = {
    # level: sstr of _gen_isolated(lv, GATE_BAND + 1000*lv + 0),
    # frozen 2026-08-12 by scratch/gatepins_freeze.py — pins the
    # standard gate's seed arithmetic + band + generator together
    3: '24*x**2 - 7',
    4: '2*(3*x*cos(x**3 + 2*x + 2) - (3*x**2 + 2)**2'
       '*sin(x**3 + 2*x + 2))*cos(x**3 + 2*x + 2)',
    5: '(108*x**5 + 72*x**3 + 54*x**2 + 9*x + 9)/sqrt(2*x**3 + x)',
    6: '2*(3*x**4 + 24*x**3 + 54*x**2 + 47*x + 13)'
       '/(x**4 + 8*x**3 + 22*x**2 + 24*x + 9)',
    7: '-4*log(x)*sin(2*x) + 2*cos(2*x)/x',
}


def test_gate_problem_grid_pinned():
    sp = pytest.importorskip("sympy")
    from llmopt.lab.gate import GATE_BAND
    from llmopt.lab.gen import _gen_isolated
    for lv, want in GATE_PINS.items():
        p = _gen_isolated(lv, GATE_BAND + 1000 * lv + 0)
        assert sp.sstr(p._expr) == want, lv
