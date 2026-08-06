"""Guards for llmopt/lab/traj.py (module 4 — UNIFICATION, not verbatim
adoption; see the module docstring's divergence/authority table).

Tiers here (desk-level, no 30B):
  - pure: import safety off-Metal, the D-1 traj+keep refusal.
  - synthetic parity (skips cleanly without mlx): a fake MoE model run
    through the FROZEN scratch instruments (moe_gt1.instrument free+traj,
    moe_gt1_arm2.instrument masked+recall) and through the unified
    patch_moe_router — traj rows byte-identical (json.dumps), stats/
    first-touch/pos equal, recall counters equal, model outputs equal.
  - restore contract: exiting the context manager restores the class
    __call__ it found on entry.

The D0 590,736-row regression and the live 3a/3b arms need the 30B
resident and run per the spec's acceptance ladder, not here.
"""

import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.lab.traj import begin_prompt, patch_moe_router

mx = pytest.importorskip("mlx.core")

N_EXP = 16
TOP_K = 2
DIM = 8


def make_cls():
    """A fresh fake sparse-MoE block CLASS per arm (all three
    instruments patch at class level; sharing one class would leak
    wrappers between comparisons)."""

    class FakeGate:
        def __init__(self, weight):
            self.weight = weight

        def __call__(self, x):
            return x @ self.weight.T

    def switch_mlp(self, x, inds):
        return x[..., None, :] * (inds[..., :, None].astype(x.dtype) + 1)

    def call(self, x):  # unrouted original, exists to be saved/restored
        return x

    cls = type("FakeMoE", (), {"switch_mlp": switch_mlp, "__call__": call})
    cls.FakeGate = FakeGate
    return cls


def make_model(cls, seed="traj-test"):
    rng = random.Random(seed)
    layers = []
    for _ in range(2):
        w = mx.array([[rng.uniform(-1, 1) for _ in range(DIM)]
                      for _ in range(N_EXP)])
        block = cls.__new__(cls)
        block.gate = cls.FakeGate(w)
        block.top_k = TOP_K
        block.norm_topk_prob = True
        layers.append(SimpleNamespace(mlp=block))
    # a non-MoE layer the discovery must skip
    layers.append(SimpleNamespace(mlp=SimpleNamespace()))
    return SimpleNamespace(model=SimpleNamespace(layers=layers))


def prompts(seed="traj-test-x"):
    """Two prompts, each a multi-token prefill then 1-token steps —
    exercises prefill / prompt_tail / decode tagging."""
    rng = random.Random(seed)

    def tok():
        return [rng.uniform(-1, 1) for _ in range(DIM)]

    out = []
    for _ in range(2):
        prefill = mx.array([[tok() for _ in range(5)]])
        steps = [mx.array([[tok()]]) for _ in range(3)]
        out.append((prefill, steps))
    return out


def run_blocks(model, batches):
    outs = []
    for x in batches:
        for layer in model.model.layers[:2]:
            x = layer.mlp(x)
            x = x.sum(axis=-2)  # collapse the expert axis for chaining
        outs.append(x.tolist())
    return outs


def drive_free(model, state, reset):
    """Run the two-prompt corpus, using `reset(state, i)` at each
    prompt boundary; returns (traj rows as dumped bytes, outputs)."""
    rows, outs = [], []
    for i, (prefill, steps) in enumerate(prompts()):
        reset(state, i)
        outs += run_blocks(model, [prefill] + steps)
        if state["traj"] is not None:
            rows += [json.dumps(r) for r in state["traj"]]
            state["traj"].clear()
    return rows, outs


def certified_reset(state, i):
    """A's driver-side resets, verbatim from scratch/moe_gt1.py main()."""
    state["prompt"] = i
    state["tpos"] = {}
    state["tail_done"] = {}


def test_traj_plus_keep_refused():
    with pytest.raises(ValueError, match="D-1"):
        patch_moe_router(object(), traj=True, keep={0: set(range(8))})


def test_free_traj_parity_vs_frozen_gt1(monkeypatch):
    from llmopt.moe.router_stats import RouterStats
    from scratch.moe_gt1 import instrument as frozen_instrument

    monkeypatch.setenv("TRAJ", "1")
    cls = make_cls()
    model = make_model(cls)
    original = cls.__call__

    state_a, _ = frozen_instrument(model)
    state_a["stats"] = RouterStats(n_experts=N_EXP)
    rows_a, outs_a = drive_free(model, state_a, certified_reset)
    wrapper_a = cls.__call__

    with patch_moe_router(model, traj=True) as state_u:
        assert cls.__call__ is not wrapper_a
        state_u["stats"] = RouterStats(n_experts=N_EXP)
        rows_u, outs_u = drive_free(model, state_u, begin_prompt)
    assert cls.__call__ is wrapper_a  # restore contract: back to entry state

    assert rows_u == rows_a  # byte-identical traj rows
    assert outs_u == outs_a
    assert state_u["first"] == state_a["first"]
    assert state_u["pos"] == state_a["pos"]
    assert state_u["stats"].counts == state_a["stats"].counts
    assert state_u["stats"].mass == state_a["stats"].mass
    cls.__call__ = original


def test_masked_recall_parity_vs_frozen_arm2():
    from scratch.moe_gt1_arm2 import instrument as frozen_instrument

    keep = {0: set(range(0, 8)), 1: set(range(4, 12))}
    cls = make_cls()
    model = make_model(cls)
    original = cls.__call__
    batches = [b for prefill, steps in prompts() for b in [prefill] + steps]

    state_b, restore = frozen_instrument(model, keep)
    outs_b = run_blocks(model, batches)
    restore()
    assert cls.__call__ is original

    with patch_moe_router(model, keep=keep) as state_u:
        outs_u = run_blocks(model, batches)
    assert cls.__call__ is original

    assert (state_u["hits"], state_u["slots"]) == (
        state_b["hits"], state_b["slots"])
    assert state_u["slots"] > 0
    assert outs_u == outs_b


def test_not_restored_is_loud(capsys):
    cls = make_cls()
    model = make_model(cls)
    with patch_moe_router(model) as _:
        cls.__call__ = lambda self, x: x  # hostile re-patch while live
    assert "INSTRUMENT_NOT_RESTORED" in capsys.readouterr().out
