"""Population LoRA (K adapters, one frozen base) vs K sequential runs.

Bar (pre-registered): the batched population step must beat K
sequential single-adapter steps on wall clock at the same total
tokens — the win is amortized base-weight traffic; the cost is K x
activations (which 36GB absorbs). Loss parity between arms is the
correctness oracle (same seeds per adapter => same numbers). Peak MB
reported for both. If sequential wins, that verdict gets recorded.

Sequential DID win (2026-07-13): see the verdict in
llmopt/train/population.py — MLX is compute-saturated at batch 4, no
slack to amortize. Trajectory parity here is init-confounded (adapter
A inits are unseeded across arms); the exact grad-equivalence oracle
lives in tests/test_population.py.

    .venv/bin/python scripts/bench_population.py [K]
"""
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx.utils import tree_flatten

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llmopt.train.population import (apply_population_lora,
                                     population_loss)

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
B, T = 4, 512  # per-adapter batch
STEPS = 8


def load_model():
    from mlx_lm import load
    model, _tok = load(MODEL)
    return model


def make_data(k_total: int, step: int, rows: slice):
    """One shared deterministic pool per step; arms take row slices —
    adapter i sees IDENTICAL tokens in both arms (the parity oracle)."""
    mx.random.seed(1000 + step)
    toks = mx.random.randint(0, 151000, (k_total * B, T))[rows]
    tgt = mx.concatenate([toks[:, 1:], mx.zeros((toks.shape[0], 1),
                          dtype=toks.dtype)], axis=1)
    return toks, tgt


def train_arm(k: int, k_total: int, rows: slice, label: str):
    model = load_model()
    n = apply_population_lora(model, k=k)
    head_w = model.model.embed_tokens.weight
    opt = optim.Adam(learning_rate=1e-4)

    def loss_fn(mdl, toks, tgt):
        hidden = mdl.model(toks)
        return population_loss(hidden, head_w, tgt.reshape(-1), k)

    lg = nn.value_and_grad(model, loss_fn)
    losses = []
    mx.reset_peak_memory()
    t0 = time.perf_counter()
    for step in range(STEPS):
        toks, tgt = make_data(k_total, step, rows)
        loss, grads = lg(model, toks, tgt)
        opt.update(model, grads)
        mx.eval(loss, model.trainable_parameters(), opt.state)
        losses.append(loss.item() / k)
    dt = time.perf_counter() - t0
    toks_total = STEPS * k * B * T
    print(f"{label:>12s}: {n} wrapped, {dt:6.1f}s "
          f"{toks_total / dt:7.0f} tok/s "
          f"peak {mx.get_peak_memory() / 2**20:6.0f}MB "
          f"loss {losses[0]:.3f}->{losses[-1]:.3f}")
    return dt, losses


def main() -> None:
    ks = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    print(f"# population bench: K={ks}, B={B}, T={T}, {STEPS} steps/arm")
    # sequential arm: K separate single-adapter runs, same data stream
    # sliced identically (adapter i sees rows [i*B:(i+1)*B] of each step)
    t_seq = 0.0
    seq_losses = []
    for i in range(ks):
        dt, ls = train_arm(1, ks, slice(i * B, (i + 1) * B),
                           label=f"seq[{i}]")
        t_seq += dt
        seq_losses.append(ls)
    dt_pop, pop_losses = train_arm(ks, ks, slice(None),
                                   label=f"pop K={ks}")
    print(f"\nsequential total {t_seq:.1f}s vs population {dt_pop:.1f}s "
          f"-> {t_seq / dt_pop:.2f}x")
    gap = max(abs(sum(s[j] for s in seq_losses) / ks - pop_losses[j])
              for j in range(STEPS))
    print(f"loss parity |seq_mean - pop| max over steps: {gap:.4f}")
    print("bar: population > 1.0x at matching loss trajectories")


if __name__ == "__main__":
    main()
