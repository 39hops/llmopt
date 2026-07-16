"""Curriculum v2.1: L4-targeted calculus shard (the residue fix).

The algebra substrate eased L4's starvation (v2-GRPO: mixed groups
1.5x easier, 6-frozen -> 8-stable) but L4's ceiling still lags — the
gap is running the integration pattern ACROSS the chain structure,
which only worked L4 step chains can teach. The phase-1 diet had
just 7.6k L4 pairs (the thinnest band); this shard multiplies that
with engine-replayed, oracle-verified L4 chains at a fresh band.

Reuses expert_iter_steps.phase_chains verbatim (fork-boxed engine
replay, streamed rows, min_pairs=2 — the round-2 lesson: mine where
the engine CHAINS, one-ply pairs taught answers not chaining), with
the output redirected to a separate shard file so the v2 diet and
this addition stay independently attributable.

    caffeinate -i .venv/bin/python -u scripts/farm_l4_calc.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import expert_iter_steps as EIS

EIS.CHAINS = Path("data/micromodel_calc_l4_shard0.jsonl")
SEED_BASE = 65_000_000  # fresh band (64M = v2 leg 2 collection)

if __name__ == "__main__":
    EIS.phase_chains(4000, SEED_BASE, levels=(4,), min_pairs=2)
