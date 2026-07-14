"""Resample diversity at stuck states: is validity starved by
sampling REDUNDANCY?

The hints A/B's mechanism claim (RESULTS 2026-07-13): at ~1%
validity, chains need resample diversity, and anything that
collapses it costs solves. This measures the claim directly, then
races the cheap levers.

Phase 1 (measure): at N root states, draw WAVES x 8 samples at the
production config (temp 0.7, fixed prompt). Report distinct-fraction
(unique first-lines / samples) and the saturation curve — if new
waves stop producing new candidates, budget is being burned on
duplicates.

Phase 2 (race): same states, same total samples, four arms:
  const   — production: temp 0.7 every stream
  ladder  — per-stream temps 0.4..1.45 (one knob, no retraining)
  rotate  — few-shot example order rotated per wave (prompt jitter)
  both    — ladder + rotate
Metric that pays: DISTINCT VERIFIED-VALID steps found per state
(verification is fork-isolated and deduped first — verify cost
scales with distinct count, another reason diversity is the right
unit). Plain validity reported too.

    .venv/bin/python scripts/bench_step_diversity.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

N_STATES = 24
WAVES = 8
B = 8
SEED0 = 9_300_000  # third fresh band
LADDER = [0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.45]


def fewshot_rotations(fewshot: str) -> list[str]:
    """Rotate example blocks (split on blank lines); the instructions
    header (first block) stays put."""
    blocks = fewshot.split("\n\n")
    head, examples = blocks[0], blocks[1:]
    rots = []
    for k in range(len(examples)):
        rots.append("\n\n".join([head] + examples[k:] + examples[:k]))
    return rots


def main() -> None:
    import sympy as sp

    from bench_step_tokens import (FEWSHOT, _gen_isolated, load,
                                   sample_batch, verify_step)

    tok, model = load("checkpoints/step_lora.pt")
    states = []
    for lv in (3, 4, 5):
        for i in range(N_STATES // 3):
            p = _gen_isolated(lv, SEED0 + 1000 * lv + i)
            if p is not None:
                states.append((lv, f"Integral({sp.sstr(p._expr)}, x)"))
    rots = fewshot_rotations(FEWSHOT)
    print(f"# diversity probe — {len(states)} states, "
          f"{WAVES}x{B} samples/state/arm, {len(rots)} fewshot "
          f"rotations available")

    arms = {
        "const":  {"temps": None,   "rotate": False},
        "ladder": {"temps": LADDER, "rotate": False},
        "rotate": {"temps": None,   "rotate": True},
        "both":   {"temps": LADDER, "rotate": True},
    }
    for arm, cfg in arms.items():
        tot = dist = valid_dist = 0
        sat_first = sat_last = 0  # new distincts in first/last wave
        for si, (lv, cur) in enumerate(states):
            seen: set[str] = set()
            new_by_wave = []
            for w in range(WAVES):
                fs = rots[w % len(rots)] if cfg["rotate"] else FEWSHOT
                prompt = fs + f"\nCurrent: {cur}\nHints: none\nStep:"
                texts, _ = sample_batch(
                    tok, model, prompt,
                    seeds=[SEED0 + 104729 * si + 7919 * (w * B + b)
                           for b in range(B)],
                    constrain=True, temps=cfg["temps"])
                new = 0
                for t in texts:
                    line = t.splitlines()[0].strip() if t else ""
                    tot += 1
                    if line and line not in seen:
                        seen.add(line)
                        new += 1
                new_by_wave.append(new)
            sat_first += new_by_wave[0]
            sat_last += new_by_wave[-1]
            dist += len(seen)
            # verify the DISTINCT candidates only (dedup-then-verify)
            for line in seen:
                expr = line.split("=>")[-1].strip()
                if not expr:
                    continue
                ok, _solved = verify_step(cur, expr)  # tuple return!
                valid_dist += bool(ok)
        n = len(states)
        print(f"{arm:>7s}: distinct {dist}/{tot} "
              f"({100 * dist / max(tot, 1):.0f}%) "
              f"valid-distinct/state {valid_dist / n:.2f} "
              f"wave1 new {sat_first / n:.1f} -> wave{WAVES} new "
              f"{sat_last / n:.1f}", flush=True)


if __name__ == "__main__":
    main()
