# Expert-iteration loop log

| round | frontier | mined (model/engine) | validity | solves (pre -> gate) | verdict | wall |
|---|---|---|---|---|---|---|
| 2/3 (manual) | F~L3/4 | +580 (88 fwd, 492 rev) +895 skips | 0.5% -> 1.0% | one-shot 13->19, steps 8->12 /30 | PROMOTE (manual judgment; gate not yet armed) | ~40m |
| 4 (manual) | eval L2/L3 | balance: 150 one-hop cap, magic skips | 0.8% | one-shot 19->0 (!), steps 12->7, chained 0 | ROLLBACK — cap unlearned finishing; train/eval shape mismatch (reverse choreography vs non-sum eval band) | ~35m |
| 1-take2 (coeff flood) | F=L5 | +11m/+0e | val 1.3->0.6 | {2: 11, 3: 10, 4: 5, 5: 7} -> {2: 6, 3: 1} | ROLLBACK: L2 regressed 11->6 — 764 uncapped coeff drills flooded the chain class, finishing halved | 104m |
| 1-take3 (class-aware balance) | F=L5 | +0m/+0e | val 1.3->1.4 | {2: 11, 3: 10, 4: 5, 5: 7} -> {2: 10, 3: 7, 4: 3} | ROLLBACK: L3 regressed 10->7 — no collapse, but coeff drills at 250-dose don't pay at eval level | 165m |

*Condition change 2026-07-13 evening: USE_HINTS=False (two-band A/B,
RESULTS) — evals faster (no oracle forks) and validity reads higher.
Rows below re-baseline; not comparable to rows above.*
