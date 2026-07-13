# Expert-iteration loop log

| round | frontier | mined (model/engine) | validity | solves (pre -> gate) | verdict | wall |
|---|---|---|---|---|---|---|
| 2/3 (manual) | F~L3/4 | +580 (88 fwd, 492 rev) +895 skips | 0.5% -> 1.0% | one-shot 13->19, steps 8->12 /30 | PROMOTE (manual judgment; gate not yet armed) | ~40m |
| 4 (manual) | eval L2/L3 | balance: 150 one-hop cap, magic skips | 0.8% | one-shot 19->0 (!), steps 12->7, chained 0 | ROLLBACK — cap unlearned finishing; train/eval shape mismatch (reverse choreography vs non-sum eval band) | ~35m |
