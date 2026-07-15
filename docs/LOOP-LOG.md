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
| 1 (hints-off) | F=L5 | +3m/+0e | val 1.6->1.1 | {2: 12, 3: 12, 4: 5, 5: 9} -> {2: 11, 3: 8, 4: 3} | ROLLBACK: L3 regressed 12->8 | 143m |
| 2 (hints-off) | F=L5 | +14m/+0e | val 1.4->1.5 | {2: 7, 3: 10, 4: 6, 5: 7} -> {2: 15, 3: 6, 4: 4} | ROLLBACK: L3 regressed 10->6 (but L2 7->15! — retrains REALLOCATE levels, not degrade uniformly) | 118m; HALT: two consecutive fails |
| control (own r23 diet) | eval L2-4 (band 8.5M) | retrain on byte-identical rounds-2/3 corpus | 1.32->1.72 | {2: 12, 3: 8, 4: 2} -> {2: 14, 3: 11, 4: 3} | PASS (harness printed FAIL via frontier=5 bug — L5 never evaluated; at the real frontier: frontier gain). VERDICT: H_DIET — own-diet retrain beats promoted everywhere; the post-r3 corpus additions were the harm | ~150m |
| diet B (r23 + 1332 finishing) | band 8.5M | reverse finishing merge, coeff benched | 1.52 | {2: 10, 3: 13, 4: 1} vs control {14, 11, 3} @ 1.72 | NO-MERGE: buys L3 (+2), pays L2 (-4) L4 (-2) — every diet buys its own shape; r23 alone stays champion retrain. Finishing pairs benched with coeff | ~75m |

*GRPO era begins (fast oracle scale — not comparable above):*
| grpo run 1 (slow oracle) | gate band 8.6M | 3.5 cycles, 64 mixed groups each | 1.38 -> 1.90 @ cycle-2 gate | {13,9,5,5} -> {15,10,6,8} | EVERY LEVEL UP — first uniform improvement after six reallocating SFT retrains | ~67m/cycle |
| grpo run 2b (fast oracle e2e, from run-1 ckpt) | gate band 8.6M | 20 cycles, 2 rollbacks (lr-halved), 8 checkpoints | **2.24 -> 5.38 (2.4x)** | {15,10,11,8} -> {18,13,10,11} | monotone-ish climb, gates green 8/10; all-pass waves 1 -> ~90/cycle; mined +6774 verified steps | ~7m/cycle |
