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

*Math-native 19M (from birth, Mac):*
| micro run 1 | gate L2-5 band 9.8M | 12 cycles, full-param GRPO from phase-1 ckpt | 76.8 -> 80.3 | baseline {12,12,5,7} -> plateau {12,12,7,7} (L2/L3 PERFECT from birth) | PLATEAU after cycle 4 — curriculum ascent next (L4-7) | ~15m/cycle incl gates |

*45M production lineage (mathnative_45m_v21 base, Mac):*
| 45m run 1 | gate L3-7 GATE_N=24 | 12 cycles, full-param, dual-clip | 54.24 -> 56.79 @ c10 | 57/120 -> **61/120** (past the 19M record 60) | host outage killed cycle 12 pre-gate; c10 best banked, +6.9k mined streamed | ~16m/cycle |
| 45m run 1b (continuation) | same | from run-1 c10 best; 12 cycles, 2 rollbacks (c4 by 0.07, c8 — the 62-solve candidate lost to the validity veto, the snapshot-before-verdict case in the flesh), mined +9,279 | 56.79 -> **59.36** @ c10 (45M lineage record) | 61/120 held all run | COMPLETE — validity ratchets, solves plateau at 61; next lever is diet (self-distillation consolidation) not more RL, echoing the 19M arc | ~17m/cycle |
| consolidation 1 (strategy B) | chain gate | ONE low-LR epoch on level-capped sidecar (24.2k rows) from promoted 61@59.36 | 59.36 -> **62.23** | 61 -> **64/120** (program record; every level >= promoted, L7 first 10) | PASS — 5.5 min of SFT beat six RL cycles; RL explores, SFT consolidates, measured | 5.5m + gate |
| 45m run 2 (from consol base) | same | 12 cycles, 2 rollbacks (c2, c8), mined +12,985 (2x run 1b — the consolidated miner is far richer) | 62.23 -> 62.43 @ c10 | 64/120 held (oscillation band 62-64) | COMPLETE — RL holds the consolidated ceiling, doesn't extend it (+0.2 over 12 cycles); all-pass waves ~50% = gradient starving at this frontier | ~19m/cycle |
| consolidation 2 (from run-2 best, refreshed sidecar) | chain gate | same recipe, seed 1, 25.6k capped rows | 62.43 -> **63.05** | 64 -> **65/120** (record; L7 11/24, third straight high) | PASS — the RL->consolidate loop REPEATS (61->64->65), diminishing but compounding | 6m + gate |
| 45m run 3 (small cycles, from consol2) | two-tier gates | 24 cycles @ --groups 16, 2 rollbacks (both snapshot-saved), mined +7,222; 10 honest gates in the wall time run 2 spent on 6 | 63.05 -> **63.82** (final gate = best; 3 validity records in-run) | 65/120 held all run | COMPLETE — recipe works (dip->halve->recover in ~40 min vs 2 h); solves ceiling untouched, as the autopsy predicts | ~8m/cycle |
| gen-5 mining (from gen-4 record base, 3080/cuda) | two-tier, solves-primary | 18 cycles @ groups 16, HALT on 2 consecutive rollbacks; snapshots banked | 64.66 -> 65.22 peak | 66 -> **68/120** @ c10 (records: L4 9, L3 22) | RL from the REBORN base climbs again (+2) where the exhausted lineage got +0 — rebirth resets gradient supply. NOVELTY AUDIT: only 514 new corpus rows — SEED_BASE constant means every run re-mines the same problem stream (fix: --seed-base per run); fresh problems need band ascent / new seeds | ~7m/cycle (cuda) |
| gen-5 mining B (fresh stream, seed 70M) | same | 12 cycles; 3 validity records (65.25/65.52/65.66); mined +4,747 raw | 64.93 -> **65.66** (all-time validity record) | 68/120 held | COMPLETE — fresh problems restored RL gradient (3 records in 12 cycles vs 2 in 18 stale) BUT corpus novelty only 2% (113 new rows): fresh problems != fresh steps; the band's step-space is ENUMERATED at 84k. Mining's two products split: gradient (delivered) vs corpus (saturated — needs L8/ODE) | ~5m/cycle |
| consolidation 3 | chain gate | same recipe, seed 2 | 63.82 -> **63.97** (record) | 65/120 (no new solves) | PASS but the loop is EXHAUSTED: 61 -> +3 -> +1 -> +0 solves — consolidation redistributes what mining contains, and mining has no new patterns at this diet. NEXT LEVER: v2.2 (autopsy-aimed) | 6m + gate |
