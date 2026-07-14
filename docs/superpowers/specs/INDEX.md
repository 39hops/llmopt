# Specs index — design docs with their outcomes

Specs are provenance, kept verbatim after execution. Status here, one
line each; verdicts live in docs/RESULTS.md, current queue in
docs/BOARD.md.

| Spec | Status |
|---|---|
| 2026-07-05 metal-splitk-decode | EXECUTED — ties mx.fast SDPA at T=32k (kernels/metal.py) |
| 2026-07-06 hce-rung1-primitive-moves | EXECUTED — rung 1 shipped, doit demoted to verifier |
| 2026-07-06 hce-calibration | EXECUTED — ρ=+0.685/+0.712, motivated NNUE |
| 2026-07-06 next-sessions-roadmap | HISTORICAL — roadmap #1 (Stockfish-for-math) completed; superseded by docs/BOARD.md |
| 2026-07-06 rotate-quantize | EXECUTED — 15-20% RTN error cut at 4/3 bits; 2-bit inversion recorded |
| 2026-07-06 task-arithmetic | EXECUTED — transfer complete, addition annihilates (honest failures logged) |
| 2026-07-06 weightspace-reader | EXECUTED — 80.8%/88.4%; source of the never-score-by-weight-distance rule |
| 2026-07-07 adaptive-k | EXECUTED — T=1.0 null → T=0.1 champion (300/360) |
| 2026-07-07 engine-optimizations | EXECUTED — sampled verification 1.65x, byte-identical |
| 2026-07-07 expert-iteration-r2 | EXECUTED — step-function-to-ceiling result (tabula rasa arc) |
| 2026-07-07 mathgen-expansion | EXECUTED — ten kinds in a day |
| 2026-07-07 mathgen-series-inequalities | PARTIAL — series landed; inequalities unconsumed |
| 2026-07-07 move-proposer | EXECUTED — 99.7% top-3; later ambushed by the markov dictionary |
| 2026-07-07 nnue-eval | EXECUTED — ρ=+0.937, wins/ties all 24 cells |
| 2026-07-07 proofs-rung | BANKED — proofs.py generator exists; Lean rung on the board |
| 2026-07-07 rung2-integration-moves | EXECUTED — u-sub/by-parts/linearity; budget-saturation chart |
| 2026-07-07 tabula-rasa | EXECUTED — self-teaching is a step function to the reachable-set ceiling |
| 2026-07-08 tcount-engine | EXECUTED — ZX rungs 0-6 (RESULTS "T-count engine") |
| 2026-07-12 step-expert-iteration | LIVE — the loop driver runs this spec (scripts/expert_loop.py, LOOP-LOG.md) |
| 2026-07-12 variational-ground-engine | EXECUTED — rung 1 shipped; structure search closed (2 fails, on the books) |
| 2026-07-13 syndrome-head | RUNNING — re-aimed at payoff 3 (representation shaping); A/B on Mac |
| 2026-07-14 step-grpo | RUNNING — run 1 cycle-2 gate GREEN on every level (first uniform improvement after six reallocating SFT retrains) |
| 2026-07-14 grpo-v2-and-unified-climb | SPEC'D — lossless verify levers (cache/batched-fork/numeric-reject), syndrome-in-RL unified climb gated on the Mac A/B |
