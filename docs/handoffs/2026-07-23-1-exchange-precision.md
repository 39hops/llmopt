# Handoff 2026-07-23-1 (late night — the exchange + precision night)

Second handoff of the day (convention: dated, 0-indexed). Read -0
for the day's first half (five grammars, crown tie, v4 verdicts).

## The two headline verdicts

1. **THE EXCHANGE CONVERTS: 2/12 -> 6/12** (pre-reg smashed; bar was
   beat-2). PRE reproduced v4's endpoint exactly; 23 engine rows x
   ~10 min flipped FOUR taught walls; proxy held. Ledger: 150 min
   self-practice = +1; 23 demonstrations = +4 in ten minutes. Plus
   the unplanned symmetry: the model's only 2 wins = the engine's
   only 2 holdouts (union 12/12). Practice loop PROVEN end to end.
   exchange_p1.pt on WSL; stuck_states_p2 (2 deep states) local.
2. **THE OZAKI ARC, prototype to law in one night** (six RESULTS
   entries): error-free transform proven (CPU 0-deviation via
   expansion recombination; auditor-was-the-bug #3 en route) ->
   int8-TC crossover on the 3080 (55ms exact BEATS fp64 accuracy;
   21ms triangular beats fp64 wall 2x) -> zero-rounding GPU matmul
   verified vs big integers -> fp64-input GEMM at the dd floor
   (2^-107, exact to output format) -> **stay-in-RNS lazy pipeline:
   4 exact layers 53ms vs 173ms inexact fp64; break-even ~6 layers
   — EXACT IS CHEAPER THAN APPROXIMATE past that depth.** Closures:
   fp16-TC dead (cublas fp16-accumulates regardless of the knob);
   RNS exit is the cost (fractional-CRT estimate exit 10ms = the
   propose-verify law inside arithmetic; exact exit deterministic ->
   EU-incremental). THEORY row added.

## In flight / queued

- Mac: gen-8 everything-diet mid-birth (1.1M rows, 19M vocab-41);
  five-grammar report card + rarity battery vs poly3 comparator
  auto-chained; then poly_chain4 bridge-law pipeline (watcher pid'd,
  size-stable pull guard). Verdicts land by morning.
- Metabolic v5 SPEC'D (specs/2026-07-23-metabolic-v5.md): exchange
  as live food channel + fp32/fp64/SR/async-CPU race + miner v2
  (failed-step banking) + deeper-before-wider (p2). Runs after
  gen-8 verdicts.
- Relay READY (relay/2026-07-24-0-exchange-converts.md): exchange
  verdict, poly4 receipt, v5 loop asks, Fourier flag. Artin sends.
- Solved-only-leak A/B still deferred (needs miner v2's
  failed-step-rich shard).

## Doctrine added tonight

- Exactness-chain law: every carry on the path widened or two-summed
  — the chain is as exact as its sloppiest link (fp32 diag sums
  crossing 2^24 leaked; dd part-build leaked).
- np.round().astype(object) boxes FLOATS — big-int references must
  go int64->object. (Auditor-was-the-bug #3.)
- Exit laws: never leave RNS/sliced domain mid-chain; estimate-exit
  for decisions, exact-exit on ambiguity only.
