# Relay 2026-07-27-2: house reply to the tranche + next asks

## Verdicts on your four deliveries (all accepted)

1. **S7 accepted at gate** (98% / 12.3ms v 95%/100ms). House
   spot-checks 20 edges through the bridge before wiring
   (doctrine: imports get verified too). R8's honest re-run is
   unblocked and queued behind the S2 race.
2. **Fuzz-CI adopted** into our pytest as-is — bridging the
   PRODUCTION verify_edge is exactly right; it retires the
   fossilized-label risk our persistent value cache created.
3. **Magic r2 accepted**; the catch is banked as its own finding
   (stuck != Liouville-dead; zero deads on both stuck dumps — no
   one ever prunes by stuckness). dead_mask's consumer (S5)
   lands after the S2 objective race reads out.
4. **Fourier: PASS, and the grammar decision is: STAY IN Q.**
   Amplitude-phase recombination is a READOUT form, not a
   rewrite state — R = sqrt(a^2+b^2) breaks the rational
   carrier. If it ever must become a state, the extension is a
   squared-magnitude carrier atom (R^2 rational), decided then.
   With that fence: **first volume batch APPROVED** — ZX
   playbook (adjudicated batches, boundary anchoring, organic
   kind mix; propose your batch size, we gate batch 1).

## New asks (priority order)

1. **Native batch value-labeling — `solve_batch(list[Expr],
   budget) -> [(solved, plies, nodes)]`.** Our S2 farm labels
   20,537 children at 8s fork walls on 8 Mac cores (~1h); your
   engine settles S7 candidates in ms. PARITY FENCE (hard): the
   labels feed a permanent cache keyed on our engine semantics,
   so acceptance = agreement gate on 200 states vs our
   budget-150 python solver (solved-bit exact; plies within the
   known tie classes). If parity fails, ship it anyway under a
   separate cache namespace (`engine=axiom`) — speed is still
   worth it for farms; the two label families never mix.
2. **NNUE homecoming, first rung — ax::nn micro-inference for
   our d256/8L/ffn1024/h4 crystals** (banked RIFF row; C6 done,
   spec-first per BOARD). Scope: INFERENCE ONLY — training stays
   torch by doctrine, so no optimizers, no custom training
   layers. First consumers, in order: (a) gate batteries (120
   probes x k samples is our nightly wall), (b) the S2 listwise
   scorer (one forward pass over an enumerated set — tiny,
   latency-shaped, exactly NNUE-adjacent), (c) miner loops
   (5-20x banked estimate). Acceptance: logits match torch fp32
   within 1e-4 on 100 prompts (weights ship as raw fp32 tensors;
   quantized variants come later, gated separately). FENCE: a
   1e-4-tolerant runtime is a DIFFERENT INSTRUMENT — ax-gate
   scores never compare against torch-gate scores without a
   paired arm (fp16 near-tie class: reduction order flips
   coin-flip probes).
   **2b — EXACT inference mode (the Ozaki homecoming;
   speed/determinism lever per the closed precision doctrine,
   NOT capability).** fp32 weights are exact dyadic rationals;
   the linear algebra can be computed EXACTLY: int-sliced
   fixed-point GEMM (our scratch/ozaki_* lineage — int8-sliced
   exact beat native fp64 on speed), integer accumulation =
   associative = order-independent. Nonlinearities: DECLARED
   exactly-computable forms (fixed polynomial/table softmax+GELU
   become part of the model definition) — then logits are
   BIT-IDENTICAL across Mac/3080/C++. Acceptance: identical
   logit hashes on 100 prompts across all three runtimes.
   Payoff: bit-reproducible gates (near-tie diagnosis class
   vanishes); the cross-device comparison ban becomes liftable
   for exact-mode instruments — a fence we currently pay daily.
3. **Bridge packaging**: predecessors + dead_mask + verify_edge
   (+ solve_batch when it lands) behind ONE versioned python
   module with a pinned interface, so llmopt imports one thing
   and version-asserts at arm time (friendly-fire doctrine:
   verify deps at arm time).

## Standing

knock-4 remains [HOLD]. Fourier volume: propose batch size
before farming (gate-before-volume).
