# SPEC: the format ladder (format x schedule at d256)

Origin: Artin's format-x-schedule riff (RIFF-LEDGER 2026-07-26,
promoted by the CE-gate finding: the gate measures CHAINING and
single-pass ties 3ep at one ply — the epoch's -8 is a composition
wound) + his delta-chaining riff (similarity-adjacent rows
in-context: "the chains acting like a superposition of where the
resulting weight should land" — house translation: in-context
analogical exposure, the worked-example effect factored across
problems). Design lessons inherited: the 07-22 chain-carry
ablation was VOIDED (thin budget + repetition asymmetry) — every
arm here uses the full gen-4 corpus content and matched
construction; the interaction law (in-BATCH similarity costs -12
at one pass) makes E's question sharp: does in-CONTEXT similarity
carry the OPPOSITE sign?

## Substrate + schedules

d256/L8/ffn1024/h4, seed 1, MPS, gen-4 corpus. Schedules:
- 1P = the v4 streaming recipe (1 pass, mixed shuffled batches,
  final-10% cooldown, surprise rider) — the wound regime, primary.
- 3E = the standard 3ep OneCycle trainer recipe.
Comparators already on disk: pairs@1P = 57 (stream4), pairs@3E =
65 (wfloor). Gate = gate_ckpt (chain gate, 120); CE-400 fixed-
sample read rides every arm (the new standing proxy instrument).

## The seven formats

1. **pairs** (control; both cells measured).
2. **traces** (A): state-linked chains (nxt->cur greedy from
   roots) serialized as `Current: root / Step: s1 / ... /Step: sk`
   — composition in-context, frames compressed. SEQ_CAP 1024.
3. **skip-pairs** (B): 50% plain pairs + 50% (s_i -> s_{i+2})
   hops (transitivity-verified free), matched row count.
4. **de-chained** (C): no two rows share a state (every-other-step
   sampling, ~66k rows) — chain-adjacency removed; dose reported
   honestly (content class unchanged, row count not matched).
5. **one-shot** (D): root -> final-answer rows (~53k, from linked
   chain ends). Gate still valid (a correct answer = 1-step
   solve); step-tokens precedent predicts a deep-level crater —
   never measured from birth.
6. **delta-chained** (E): sequences of 4 pair-rows selected by an
   embedding-similarity walk (mean-pooled wfloor_d256 hidden
   states, round-3 precedent; cosine kNN, greedy walk, similarity
   floor = the delta knob). SEQ_CAP 1024.
7. **random-packed** (E0): same 4-pairs-per-sequence structure,
   random selection — E's mandatory control (separates context-
   packing from similarity).

## Cells + order

Primary (1P): A, B, C, D, E, E0 = 6 births (~20-25 min each).
Interaction column (3E): A, E, E0 = 3 births (~35 min each).
Others' 3E cells fire only if their 1P cell moves >= the bar.

## Pre-registered readings (sigma_d256 ~1.0, directional bar 3,
strong bar 5)

- PRIMARY: traces@1P >= 61 => the epoch's -8 was largely FORMAT
  (composition-in-context substitutes for revisits) — streaming
  reopens, and "the epoch is load-bearing" retracts to "the PAIR
  format needs revisits." traces@1P ~57 => format-neutral at one
  pass; the epoch survives its strongest challenge yet.
- E vs E0 (the delta test): E > E0 + 3 => IN-CONTEXT similarity
  PAYS where in-batch similarity costs — the interaction law
  splits into two opposite-signed laws (the headline if it
  lands). E ~ E0 => juxtaposition is inert; the analogy story
  dies cheap. E < E0 - 3 => similarity hurts in-context too —
  the interaction law generalizes across dimensions.
- E0 vs pairs@1P prices context-packing alone.
- one-shot@1P: predicted crater (<= 40) with L5-7 near zero —
  the from-birth leg of step-tokens.
- de-chained@1P vs pairs@1P: if flat, chain-ADJACENCY in the diet
  never mattered (pairs are already shuffled); if down, state
  overlap across rows was quietly load-bearing (dose fence
  noted: 66k vs 132k rows — a drop books as adjacency-OR-dose,
  decided by a row-matched follow-up only if it moves).
- skip-pairs@1P vs pairs@1P: compression axis; prediction from
  the step-dropout bank: small positive or flat.
- CE-400 across all arms: does CE keep tracking the gate across
  FORMAT variation, or is this morning's 4/4 rank agreement
  schedule-specific? (Free leg of the CE-gate corpus.)

## Fences

Token mass differs by format (traces/one-shot compress frames;
packed sequences pad less) — reported per arm, never hidden;
row-content held to the same gen-4 corpus everywhere except C/D's
structural subsetting (named above). n=1 per cell at d256 (sigma
1.0 measured, n=3). Stitched traces may cross problems at state
collisions — verified-coherent by construction, noted. E's
embedding model is the pairs-trained control crystal (no
circularity: it only orders rows, never labels them).
