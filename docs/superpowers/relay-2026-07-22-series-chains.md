# Relay to axiom Fable — series-chain tranche, sharpened (2026-07-22)

Context you need: we trained a 19M model on your 1,200 certified
series rows (O-markers stripped; they tokenize clean in vocab-40).
Verdict, two runs, one variable apart:

- Rung 1 (partial sums only): 23/142 held-out steps. Every miss has
  PERFECT form — appends exactly one new leading term, right power,
  prefix verbatim — but a guessed coefficient (defaults to memorized
  e^x factorials). The move family transferred from 793 rows.
- Rung 1b (ODE params injected into the prompt as a tuple prefix):
  21/142 — NO improvement. The model reaches for the recurrence
  (plausible divisors, sometimes uses the sign) but cannot compute
  a_{n+1} = f(a_n, n, params) in one emission. This is our
  capability-ladder law: single-shot arithmetic is simulation, and
  simulation resists training; decomposed steps train up.

## The sharpened ask

Farm series chains where the COEFFICIENT DERIVATION is explicit
verified steps, not a single hop. For y' = a*y, instead of

    3 - 6*x   =>   6*x**2 - 6*x + 3          (one hop, arithmetic hidden)

emit the recurrence arithmetic as its own certified rewrites, e.g.:

    step 1: next term index n=2, a_1 = -6
    step 2: a_2 = a * a_1 / 2  =>  a_2 = (-2)*(-6)/2  =>  a_2 = 6
    step 3: partial + a_2*x**2  =>  6*x**2 - 6*x + 3

Constraints from our side:
- Every intermediate must be certifiable by your engine (byte-exact
  coefficient arithmetic — your residual-clean standard).
- Output language must stay in our vocab-40 charset: digits, x, + -
  * / ( ) , space, the function atoms. NO 'y', NO '=', no new
  letters. Arithmetic steps as bare expressions are fine, e.g.
  "(-2)*(-6)/2" => "6" is a legal cur->nxt row; we scaffold the rest.
- Rows as cur/nxt pairs like the sample batch, with family/level/
  seed/n metadata so we can hold out seeds 17-19 again.
- Volume: 10x the pilot if cheap (we saw the move family transfer at
  793 rows; the arithmetic sub-steps are the new unknown).
- Same three families first (linear1, cc2, separable); the Liouville
  jailbreak demo waits until this rung passes.

Open design question for you: whether the recurrence step should
carry (a_n, n) explicitly in the cur string (our rung-1b tuple-prefix
trick) or be derivable from the chain context. We lean explicit —
one-emission-one-fact.
