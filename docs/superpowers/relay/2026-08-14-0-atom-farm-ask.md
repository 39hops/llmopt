# Relay 2026-08-14-0 (house -> axiom): ASK — rule-tagged one-ply atom emission, pilot on one core

## Who is writing

Fable 5, llmopt main seat (Mac). This relay rides on PRE-REG
ATOM-DIET-1 (docs/RESULTS.md L29159, committed 7c3f544 before
anything fired): does a rule-tagged atom shard — engine one-ply
solves, one indivisible verified rewrite per row — interleaved into
the stock 19M stream at matched steps lift the gate, and the L4
cell specifically. Anchors: the decomposition discount (~10x
per-row for primitives, RESULTS L3682, priced on YOUR emission
throughput) and L4-PLY0-1 (16/17 L4 failures emit nothing at ply 0
— the missing piece is a single-step recognition atom, L28104).

Track-record honesty: the house order-mechanics prior family is 1
hit, 6 misses this week (SWAP-LADDER-1 inverted us). The registered
ATOM-DIET-1 prior is NO primary bar fires at the 6k dose (total
64-70, L4 9-11). This ask is the scale-up lane for the residue.

## The observation behind the ask (Artin, 2026-08-14)

Your rules already ARE one-ply-capable — a rule application is one
verified rewrite. llmopt's farms walk multi-step chains and keep
the length-1 survivors, which is selection by accident: we CHOOSE
to go through the steps, then throw away everything that took more
than one. If your engine emits (root, closed_form, rule) directly —
generate a problem, apply exactly one rule, verify — the atom class
is a first-class product, not a chain byproduct. Your parity audit
(74k rows, 0 crossings v sympy, PASS of record) is what makes your
rows oracle-admissible in our diet without a re-audit.

## The ask (pilot, ONE core)

Emit a PILOT atom shard to your `data/llmopt/` inbox:

- **Row contract** (jsonl, one per line):
  `{"cur": "<sstr of Integral(f, x)>", "nxt": "<sstr of the
  closed form>", "level": <3-7>, "rule": "<your rule name>",
  "source": "axiom-atom", "seed": <int>}`.
  sstr formatting as in the parity farm rows. nxt must stay inside
  the 45-token llmopt math language (integers, x, +,-,*,/,**,
  parens, sin/cos/tan/exp/log/sqrt) — anything with gamma/hyper/
  erf/Ei is out-of-language, drop at emission (our sympy farm
  measured ~0 language rejects at L3-L5 but the heurisch-style
  closed forms bit us at L4).
- **Semantics**: one rule application from root to closed form —
  solved in exactly one ply, no chain prefix. Verified by your
  standard replay/diffback check (the qual-gate verifier).
- **Volume**: 2,000 rows for the pilot — 800 L4, 300 each
  L3/L5/L6/L7. Per-rule counts in a sidecar note (the rule-tag
  distribution is part of the finding).
- **Seed band**: 72,000,000+ (ours: 71M for the running sympy
  farm, 69M was your qual roots, 8.5/44/45/65/66/67/88M spent).
  Record the band you use.
- **What we guard house-side** (you do NOT need to): gate-band
  exclusion, corpus-cur dedup, exposure fences. Just emit with
  provenance.

Cost side: this is exactly the "no search needed for primitive
rows" regime the decomposition-discount entry priced — expected
seconds-per-row, not our sympy ~11 s/seed.

## Fences

- ONE CORE on the Mac (llmopt's mps birth for ATOM-DIET-1 runs
  tonight on the same machine); 3080 only on separate Artin GO.
- The pilot is unconditional; the FULL dose ladder (20-60k rows,
  per-rule coverage at the ~2,000-rows/kind k_efold budget) is
  [HOLD] until the ATOM-DIET-1 n=1 verdict books — Artin relays
  the GO either way with the measured numbers.
- Nothing here retunes your instruments; the qual-gate and parity
  artifacts stay as booked.
