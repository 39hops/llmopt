# The closed-system-native transformer — rungs 1-3 (specced 2026-07-25)

Provenance: Artin's riff (RIFF-LEDGER 2026-07-25) + reviewer survey
(same night). Litmus governing every rung: seed STATISTICS that
training error-corrects = pays (warm-birth); impose microstructure /
freeze / tell = null (hints x2, prior-wash, payoff-3, canonical-sort).
One variable per rung, paired vanilla twin, same seed/device/diet.

## Rung 1 — PREFIX NOTATION (the enabling substrate)

WHAT: serialize every expression prefix/Polish (`* 7 x` for `7*x`);
same atoms, no parens tokens in expressions; Current:/Step: frame
unchanged. Infix twin = the control.
BUILD (Fable): (a) sympy->prefix serializer + prefix->sympy parser
(~30 lines each; parser needed by every oracle touchpoint);
(b) diet converter (cur/nxt -> prefix at build, strict-encode
gated); (c) probe/gate scorers parse prefix before oracle calls.
Nothing else changes — trainer/model untouched.
PRE-REG: (i) math gate within noise of twin (>= twin-2); (ii)
median sequence >= 20% shorter; (iii) SECONDARY, where variance
lives: psub/padd/ibridge kind-accuracy vs twin (does removing
parens bookkeeping move the operand-complexity wall?); (iv) rider:
int3 quantized gate delta (delimiter-outlier removal -> predicted
MORE robust).
ADOPT IF (i)+(ii); the emission-wall and quant reads bank either way.
COST: 19M birth + twin, Mac or 3080. SLOT: Mac, behind the width
ladder (serializer written first, ~1-2h Fable work).

## Rung 2 — SUBTREE-HASH-KEYED TREE-PE (on the prefix substrate)

WHAT: positional encoding for expression tokens keyed by CONTENT
(subtree hash), not sequence index or tree path. Reviewer's
correction, adopted: tree-PATHS are not rewrite-invariant (root
restructures shift every path); subtree-hash identity IS the EU/
magic-boards 95%-shared currency — untouched subtrees keep their
encoding across rewrites by construction.
DESIGN NOTES: hybrid PE (sequence-PE for the Current:/Step: frame
tokens — they have no tree node; hash-PE inside expressions);
decoding order resolved by prefix (a node's children follow it —
the hash of the emitting subtree is computable incrementally);
hash -> encoding via a small learned embedding of hash buckets
(statistics-seeded, trainable — never frozen).
PRE-REG: (a) NORTH STAR: zero-epoch gate > 0/120 (v0 template
reads exactly 0; any structural-prior gain drags the zero); (b)
3ep gate >= prefix twin; (c) EU rider: wave-scoring wall with
cached subtree encodings (the 95%-shared claim, finally cashable).
RISKS NAMED: the frame/expression PE seam is the failure surface;
losing the learned "answer-after-Step:" anchor.
COST: model-code change (PE module) + 2 births. SLOT: after rung-1
adopts (undefinable on infix).

## Rung 3 — ATTENTION-INIT FROM TREE-ADJACENCY (init-only, ever)

WHAT: initialize (never freeze, never bias at inference) QK
patterns from measured structure: parent-child tree adjacency +
rule-bigram statistics, with UNSEEN-MASS SMOOTHING (the median-
mass lesson — prior-wash suppresses exactly the exploration new
capability needs).
PRE-REG: graded on EP1 SPEED, not ceiling (warm-birth: inits are
time machines; expect COLD to catch up by ep3 — the win is
compute, scaling with birth cost). Bar: ep1 gate >= cold-ep1 + 5
(warm-birth's FFN leg bought +8).
FORBIDDEN FORMS (do not implement): valuations as prompt features
(hints x2 null); frozen attention bias (prior-wash); syndrome aux
head (payoff-3 null); any canonical neuron/position sort (gauge).
COST: init-fn + 2 births. SLOT: after rung-2 (compounds with
hash-PE; also meaningful directly on prefix if rung-2 stalls).

## Program grading

Each rung banks its verdict separately (RESULTS, pre-reg before
launch). The program's composite promise (Artin): quicker AND
ceiling-lifting — rung 1 buys wall on every future birth, rung 2
aims at the emission wall + the zero-epoch north star, rung 3 buys
birth compute. If all three adopt, the vanilla transformer is gone
from the lab and every component earns its place by measurement.
