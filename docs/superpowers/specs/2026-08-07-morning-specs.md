# Spec 2026-08-07: three grounded pre-reg designs (morning agents, Fable-verified)

All three fire on Artin GO, each independently. Grounding by an Opus
desk agent (read-only, ~90s desk compute), key numbers spot-checkable
from the named rules.

## 1. EX-ANAT-3 — carrier anatomy (Mac; the identity era's mechanism rung)

SUBJECTS now measured (rederived by the exact ex1_swap rule):
- 1,657 distinct (layer, expert) carriers across the three bins
  (27.0% of the expert bank; ~34.5/layer named at least once).
- Carriers are largely BIN-SPECIFIC: 1,094 named by exactly one bin,
  483 by two, only 80 by all three (pairwise Jaccard 0.18-0.20).
  315 carriers flip hi/lo side across bins — side is not a carrier
  property.
- Global demand rank: median 47/128 (uniform null 63.5) — mildly
  high-demand, spanning the whole range; EXCLUSIVITY does the
  selecting, not global demand.
- Structure membership: 0/1,657 in the symbolic core (verified by
  construction); 64.0% inside the GT-1 45.3% crest mask (1.41x base
  rate); vonly ~52% by class construction (not independent).
DESIGN CONSEQUENCE: two subject tiers — (a) the 80 TRI-BIN
INVARIANT carriers (the strongest candidates for "what carriers
compute"); (b) per-bin sets (~760 slots). TRAJ lenses (the banked
DeepSeek set): token-specialization signature in free routing
(which token classes fire a carrier — TRAJ v3 instrument ready),
demand-rank-within-exclusives, layer-position profile. Rungs:
1. DESK: itinerary census of the 80 invariants from the EXISTING
   traj artifacts (moe_gt1_traj_v2 + gt2/gt3/gt4 domain trajs) —
   which phases/token positions route through them, free.
2. GATE (pre-reg): mask ONLY the 80 invariants out of the full
   model (and out of the crest mask) — if carriers compute
   something load-bearing, this tiny 80-expert deletion should
   cost measurably; paired with 80 random matched-rank deletions.
3. Optional: forced-residency error-class shift (EX-ANAT-2's
   framing) on a named subset.

## 2. Params ladder — P-CAPACITY-2 (3080 box, CPU; deterministic)

Anchor: 12,518 cycle-mean at 60,224 params / 8 windows / 1000 steps
(DIET-BRIDGE; steps/windows/schedule all failed to break it —
PLATEAU-BREAK + P-STEP-BOUND-2).
FENCE THAT SHAPES THE DESIGN (agent-caught): DIM is NOT a free axis
(GRAVMOE-BRUTE: D=128 at default ACT_CLAMP clamps 23% at init;
ACLAMP ceiling ~46,341 rms overflow class). Hold D=64; move FFN and
NBLK only.
ARMS (all NWIN=8, SHIFT=14, seed 17, STEPS=1000 — the anchor's
schedule; one variable each):
  P-SMALL  NBLK=1            31,424 params (0.52x)
  P-WIDE   FFN=256, NBLK=2  109,376 params (1.82x)
  P-DEPTH  NBLK=4, FFN=128  117,824 params (1.96x)
P-WIDE v P-DEPTH double as a depth-v-width pair at matched capacity
class for free.
BARS: P-CAPACITY-2 fires iff (P-WIDE or P-DEPTH) final <= 11,266
(the same 10% class the prior rungs missed) AND P-SMALL regresses
ABOVE 12,518 (monotone capacity reading, not noise). IMPLEMENTATION
FENCE (agent-caught): the detbwd_plateau post-import knob pattern
does NOT work for DIM/FFN/NBLK — those are module globals bound at
import of detbwd_r2b/detbwd_mb; the params driver must set
os.environ BEFORE importing detbwd_diet.

## 3. Multiplicity census — MULT-0 (Mac; rung 1 of the reverse-propose ladder)

The pincer closure's fence made measurable (RIFF 2026-08-07): does
the reverse model emit >= 2 DISTINCT verified candidates?
ZERO-COMPUTE PASS first: logs/tenet_r1b_{F,FR,LR}.jsonl rev_score
ledgers reproduce the forward 8/120 figure free (no distribution).
THE REAL CENSUS: reuse the D1b frame VERBATIM minus one
short-circuit — scratch/tenet_d1_revgate.py rev_gate_eval
(poststep) already does first-encodable-child prompting, B=8 wave
sampling, dedup, the verified-AND-distinct fence, and REPLAY
minting (the direction-honest criterion); its `if done: continue`
stops at the first mint. New thin driver
scratch/tenet_mult_census.py (import-and-override, the
detbwd_plateau pattern — never edit the cited gate file) counts
mints per problem instead of stopping. Subjects: the sha-pinned
checkpoints sym_birth_dense_revcert.pt (0d5ece32...) + the
fwdcert control (272e47d2...).
READOUTS to register: multiplicity histogram m per problem/level;
P(m >= 2) headline (the ranker's intervention surface); per-level
carries the open L4=0 anomaly; miss/err census as the
malformed-emission control. RUNTIME: ~6 min/arm at B=8 (~12 min
rev + fwd control); B=32 sweep ~25 min/arm if the B leg is wanted
same-session. GATE FOR RUNG 2 (chooser calibration): P(m >= 2)
materially above the forward control's — exact bar set in the
census pre-reg once the zero-compute pass anchors the scale.
