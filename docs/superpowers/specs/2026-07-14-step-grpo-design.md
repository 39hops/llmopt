# GRPO at the frontier band — sustained RL over verified steps

Provenance: Artin GO 2026-07-14 ("RL over those steps could be
insanely strong within the steps themselves"). Converging evidence:
(1) four consecutive gate rollbacks show from-scratch SFT rounds are
a reallocation lottery (round-2 texture: L2 7->15 while L3 10->6);
GRPO updates the CURRENT promoted policy incrementally — no lottery.
(2) SFT discards 99% of samples at ~1% validity; GRPO also learns
from failures (down-weight). (3) MAI-Thinking-1's sustained-RL
finding (banked 2026-07-14). (4) Our oracle gives DENSE EXACT
per-step rewards — no credit smearing; almost nobody has this.

## Why "frontier band" is load-bearing, not garnish

Binary rewards + group-relative advantage => gradient is NONZERO
only for MIXED groups (some of the wave verified, some not).
All-fail groups (the famine: 0.33 valid/64 samples at hard states)
and all-pass groups teach nothing. The learning signal lives
exactly where variance lives — the starved-judge law now governing
the gradient itself. So state selection IS the algorithm:

- Draw problems at levels where the promoted model's solve rate is
  in the 20-80% band (the loop's existing frontier() logic).
- Walk chains with the current policy; at each stuck state run one
  wave (B=8, the existing sample_batch — waves ARE GRPO groups).
- Dedup-then-verify the wave (the diversity-probe discipline);
  reward 1.0 per verified step, 0.0 otherwise. v0 keeps rewards
  binary; a finishing bonus (+0.5 for Integral-free nxt) is the
  first knob to try if L2/finishing sags — banked, not default.
- KEEP only mixed groups for the update batch; all-fail/all-pass
  groups are free to discard (zero advantage anyway — this also
  caps verify cost, since dedup makes all-dup waves cheap).

## Update rule (pieces already in train/preference.py)

- Per-sequence logp of each sampled completion recorded AT SAMPLING
  TIME (sample_batch extension: return per-token logps; sum over
  generated tokens) => logp_old.
- advantages = grpo_advantages(rewards, group_ids)  # z-score/group
- loss = grpo_loss(logp_new, logp_old, adv, clip=0.2)
- One inner epoch per collected batch (ratio ~= 1, clip rarely
  binds — deliberate: stability over sample reuse).
- LoRA params only (r=16 existing adapter continued, NOT re-init),
  lr 5e-6 (RL lr << SFT lr; the 2e-4-diverged scar transfers).
- No KL-to-ref term in v0; the clip + tiny lr + gate tripwire are
  the drift guards. Add KL only if the gate catches drift.

## The climb loop

    collect 64 mixed groups -> 1 update epoch -> repeat;
    every 4 cycles: trimmed evaluate (n_per=24, budget 512, fresh
    band) -> checkpoint if validity >= last checkpoint - 0.1 else
    ROLLBACK to last checkpoint and halve lr (one retry, then HALT).

Verified steps found during collection append to the corpus
(source "grpo") — mining stays free, the flywheel is unchanged.

## Pre-registered bars (first sustained run, ~6h on the 3080)

- PRIMARY: held-out validity (the direct optimization target)
  > promoted baseline + 0.3pts. Validity is what the reward IS; if
  it doesn't move, the machinery is broken or the frontier is
  mis-selected — either is a finding.
- Solves >= baseline - 1 on the same band (no finishing collapse).
- Health meter: mixed-group rate over time. RISING = the frontier
  is expanding under the policy (the hill-climb working);
  FALLING toward all-fail = curriculum drifted too hard.
- Kill: two consecutive gate rollbacks (same as the SFT loop).

## Sequencing

1. Control round verdict lands first (H_diet vs H_variance) — if
   H_variance, GRPO jumps the queue over tournament-SFT (it answers
   variance structurally, not statistically).
2. Implementation: scripts/step_grpo.py (driver) + sample_batch
   logp extension. 3080 (CUDA); Mac carries the syndrome-head A/B
   meanwhile. Syndrome-head result composes: if multi-task helps,
   the GRPO base checkpoint starts from the multi-task adapter.
3. Self-distillation (MAI's other trick) banks as the follow-on:
   the climbing policy periodically re-mines its own solved chains
   into SFT data — our engine-replay seeding, closed on itself.

## Relation to banked threads

- Engine-regret hook: GRPO's mixed-group filter IS regret at the
  right unit cost (wave-level, ~50:1).
- Population/tournament gate: superseded if GRPO holds; revisit
  only if RL nulls.
- Latent-between-anchors: unchanged; GRPO trains the proposal
  distribution those anchors gate.
